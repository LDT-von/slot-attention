"""Analyze an explicitly bound private ROI/attention case; no model execution.

This entry point consumes previously captured attention. A declared provenance
record and SHA-256 binding detect accidental row reordering and file replacement;
they cannot independently establish the truth of the user's extraction records,
slide identity, biological annotations, or checkpoint provenance.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .roi_alignment import align_patch_annotations
from .roi_metrics import analyze_attention
from . import roi_visualization as viz


def rows_sha256(rows):
    """Hash ordered, full row records using the trace export's canonical JSON."""
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private(path, root, *, exists=False):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("case, trace, and output must be children of local_evidence")
    if exists and not path.is_file():
        raise FileNotFoundError(path)
    return path


def _scalar_text(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a scalar string, not an object array")
    item = array.item()
    return item.decode("ascii") if isinstance(item, bytes) else item


def _hex(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _provenance(prov, *, trace_required):
    if not isinstance(prov, dict):
        raise ValueError("Case provenance is required")
    if prov.get("input_protocol") not in ("original_full_validation", "fixed_4096_diagnostic", "synthetic_fixture"):
        raise ValueError("An explicit input protocol is required")
    if prov.get("split_role") != "validation":
        raise ValueError("This diagnostic requires a declared validation split")
    for key in ("checkpoint_sha256", "source_sha256", "signature_sha256", "split_sha256"):
        _hex(prov.get(key), key)
    if trace_required:
        _hex(prov.get("trace_sha256"), "trace_sha256")
    for key in ("fold", "training_seed", "evaluation_seed"):
        value = prov.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("study", "feature_encoder", "checkpoint_selection"):
        if not isinstance(prov.get(key), str) or not prov[key].strip():
            raise ValueError(f"{key} must be explicit")


def _rings(geometry):
    """Flatten Polygon/MultiPolygon boundaries for plotting, including holes."""
    rings = []
    for obj in geometry.get("geometries", []):
        if obj.get("type") == "Polygon":
            rings.extend(obj["coordinates"])
        elif obj.get("type") == "MultiPolygon":
            for polygon in obj["coordinates"]:
                rings.extend(polygon)
        else:
            raise ValueError("ROI plotting supports Polygon and MultiPolygon boundaries")
    return rings


def _compact(metrics):
    """Keep summary output bounded without discarding the saved native trace."""
    for group in [metrics["head_mean"], *metrics["by_head"]]:
        group.pop("real_attention", None)
    return metrics


def save_captured_case(capture, case, output_dir, *, thumbnail=None, patch_tiles=None, private_root=None):
    """Save an in-memory verified trace and case metadata without another forward.

    Capture must be produced by ``roi_trace.capture_verified``. The caller still
    owns the true identity of its model, original checkpoint, feature rows and
    annotations. This helper never opens those inputs or starts an experiment.
    """
    if private_root is None:
        private_root = Path(__file__).resolve().parents[1] / "local_evidence"
    output = _private(output_dir, private_root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing capture directory")
    if not isinstance(capture, dict) or capture.get("schema") != "slotspe_roi_verified_trace_v1":
        raise ValueError("A capture_verified result is required")
    checks = capture.get("checks", {})
    for name in ("entire_forward_output_exact", "slot_and_decoder_output_exact",
                 "post_forward_rng_exact", "outer_rng_restored", "model_state_unchanged"):
        if checks.get(name) is not True:
            raise ValueError("Capture equality or state checks are missing")
    prepared = deepcopy(case)
    if not isinstance(prepared, dict) or prepared.get("schema") != "slotspe_roi_case_v1":
        raise ValueError("Unsupported case schema")
    rows = capture.get("row_manifest")
    if prepared.get("patches") != rows or rows_sha256(rows) != capture.get("row_manifest_sha256"):
        raise ValueError("Case rows must exactly match the captured row manifest")
    if prepared.get("pathway_names") != capture.get("pathway_names"):
        raise ValueError("Case pathway order must exactly match the captured order")
    prov = prepared.get("provenance")
    _provenance(prov, trace_required=False)
    if prov["input_protocol"] != "synthetic_fixture":
        purity = capture.get("native_purity")
        if not isinstance(purity, dict) or not purity.get("checks") or not all(purity["checks"].values()):
            raise ValueError("Real cases require source-bound native-model purity verification")
        if purity.get("source_sha256", {}).get("models/SlotSPE.py") != prov["source_sha256"]:
            raise ValueError("Declared native source SHA-256 does not match the captured model")

    modalities = capture["modalities"]
    arrays = {"row_manifest_sha256": np.asarray(capture["row_manifest_sha256"]),
              "wsi_input_sha256": np.asarray(_hex(checks.get("wsi_input_sha256"), "wsi_input_sha256")),
              "omic_input_sha256": np.asarray(_hex(checks.get("omic_input_sha256"), "omic_input_sha256")),
              "logits": np.asarray(capture["logits"]),
              "rng_entry_cpu": np.asarray(capture["rng_entry"]["cpu"])}
    for device, state in capture["rng_entry"]["cuda"].items():
        if not isinstance(device, int) or device < 0:
            raise ValueError("CUDA RNG device must be a nonnegative integer")
        arrays[f"rng_entry_cuda_{device}"] = np.asarray(state)
    for modality in ("wsi", "omic"):
        item = modalities[modality]
        for name in ("last_attention_by_head", "attention_by_iteration", "slot_states", "slots",
                     "slot_gate", "selected_mask"):
            arrays[f"{modality}_{name}"] = np.asarray(item[name])
    if thumbnail is not None:
        arrays["thumbnail"] = np.asarray(thumbnail)
    if isinstance(patch_tiles, dict):
        if not patch_tiles or any(isinstance(key, bool) or not isinstance(key, (int, np.integer)) for key in patch_tiles):
            raise ValueError("sparse patch tiles require integer input-row keys")
        order = sorted(patch_tiles)
        arrays["tile_row_indices"] = np.asarray(order, dtype=np.int64)
        arrays["tile_images"] = np.stack([patch_tiles[i] for i in order])
    elif patch_tiles is not None:
        arrays["patch_tiles"] = np.asarray(patch_tiles)
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise ValueError("Object arrays cannot be serialized safely")
    aligned = align_patch_annotations(rows, prepared["slides"], prepared["polygons"],
                                      expected_token_count=arrays["wsi_last_attention_by_head"].shape[-1])
    if prov["input_protocol"] == "original_full_validation" and not all(aligned["real_mask"]):
        raise ValueError("Original full-validation input must not contain synthetic padding")
    if prov["input_protocol"] == "fixed_4096_diagnostic" and len(rows) != 4096:
        raise ValueError("Fixed 4096 diagnostic input must contain exactly 4096 rows")
    if arrays["omic_last_attention_by_head"].shape[-1] != len(prepared["pathway_names"]):
        raise ValueError("Captured OMIC token count differs from pathway metadata")
    output.mkdir(parents=True, exist_ok=False)
    trace_path = output / "trace.npz"
    np.savez_compressed(trace_path, **arrays)
    prov["trace_sha256"] = _sha(trace_path)
    prov["capture_checks"] = {"outputs_equal": True, "rng_equal": True, "weights_unchanged": True}
    prov["captured_model_state_sha256"] = checks["model_state_sha256"]
    prov["wsi_input_sha256"] = checks["wsi_input_sha256"]
    prov["omic_input_sha256"] = checks["omic_input_sha256"]
    prov["native_purity"] = capture.get("native_purity")
    prov["rng_entry_sha256"] = checks["rng_entry_sha256"]
    prov["capture_declaration"] = "model/checkpoint and real-world mapping supplied by caller; not independently audited"
    case_path = output / "case.json"
    case_path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"case": str(case_path), "trace": str(trace_path),
            "row_manifest_sha256": capture["row_manifest_sha256"], "trace_sha256": prov["trace_sha256"]}


def analyze_case(case_path, trace_path, output_dir, *, private_root=None):
    """Analyze one user-prepared case with no training or model forward.

    All three paths must be within ``local_evidence``; output must not exist.
    The NPZ is loaded with pickle disabled. No patient data leave that directory.
    """
    if private_root is None:
        private_root = Path(__file__).resolve().parents[1] / "local_evidence"
    root = Path(private_root).resolve()
    case_path = _private(case_path, root, exists=True)
    trace_path = _private(trace_path, root, exists=True)
    output = _private(output_dir, root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing output directory")
    case = json.loads(case_path.read_text(encoding="utf-8-sig"))
    if case.get("schema") != "slotspe_roi_case_v1":
        raise ValueError("Unsupported case schema")
    prov = case.get("provenance")
    _provenance(prov, trace_required=True)
    if prov["input_protocol"] != "synthetic_fixture":
        purity = prov.get("native_purity")
        if not isinstance(purity, dict) or not purity.get("checks") or not all(purity["checks"].values()):
            raise ValueError("Real cases require declared native-model purity checks")
        if purity.get("source_sha256", {}).get("models/SlotSPE.py") != prov.get("source_sha256"):
            raise ValueError("Declared source hash differs from native-model purity record")
    if _sha(trace_path) != prov["trace_sha256"]:
        raise ValueError("Trace file SHA-256 mismatch")
    checks = prov.get("capture_checks")
    required = ("outputs_equal", "rng_equal", "weights_unchanged")
    if not isinstance(checks, dict) or any(checks.get(k) is not True for k in required):
        raise ValueError("Capture must declare verified output, RNG and weight equality")

    patches = case.get("patches")
    slides = case.get("slides")
    polygons = case.get("polygons")
    if not isinstance(patches, list) or not isinstance(slides, dict) or not isinstance(polygons, dict):
        raise ValueError("patches, slides and polygons are required")
    digest = rows_sha256(patches)
    with np.load(trace_path, allow_pickle=False) as arrays:
        required_keys = {"wsi_last_attention_by_head", "omic_last_attention_by_head", "row_manifest_sha256",
                         "wsi_selected_mask", "omic_selected_mask",
                         "wsi_input_sha256", "omic_input_sha256"}
        if not required_keys.issubset(arrays.files):
            raise ValueError("Trace lacks WSI/OMIC attention or row-manifest binding")
        if _scalar_text(arrays["row_manifest_sha256"], "row_manifest_sha256") != digest:
            raise ValueError("Trace-to-row mapping SHA-256 mismatch")
        for name in ("wsi_input_sha256", "omic_input_sha256"):
            if _scalar_text(arrays[name], name) != _hex(prov.get(name), name):
                raise ValueError(f"Trace-to-provenance {name} mismatch")
        wsi = arrays["wsi_last_attention_by_head"].copy()
        omic = arrays["omic_last_attention_by_head"].copy()
        selected_masks = {name: arrays[f"{name}_selected_mask"].copy() for name in ("wsi", "omic")}
        thumbnail = arrays["thumbnail"].copy() if "thumbnail" in arrays.files else None
        tiles = arrays["patch_tiles"].copy() if "patch_tiles" in arrays.files else None
        if "tile_images" in arrays.files or "tile_row_indices" in arrays.files:
            if tiles is not None or not {"tile_images", "tile_row_indices"}.issubset(arrays.files):
                raise ValueError("Use either full tiles or both sparse tile arrays")
            tile_ids = arrays["tile_row_indices"].copy()
            tile_images = arrays["tile_images"].copy()
            if (tile_ids.ndim != 1 or tile_ids.dtype.kind not in "iu" or
                    len(tile_ids) != len(tile_images) or len(set(tile_ids.tolist())) != len(tile_ids)):
                raise ValueError("Sparse tile indices must be unique and match their images")
            tiles = {int(key): value for key, value in zip(tile_ids, tile_images)}

    if wsi.ndim != 3 or omic.ndim != 3:
        raise ValueError("Attention arrays must have [heads, slots, tokens] shape")
    selected = {}
    for name, attention in (("wsi", wsi), ("omic", omic)):
        mask = selected_masks[name]
        if (mask.shape != (attention.shape[1],) or not np.isfinite(mask).all()
                or not np.isin(mask, (0, 1)).all() or not np.any(mask)):
            raise ValueError(f"{name} selected mask must contain one binary value per slot")
        selected[name] = np.flatnonzero(mask).tolist()
    aligned = align_patch_annotations(patches, slides, polygons, expected_token_count=wsi.shape[-1])
    real = np.asarray(aligned["real_mask"], dtype=bool)
    tumor = np.asarray(aligned["tumor_mask"], dtype=bool)
    if prov["input_protocol"] == "original_full_validation" and not real.all():
        raise ValueError("Original full-validation input must not contain synthetic padding")
    if prov["input_protocol"] == "fixed_4096_diagnostic" and len(real) != 4096:
        raise ValueError("Fixed 4096 diagnostic input must contain exactly 4096 rows")
    pathways = case.get("pathway_names")
    if not isinstance(pathways, list) or len(pathways) != omic.shape[-1]:
        raise ValueError("One pathway name is required per OMIC input token")
    if any(not isinstance(n, str) or not n.strip() for n in pathways) or len(set(pathways)) != len(pathways):
        raise ValueError("Pathway names must be unique, nonempty strings")
    wsi_metrics = analyze_attention(wsi, real, tumor)
    omic_metrics = analyze_attention(omic, np.ones(omic.shape[-1], dtype=bool),
                                     np.zeros(omic.shape[-1], dtype=bool))
    if wsi_metrics["head_mean"]["status"] != "ok" or omic_metrics["head_mean"]["status"] != "ok":
        raise ValueError("Real input attention must be evaluable before plotting")
    sensitivity = {}
    for threshold, mask in aligned["sensitivity"].items():
        result = analyze_attention(wsi, real, np.asarray(mask, dtype=bool))
        sensitivity[threshold] = result["head_mean"]["tumor"]

    wsi_group = wsi_metrics["head_mean"]
    weights = np.zeros((wsi.shape[1], wsi.shape[2]), dtype=np.float64)
    weights[:, wsi_group["real_input_row_indices"]] = wsi_group["real_attention"]
    top = np.asarray([slot["input_row_indices"] for slot in wsi_group["top"]["slots"]], dtype=np.int64)
    figures = {}
    figure_meta = {}
    figures["overlap_matrix"], figure_meta["overlap_matrix"] = viz.render_overlap_matrix(wsi_group["overlap"])
    figures["pathway_heatmap"], figure_meta["pathway_heatmap"] = viz.render_pathway_heatmap(
        np.asarray(omic_metrics["head_mean"]["real_attention"]), pathways)
    figures["top_gallery"], figure_meta["top_gallery"] = viz.render_top_gallery(tiles, top, real, tumor)
    slide_id = case.get("display_slide_id")
    if thumbnail is not None:
        if slide_id not in polygons or slide_id not in slides:
            raise ValueError("Thumbnail needs an exact annotated display slide ID")
        if case.get("thumbnail_slide_id") != slide_id:
            raise ValueError("Thumbnail slide ID must match the display slide")
        slide = slides[slide_id]
        size = (slide["width"], slide["height"])
        boxes = np.array([p["box"] if p.get("box") is not None else [np.nan] * 4 for p in patches], dtype=float)
        visible = np.array([p["is_real"] and p["slide_id"] == slide_id and p["box"] is not None
                            for p in patches], dtype=bool)
        rings = _rings(polygons[slide_id])
        figures["roi_overlay"], figure_meta["roi_overlay"] = viz.render_roi_overlay(
            thumbnail, size, rings, boxes, visible, real)
        figures["slot_heatmaps"], figure_meta["slot_heatmaps"] = viz.render_slot_heatmaps(
            thumbnail, size, rings, boxes, weights, visible, real)
    else:
        for key in ("roi_overlay", "slot_heatmaps"):
            figures[key] = None
            figure_meta[key] = {"status": "missing_thumbnail"}

    if prov["input_protocol"] == "synthetic_fixture":
        for fig in figures.values():
            if fig is not None:
                existing = fig._suptitle
                suffix = "\n" + existing.get_text() if existing is not None else ""
                fig.suptitle("SYNTHETIC DATA - NO PATIENT" + suffix, color="darkred", fontsize=11)

    # Plot helper itself additionally enforces that output is in this checkout's
    # local_evidence; tests can monkeypatch its __file__ to a temporary checkout.
    receipt = viz.save_figures(figures, output)
    summary = {
        "schema": "slotspe_roi_case_analysis_v1",
        "case_file_sha256": _sha(case_path), "trace_file_sha256": _sha(trace_path),
        "row_manifest_sha256": digest, "provenance": prov,
        "provenance_scope": "file hashes checked; extraction, slide identity and capture declarations not independently audited",
        "label_scope": "annotation-positive versus unknown; not normal classification",
        "no_model_forward_or_training": True,
        "alignment": aligned, "wsi": _compact(wsi_metrics), "omic": _compact(omic_metrics),
        "selected_slot_indices_zero_based": selected,
        "selected_wsi_overlap": np.asarray(wsi_group["overlap"])[np.ix_(selected["wsi"], selected["wsi"])].tolist(),
        "selected_slot_scope": "native selection masks; indices describe selected slots, not gene attribution",
        "roi_threshold_sensitivity_head_mean": sensitivity,
        "pathway_names": pathways, "figures": figure_meta, "figure_files": receipt,
    }
    (output / "case_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "completed", "output_dir": str(output), "n_real": int(real.sum()),
            "n_annotated_tumor": int(tumor.sum()), "tumor_status": wsi_group["tumor"]["status"],
            "missing_figures": receipt["missing"], "patient_data_publicly_exported": False}


def analyze_cohort(manifest_path, output_dir, *, private_root=None):
    """Aggregate hash-bound case summaries inside local_evidence only."""
    from .roi_cohort import summarize_cohort

    if private_root is None:
        private_root = Path(__file__).resolve().parents[1] / "local_evidence"
    root = Path(private_root).resolve()
    manifest_path = _private(manifest_path, root, exists=True)
    output = _private(output_dir, root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing cohort output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != "slotspe_roi_cohort_manifest_v1" or not isinstance(manifest.get("records"), list):
        raise ValueError("Unsupported cohort manifest")
    records = []
    seen = set()
    for row in manifest["records"]:
        if not isinstance(row, dict) or not isinstance(row.get("case_key"), str) or not row["case_key"].strip():
            raise ValueError("Each cohort record requires a private case_key")
        location = row.get("summary_path")
        if not isinstance(location, str) or not location:
            raise ValueError("Each cohort record requires summary_path")
        summary_path = Path(location)
        if not summary_path.is_absolute():
            summary_path = manifest_path.parent / summary_path
        summary_path = _private(summary_path, root, exists=True)
        if summary_path in seen:
            raise ValueError("A case summary cannot be included twice")
        seen.add(summary_path)
        if _sha(summary_path) != _hex(row.get("sha256"), "summary sha256"):
            raise ValueError("Case summary SHA-256 mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        records.append({"case_key": row["case_key"], "summary": summary})
    result = summarize_cohort(records)
    result["manifest_sha256"] = _sha(manifest_path)
    result["provenance_scope"] = "individual summary hashes checked; real-world identities and mappings remain caller attestations"
    output.mkdir(parents=True, exist_ok=False)
    (output / "cohort_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": result["status"], "output_dir": str(output),
            "patient_count_input": result["patient_count_input"],
            "patient_count_complete": result["patient_count_complete"],
            "patient_count_incomplete": result["patient_count_incomplete"]}
