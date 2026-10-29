"""End-to-end checks with generated inputs, images and weights only."""

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("shapely")
pytest.importorskip("matplotlib")

from slotspe_proof import roi_visualization as viz
from slotspe_proof.roi_trace import capture_verified
from slotspe_proof.roi_workflow import analyze_case, analyze_cohort, rows_sha256, save_captured_case


@pytest.fixture
def generated_capture():
    from slotspe_proof.model import load_source, make_args

    root = Path(__file__).resolve().parents[1]
    source = load_source(root / "third_party" / "SlotSPE")
    args = make_args([2, 3], encoding_dim=8)
    args.wsi_projection_dim = 32
    args.slot_iters = 2
    torch.manual_seed(2402)
    model = source.SlotSPE(args).eval()
    x = torch.randn(1, 6, 8)
    x[0, 5] = 0
    batch = {"x_wsi": x, "x_omic1": torch.ones(1, 2),
             "x_omic2": torch.ones(1, 3), "omic_missing": False}
    rows = [
        {"token_index": i, "is_real": True, "slide_id": "fake-slide-a",
         "feature_row": i, "box": [i * 20, 0, 20, 20]}
        for i in range(4)
    ] + [
        {"token_index": 4, "is_real": True, "slide_id": "fake-slide-b", "feature_row": 0, "box": None},
        {"token_index": 5, "is_real": False, "slide_id": None, "feature_row": None, "box": None},
    ]
    names = ["fake-pathway-a", "fake-pathway-b"]
    capture = capture_verified(model, batch, row_manifest=rows, pathway_names=names)
    case = {
        "schema": "slotspe_roi_case_v1", "patches": rows, "pathway_names": names,
        "slides": {
            "fake-slide-a": {"feature_count": 4, "mapping_verified": True, "mapping_sha256": "a" * 64,
                             "coordinate_space": "level0", "width": 100, "height": 100},
            "fake-slide-b": {"feature_count": 1, "mapping_verified": True, "mapping_sha256": "b" * 64},
        },
        "polygons": {"fake-slide-a": {"coordinate_space": "level0", "geometries": [
            {"type": "Polygon", "coordinates": [[[0, 0], [40, 0], [40, 20], [0, 20], [0, 0]]]},
        ]}},
        "display_slide_id": "fake-slide-a", "thumbnail_slide_id": "fake-slide-a",
        "provenance": {"input_protocol": "synthetic_fixture", "split_role": "validation",
                       "checkpoint_sha256": "c" * 64, "source_sha256": "d" * 64,
                       "signature_sha256": "e" * 64, "split_sha256": "f" * 64,
                       "study": "invented", "feature_encoder": "generated vectors", "fold": 0, "training_seed": 3,
                       "evaluation_seed": 2402, "checkpoint_selection": "generated only"},
    }
    return capture, case


def test_original_model_to_private_case_to_metrics_and_figures(tmp_path, monkeypatch, generated_capture):
    capture, case = generated_capture
    private = tmp_path / "local_evidence"
    private.mkdir()
    monkeypatch.setattr(viz, "__file__", str(tmp_path / "slotspe_proof" / "roi_visualization.py"))
    assert rows_sha256(case["patches"]) == capture["row_manifest_sha256"]
    files = save_captured_case(capture, case, private / "capture", private_root=private,
                               thumbnail=np.full((100, 100, 3), 220, dtype=np.uint8))
    receipt = analyze_case(files["case"], files["trace"], private / "analysis", private_root=private)
    assert receipt["n_real"] == 5 and receipt["n_annotated_tumor"] == 2
    assert receipt["tumor_status"] == "not_evaluable_insufficient_annotated_tumor"
    assert receipt["missing_figures"] == ["top_gallery"]
    summary = json.loads((private / "analysis" / "case_summary.json").read_text(encoding="utf-8"))
    assert summary["alignment"]["counts"]["unknown"] == 3
    assert summary["alignment"]["counts"]["padding"] == 1
    assert summary["wsi"]["head_mean"]["tumor"]["q"] == pytest.approx(2 / 5)
    assert summary["wsi"]["n_padding"] == 1
    assert summary["omic"]["n_real"] == 2
    selected = summary["selected_slot_indices_zero_based"]
    assert selected["wsi"] == np.flatnonzero(capture["modalities"]["wsi"]["selected_mask"]).tolist()
    assert selected["omic"] == np.flatnonzero(capture["modalities"]["omic"]["selected_mask"]).tolist()
    assert np.asarray(summary["selected_wsi_overlap"]).shape == (len(selected["wsi"]),) * 2
    assert "not independently audited" in summary["provenance_scope"]
    for name in ("roi_overlay", "slot_heatmaps", "overlap_matrix", "pathway_heatmap"):
        assert (private / "analysis" / (name + ".png")).is_file()


def test_row_tampering_is_rejected_before_any_output(tmp_path, generated_capture):
    capture, case = generated_capture
    private = tmp_path / "local_evidence"
    private.mkdir()
    files = save_captured_case(capture, case, private / "capture", private_root=private)
    case_path = Path(files["case"])
    changed = json.loads(case_path.read_text(encoding="utf-8"))
    changed["patches"][0]["box"][0] = 1
    case_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping SHA-256 mismatch"):
        analyze_case(case_path, files["trace"], private / "result", private_root=private)
    assert not (private / "result").exists()


def test_trace_and_output_guards(tmp_path, generated_capture):
    capture, case = generated_capture
    private = tmp_path / "local_evidence"
    private.mkdir()
    with pytest.raises(ValueError, match="local_evidence"):
        save_captured_case(capture, case, tmp_path / "public", private_root=private)
    files = save_captured_case(capture, case, private / "capture", private_root=private)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_captured_case(capture, case, private / "capture", private_root=private)
    trace = Path(files["trace"])
    with trace.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Trace file SHA-256 mismatch"):
        analyze_case(files["case"], trace, private / "result", private_root=private)
    assert not (private / "result").exists()


def test_plan_does_not_need_or_read_declared_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = subprocess.check_output([sys.executable, "-B", str(root / "scripts" / "analyze_slotspe_annotated_roi.py"),
                                      "--plan", "--case", str(tmp_path / "nonexistent")], cwd=root)
    assert json.loads(output)["action"] == "plan_only"
    assert list(tmp_path.iterdir()) == []


def test_private_cohort_manifest_counts_one_patient_not_twenty_repeats(tmp_path, monkeypatch, generated_capture):
    capture, case = generated_capture
    private = tmp_path / "local_evidence"
    private.mkdir()
    monkeypatch.setattr(viz, "__file__", str(tmp_path / "slotspe_proof" / "roi_visualization.py"))
    files = save_captured_case(capture, case, private / "capture", private_root=private)
    analyze_case(files["case"], files["trace"], private / "one", private_root=private)
    summary = json.loads((private / "one" / "case_summary.json").read_text(encoding="utf-8"))
    entries = []
    for seed in range(2026102700, 2026102720):
        summary["provenance"]["evaluation_seed"] = seed
        summary["provenance"]["rng_entry_sha256"] = f"{seed:064x}"
        path = private / f"generated_{seed}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        entries.append({"case_key": "invented-private-case", "summary_path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = private / "cohort.json"
    manifest.write_text(json.dumps({"schema": "slotspe_roi_cohort_manifest_v1", "records": entries}), encoding="utf-8")
    result = analyze_cohort(manifest, private / "cohort_result", private_root=private)
    assert result["patient_count_input"] == result["patient_count_complete"] == 1
    contents = (private / "cohort_result" / "cohort_summary.json").read_text(encoding="utf-8")
    assert "invented-private-case" not in contents
    assert json.loads(contents)["observation_count_input"] == 20
    entries[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"schema": "slotspe_roi_cohort_manifest_v1", "records": entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        analyze_cohort(manifest, private / "rejected", private_root=private)
    assert not (private / "rejected").exists()
