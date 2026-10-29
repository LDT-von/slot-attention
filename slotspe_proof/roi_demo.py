"""Reproducible synthetic-only demonstration; never loads patient data."""

import hashlib
from pathlib import Path

import numpy as np

from .model import load_source, make_args
from .roi_trace import capture_verified
from .roi_workflow import _private, analyze_case, save_captured_case


def run_demo(output_dir):
    """Generate an invented slide, feature bag and model; emit private figures."""
    import torch

    root = Path(__file__).resolve().parents[1]
    private = root / "local_evidence"
    output = _private(output_dir, private)
    if output.exists():
        raise FileExistsError("Synthetic demo output directory already exists")
    torch.manual_seed(2026102700)
    args = make_args([2, 3], encoding_dim=8)
    args.wsi_projection_dim = 32
    args.slot_iters = 2
    source = load_source(root / "third_party" / "SlotSPE")
    model = source.SlotSPE(args).eval()
    features = torch.randn(1, 30, 8)
    features[0, 27:] = 0
    batch = {"x_wsi": features, "x_omic1": torch.tensor([[0.2, 0.8]]),
             "x_omic2": torch.tensor([[0.3, 0.5, 0.7]]), "omic_missing": False}
    patches = []
    for index in range(30):
        if index < 27:
            patches.append({"token_index": index, "is_real": True, "slide_id": "invented-slide",
                            "feature_row": index, "box": [index % 6 * 100, index // 6 * 100, 100, 100]})
        else:
            patches.append({"token_index": index, "is_real": False, "slide_id": None,
                            "feature_row": None, "box": None})
    names = ["invented-pathway-a", "invented-pathway-b"]
    capture = capture_verified(model, batch, row_manifest=patches, pathway_names=names,
                               source_root=root / "third_party" / "SlotSPE")
    source_hash = hashlib.sha256((root / "third_party/SlotSPE/models/SlotSPE.py").read_bytes()).hexdigest()
    case = {
        "schema": "slotspe_roi_case_v1", "patches": patches, "pathway_names": names,
        "slides": {"invented-slide": {"feature_count": 27, "mapping_verified": True,
                   "mapping_sha256": "a" * 64, "coordinate_space": "level0", "width": 600, "height": 500}},
        "polygons": {"invented-slide": {"coordinate_space": "level0", "geometries": [
            {"type": "Polygon", "coordinates": [[[0, 0], [500, 0], [500, 400], [0, 400], [0, 0]]]},
        ]}},
        "display_slide_id": "invented-slide", "thumbnail_slide_id": "invented-slide",
        "provenance": {"input_protocol": "synthetic_fixture", "split_role": "validation",
                       "checkpoint_sha256": capture["checks"]["model_state_sha256"],
                       "source_sha256": source_hash, "signature_sha256": "b" * 64, "split_sha256": "c" * 64,
                       "study": "invented", "feature_encoder": "generated vectors", "fold": 0,
                       "training_seed": 3, "evaluation_seed": 2026102700,
                       "checkpoint_selection": "generated random model state; no checkpoint file"},
    }
    rng = np.random.default_rng(4812)
    thumbnail = np.full((500, 600, 3), 245, dtype=np.uint8)
    tiles = np.full((30, 16, 16, 3), 245, dtype=np.uint8)
    for index in range(27):
        color = rng.integers(70, 220, size=3, dtype=np.uint8)
        x, y, width, height = patches[index]["box"]
        thumbnail[y:y + height, x:x + width] = color
        tiles[index] = color
    output.mkdir(parents=True, exist_ok=False)
    files = save_captured_case(capture, case, output / "capture", thumbnail=thumbnail, patch_tiles=tiles)
    result = analyze_case(files["case"], files["trace"], output / "analysis")
    result.update({"data_kind": "invented_synthetic_only", "patient_data_used": False,
                   "checkpoint_file_loaded": False, "training_steps": 0})
    return result
