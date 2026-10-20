"""Read-only execution probes of the original SlotSPE data protocol.

No CSV, checkpoint, or original source is changed. The only counterfactual
modifies a copied clinical DataFrame in memory. No model is trained.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sksurv.util import Surv


SOURCE_ROOT = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main")
OUTPUT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = SOURCE_ROOT / "dataset_csv"
WSI_ROOT = Path(r"E:\TCGA-huggingface\blca\uni\pt_files")
LABEL = "survival_months_dss"
CENSOR = "censorship_dss"
N_BINS = 4
EPS = 1e-6


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges_quantile(frame: pd.DataFrame) -> np.ndarray:
    events = frame.loc[frame[CENSOR] < 1, LABEL]
    _, edges = pd.qcut(events, q=N_BINS, retbins=True, labels=False)
    edges[0] = events.min() - EPS
    edges[-1] = events.max() + EPS
    return edges


def edges_equal_width(frame: pd.DataFrame) -> np.ndarray:
    _, edges = pd.cut(frame[LABEL], bins=N_BINS, retbins=True, labels=False,
                      right=False, include_lowest=True)
    return edges


def describe_bins(edges: np.ndarray, frame: pd.DataFrame) -> dict:
    labels = pd.cut(frame[LABEL], bins=edges, labels=False, right=False,
                    include_lowest=True)
    return {
        "edges": edges.tolist(),
        "label_counts_0_to_3": [int((labels == i).sum()) for i in range(N_BINS)],
        "out_of_range_count": int(labels.isna().sum()),
    }


def main() -> None:
    sys.dont_write_bytecode = True
    dataset_path = SOURCE_ROOT / "dataset" / "dataset_survival.py"
    core_path = SOURCE_ROOT / "utils" / "core_utils.py"
    clinical_path = DATA_ROOT / "clinical" / "all" / "blca.csv"
    split_path = DATA_ROOT / "splits" / "5fold" / "blca" / "fold_0.csv"
    spec = importlib.util.spec_from_file_location("slotspe_original_dataset_probe", dataset_path)
    dataset_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(dataset_module)
    factory = dataset_module.SurvivalDatasetFactory(
        study="blca", data_path=str(DATA_ROOT), rna_format="Pathways",
        label_col=LABEL, signature="combine", n_bins=N_BINS,
        eps=EPS, num_patches=4096,
    )
    # This invokes the original split-loading implementation, without __getitem__.
    train_data = dataset_module.SurvivalDataset(factory, str(WSI_ROOT), "train", 0, 1024)
    val_data = dataset_module.SurvivalDataset(factory, str(WSI_ROOT), "val", 0, 1024)
    full_df = factory.clinical_df.copy(deep=True)
    train_df = train_data.label_df.copy(deep=True)
    val_df = val_data.label_df.copy(deep=True)
    train_ids = set(train_df["case id"])
    val_ids = set(val_df["case id"])
    rna_ids = set(factory.gene_data_df.columns)

    bins = {
        "actual_original_full_cohort_equal_width": describe_bins(factory.bins, full_df),
        "full_uncensored_quantiles_computed_then_discarded": describe_bins(edges_quantile(full_df), full_df),
        "train_only_equal_width_applied_to_full": describe_bins(edges_equal_width(train_df), full_df),
        "train_only_uncensored_quantiles_applied_to_full": describe_bins(edges_quantile(train_df), full_df),
    }
    expected_current_edges = edges_equal_width(full_df)
    assert np.array_equal(factory.bins, expected_current_edges)

    # Select one actual validation patient and change only this copied time.
    selected = val_df.loc[val_df[LABEL].idxmax(), "case id"]
    altered = copy.copy(factory)
    altered.clinical_df = full_df.drop(columns=["label"]).copy(deep=True)
    selected_mask = altered.clinical_df["case id"] == selected
    old_time = float(altered.clinical_df.loc[selected_mask, LABEL].iloc[0])
    new_time = float(full_df[LABEL].max() * 1000.0 + 1.0)
    altered.clinical_df.loc[selected_mask, LABEL] = new_time
    altered._disc_label(altered._get_uncensored_data())
    before_labels = full_df.set_index("case id")["label"]
    after_labels = altered.clinical_df.set_index("case id")["label"]
    train_before = before_labels.loc[train_df["case id"]]
    train_after = after_labels.loc[train_df["case id"]]
    train_time_before = full_df.set_index("case id").loc[train_df["case id"], LABEL]
    train_time_after = altered.clinical_df.set_index("case id").loc[train_df["case id"], LABEL]
    assert train_time_before.equals(train_time_after)
    altered_train = altered.clinical_df[altered.clinical_df["case id"].isin(train_ids)]

    # Execute the exact original function body, avoiding unrelated trainer imports.
    tree = ast.parse(core_path.read_text(encoding="utf-8"))
    original_function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name == "_extract_survival_metadata")
    isolated_module = ast.Module(body=[original_function], type_ignores=[])
    namespace = {"Surv": Surv}
    exec(compile(isolated_module, str(core_path), "exec"), namespace)
    extract = namespace["_extract_survival_metadata"]
    metadata_all = extract(factory)
    metadata_train = extract(SimpleNamespace(clinical_df=train_df, censorship_var=CENSOR, label_col=LABEL))
    metadata_val = extract(SimpleNamespace(clinical_df=val_df, censorship_var=CENSOR, label_col=LABEL))

    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Original SlotSPE BLCA DSS fold 0, real metadata, read-only execution; no model training.",
        "python": {"executable": sys.executable, "version": sys.version},
        "sources": {
            str(p): {"sha256": file_hash(p)}
            for p in [dataset_path, core_path, clinical_path, split_path]
        },
        "counts": {
            "factory_full_clinical_after_dropna": len(full_df),
            "original_train_dataset": len(train_data),
            "original_validation_dataset": len(val_data),
            "train_validation_overlap": len(train_ids & val_ids),
            "train_missing_rna": len(train_ids - rna_ids),
            "validation_missing_rna": len(val_ids - rna_ids),
        },
        "bins": bins,
        "validation_time_counterfactual": {
            "patient_split": "validation",
            "selected_public_tcga_case_id": selected,
            "old_survival_months": old_time,
            "in_memory_new_survival_months": new_time,
            "original_bins": factory.bins.tolist(),
            "counterfactual_bins": altered.bins.tolist(),
            "training_patient_times_changed": int((train_time_before != train_time_after).sum()),
            "training_patient_discrete_labels_changed": int((train_before != train_after).sum()),
            "training_label_counts_before": [int((train_before == i).sum()) for i in range(N_BINS)],
            "training_label_counts_after": [int((train_after == i).sum()) for i in range(N_BINS)],
            "train_only_equal_width_edges_unchanged": bool(np.array_equal(edges_equal_width(train_df), edges_equal_width(altered_train))),
            "train_only_quantile_edges_unchanged": bool(np.array_equal(edges_quantile(train_df), edges_quantile(altered_train))),
            "files_modified": [],
        },
        "survival_metadata_original_function": {
            "execution_method": "Execute unchanged AST function body from utils/core_utils.py with sksurv.util.Surv.",
            "source_lines": [original_function.lineno, original_function.end_lineno],
            "returned_full_factory_count": len(metadata_all),
            "returned_train_only_wrapper_count": len(metadata_train),
            "returned_validation_only_wrapper_count": len(metadata_val),
            "full_factory_includes_validation_patients": val_ids.issubset(set(full_df["case id"])),
            "full_factory_event_count": int(metadata_all["event"].sum()),
            "train_event_count": int(metadata_train["event"].sum()),
            "validation_event_count": int(metadata_val["event"].sum()),
        },
        "interpretation_limits": [
            "The observed relabeling proves validation outcomes affect training targets in this implementation.",
            "This is a protocol/implementation issue, not evidence of a new model module or evidence that OT improves performance.",
            "No trained checkpoint, predictive performance, C-index inflation magnitude, or corrected-model performance is evaluated here.",
            "Train-only reference bins show the boundary source; out-of-range counts are disclosed rather than silently clipping held-out times.",
        ],
    }
    destination = OUTPUT_ROOT / "data_protocol_results.json"
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "counts": results["counts"],
                      "changed_train_labels": results["validation_time_counterfactual"]["training_patient_discrete_labels_changed"],
                      "metadata_count": len(metadata_all)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
