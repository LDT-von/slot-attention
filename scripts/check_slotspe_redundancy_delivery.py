"""Opt-in full queue smoke on invented CSV/HDF5 fixtures, never patient data.

The encoder label in the fixture exercises the UNI2-h input contract only;
these random eight-dimensional tensors were not produced by a WSI encoder.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local_evidence/redundancy_delivery_generated_20261022_run01"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    if not parser.parse_args().synthetic:
        print("--synthetic runs all 14 queue stages with seven invented cases, CPU, native 256/8/10; no patient inputs")
        return
    if OUT.exists():
        raise FileExistsError("preserve earlier smoke evidence; fixed generated output already exists")
    import numpy as np
    import pandas as pd
    import h5py

    metadata, features = OUT / "invented_metadata", OUT / "invented_features"
    for sub in ("clinical/all", "splits/5fold/blca", "raw_rna_data_inter", "signatures"):
        (metadata / sub).mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True)
    ids = [f"INVENTED_QUEUE_CASE_{index}" for index in range(7)]
    pd.DataFrame({"case id": ids, "survival_months_dss": [1., 2., 3., 4., 5., 6., 8.],
                  "censorship_dss": [0] * 7, "wsi": [value + ".svs" for value in ids]}).to_csv(metadata / "clinical/all/blca.csv", index=False)
    pd.DataFrame({"train": pd.Series(ids[:5]), "val": pd.Series(ids[5:])}).to_csv(metadata / "splits/5fold/blca/fold_0.csv", index=False)
    genes = [f"INVENTED_GENE_{index}" for index in range(6)]
    generator = np.random.default_rng(51623)
    pd.DataFrame({pid: generator.normal(size=6) for pid in ids}, index=genes).to_csv(metadata / "raw_rna_data_inter/blca_rna_inter.csv")
    pd.DataFrame({"INVENTED_PATH_1": pd.Series(genes[:2]), "INVENTED_PATH_2": pd.Series(genes[1:4]),
                  "INVENTED_PATH_3": pd.Series(genes[2:])}).to_csv(metadata / "signatures/combine_signatures.csv", index=False)
    for index, pid in enumerate(ids):
        values = generator.normal(size=(10, 8)).astype("float32")
        with h5py.File(features / (pid + ".h5"), "w") as handle:
            handle.create_dataset("features", data=values if index % 2 else values[None])
    config = copy.deepcopy(json.loads((ROOT / "configs/slotspe_redundancy_queue.json").read_text(encoding="utf-8")))
    stage = next(item for item in config["stages"] if item["id"] == "real_data_diagnostic")
    stage.update(epochs=1, real_patches=8, batch_size=2, eval_repeats=2)
    config["execution"] = {**config.get("execution", {}), "metadata_root": str(metadata),
                            "features_root": str(features), "device": "cpu", "torch_threads": 1}
    profile = OUT / "invented_fixture_config.json"
    profile.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    queue_output = OUT / "queue"
    command = [sys.executable, "-B", str(ROOT / "scripts/run_slotspe_redundancy.py"),
               "--config", str(profile), "--output-root", str(queue_output), "--run"]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        raise RuntimeError("generated full queue failed; retained private logs and partial stage")
    summary = json.loads((queue_output / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((queue_output / "queue_manifest.json").read_text(encoding="utf-8"))
    if summary["completed_unique_training_runs"] != 11 or manifest["status"] != "completed":
        raise RuntimeError("full queue smoke is incomplete")
    # Verify completed-stage resume without retraining any generated case.
    resumed = subprocess.run(command + ["--resume"], cwd=ROOT)
    if resumed.returncode:
        raise RuntimeError("completed generated queue did not resume cleanly")
    counts = {}
    for row in summary["arms"].values():
        for key, value in row["counters"].items():
            counts[key] = counts.get(key, 0) + value
    receipt = {"schema": "slotspe_generated_delivery_smoke_v1", "status": "completed_generated_only",
               "data": "seven invented cases and random 8-D features, not actual UNI2-h or patient data",
               "stages": len(manifest["records"]), "training_arms": list(summary["arms"]),
               "native_projection_dim": 256, "native_slots": 8, "native_slot_iterations": 10,
               "epochs": 1, "patches": 8, "eval_repeats": 2, "counters": counts,
               "source_and_configuration_pins_verified": True, "completed_resume_retrained": False,
               "patient_runs": 0, "server_runs": 0,
               "private_summary_sha256": hashlib.sha256((queue_output / "summary.json").read_bytes()).hexdigest()}
    (OUT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
