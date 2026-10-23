"""Audit saved full-model gate observations without torch or model execution.

The result and its patient identifiers remain in ignored local_evidence. This
auditor reads only JSON, raw file bytes and local upstream Git blobs. Any failed
check returns a nonzero exit status and writes a failed receipt, never a pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = REPO / "local_evidence/gate_collapse_audit_20261022/full_model_rng01/gate_diagnostic.json"
MODALITIES = ("wsi", "omic")
CONTROL_KEYS = (
    "temperature_1_same_noise_gate_sum",
    "no_noise_temperature_0_01_gate_sum",
    "masked_softmax_same_mask_gate_sum",
)
TOLERANCES = {
    "gate_weight_sum_abs": 2e-7,
    "float32_formula_abs": 2e-6,
    "float32_mass_abs": 2e-6,
    "control_gate_sum_abs": 3e-7,
    "float64_abs": 2e-12,
    "summary_abs": 2e-12,
    "summary_rel": 2e-12,
}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def reject_nonfinite(value):
    raise AuditError("Nonfinite JSON constant")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicates, parse_constant=reject_nonfinite)


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label + " is not numeric")
    require(math.isfinite(value), label + " is not finite")
    return float(value)


def same(actual, expected, label, abs_tol=None, rel_tol=None):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + " object fields differ")
        for key in expected:
            same(actual[key], expected[key], label + "." + str(key), abs_tol, rel_tol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " list shape differs")
        for index, value in enumerate(expected):
            same(actual[index], value, label + "[" + str(index) + "]", abs_tol, rel_tol)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, label + " value differs")
    elif isinstance(expected, int):
        require(isinstance(actual, int) and not isinstance(actual, bool) and actual == expected,
                label + " integer differs")
    else:
        actual = number(actual, label)
        require(math.isclose(actual, float(expected),
                             abs_tol=TOLERANCES["summary_abs"] if abs_tol is None else abs_tol,
                             rel_tol=TOLERANCES["summary_rel"] if rel_tol is None else rel_tol),
                label + " numeric value differs")


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(),
            "Invalid distribution input")
    n = len(values)
    counts = [int(np.count_nonzero(values < threshold)) for threshold in (.5, .01)]
    return {
        "n": n, "median": float(np.median(values)), "mean": float(np.mean(values)),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "below_0_5_count": counts[0], "below_0_5_fraction": counts[0] / n,
        "below_0_01_count": counts[1], "below_0_01_fraction": counts[1] / n,
    }


def softmax(values):
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def contained(path, root, label):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()), label + " escapes required directory")
    return path


def git(clone, *args):
    completed = subprocess.run(["git", "-C", str(clone), *args], capture_output=True, check=True)
    return completed.stdout


def check_provenance(result, result_path, source_path):
    receipt = read(source_path)
    same(result["source_verification"], receipt, "source_verification")
    comparison = receipt["comparison"]
    require(comparison["runtime_difference_count"] == 0 and
            comparison["runtime_file_equal_count"] == comparison["runtime_file_count"] == 10,
            "Official runtime comparison did not pass")
    require(receipt["official_pinned_commit"] == "02051a2083add7b427727e0200e4263516903561",
            "Unexpected official fixed commit")
    clone = contained(REPO / receipt["clone_relative_path"], REPO / "local_evidence", "Official clone")
    tree = git(clone, "rev-parse", receipt["official_pinned_commit"] + "^{tree}").decode().strip()
    same(tree, receipt["official_pinned_tree_sha1"], "official_pinned_tree_sha1")
    require(not git(clone, "status", "--porcelain").strip(), "Official clone worktree changed")
    watched = {result_path: sha(result_path), source_path: sha(source_path)}
    runtime_paths, runtime_count = set(), 0
    for entry in receipt["file_comparisons"]:
        relative = entry["path"]
        vendor = contained(REPO / "third_party/SlotSPE" / relative, REPO / "third_party/SlotSPE", "Vendor file")
        official = contained(clone / relative, clone, "Official file")
        vendor_hash, official_hash = sha(vendor), sha(official)
        same(vendor_hash, entry["vendored_sha256"], "vendor file hash")
        same(official_hash, entry["official_sha256"], "official file hash")
        pinned_blob = git(clone, "show", receipt["official_pinned_commit"] + ":" + relative)
        require(official.read_bytes() == pinned_blob, "Official checkout differs from fixed Git blob")
        same(hashlib.sha1(b"blob " + str(len(pinned_blob)).encode() + b"\0" + pinned_blob).hexdigest(),
             entry["official_git_blob_sha1"], "official Git blob identity")
        watched[vendor], watched[official] = vendor_hash, official_hash
        if entry["classification"] != "documentation":
            require(vendor_hash == official_hash, "Runtime file differs from official commit")
            runtime_count += 1
            runtime_paths.add(vendor.relative_to(REPO).as_posix())
    require(runtime_count == 10, "Runtime enumeration count differs")
    input_hashes = result["input_sha256"]
    require(isinstance(input_hashes, dict) and len(input_hashes) == 15, "Unexpected input hash inventory")
    require(runtime_paths.issubset(input_hashes), "Runtime source missing from input hashes")
    by_basename = {}
    for relative, expected in input_hashes.items():
        path = contained(REPO / relative, REPO, "Recorded input")
        require(path.is_file(), "Recorded input missing")
        same(sha(path), expected, "Recorded input SHA256")
        watched[path] = expected
        by_basename.setdefault(path.name, []).append(path)
    required = ("observations.json", "paired_blca_fold0.pt", "last.pt", "eval_repeats.json", "manifest.json")
    require(all(len(by_basename.get(name, [])) == 1 for name in required), "Required patient input inventory differs")
    observations = read(by_basename["observations.json"][0])
    for name in required[1:]:
        path = by_basename[name][0]
        expected = [digest for old, digest in observations["source_inputs_sha256"].items()
                    if Path(old.replace("\\", "/")).name == name]
        require(len(expected) == 1 and sha(path) == expected[0], "Historical input provenance differs")
    historical = read(by_basename["eval_repeats.json"][0])
    same([entry["eval_seed"] for entry in historical], result["protocol"]["eval_seeds"],
         "Historical eval seed sequence")
    ids = [row["patient_id"] for row in observations["records"]]
    require(ids == [row["patient_id"] for row in result["per_patient"]], "Historical patient order differs")
    runner = REPO / "scripts/check_slotspe_gate_full_rng.py"
    runner_hash = sha(runner)
    same(runner_hash, result["checks"]["runner_sha256"], "Full RNG runner SHA256")
    watched[runner] = runner_hash
    return watched, {
        "source_receipt_sha256": watched[source_path], "official_pinned_commit": receipt["official_pinned_commit"],
        "official_tree_sha1": tree, "official_clone_clean": True,
        "runtime_files_verified": runtime_count, "input_sha256_verified_count": len(input_hashes),
        "fixed_git_blobs_verified_count": len(receipt["file_comparisons"]),
        "checkpoint_sha256": sha(by_basename["last.pt"][0]),
        "cache_sha256": sha(by_basename["paired_blca_fold0.pt"][0]), "runner_sha256": runner_hash,
    }


def recompute(result):
    same(result["schema"], "slotspe_readonly_full_model_gate_rng_v1", "schema")
    same(result["status"], "completed", "status")
    protocol = result["protocol"]
    for key, expected in {"patients": 71, "fold": 0, "training_seed": 3, "epochs": 3,
                          "patches": 256, "slots": 8, "k": 2, "repeats": 20,
                          "temperature": .01, "epsilon": 1e-8}.items():
        same(protocol[key], expected, "protocol." + key)
    seeds = list(range(3080000, 3080020))
    same(protocol["eval_seeds"], seeds, "protocol.eval_seeds")
    batch_size = protocol["batch_size"]
    require(isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0,
            "Invalid batch size")
    rows = result["per_patient"]
    require(isinstance(rows, list) and len(rows) == 71, "Patient row count differs")
    ids = [row["patient_id"] for row in rows]
    require(all(isinstance(value, str) and value for value in ids) and len(set(ids)) == 71,
            "Patient identifiers are invalid or duplicated")
    maxima = Counter()
    per_modality = {modality: [] for modality in MODALITIES}
    double_sums = {modality: [] for modality in MODALITIES}

    def difference(label, actual, expected, tolerance):
        error = abs(number(actual, label) - float(expected))
        maxima[label] = max(maxima[label], error)
        require(error <= tolerance, label + " exceeds numerical tolerance")

    for patient_index, row in enumerate(rows):
        require(set(row["modalities"]) == set(MODALITIES), "Modality fields differ")
        for modality in MODALITIES:
            patient = row["modalities"][modality]
            reps = patient["repeats"]
            require(isinstance(reps, list) and len(reps) == 20, "Patient repeat count differs")
            derived = []
            for repeat, record in enumerate(reps):
                # Row indices, never identifiers, are used in error messages.
                label = f"row[{patient_index}].{modality}.repeat[{repeat}]"
                same(record["repeat"], repeat, label + ".repeat")
                same(record["eval_seed"], seeds[repeat], label + ".eval_seed")
                scores = np.asarray(record["scores"], dtype=np.float64)
                weights = np.asarray(record["gate_weights"], dtype=np.float64)
                require(scores.shape == weights.shape == (8,) and np.isfinite(scores).all() and
                        np.isfinite(weights).all(), label + " invalid score/weight vectors")
                selected = record["selected_indices"]
                require(isinstance(selected, list) and len(selected) == 2 and len(set(selected)) == 2 and
                        all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 8 for i in selected),
                        label + " invalid retained indices")
                same(record["keep_count"], 2, label + ".keep_count")
                mask = np.zeros(8, dtype=bool)
                mask[selected] = True
                require(np.all(weights >= 0) and np.max(np.abs(weights[~mask])) <= 2e-7,
                        label + " gate weight outside hard selection")
                top2 = record["deterministic_top2_indices"]
                require(isinstance(top2, list) and len(top2) == 2 and len(set(top2)) == 2 and
                        all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < 8 for i in top2),
                        label + " invalid deterministic Top2 indices")
                topmask = np.zeros(8, dtype=bool)
                topmask[top2] = True
                # Native TopK tie-breaking may differ from NumPy. Its indices
                # are valid iff no omitted score exceeds a retained score.
                require(np.min(scores[topmask]) >= np.max(scores[~topmask]),
                        label + " deterministic Top2 violates score boundary")
                require(scores[top2[0]] >= scores[top2[1]], label + " deterministic Top2 order differs")
                overlap = len(set(selected).intersection(top2))
                same(record["deterministic_top2_overlap"], overlap, label + ".top2_overlap")
                retained_top1 = int(np.argmax(scores)) in selected
                same(record["top1_retained"], retained_top1, label + ".top1_retained")
                gate_sum = number(record["gate_sum"], label + ".gate_sum")
                difference("weight_sum_vs_native_float32_gate_sum", gate_sum, np.sum(weights),
                           TOLERANCES["gate_weight_sum_abs"])
                effective = scores / .01
                shifted = effective - np.max(effective)
                exponentials = np.exp(shifted)
                mass = float(np.sum(exponentials[mask]) / np.sum(exponentials))
                logmass = float(np.log(np.sum(exponentials[mask])) - np.log(np.sum(exponentials)))
                raw_mass = float(np.sum(softmax(scores)[mask]))
                difference("temperature_mass_float64", record["selected_temperature_mass_float64"], mass,
                           TOLERANCES["float64_abs"])
                difference("temperature_logmass_float64", record["log_selected_temperature_mass_float64"], logmass,
                           TOLERANCES["float64_abs"])
                difference("temperature_mass_float32", record["selected_temperature_mass_float32"], mass,
                           TOLERANCES["float32_mass_abs"])
                difference("raw_score_mass_float64", record["selected_raw_score_mass_float64"], raw_mass,
                           TOLERANCES["float64_abs"])
                formula = softmax(effective) * mask / (mass + 1e-8)
                difference("gate_vector_float64_formula", float(np.max(np.abs(weights - formula))), 0,
                           TOLERANCES["float32_formula_abs"])
                quiet_mass = float(np.sum(softmax(effective)[top2]))
                controls = {key: number(record[key], label + "." + key) for key in CONTROL_KEYS}
                for key, expected in zip(CONTROL_KEYS, (raw_mass / (raw_mass + 1e-8),
                                                        quiet_mass / (quiet_mass + 1e-8), 1.)):
                    difference(key + "_vs_numeric_formula", controls[key], expected,
                               TOLERANCES["control_gate_sum_abs"])
                derived.append({"gate_sum": gate_sum, "selected_temperature_mass_float64": mass,
                                "selected_raw_score_mass_float64": raw_mass, "selected_indices": selected,
                                "deterministic_top2_overlap": overlap, "top1_retained": retained_top1,
                                **controls})
                double_sums[modality].append(float(np.sum(weights)))
            unique_sets = len({tuple(sorted(record["selected_indices"])) for record in derived})
            same(patient["unique_retained_sets"], unique_sets, "patient unique retained set count")
            patient_dist = distribution([record["gate_sum"] for record in derived])
            same(patient["gate_sum_across_rng"], patient_dist, "patient gate distribution")
            per_modality[modality].append({"repeats": derived, "unique_retained_sets": unique_sets})
    summaries = {}
    for modality, patients in per_modality.items():
        pooled = [record for patient in patients for record in patient["repeats"]]
        summaries[modality] = {
            "primary_first_rng_unique_patients": distribution([p["repeats"][0]["gate_sum"] for p in patients]),
            "all_rng_patient_draws_not_independent_patients": distribution([r["gate_sum"] for r in pooled]),
            "unique_patients_ever_below_0_5": sum(any(r["gate_sum"] < .5 for r in p["repeats"]) for p in patients),
            "unique_patients_ever_below_0_01": sum(any(r["gate_sum"] < .01 for r in p["repeats"]) for p in patients),
            "selected_temperature_mass": distribution([r["selected_temperature_mass_float64"] for r in pooled]),
            "selected_raw_score_mass": distribution([r["selected_raw_score_mass_float64"] for r in pooled]),
            "top2_overlap_histogram": dict(Counter(str(r["deterministic_top2_overlap"]) for r in pooled)),
            "top2_overlap_mean": float(np.mean([r["deterministic_top2_overlap"] for r in pooled])),
            "top1_missed_fraction": float(np.mean([not r["top1_retained"] for r in pooled])),
            "unique_retained_sets": distribution([p["unique_retained_sets"] for p in patients]),
            "controls": {key: distribution([record[key] for record in pooled]) for key in CONTROL_KEYS},
        }
    same(result["summary"], summaries, "summary")
    checks = result["checks"]
    for key in ("historical_full_logits_replay_passed", "sources_weights_inputs_unchanged",
                "hooks_removed_and_settings_restored"):
        same(checks[key], True, "checks." + key)
    require(all(value is True for value in checks["model_purity"].values()) and len(checks["model_purity"]) == 5,
            "Native model purity receipt did not pass")
    same(checks["model_state_sha256_before"], checks["model_state_sha256_after"], "state digest identity")
    require(len(checks["model_state_sha256_before"]) == 64, "Invalid state digest")
    batch_forwards = math.ceil(71 / batch_size) * 20
    expected_counts = {"training_runs": 0, "optimizer_steps": 0, "full_model_batch_forwards": batch_forwards,
                       "full_model_patient_forwards": 1420, "new_slot_generation_patient_pairs": 1420,
                       "paired_control_decoder_batch_forwards": batch_forwards * 4,
                       "server_execution": False, "new_server_raw_receipt": False}
    for key, expected in expected_counts.items():
        same(checks[key], expected, "checks." + key)
    require(set(checks["error_maxima"]) == {"historical_full_logits_max_abs", "native_gate_formula_max_abs"},
            "Unexpected runner error receipt fields")
    require(all(0 <= number(value, "runner error") <= 1e-6 for value in checks["error_maxima"].values()),
            "Runner exact-replay errors exceeded tolerance")
    return summaries, dict(maxima), {modality: distribution(values) for modality, values in double_sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--source-verification", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result_path = contained(args.result, REPO / "local_evidence", "Result")
    source_path = (args.source_verification or result_path.parent.parent / "source_verification.json").resolve()
    output = contained(args.output or result_path.parent / "result_audit.json", result_path.parent, "Audit output")
    require(output != result_path and output != source_path, "Audit cannot replace its inputs")
    require(not output.exists(), "Fresh audit output required")
    require(subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(output)],
                           capture_output=True).returncode == 0, "Audit output must be Git ignored")
    report = {"schema": "slotspe_full_model_gate_result_independent_audit_v1", "status": "failed",
              "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
              "audit_script_sha256": sha(__file__), "numpy": str(np.__version__),
              "model_execution": False, "torch_imported": False, "training_runs": 0,
              "tolerances": TOLERANCES}
    code = 1
    try:
        result_hash = sha(result_path)
        result = read(result_path)
        summary, maxima, double_sums = recompute(result)
        watched, provenance = check_provenance(result, result_path, source_path)
        status_path = result_path.parent / "execution_status.json"
        status = read(status_path)
        same(status["status"], "completed", "execution.status")
        same(status["result_sha256"], result_hash, "execution.result_sha256")
        for key in ("full_model_batch_forwards", "paired_control_decoder_batch_forwards", "training_runs", "error_maxima"):
            same(status[key], result["checks"][key], "execution." + key)
        watched[status_path] = sha(status_path)
        for path, digest in watched.items():
            require(sha(path) == digest, "Audited file changed while audit was running")
        require("torch" not in sys.modules, "Auditor imported torch")
        report.update({"status": "passed", "result_sha256": result_hash,
                       "input_sha256": dict(result["input_sha256"]),
                       "patients": 71, "modalities": 2, "repeats_per_patient": 20,
                       "gate_records_audited": 2840, "summary_disagreements": 0,
                       "provenance": provenance, "unchanged_file_hashes_rechecked": len(watched),
                       "numeric_error_maxima": maxima, "independent_summary": summary,
                       "independent_double_precision_weight_sum_distributions": double_sums,
                       "execution_receipt_matches": True, "state_digest_before_after_equal": True,
                       "limitations": [
                           "Native scalar gate_sum is checked against stored weights within float32 tolerance; native scalar summaries are then recomputed independently.",
                           "Float64 score masses and deterministic Top2 are independently computed from each repeat's scores, not from the first repeat.",
                           "Control vectors and full-model logits are not stored here; control scalar formulas and summaries are audited, while exact logits replay remains a runner receipt claim.",
                           "Matching state digests is checked from the saved receipt; the auditor does not reload a model or independently compute its state digest.",
                           "Audit consistency is not evidence of predictive harm or correction efficacy."]})
        code = 0
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    output.write_text(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "patients": report.get("patients"),
                      "gate_records_audited": report.get("gate_records_audited"),
                      "failure": report.get("failure"), "output": str(output)}, ensure_ascii=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
