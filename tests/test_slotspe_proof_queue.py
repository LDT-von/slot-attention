"""Contracts for a server task. All data below are synthetic test fixtures."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os
import zipfile

import pytest

from survot_rank.research.slotspe_proof import queue

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/SlotSPE"


def configuration(tmp_path, **updates):
    return queue.normalize_config({"source_root": str(SOURCE),
        "metadata_root": str(tmp_path / "metadata"), "features_root": str(tmp_path / "features"),
        "device": "cpu", "folds": [0], "seeds": [3], "epochs": 1,
        "patches": 8, "batch_size": 2, "eval_repeats": 2, **updates})


def metadata_fixture(tmp_path, *, rna=False):
    import pandas as pd
    m = tmp_path / "metadata"
    (m / "clinical/all").mkdir(parents=True)
    (m / "splits/5fold/blca").mkdir(parents=True)
    ids = [f"SYNTHETIC_CASE_{i}" for i in range(7)]
    pd.DataFrame({"case id": ids, "survival_months_dss": [1., 2., 3., 4., 5., 6., 12.],
        "censorship_dss": [0] * 7, "wsi": [pid + ".svs" for pid in ids]}).to_csv(m / "clinical/all/blca.csv", index=False)
    pd.DataFrame({"train": pd.Series(ids[:5]), "val": pd.Series(ids[5:])}).to_csv(m / "splits/5fold/blca/fold_0.csv", index=False)
    if rna:
        (m / "raw_rna_data_inter").mkdir()
        (m / "signatures").mkdir()
        pd.DataFrame({pid: [float(i + 1), float(i + 2), float(i + 3)] for i, pid in enumerate(ids)},
            index=["SYN_G1", "SYN_G2", "SYN_G3"]).to_csv(m / "raw_rna_data_inter/blca_rna_inter.csv")
        pd.DataFrame({"SYN_PATH_1": ["SYN_G1", "SYN_G2"], "SYN_PATH_2": ["SYN_G2", "SYN_G3"]}).to_csv(m / "signatures/combine_signatures.csv", index=False)
    return m, ids


def test_plan_is_import_and_data_free(tmp_path, monkeypatch):
    monkeypatch.delenv("SLOTSPE_METADATA_ROOT", raising=False)
    monkeypatch.delenv("SLOTSPE_FEATURES_ROOT", raising=False)
    command = [sys.executable, "-B", "-c",
        "import sys; from survot_rank.research.slotspe_proof.queue import main; "
        "main(['--plan']); assert 'torch' not in sys.modules; assert 'pandas' not in sys.modules"]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    plan = json.loads(completed.stdout)
    # The shipped default queue config is the clean original-only fold0/seed3 run
    # (configs/slotspe_proof_server.json); the expected run count follows that config
    # instead of a hardcoded number from an older 5-fold/3-seed default.
    default_config = queue.normalize_config(json.loads(
        (ROOT / "configs/slotspe_proof_server.json").read_text(encoding="utf-8")))
    expected_runs = (len(queue.training_arms(default_config))
                     * len(default_config["folds"]) * len(default_config["seeds"]))
    assert plan["config"]["experiment"] == default_config["experiment"]
    assert plan["training_arms"] == queue.training_arms(default_config)
    assert plan["training_runs"] == expected_runs == 1
    assert plan["patient_work_executed"] is False
    assert plan["unresolved_environment_variables"]


@pytest.mark.parametrize("field,value", [("folds", [0, 0]), ("seeds", []), ("epochs", 0),
    ("patches", True), ("eval_repeats", 1), ("study", "../blca"), ("device", "cuda:x"), ("lr", float("nan"))])
def test_invalid_plan_rejected(tmp_path, field, value):
    with pytest.raises(ValueError):
        configuration(tmp_path, **{field: value})


def test_grid_and_budget_are_parameters(tmp_path):
    plan = queue.make_plan(configuration(tmp_path, folds=[1, 4], seeds=[7, 19], epochs=13), tmp_path / "out")
    assert plan["training_runs"] == 8
    assert plan["same_weights_eval_controls"] == 4
    assert [s["id"] for s in plan["stages"] if s["kind"] == "pair"] == [
        "fold_1_seed_7", "fold_1_seed_19", "fold_4_seed_7", "fold_4_seed_19"]


def test_unset_server_paths_fail_before_any_output(tmp_path):
    config = configuration(tmp_path, metadata_root="${UNSET_SLOTSPE_PROOF_TEST_VARIABLE}")
    plan = queue.make_plan(config, tmp_path / "out")
    with pytest.raises(ValueError, match="Unresolved"):
        queue.execute(plan, runtime_check=False)
    assert not (tmp_path / "out").exists()


def test_resume_verifies_and_skips_not_blind_reuse(tmp_path):
    metadata_fixture(tmp_path)
    config = configuration(tmp_path)
    plan = queue.make_plan(config, tmp_path / "out", checks_only=True)
    calls = []
    def fake(command, stage, root, log):
        calls.append(stage["id"])
        for path in queue._stage_files(root, stage):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"fixture": true}', encoding="utf-8")
        return 0
    queue.execute(plan, executor=fake, runtime_check=False)
    assert calls == ["protocol", "reconstruction", "mechanisms", "summarize"]
    calls.clear()
    queue.execute(plan, resume=True, executor=fake, runtime_check=False)
    assert calls == []
    (tmp_path / "out/checks/protocol_checks.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output changed"):
        queue.execute(plan, resume=True, executor=fake, runtime_check=False)


def test_resume_rejects_changed_input_and_nonempty_new_run(tmp_path):
    m, _ = metadata_fixture(tmp_path)
    config = configuration(tmp_path)
    plan = queue.make_plan(config, tmp_path / "out", checks_only=True)
    def fake(command, stage, root, log):
        for path in queue._stage_files(root, stage):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        return 0
    queue.execute(plan, executor=fake, runtime_check=False)
    with pytest.raises(FileExistsError):
        queue.execute(plan, executor=fake, runtime_check=False)
    with (m / "clinical/all/blca.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="changed"):
        queue.execute(plan, resume=True, executor=fake, runtime_check=False)


@pytest.mark.parametrize("target", ["resolved_config.json", "queue_manifest.json"])
def test_resume_rejects_edited_saved_configuration(tmp_path, target):
    metadata_fixture(tmp_path)
    plan = queue.make_plan(configuration(tmp_path), tmp_path / "out", checks_only=True)
    def fake(command, stage, root, log):
        for path in queue._stage_files(root, stage):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        return 0
    queue.execute(plan, executor=fake, runtime_check=False)
    path = tmp_path / "out" / target
    saved = json.loads(path.read_text(encoding="utf-8"))
    (saved if target == "resolved_config.json" else saved["config"])["epochs"] = 99
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        queue.execute(plan, resume=True, executor=fake, runtime_check=False)


def test_mid_stage_config_edit_cannot_receive_completion_receipt(tmp_path):
    metadata_fixture(tmp_path)
    plan = queue.make_plan(configuration(tmp_path), tmp_path / "out", checks_only=True)
    def mutate(command, stage, root, log):
        for path in queue._stage_files(root, stage):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        saved = json.loads((root / "resolved_config.json").read_text(encoding="utf-8"))
        saved["epochs"] = 99
        (root / "resolved_config.json").write_text(json.dumps(saved), encoding="utf-8")
        return 0
    with pytest.raises(RuntimeError, match="configuration changed"):
        queue.execute(plan, executor=mutate, runtime_check=False)
    assert not (tmp_path / "out/receipts/protocol.json").exists()
    assert json.loads((tmp_path / "out/queue_status.json").read_text())["status"] == "failed"
    # A child also verifies the saved config before importing model/data code.
    completed = subprocess.run([sys.executable, "-B", "-m", queue.PACKAGE + ".worker",
        "--config", str(tmp_path / "out/resolved_config.json"), "--output-root", str(tmp_path / "out"),
        "--stage", "protocol", "--checks-only"], cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "configuration changed" in completed.stderr


def test_failure_stops_queue_and_retains_log(tmp_path):
    metadata_fixture(tmp_path)
    plan = queue.make_plan(configuration(tmp_path), tmp_path / "out", checks_only=True)
    calls = []
    def fail(command, stage, root, log):
        calls.append(stage["id"])
        log.write("deliberate synthetic test failure")
        return 9
    with pytest.raises(RuntimeError, match="exit 9"):
        queue.execute(plan, executor=fail, runtime_check=False)
    assert calls == ["protocol"]
    assert not (tmp_path / "out/.queue.lock").exists()
    assert json.loads((tmp_path / "out/queue_status.json").read_text())["status"] == "failed"
    assert "deliberate" in (tmp_path / "out/logs/protocol_attempt_1.log").read_text()


@pytest.mark.parametrize("experiment", ["reproduction", "attribution_abc"])
def test_full_synthetic_interface_in_fresh_worker_processes(tmp_path, experiment):
    """Exercise real source/import/trainer/file contracts on invented inputs."""
    torch = pytest.importorskip("torch")
    m, ids = metadata_fixture(tmp_path, rna=True)
    features = tmp_path / "features"
    features.mkdir()
    generator = torch.Generator().manual_seed(909)
    for pid in ids:
        torch.save(torch.randn(10, 16, generator=generator), features / (pid + ".pt"))
    config = configuration(tmp_path, experiment=experiment)
    profile = tmp_path / "synthetic_profile.json"
    profile.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "synthetic_results"
    completed = subprocess.run([sys.executable, "-B", "-u", str(ROOT / "scripts/run_slotspe_proof.py"),
        "--config", str(profile), "--run", "--output-root", str(out)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads((out / "proof_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    pair = summary["patients"]["results"][0]
    assert pair["counts"]["train"] == 5 and pair["counts"]["val"] == 2
    assert len(pair["arms"]) == (3 if experiment == "attribution_abc" else 2)
    assert pair["arms"][0]["same_initial_state_sha256"] == pair["arms"][1]["same_initial_state_sha256"]
    if experiment == "reproduction":
        assert pair["same_weights_eval_fix"]["new_optimizer_steps"] == 0
        assert pair["same_weights_eval_fix"]["patient_risk_range_max"] == 0
    else:
        assert pair["same_weights_eval_fix"] is None
        assert all(pair["checks"].values())
        assert [row["comparison"] for row in pair["contrasts"]] == ["B_minus_A", "C_minus_B"]
        assert json.loads((out / "pairs/fold_0/seed_3/matched_state_bc_checks.json").read_text())["status"] == "passed"
    for arm in pair["arms"]:
        probe = json.loads((out / "pairs/fold_0/seed_3" / arm["arm"] / "patient_slot_intervention.json").read_text())
        assert probe["patient_ids"] == ids[5:]
        assert probe["checkpoint_sha256"] == arm["checkpoint_sha256"]
    from survot_rank.research.slotspe_proof.acceptance import audit_return
    returned = audit_return(out)
    assert returned["status"] == "artifact_contract_passed"
    assert returned["evidence_gaps"] == []
    # Summarization must reject modified scientific outputs, not publish means.
    predictions = out / "pairs/fold_0/seed_3" / pair["arms"][0]["arm"] / "validation_predictions.csv"
    predictions.write_text(predictions.read_text().replace(ids[5], "WRONG_PATIENT"), encoding="utf-8")
    from survot_rank.research.slotspe_proof.patient import summarize
    with pytest.raises(RuntimeError, match="patient identities"):
        summarize(config, out)
    with pytest.raises(ValueError, match="hash differs"):
        audit_return(out)


@pytest.mark.parametrize("experiment", ["reproduction", "attribution_abc"])
def test_self_contained_delivery_has_no_repository_or_audit_dependency(tmp_path, experiment):
    torch = pytest.importorskip("torch")
    archive = tmp_path / "independent_task.zip"
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/build_slotspe_proof_delivery.py"),
        "--output", str(archive)], cwd=ROOT, capture_output=True, check=True)
    unpack = tmp_path / "standalone task space"
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpack)
        delivered = json.loads(z.read("slotspe_proof_task_20261019/DELIVERY_MANIFEST.json"))
        import hashlib
        assert all(hashlib.sha256(z.read(delivered["prefix"] + path)).hexdigest() == digest
                   for path, digest in delivered["files_sha256"].items())
    task = unpack / "slotspe_proof_task_20261019"
    assert not (task / ".review").exists()
    # The included tests can rebuild a delivery without the old checkout.
    subprocess.run([sys.executable, "-B", str(task / "scripts/build_slotspe_proof_delivery.py"),
        "--output", str(tmp_path / "rebuilt_from_delivery.zip")], cwd=task,
        env={**os.environ, "PYTHONPATH": ""}, capture_output=True, check=True)
    m, ids = metadata_fixture(tmp_path, rna=True)
    features = tmp_path / "features"
    features.mkdir()
    for i, pid in enumerate(ids):
        torch.save(torch.full((10, 16), float(i + 1)) + torch.eye(10, 16), features / (pid + ".pt"))
    config = configuration(tmp_path, source_root=str(task / "third_party/SlotSPE"), experiment=experiment)
    profile = tmp_path / "packaged_profile.json"
    profile.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run([sys.executable, "-B", "-u", str(task / "scripts/run_slotspe_proof.py"),
        "--config", str(profile), "--run", "--output-root", str(tmp_path / "packaged_results")],
        cwd=task, env={**os.environ, "PYTHONPATH": ""}, capture_output=True,
        text=True, encoding="utf-8", timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "packaged_results/proof_summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["patients"]["completed_pairs"] == 1
    assert report["patients"]["results"][0]["source_unchanged"]
    # Offline acceptance does not require the original server metadata paths.
    accepted = subprocess.run([sys.executable, "-B", str(task / "scripts/audit_slotspe_return.py"),
        "--root", str(tmp_path / "packaged_results"), "--delivery", str(archive)],
        cwd=task, env={**os.environ, "PYTHONPATH": ""}, capture_output=True,
        text=True, encoding="utf-8", timeout=60)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert json.loads(accepted.stdout)["implementation_matches_delivery"]
    returned = tmp_path / "server_return.zip"
    packed = subprocess.run([sys.executable, "-B", str(task / "scripts/pack_slotspe_return.py"),
        "--root", str(tmp_path / "packaged_results"), "--output", str(returned)],
        cwd=task, env={**os.environ, "PYTHONPATH": ""}, capture_output=True,
        text=True, encoding="utf-8", timeout=60)
    assert packed.returncode == 0, packed.stdout + packed.stderr
    with zipfile.ZipFile(returned) as z:
        assert not any("prepared/cache/" in name for name in z.namelist())
        assert any(name.endswith("last.pt") for name in z.namelist())
        z.extractall(tmp_path / "returned")
    from survot_rank.research.slotspe_proof.acceptance import audit_return
    received_root = tmp_path / "returned/slotspe_server_return"
    assert audit_return(received_root, archive)["status"] == "artifact_contract_passed"
    saved_status = json.loads((received_root / "queue_status.json").read_text())
    saved_status["status"] = "failed"
    (received_root / "queue_status.json").write_text(json.dumps(saved_status))
    with pytest.raises(ValueError, match="incomplete"):
        audit_return(received_root)


def test_abc_plan_separates_training_and_reproduction(tmp_path):
    config = configuration(tmp_path, experiment="attribution_abc", eval_repeats=20)
    plan = queue.make_plan(config, tmp_path / "out")
    assert plan["training_runs"] == 3
    assert plan["training_arms"] == ["A", "B", "C"]
    assert plan["same_weights_eval_controls"] == 0
    assert any(stage["kind"] == "attribution" for stage in plan["stages"])
    with pytest.raises(ValueError, match="experiment"):
        configuration(tmp_path, experiment="unknown")
