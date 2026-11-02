"""Tests for the anti-collapse queue's own resume and summary logic.

The queue failed after its first wave for three independent reasons, and
each of them is guarded here:

1.  The stage log was opened in exclusive mode, so a retried stage raised
    FileExistsError before its body ran.  The log must be appendable.
2.  The fold and root summaries were delegated to the completed queue's
    helpers, which hard-code that queue's eleven arms and its own fold
    list.  This queue has two arms, so the summary must read only its own
    arms.
3.  The shared arm view reads the C-index out of ``evaluation_summary``,
    but this queue's engine writes the slot diagnostics there and leaves
    the C-index aggregation in ``eval_repeats.json``.  The summary must
    therefore read the C-index from this queue's own artifacts, and the
    first repeat must reproduce the single-pass Protocol A value exactly.

The resume reconciliation is guarded too: a stage whose artifacts are
complete must be recognised as completed, and a stage whose artifacts are
incomplete must not be silently accepted.
"""
import json
from pathlib import Path

import pytest

from slotspe_proof import redundancy_anticollapse_multifold as queue


def _write_repeats(path, scores):
    """Write an eval_repeats.json in this queue's schema."""
    repeats = []
    for index, score in enumerate(scores):
        repeats.append({"repeat": index, "nll": 0.5, "cindex": score,
                        "cindex_status": "computed",
                        "risks": [float(index) + 0.01 * position for position in range(4)]})
    path.write_text(json.dumps(repeats), encoding="utf-8")


def _make_arm_folder(root, fold, arm, scores, epochs=3):
    folder = root / "folds" / str(fold) / "arms" / arm
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "best.pt").write_bytes(b"checkpoint")
    (folder / "last.pt").write_bytes(b"checkpoint")
    (folder / "curve.json").write_text(json.dumps([
        {"epoch": 1, "total_loss": 2.0, "core_gradient_norm": 1.0,
         "pre_clip_gradient_norm": 3.0, "gradient_was_clipped": True}]), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps({
        "status": "completed", "arm": arm, "fold": fold, "seed": 3,
        "best_epoch": 2, "best_epoch_cindex_single_pass": scores[0],
        "elapsed_seconds": 10.0,
        "protocol": {"initial_core_hash": "core", "initial_probe_hash": "probe"},
        "counters": {"root_forwards": 1},
        "evaluation_summary": {"wsi": {"effective_rank": 5.0}, "omic": {"effective_rank": 4.0}},
        "slot_diagnostics": {"wsi": {"effective_rank": 5.0}, "omic": {"effective_rank": 4.0}},
    }), encoding="utf-8")
    _write_repeats(folder / "eval_repeats.json", scores)
    (folder / "epoch_validation.json").write_text(json.dumps(
        [{"epoch": index + 1, "cindex": score, "cindex_status": "computed"}
         for index, score in enumerate(scores)]), encoding="utf-8")
    (folder / "diagnostics.pt").write_bytes(b"diagnostics")
    (folder / "validation_predictions.csv").write_text(
        "patient_id,time,censor,label,risk_repeat_00\n", encoding="utf-8")
    return folder


def test_the_stage_log_is_appendable_so_a_retried_stage_can_start(tmp_path):
    """A retained log from an earlier attempt must not stop a retry.

    The exclusive open mode raised FileExistsError before the stage body
    ran, which is what left every stage of the first wave recorded as
    failed even though six of them had finished training.
    """
    root = tmp_path / "queue"
    stage = {"id": "arm_fold_0_A0_clip", "kind": "arm", "fold": 0, "arm": "A0_clip"}
    log = root / "logs" / (stage["id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("earlier attempt\n", encoding="utf-8")

    # The spawn function is exercised for its log handling only; the
    # subprocess is replaced so that the test never trains a model.
    import subprocess

    original = subprocess.Popen

    class _Fake:
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(command, **kwargs):
        kwargs["stdout"].write("second attempt\n")
        kwargs["stdout"].flush()
        return _Fake()

    subprocess.Popen = fake_popen
    try:
        queue._spawn_stage(stage, {}, root, device=None)
    finally:
        subprocess.Popen = original

    text = log.read_text(encoding="utf-8")
    assert "earlier attempt" in text, "the earlier attempt must stay visible"
    assert "second attempt" in text, "the retry must be able to write"


def test_the_summary_reads_the_cindex_from_this_queue_own_schema(tmp_path):
    """The shared arm view would report a null C-index for these arms.

    This queue's engine writes the slot diagnostics into
    ``evaluation_summary`` and leaves the C-index aggregation in
    ``eval_repeats.json``, so the summary must read it from there.
    """
    root = tmp_path / "queue"
    scores = [0.70, 0.72, 0.74]
    _make_arm_folder(root, 0, "A0_clip", scores)

    view = queue._arm_view_anti(root, 0, "A0_clip")

    assert view["cindex_mean_over_repeats"] == pytest.approx(sum(scores) / len(scores))
    assert view["cindex_defined_repeats"] == 3
    assert view["evaluation_repeats"] == 3
    # Protocol A is the single validation pass at the best epoch, which is
    # exactly repeat 0 of the repeated evaluation.
    assert view["best_epoch_cindex_single_pass"] == pytest.approx(scores[0])
    assert view["cindex_repeat_min"] == pytest.approx(0.70)
    assert view["cindex_repeat_max"] == pytest.approx(0.74)
    # The slot diagnostics must survive the schema difference.
    assert view["slot_diagnostics"]["wsi"]["effective_rank"] == 5.0
    # The pre-clip gradient norm is the spike evidence and must be kept
    # separately from the clipped value that the curve reports.
    assert view["pre_clip_gradient_norm_max"] == pytest.approx(3.0)
    assert view["clipped_steps"] == 1


def test_the_summary_reads_only_this_queue_own_two_arms(tmp_path):
    """The completed queue's helpers hard-code its eleven arms.

    Delegating to them made this queue fail on a missing ``arms/S0``
    folder, so the summary must read only the declared anti-collapse arms.
    """
    root = tmp_path / "queue"
    for fold in range(5):
        for arm in queue.ANTI_COLLAPSE_ARMS:
            _make_arm_folder(root, fold, arm, [0.70, 0.71])

    runtime = {"epochs": 30, "patches": 4096, "eval_repeats": 2, "batch_size": 8,
               "splits_dir": "5fold", "lr": 5e-4, "probe_lr": 5e-4,
               "projection_dim": 256, "slot_iters": 10, "content_target_dim": 32,
               "content_clusters": 8, "gradient_clip_norm": 1.0}
    for fold in range(5):
        queue._summarize_fold(root, runtime, fold)

    result = queue._summarize_all(root, runtime)

    assert set(result["per_arm"]) == set(queue.ANTI_COLLAPSE_ARMS)
    assert len(result["per_arm"]) == 2
    for arm, entry in result["per_arm"].items():
        assert entry["cindex_five_fold"]["fold_mean"] == pytest.approx(0.705)
        assert set(entry["cindex_per_fold"]) == {0, 1, 2, 3, 4}
    assert (root / "summary.txt").is_file()
    assert (root / "summary.json").is_file()


def test_a_fold_whose_arms_are_missing_stops_the_summary(tmp_path):
    """A missing fold must fail loudly rather than be skipped silently."""
    root = tmp_path / "queue"
    for fold in range(4):
        for arm in queue.ANTI_COLLAPSE_ARMS:
            _make_arm_folder(root, fold, arm, [0.70, 0.71])

    runtime = {"epochs": 30, "patches": 4096, "eval_repeats": 2, "batch_size": 8,
               "splits_dir": "5fold", "lr": 5e-4, "probe_lr": 5e-4,
               "projection_dim": 256, "slot_iters": 10, "content_target_dim": 32,
               "content_clusters": 8, "gradient_clip_norm": 1.0}
    for fold in range(4):
        queue._summarize_fold(root, runtime, fold)
    with pytest.raises(Exception):
        queue._summarize_all(root, runtime)


def test_the_resume_reconciliation_accepts_a_complete_stage(tmp_path):
    """A stage that failed on a launcher error but produced complete
    artifacts must be recognised as completed instead of blocking the
    resume or being re-run over its own outputs."""
    root = tmp_path / "queue"
    _make_arm_folder(root, 0, "A0_clip", [0.70, 0.71])
    stage = {"id": "arm_fold_0_A0_clip", "kind": "arm", "fold": 0, "arm": "A0_clip"}

    reconciled = queue._reconcile_failed_records(root, {"arm_fold_0_A0_clip": {
        "status": "failed", "type": "FileExistsError", "error": "log exists"}}, [stage])

    assert reconciled["arm_fold_0_A0_clip"]["status"] == "completed"
    assert reconciled["arm_fold_0_A0_clip"]["reconciled_from"] == "failed"
    assert reconciled["arm_fold_0_A0_clip"]["artifact_sha256"]


def test_the_resume_reconciliation_refuses_an_incomplete_stage(tmp_path):
    """A stage whose artifacts are incomplete must stay failed, so that the
    queue never treats a half-trained arm as a completed one."""
    root = tmp_path / "queue"
    folder = root / "folds" / "0" / "arms" / "A0_clip"
    folder.mkdir(parents=True)
    (folder / "summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    stage = {"id": "arm_fold_0_A0_clip", "kind": "arm", "fold": 0, "arm": "A0_clip"}
    records = {"arm_fold_0_A0_clip": {"status": "failed", "type": "FileExistsError",
                                      "error": "log exists"}}

    reconciled = queue._reconcile_failed_records(root, records, [stage])

    assert reconciled["arm_fold_0_A0_clip"]["status"] == "failed"


def test_the_release_quarantines_the_incomplete_folder_instead_of_deleting_it(tmp_path):
    """A genuinely failed stage must be re-runnable without losing data.

    The queue refuses to write into a folder that already holds content,
    so the incomplete folder is moved aside under a quarantine name.  The
    original files must still be there afterwards.
    """
    root = tmp_path / "queue"
    folder = root / "folds" / "3" / "arms" / "A0_clip"
    folder.mkdir(parents=True)
    (folder / "curve.json").write_text("[]", encoding="utf-8")
    stage = {"id": "arm_fold_3_A0_clip", "kind": "arm", "fold": 3, "arm": "A0_clip"}

    quarantine = queue._clear_failed_stage(root, stage, {"status": "failed"})

    assert quarantine is not None
    assert quarantine.is_dir()
    assert (quarantine / "curve.json").read_text(encoding="utf-8") == "[]"
    # The original location must be free for the re-run.
    assert not folder.exists()


def test_a_resumed_wave_does_not_wait_for_an_already_satisfied_dependency(tmp_path):
    """A resumed wave must not deadlock on a finished dependency.

    The fold 3 initialization was completed by an earlier attempt, but the
    fold 3 arms were not.  The shared scheduler only knows about the
    stages it spawns itself, so without the already-satisfied set the
    wave would wait forever for a dependency that is already on disk.
    """
    from slotspe_proof import redundancy_multifold as base

    root = tmp_path / "queue"
    order = [{"id": "init_fold_3", "kind": "initialize", "fold": 3},
             {"id": "arm_fold_3_A0_clip", "kind": "arm", "fold": 3, "arm": "A0_clip"}]
    spawned = []

    def fake_spawn(stage, runtime, root_arg, device=None):
        spawned.append(stage["id"])
        (root_arg / "logs").mkdir(parents=True, exist_ok=True)
        handle = open(root_arg / "logs" / (stage["id"] + ".log"), "w", encoding="utf-8")

        class _Fake:
            returncode = 0

            def poll(self):
                return 0

        return stage, _Fake(), handle

    original_spawn = queue._spawn_stage
    queue._spawn_stage = fake_spawn
    try:
        queue._run_wave(order, {"arm_fold_3_A0_clip"}, {}, root, None, 1,
                        satisfied={"init_fold_3"})
    finally:
        queue._spawn_stage = original_spawn

    assert spawned == ["arm_fold_3_A0_clip"], \
        "the arm must start even though its dependency was finished earlier"


def test_the_release_never_moves_a_fold_directory_that_holds_its_arms(tmp_path):
    root = tmp_path / "queue"
    fold = root / "folds" / "0"
    (fold / "arms" / "A0_clip").mkdir(parents=True)
    (fold / "arms" / "A0_clip" / "summary.json").write_text("{}", encoding="utf-8")
    (fold / "summary.json").write_text("{}", encoding="utf-8")
    stage = {"id": "summarize_fold_0", "kind": "summarize_fold", "fold": 0}

    quarantine = queue._clear_failed_stage(root, stage, {"status": "failed"})

    assert quarantine is None
    assert (fold / "summary.json").exists(), "an unrelated file must not be deleted"
    assert (fold / "arms" / "A0_clip" / "summary.json").is_file(), \
        "the finished arm must not be moved away"
