"""Regression tests for the padding-policy decision guard.

The guard enforces the recorded 2026-10-23 18:46 decision that the eleven arms
keep the official zero-padding behaviour and that a padding mask stays an
independent correction arm. A guard that silently stops guarding would be worse
than no guard, so these tests verify both directions: the clean repository
passes, and each violation is actually caught.

These are GENERATED structural tests. They read source text and the queue
config. They import no patient data and assert nothing about patient
performance.
"""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "padding_guard", REPO / "scripts" / "check_slotspe_padding_policy_guard.py")
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


@pytest.fixture
def sandbox(tmp_path):
    """A throwaway copy of the code and config the guard inspects."""
    for name in ("slotspe_proof", "configs"):
        shutil.copytree(REPO / name, tmp_path / name)
    return tmp_path


def findings_in(sandbox, monkeypatch):
    monkeypatch.setattr(GUARD, "REPO", sandbox)
    return (GUARD.check_no_padding_mask_in_arms()
            + GUARD.check_correction_arms_not_in_queue()
            + GUARD.check_official_padding_is_reproduced())


def test_clean_repository_satisfies_the_decision(monkeypatch):
    """The real tree must currently pass every part of the guard."""
    monkeypatch.setattr(GUARD, "REPO", REPO)
    assert GUARD.check_no_padding_mask_in_arms() == []
    assert GUARD.check_correction_arms_not_in_queue() == []
    assert GUARD.check_official_padding_is_reproduced() == []


def test_mask_in_the_allocation_arm_path_is_caught(sandbox, monkeypatch):
    path = sandbox / "slotspe_proof" / "native_allocation.py"
    text = path.read_text(encoding="utf-8")
    anchor = "noises=noises,\n        )"
    assert anchor in text
    # Inject on the continuation line, which a naive single-line scan misses.
    path.write_text(text.replace(anchor, "noises=noises, token_mask=tm,\n        )"), encoding="utf-8")
    found = findings_in(sandbox, monkeypatch)
    assert any("token_mask" in f for f in found), found


def test_mask_in_the_S5_pruning_arm_path_is_caught(sandbox, monkeypatch):
    path = sandbox / "slotspe_proof" / "native_pruning.py"
    text = path.read_text(encoding="utf-8")
    assert "replay_with_assignment(" in text
    path.write_text(text.replace("iterations=attention.iters - 1)",
                                   "iterations=attention.iters - 1, token_mask=tm)"), encoding="utf-8")
    found = findings_in(sandbox, monkeypatch)
    assert any("token_mask" in f and "native_pruning" in f for f in found), found


def test_non_zero_padding_is_caught(sandbox, monkeypatch):
    """Padding must stay all-zero, or every padded key and value changes."""
    path = sandbox / "slotspe_proof" / "patient.py"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("dtype=patches.dtype)", "dtype=patches.dtype) + 1.0)"), encoding="utf-8")
    found = findings_in(sandbox, monkeypatch)
    assert any("no longer plain zeros" in f for f in found), found


def test_reintroduced_patient_exclusion_is_caught(sandbox, monkeypatch):
    path = sandbox / "slotspe_proof" / "patient.py"
    path.write_text(path.read_text(encoding="utf-8")
                    + "\n# fewer_real_patches_than_configured_no_padding\n", encoding="utf-8")
    found = findings_in(sandbox, monkeypatch)
    assert any("patient-exclusion" in f for f in found), found


def test_a_mask_arm_cannot_enter_the_queue(sandbox, monkeypatch):
    path = sandbox / "configs" / "slotspe_redundancy_queue.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["stages"][1]["variants"].append("P1_mask")
    config["requested_scope"]["selected_variants"].append("P1_mask")
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    found = findings_in(sandbox, monkeypatch)
    assert any("correction arm" in f for f in found), found


def test_mask_exemption_is_scoped_and_documented():
    """The one exempt module must carry a reason, and must be off the arm path."""
    assert set(GUARD.MASK_EXEMPT_MODULES) == {"prototype_assignment.py"}
    for reason in GUARD.MASK_EXEMPT_MODULES.values():
        assert reason.strip(), "an exemption without a stated reason is not allowed"
    engine = (REPO / "slotspe_proof" / "redundancy_engine.py").read_text(encoding="utf-8")
    assert "prototype_assignment" not in engine, \
        "the exempt module must not be reachable from the eleven-arm engine"


def test_guard_does_not_import_torch():
    """The plan-only default must stay free of torch and patient arrays."""
    assert "import torch" not in (REPO / "scripts" / "check_slotspe_padding_policy_guard.py").read_text(encoding="utf-8")