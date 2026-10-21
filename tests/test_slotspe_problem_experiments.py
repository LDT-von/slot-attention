"""Measurement and execution guards on synthetic arrays only; no native model."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from survot_rank.research.slotspe_proof import problem_experiments as p


def test_plan_does_not_import_torch_or_scipy():
    code = "from survot_rank.research.slotspe_proof.problem_experiments import config_at,DEFAULT_CONFIG,plan; import sys; plan(config_at(DEFAULT_CONFIG)); assert 'torch' not in sys.modules; assert 'scipy' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], cwd=p.REPO, check=True, capture_output=True)


def test_default_plan_separates_training_and_missing_semantic_prerequisites():
    config = p.config_at(p.DEFAULT_CONFIG)
    output = p.plan(config)
    assert output["training_runs"] == 0
    assert output["automatic_problem_verdict"] is False
    assert output["missing_sidecars"] == ["annotations", "paired_features"]
    assert "scale" in output["questions"]["P6"]
    assert "not_primary_endpoints" in output


def test_capacity_plan_declares_distinct_training_phase():
    output = p.plan(p.config_at(p.DEFAULT_CONFIG), "capacity_train")
    assert output["training_runs"] == 3
    assert output["capacity_factor"]["fixed_keep_k"] == 2


def test_refuses_public_or_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "REPO", tmp_path)
    with pytest.raises(ValueError):
        p.private_output(tmp_path / "docs" / "results")
    existing = tmp_path / "local_evidence" / "already_here"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        p.private_output(existing)
    with pytest.raises(ValueError):
        p.private_output(tmp_path / "local_evidence" / ".." / "public")


def test_annotation_must_have_external_provenance_and_exact_rows():
    with pytest.raises(ValueError):
        p.check_annotation({"token_groups": [0, 1]}, 2)
    item = {"review_provenance": "synthetic-independent-labeler", "label_definition": "two synthetic types", "token_groups": [0, 1]}
    assert p.check_annotation(item, 2) == [0, 1]
    with pytest.raises(ValueError):
        p.check_annotation(item, 3)


def test_semantic_matching_does_not_reuse_scoring_tokens():
    # Matching columns agree, but heldout content assignment reverses.
    a = np.array([[1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1]], dtype=float)
    b = np.array([[1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 0, 1, 1, 0]], dtype=float)
    labels = ["A", "A", "B", "B", "A", "A", "B", "B"]
    output = p.semantic_stability(a, b, labels)
    assert output["matched_content_total_variation"] == pytest.approx(1)
    assert output["matching_ambiguous"] is False


def test_semantic_permutation_is_recovered_with_mixed_category_types():
    a = np.array([[1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1]], dtype=float)
    labels = [1, 1, "B", "B", 1, 1, "B", "B"]
    output = p.semantic_stability(a, a[::-1], labels)
    assert output["matched_content_total_variation"] == pytest.approx(0)
    assert output["same_index_content_total_variation"] == pytest.approx(1)


def test_empty_semantic_scoring_mass_rejected():
    a = np.array([[1, 0, 1, 0], [0.2, 0, 0.8, 0]], dtype=float)
    with pytest.raises(ValueError, match="empty attention row"):
        p.semantic_stability(a, a, [0, 0, 1, 1])


def test_no_labels_is_missing_evidence_not_a_negative_finding():
    assert p.semantic_stability(np.ones((2, 4)), np.ones((2, 4)), None)["status"].startswith("not_testable")


def test_objective_hook_registration_failure_cleans_existing_hooks():
    torch = pytest.importorskip("torch")
    class IncompleteSyntheticModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.slot_decoder_wsi = torch.nn.Identity()
    model = IncompleteSyntheticModel()
    with pytest.raises(AttributeError):
        p.gradient_probe(model, {}, 1)
    assert len(model.slot_decoder_wsi._forward_hooks) == 0
    assert not model.training


def test_config_unknown_keys_and_boolean_counts_rejected(tmp_path):
    config = p.read(p.DEFAULT_CONFIG)
    config["capacity_counts"] = [True, 8]
    location = tmp_path / "config.json"
    location.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        p.config_at(location)
    config = p.read(p.DEFAULT_CONFIG)
    config["silent_new_factor"] = 99
    location.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        p.config_at(location)


def test_capacity_dispatch_uses_fresh_child_and_keeps_outer_receipt(tmp_path, monkeypatch):
    from survot_rank.research.slotspe_proof import problem_capacity
    from survot_rank.research.slotspe_proof.model import make_args, MODEL_SOURCE_FILES
    monkeypatch.setattr(p, "REPO", tmp_path)
    source = tmp_path / "source"
    for relative in MODEL_SOURCE_FILES:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic provenance fixture only\n")
    input_root = tmp_path / "inputs"
    (input_root / "prepared").mkdir(parents=True)
    manifest = {"omic_sizes": [3], "encoding_dim": 1,
                "source_hashes": {name: p.sha(source / name) for name in MODEL_SOURCE_FILES},
                "folds": {"0": {"train": [], "val": [], "binning": {"validation_outcomes_used_for_training_bins": False}}}}
    p.write(input_root / "prepared/manifest.json", manifest)
    args_file = tmp_path / "args.json"
    p.write(args_file, vars(make_args([3], encoding_dim=1)))
    config = p.read(p.DEFAULT_CONFIG)
    config.update(source_root=str(source), input_root=str(input_root), model_args_file=str(args_file), device="cpu")
    def no_patient_capacity(config, *, output_root, **kwargs):
        assert not output_root.exists()
        assert (output_root.parent / "execution_receipt.json").exists()
        output_root.mkdir()
        return {"status": "synthetic_dispatch_only"}
    monkeypatch.setattr(problem_capacity, "run_capacity", no_patient_capacity)
    output = tmp_path / "local_evidence" / "synthetic_dispatch"
    result = p.execute(config, "capacity_train", output)
    assert result["status"] == "completed"
    assert (output / "capacity_training").exists()


def test_capacity_args_keep_numeric_decoder_factor_recipe(tmp_path):
    from survot_rank.research.slotspe_proof.model import make_args
    location = tmp_path / "args.json"
    p.write(location, {"model_args": vars(make_args([3], encoding_dim=1)), "decoder_k": {"wsi": {"actual_value": 2}, "omic": {"actual_value": 2}}})
    args = p.load_args(location, {"omic_sizes": [3], "encoding_dim": 1})
    assert args._diagnostic_decoder_k["omic"]["actual_value"] == 2


def test_nuisance_controls_preserve_anchor_initial_and_equal_doses(monkeypatch):
    torch = pytest.importorskip("torch")
    from survot_rank.research.slotspe_proof import problem_trace
    x = torch.arange(18, dtype=torch.float32).reshape(1, 18, 1)
    initial = torch.ones(1, 2, 1)
    calls = []
    def synthetic_uniform_replay(module, inputs, state, iterations=None):
        assert state is initial
        calls.append(inputs[0, :, 0].tolist())
        return {"attentions": [torch.ones(1, 1, 2, inputs.shape[1]) / inputs.shape[1]]}
    monkeypatch.setattr(problem_trace, "replay_slots", synthetic_uniform_replay)
    roles = ["signal"] * 12 + ["candidate_nuisance"] * 6
    output = p.nuisance_probe(object(), x, initial, {"roles": roles, "role_scope": "acquisition_artifact"})
    anchor = calls[0]
    assert len(output["dose_results"]) == 9
    for index in range(1, len(calls), 2):
        nuisance, relevant = calls[index:index + 2]
        assert nuisance[:len(anchor)] == relevant[:len(anchor)] == anchor
        assert len(nuisance) == len(relevant)
        assert set(nuisance[len(anchor):]) <= set(range(12, 18))
        assert set(relevant[len(anchor):]) <= set(range(8, 12))


def test_prognostic_irrelevance_is_not_inferred_from_visual_role():
    torch = pytest.importorskip("torch")
    output = p.nuisance_probe(object(), torch.ones(1, 6, 2), torch.ones(1, 2, 2),
                              {"roles": ["signal"] * 3 + ["candidate_nuisance"] * 3,
                               "role_scope": "prognostic_irrelevance_claim"})
    assert output["status"] == "not_testable_unsubstantiated_prognostic_irrelevance"
