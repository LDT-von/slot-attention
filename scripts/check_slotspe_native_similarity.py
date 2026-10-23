"""Replay only the already-generated S0--S4 toy outputs to audit slot cosine.

This has no patient loader, training loop, optimizer or server code. It verifies
the synthetic manifest and byte hashes before loading invented inputs/weights.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-replay", action="store_true")
    parser.add_argument("--input", default="local_evidence/native_allocation_20261022_run01")
    parser.add_argument("--output", default="local_evidence/native_similarity_20261022_replay01")
    args = parser.parse_args()
    if not args.synthetic_replay:
        print("Plan only: 15 existing invented-data models x 3 paired draws; report WSI/omic all8 and native-selected2 cosine. No training/patients.")
        return
    private = (ROOT / "local_evidence").resolve()
    source, dest = (ROOT / args.input).resolve(), (ROOT / args.output).resolve()
    if not source.is_relative_to(private) or not dest.is_relative_to(private) or dest == private:
        raise ValueError("input/output must be private local_evidence children")
    if dest.exists():
        raise FileExistsError("preserve prior output; choose a new output directory")
    manifest = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    published_path = ROOT / "docs/research/slotspe_native_assignment_training_20261022/aggregate_summary.json"
    published = json.loads(published_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "native_slotspe_generated_allocation_smoke_v1" or
            manifest.get("status") != "completed_generated_data_only" or
            manifest["patient_runs"] != 0 or manifest["server_runs"] != 0 or
            manifest["plan"]["data"] != "invented only; new random weights; CPU float32" or
            sha(source / "summary.json") != published["source_summary_sha256"] or
            manifest["raw_file_sha256"] != published["raw_synthetic_file_sha256"]):
        raise ValueError("requires the pinned, previously generated-data-only manifest")
    for name, expected in manifest["raw_file_sha256"].items():
        if Path(name).name != name or not name.endswith(".pt") or sha(source / name) != expected:
            raise ValueError("synthetic input hash/path mismatch")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("native source changed")
    dest.mkdir(parents=True)

    import torch
    from torch.nn import functional as F
    from survot_rank.research.slotspe_proof.model import build_model, make_args
    from survot_rank.research.slotspe_proof.native_allocation import NativeAllocation, VARIANTS

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds = manifest["plan"]["seeds"]
    old = {(row["seed"], row["variant"]): row for row in manifest["runs"]}
    result = {
        "schema": "native_slotspe_generated_slot_similarity_replay_v1",
        "recorded_at": datetime.now().astimezone().isoformat(),
        "source_summary_sha256": sha(source / "summary.json"),
        "script_sha256": sha(Path(__file__)),
        "definition": "Within each generated example, cosine of all unordered SA output slot pairs; average examples, draws, then three seeds. No cross-modal cosine.",
        "selected_definition": "Same raw SA slots indexed by the actual native decoder hard_keep Top2; not decoder.map outputs. Sampled selection may differ across arms.",
        "update_definition": "Last-iteration attention-weighted native V, after combine_heads, before GRU/MLP. Descriptive location probe, not causal attribution.",
        "scope": "32-dim, 8slots, 3iterations, existing 24-update generated-data models; no new training",
        "per_seed": [], "means": [],
        "counts": {"model_forwards": 0, "full_loss_backwards": 0, "optimizer_steps": 0, "patient_runs": 0, "server_runs": 0},
        "extra_global_RNG_consumption": False,
    }

    def per_example_cos(tensor):
        unit = F.normalize(tensor, dim=-1)
        pairs = torch.triu_indices(unit.shape[1], unit.shape[1], offset=1)
        return (unit[:, pairs[0]] * unit[:, pairs[1]]).sum(-1).mean(-1)

    for seed in seeds:
        generated = torch.load(source / f"generated_seed_{seed}.pt", map_location="cpu", weights_only=True)
        holdout = {key: value[64:] if torch.is_tensor(value) else value
                   for key, value in generated["invented_data"].items()}
        for arm in VARIANTS:
            model_args = make_args([4] * 4, encoding_dim=8)
            model_args.wsi_projection_dim = 32
            model_args.slot_iters = 3
            model = build_model(ROOT / "third_party/SlotSPE", model_args, "original")
            checkpoint = torch.load(source / f"toy_{arm}_seed_{seed}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["state"], strict=True)
            model.eval()
            captured_inputs, kept = {}, {}
            handles = []
            for modality in ("wsi", "omic"):
                attention = getattr(model, f"slot_attention_{modality}")
                decoder = getattr(model, f"slot_decoder_{modality}")

                def save_input(_module, inputs, _modality=modality):
                    captured_inputs[_modality] = inputs[0].detach()

                def save_keep(_module, _inputs, outputs, _modality=modality):
                    kept[_modality] = outputs[2].detach().bool()

                handles.append(attention.register_forward_pre_hook(save_input))
                handles.append(decoder.register_forward_hook(save_keep))
            rows, raw = [], []
            try:
                with NativeAllocation(model, arm, capture=True) as adapter:
                    for repeat in range(3):
                        torch.manual_seed(seed * 100000 + 50000 + repeat)
                        adapter.set_noise_seed(seed * 100000 + 70000 + repeat)
                        captured_inputs.clear()
                        kept.clear()
                        with torch.no_grad():
                            logits, auxiliary = model(**holdout)
                            result["counts"]["model_forwards"] += 1
                            if not torch.allclose(logits, checkpoint["final_logits"][repeat], atol=1e-6, rtol=1e-6):
                                raise RuntimeError("prior generated model replay failed")
                            row, raw_row = {}, {}
                            assert len(adapter.records) == 2
                            for record in adapter.records:
                                modality = record["modality"]
                                assert record["call_index"] == 0
                                slots = record["final"]
                                mask = kept[modality]
                                assert (mask.sum(-1) == 2).all()
                                selected = slots[mask].reshape(slots.shape[0], 2, slots.shape[-1])
                                attention = getattr(model, f"slot_attention_{modality}")
                                values = attention.split_heads(attention.to_v(attention.norm_input(captured_inputs[modality])))
                                weighted = torch.einsum("bhnd,bhsn->bhsd", values, record["attentions"][-1])
                                update = attention.combine_heads(attention.merge_heads(weighted))
                                all_cos, selected_cos, update_cos = map(per_example_cos, (slots, selected, update))
                                row[modality + "_all8_cos"] = float(all_cos.mean())
                                row[modality + "_selected2_cos"] = float(selected_cos.mean())
                                row[modality + "_value_update_cos"] = float(update_cos.mean())
                                raw_row[modality] = {"all8_cos": all_cos, "selected2_cos": selected_cos,
                                                     "value_update_cos": update_cos, "hard_keep": mask, "slots": slots}
                            rows.append(row)
                            raw.append(raw_row)
            finally:
                for handle in handles:
                    handle.remove()
            means = {key: statistics.mean(row[key] for row in rows) for key in rows[0]}
            expected = old[seed, arm]["final_holdout"]
            for modality in ("wsi", "omic"):
                if abs(means[modality + "_all8_cos"] - expected[modality + "_0_slot_cosine"]) > 1e-6:
                    raise RuntimeError("published all-slot metric does not match replay")
            result["per_seed"].append({"seed": seed, "variant": arm, "metrics": means, "replays_match_saved_logits": True})
            torch.save(raw, dest / f"toy_slot_metrics_{arm}_seed_{seed}.pt")
            print(f'{arm}/seed{seed}: all8 WSI={means["wsi_all8_cos"]:.6f} OMIC={means["omic_all8_cos"]:.6f}', flush=True)
    for arm in VARIANTS:
        rows = [row for row in result["per_seed"] if row["variant"] == arm]
        means = {key: statistics.mean(row["metrics"][key] for row in rows) for key in rows[0]["metrics"]}
        base = [row for row in result["per_seed"] if row["variant"] == "S0"]
        delta = {key: statistics.mean(row["metrics"][key] - baseline["metrics"][key] for row, baseline in zip(rows, base)) for key in means}
        result["means"].append({"variant": arm, "metrics": means, "paired_delta_to_S0": delta})
    result["native_source_hashes_match"] = all(sha(ROOT / name) == expected for name, expected in manifest["source_sha256"].items())
    assert result["native_source_hashes_match"]
    result["raw_generated_metrics_sha256"] = {p.name: sha(p) for p in sorted(dest.glob("*.pt"))}
    result["status"] = "completed_generated_only_no_new_training"
    (dest / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    print(json.dumps(result["means"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
