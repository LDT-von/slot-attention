"""Single explicit queue stage; planning never imports this module."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def summarize(config, root, checks_only):
    from .queue import _json, _write
    if config.get("experiment") == "original_audit":
        from .patient import summarize as summarize_patients
        protocol = _json(root / "checks/protocol_checks.json")
        patient = None if checks_only else summarize_patients(config, root)
        if patient is not None and patient["status"] != "completed":
            raise RuntimeError("Native patient audit incomplete")
        result = {"status": "completed", "task_name": "slotspe_problem_proof", "experiment": "original_audit",
            "checks_only": checks_only, "transport_added": False, "training_arms": ["original"],
            "repaired_models_constructed": False, "protocol_dependence": protocol["held_out_label_dependence"],
            "patients": patient, "scope": "Unmodified native model; diagnostic preparation and controlled interventions disclosed separately",
            "claims_not_supported": ["Patient improvement", "Correction attribution", "Clinical causality", "Guaranteed problem occurrence"]}
        _write(root / "proof_summary.json", result, exclusive=True)
        (root / "proof_summary.txt").write_text("SlotSPE 原版单臂问题检查\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result
    checks = {name: _json(root / "checks" / (name + "_checks.json"))
              for name in ("protocol", "reconstruction", "mechanisms")}
    patient = None
    if not checks_only:
        from .patient import summarize as summarize_patients
        patient = summarize_patients(config, root)
        if patient["status"] != "completed":
            raise RuntimeError("Patient batch is incomplete; refusing to publish complete means")
    result = {"status": "completed", "task_name": "slotspe_problem_proof", "checks_only": checks_only,
        "experiment": config.get("experiment", "reproduction"),
        "attribution_checks": _json(root / "checks/attribution_checks.json") if config.get("experiment") == "attribution_abc" else None,
        "protocol_dependence": checks["protocol"]["held_out_label_dependence"],
        "protocol": [{"fold": row["fold"], "label_dependence": row["held_out_label_dependence"],
            "interventions": row["interventions"]} for row in checks["protocol"]["folds"]],
        "mechanism_observations": checks["mechanisms"]["records"],
        "reconstruction": {"constant_zero_dropout_off": checks["reconstruction"]["constant_zero_reconstruction_dropout_off"],
            "same_construction_zero_training_dropout": checks["reconstruction"]["same_construction_zero_with_original_training_dropout"],
            "claims_not_supported": checks["reconstruction"]["claims_not_supported"]},
        "patients": patient, "transport_added": False,
        "claims_not_supported": ["OT necessity", "Novel method efficacy", "Independent-test generalization gain",
            "Actual patient representation collapse from a deterministic reconstruction counterexample"]}
    _write(root / "proof_summary.json", result, exclusive=True)
    lines = ["SlotSPE 证明实验：服务器本批次汇总", "",
        "源代码：" + config["source_root"], "队列：" + config["study"],
        "训练折/种子：" + str(config["folds"]) + " / " + str(config["seeds"]),
        "元数据协议依赖验证结局：" + result["protocol_dependence"],
        "重构常量反例（关闭dropout）：" + result["reconstruction"]["constant_zero_dropout_off"],
        "同一构造在原训练dropout下仍为零：" + result["reconstruction"]["same_construction_zero_training_dropout"],
        "这些构造不证明真实患者训练已经坍缩。", ""]
    if patient:
        lines += ["真实患者训练已完成的配对任务：" + str(patient["completed_pairs"]),
            "每臂轮数：" + str(config["epochs"]), "检查统计：" + json.dumps(patient["checks"], ensure_ascii=False),
            "验证指标仅为描述性结果，未使用独立测试集。"]
    else:
        lines += ["本批次仅执行元数据与合成检查，未进行患者训练。"]
    lines += ["", "original为原源码模型；readout_fix为同参数工程修正；eval_fix只更改原权重评估规则。",
        "A/B/C为独立归因模式：A原self汇总，B选中汇总且仅断开主损失的选择梯度，C同前向接通；辅助路径相同。",
        "cross仍是原全槽分支。两训练臂使用同一配对患者、训练折分箱、初始化及预算。",
        "observed/not_observed如实报告，不强制要求原版失败。", "未添加运输插件或任何DCT目标；不据此宣称OT必要或新方法有效。",
        "逐患者预测、原始记录、源码哈希、checkpoint与日志保存在本批次目录。"]
    (root / "proof_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--stage", required=True, choices=("protocol", "reconstruction", "mechanisms", "attribution", "prepare", "pair", "summarize"))
    p.add_argument("--fold", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--checks-only", action="store_true")
    args = p.parse_args(argv)
    root = args.output_root.resolve()
    from .queue import _json, _verify_resolved_config
    manifest = _json(root / "queue_manifest.json")
    if args.config.resolve() != root / "resolved_config.json":
        raise RuntimeError("Worker config must be the batch's saved resolved configuration")
    config = _verify_resolved_config(root, manifest["config"])
    stage = {"kind": args.stage}
    if args.stage == "pair":
        stage.update(fold=args.fold, seed=args.seed)
    allowed = [{key: value for key, value in item.items() if key != "id"} for item in manifest["stages"]]
    if args.checks_only != manifest["checks_only"] or stage not in allowed:
        raise RuntimeError("Worker stage does not match the saved batch plan")
    if args.stage in ("reconstruction", "mechanisms", "attribution", "pair"):
        import torch
        torch.set_num_threads(config.get("torch_threads", 2))
    if args.stage == "protocol":
        from .checks import run_protocol_checks
        result = run_protocol_checks(config["source_root"], config["metadata_root"], config["study"],
            config["folds"], root / "checks", label_col=config["label_col"], n_bins=4)
    elif args.stage == "reconstruction":
        from .checks import run_reconstruction_checks
        result = run_reconstruction_checks(config["source_root"], root / "checks")
    elif args.stage == "mechanisms":
        from .mechanisms import run_mechanism_checks
        result = run_mechanism_checks(config["source_root"], root / "checks", config["seeds"])
    elif args.stage == "attribution":
        from .attribution import run_attribution_checks
        result = run_attribution_checks(config["source_root"], root / "checks", config["seeds"])
    elif args.stage == "prepare":
        from .patient import prepare
        result = prepare(config, root)
    elif args.stage == "pair":
        if args.fold is None or args.seed is None:
            p.error("pair requires --fold and --seed")
        from .patient import run_pair
        result = run_pair(config, root, args.fold, args.seed)
    else:
        result = summarize(config, root, args.checks_only)
    print(json.dumps({"stage": args.stage, "status": result.get("status", "completed"),
        "output_root": str(root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
