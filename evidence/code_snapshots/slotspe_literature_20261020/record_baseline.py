"""Record static baseline evidence; never imports or executes the model."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"C:\Users\栋栋\Desktop\特征解耦论文\SlotSPE-main\SlotSPE-main")
OUT = Path(r"E:\DCT-Reg\docs\research\slotspe_literature_20261020")
files = ["models/SlotSPE.py", "models/slot_attention.py", "models/transformer.py", "models/omics_encoder.py", "utils/process_args.py", "scripts/SlotSPE.sh"]
records = []
for filename in files:
    path = BASE / filename
    raw = path.read_bytes()
    code = raw.decode("utf-8-sig")
    if path.suffix == ".py":
        ast.parse(code)
    records.append({"relative_path": filename, "absolute_path": str(path), "sha256_before": hashlib.sha256(raw).hexdigest(), "sha256_after": hashlib.sha256(path.read_bytes()).hexdigest()})
args_code = (BASE / "utils/process_args.py").read_text(encoding="utf-8-sig")
defaults = {}
for node in ast.walk(ast.parse(args_code)):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args and isinstance(node.args[0], ast.Constant):
        name = node.args[0].value
        for kw in node.keywords:
            if kw.arg == "default":
                try:
                    defaults[name] = ast.literal_eval(kw.value)
                except (ValueError, TypeError):
                    pass
report = {
    "task": "SPE-08", "stage": "P1 static existence and forward-scope check", "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "original_repository": str(BASE), "method": "AST parse and exact source read; no model execution", "patient_data_read": False, "checkpoint_loaded": False,
    "defaults": {name: defaults[name] for name in ["--slot_num_wsi", "--slot_num_omics", "--topk_ratio", "--slot_iters"]},
    "default_candidate_slots_per_modality": defaults["--slot_num_wsi"], "default_selected_slots_per_modality": max(1, int(defaults["--slot_num_wsi"] * defaults["--topk_ratio"])),
    "claims": [
        {"claim": "候选槽数量是配置值，主模型没有按患者推断数量", "status": "source_confirmed", "source": "models/SlotSPE.py:173-194,283-285; models/slot_attention.py:69-80"},
        {"claim": "默认Top-K掩码每患者为2，但不是最终预测的总槽容量", "status": "source_confirmed", "source": "models/SlotSPE.py:27,289-300"},
        {"claim": "cross分支读取全部候选槽", "status": "source_confirmed", "source": "models/SlotSPE.py:294"},
        {"claim": "固定8候选槽造成真实患者预后信息损失", "status": "not_yet_established", "source": None},
        {"claim": "固定Top-2造成真实患者预后信息损失", "status": "not_yet_established", "source": None}
    ],
    "boundary": "这是默认配置与读取路径事实；CLI可覆盖槽数。不能把候选数量、Top-K掩码数量、生物学事件数量混为一谈，也不能把cross全槽读取说成Top-K过滤失败。",
    "source_files": records,
    "all_source_files_unchanged": all(r["sha256_before"] == r["sha256_after"] for r in records),
}
(OUT / "p1_source_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"defaults": report["defaults"], "all_source_files_unchanged": report["all_source_files_unchanged"], "patient_data_read": False}, ensure_ascii=False))
