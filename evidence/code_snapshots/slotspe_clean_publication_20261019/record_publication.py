import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime

ROOT=Path(r"E:\DCT-Reg")
AUDIT=Path(__file__).parent
plan=json.loads((AUDIT/"publication_plan.json").read_text(encoding="utf-8"))
commit_info=json.loads((AUDIT/"commit.json").read_text(encoding="utf-8"))
WORK=Path(plan["worktree"])
commit=commit_info["commit"]
def git(*args, cwd=WORK):
    return subprocess.check_output(["git",*args],cwd=cwd)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
assert git("rev-parse","HEAD").decode().strip()==commit
assert git("rev-parse","origin/main").decode().strip()==commit
assert git("rev-parse","HEAD",cwd=ROOT).decode().strip()==plan["primary_head_before"]
assert hashlib.sha256(git("ls-files","-s","-z",cwd=ROOT)).hexdigest()==plan["primary_index_semantic_sha256"]
assert not git("status","--porcelain").strip()
scope=commit_info["tree_proof"]["scope"]
for relative in scope:
    expected=(WORK/relative).read_bytes()
    actual=git("show",commit+":"+relative)
    if relative.endswith(".zip"):
        assert actual==expected
    else:
        assert actual==expected.replace(b"\r\n",b"\n"), relative
archive="docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip"
assert hashlib.sha256(git("show",commit+":"+archive)).hexdigest()==plan["frozen_archive_sha256"]
now=datetime.now().astimezone().isoformat(timespec="seconds")
receipt={"recorded_at":now,"status":"remote_publication_verified","commit":commit,"remote":"https://github.com/LDT-von/DCT-Reg.git",
    "remote_branch":"refs/heads/main","remote_ref_verified":True,"origin_main_and_isolated_head_equal":True,
    "archive":archive,"archive_sha256":plan["frozen_archive_sha256"],"scope":scope,"tree_verification":commit_info["tree_proof"],
    "isolated_publication_tests":{"passed":2,"scope":"synthetic_metadata_and_tensors_only","xml":"docs/research/slotspe_clean_original_20261019/publication_tests.xml"},
    "primary_head_preserved":plan["primary_head_before"],"primary_index_preserved":True,
    "server_execution":{"dispatched":False,"verified_running":False,"raw_results_received":False,
        "reason":"Available connected chats/hosts are local; no server-side AI execution channel is identified.",
        "handoff":"Server AI should fetch this commit, extract the frozen ZIP to a new directory, then doctor/plan/run the clean-original config."},
    "patient_jobs_started_by_this_window":False}
dest=ROOT/"docs/research/slotspe_clean_original_20261019"
(dest/"publication_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for name in ("publication_tests.xml","publication_checks.json"):
    shutil.copy2(WORK/"docs/research/slotspe_clean_original_20261019"/name,dest/name)
paths=[ROOT/"docs/THREE_LINE_PROGRESS.json",ROOT/"docs/三条研究路线_工作总览_20261019.txt",ROOT/"docs/SLOTSPE_PROOF_TASK.json"]
backup=ROOT/"docs/history"/("slotspe_clean_publication_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
backup.mkdir(parents=True)
for path in paths:
    shutil.copy2(path,backup/path.name)
p=json.loads(paths[0].read_text(encoding="utf-8-sig"))
before=copy.deepcopy(p)
t=next(t for t in p["tasks"] if t["id"]=="SPE-09")
t.update(status="completed_published",publication_commit=commit,remote_published=True,
    publication_receipt="docs/research/slotspe_clean_original_20261019/publication_receipt.json")
t["evidence_refs"].append("EV-SPE-CLEAN-PUBLISH-20261019")
t["next_action"]="服务器AI拉取main@"+commit[:7]+"并解压原版单臂包执行；尚无可调用服务器AI通道和启动/结果回执。"
t=next(t for t in p["tasks"] if t["id"]=="SPE-07")
t.update(publication_commit=commit,server_dispatched_by_this_window=False,server_result_audited=False)
t["evidence_refs"].append("EV-SPE-CLEAN-PUBLISH-20261019")
t["next_action"]="服务器AI拉取main@"+commit[:7]+"，解压独立包，doctor→plan→run；原版单臂fold0/seed3、3轮、256真实patch、20次评估；原始产物回传后验收。"
line=next(x for x in p["lines"] if x["id"]=="L2")
line["current_clean_original_handoff"].update(publication_commit=commit,remote_published=True,server_dispatched=False)
p["evidence"].append({"id":"EV-SPE-CLEAN-PUBLISH-20261019","type":"normal_push_remote_ref_tree_and_frozen_zip_verified",
    "details":"26个SlotSPE交付/证据/入口文件已发布；2385条基线路径均保留；隔离2项原版闭环检查通过。服务器尚无启动回执和原始产物。",
    "paths":["docs/research/slotspe_clean_original_20261019/publication_receipt.json"],"commit":commit,
    "local_file_sha256":{"docs/research/slotspe_clean_original_20261019/publication_receipt.json":sha(dest/"publication_receipt.json")}})
p["verification_snapshot"]["last_verified_remote_main"]=commit
p["change_log"].append({"date":"2026-10-19","scope":"SlotSPE_clean_original_publication","details":"正常推送main@"+commit[:7]+"并验证远端引用/26文件/任务包哈希；主工作区未重置，未调用服务器或启动患者任务。","backup":backup.relative_to(ROOT).as_posix()})
assert [t for t in p["tasks"] if not t["id"].startswith("SPE-")]==[t for t in before["tasks"] if not t["id"].startswith("SPE-")]
assert [x for x in p["lines"] if x["id"]!="L2"]==[x for x in before["lines"] if x["id"]!="L2"]
paths[0].write_text(json.dumps(p,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
human=paths[1].read_text(encoding="utf-8-sig")
marker="SPE-09本地完成原版入口、自然门控记录及原始数值验收；SPE-07等待服务器AI和原始回传。"
assert human.count(marker)==1
human=human.replace(marker,"SPE-09已发布origin/main@"+commit[:7]+"；26文件和冻结包哈希核验通过，隔离原版闭环2项通过。SPE-07待服务器启动回执和原始回传，尚未派发至可调用的服务器AI通道。")
paths[1].write_text(human,encoding="utf-8")
task=json.loads(paths[2].read_text(encoding="utf-8-sig"))
task["delivery"].update(remote_published_this_work=True,source_commit=commit,remote_branch="origin/main",publication_receipt="docs/research/slotspe_clean_original_20261019/publication_receipt.json",
    archive_build_snapshot_note="ZIP bytes unchanged. Its descriptor/receipt records pre-publication build state; this receipt verifies later publication.")
paths[2].write_text(json.dumps(task,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
assert sha(ROOT/archive)==plan["frozen_archive_sha256"]
print(json.dumps({"published_commit":commit,"files_verified":len(scope),"server_started":False,"receipt":str(dest/"publication_receipt.json"),"primary_head_preserved":True},ensure_ascii=False))
