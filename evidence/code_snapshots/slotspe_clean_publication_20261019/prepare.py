"""Explicit SlotSPE-only release from the validated frozen delivery."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(r"E:\DCT-Reg")
WORK = Path(r"C:\Users\栋栋\.codex\worktrees\slotspe-clean-server-release\DCT-Reg")
AUDIT = Path(__file__).parent
BASE = "cc81df1586b4b0e437cbdffe04c923ed6546fadf"
ZIP = ROOT/"docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip"
expected_zip = "a690578675d51c1328c92cac8d7dc41dc6494fbd179e53476763df06621007a4"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ZIP) == expected_zip
def git(*args, cwd=WORK):
    return subprocess.check_output(["git", *args], cwd=cwd).decode("utf-8").strip()
assert git("rev-parse", "HEAD") == BASE
assert not git("status", "--porcelain")
main_head = git("rev-parse", "HEAD", cwd=ROOT)
index_sha = hashlib.sha256(subprocess.check_output(["git", "ls-files", "-s", "-z"], cwd=ROOT)).hexdigest()
subprocess.run(["git", "switch", "-c", "codex/slotspe-clean-server-20261019"], cwd=WORK, check=True)

frozen = ["survot_rank/research/slotspe_proof/acceptance.py", "survot_rank/research/slotspe_proof/interventions.py",
    "survot_rank/research/slotspe_proof/native_audit.py", "survot_rank/research/slotspe_proof/patient.py",
    "survot_rank/research/slotspe_proof/queue.py", "survot_rank/research/slotspe_proof/worker.py",
    "scripts/build_slotspe_proof_delivery.py", "configs/slotspe_clean_original_server.json",
    "tests/test_slotspe_clean_original.py", "docs/SLOTSPE_PROOF_TASK.json",
    "docs/SlotSPE证明任务_服务器一键运行.txt"]
with zipfile.ZipFile(ZIP) as z:
    manifest = json.loads(z.read(next(n for n in z.namelist() if n.endswith("/DELIVERY_MANIFEST.json"))))
    for relative in frozen:
        data = z.read(manifest["prefix"]+relative)
        assert hashlib.sha256(data).hexdigest() == manifest["files_sha256"][relative]
        target = WORK/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

extras = ["docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip",
    "docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip.sha256",
    "docs/research/slotspe_eight_issues_20261019/audit.json",
    "docs/research/slotspe_eight_issues_20261019/synthetic_checks.json",
    "docs/research/slotspe_eight_issues_20261019/run_synthetic_audit.py",
    "docs/research/slotspe_eight_issues_20261019/核查结论.txt",
    "docs/research/slotspe_clean_original_20261019/delivery_receipt.json",
    "docs/research/slotspe_clean_original_20261019/evidence_and_contract.json",
    "docs/research/slotspe_clean_original_20261019/full_suite_before_hardening.xml",
    "docs/research/slotspe_clean_original_20261019/final_critical_checks.xml"]
for relative in extras:
    target = WORK/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT/relative).read_bytes())

# Existing shared entries are edited from the remote base, never replaced by
# the primary checkout's unrelated V2/transport changes.
readme = WORK/"README.md"
text = readme.read_text(encoding="utf-8-sig")
marker = "# DCT-Reg v3.10"
assert text.count(marker) == 1
readme.write_text("""## SlotSPE 最高优先级：干净原版服务器问题复现（2026-10-19）

- [新独立任务包](docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip)、[SHA256](docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip.sha256)、[执行与回传说明](docs/SlotSPE证明任务_服务器一键运行.txt)。
- 服务器AI请解压新包到独立目录，再使用 `configs/slotspe_clean_original_server.json`，按 doctor → plan → run 执行。
- 只训练原版一臂：fold0、seed3、3轮、256真实patch、batch8、同末轮权重重复评估20次。原版10份文件在包内逐字节封存。
- [本地原始诊断](docs/research/slotspe_eight_issues_20261019/synthetic_checks.json)与[单臂检查记录](docs/research/slotspe_clean_original_20261019/evidence_and_contract.json)均非服务器患者结果。
- 当前等待服务器启动回执与原始产物；原两臂及A/B/C包保留用于后续修正归因。其他研究线继续使用各自入口。

"""+text[text.index(marker):], encoding="utf-8")
agents = WORK/"AGENTS.md"
text = agents.read_text(encoding="utf-8-sig")
marker = "## 当前纯运输插拔范围"
assert text.count(marker) == 1
agents.write_text("""## SlotSPE最高优先级：干净原版单臂检查（2026-10-19）

- 当前入口为 `docs/SLOTSPE_PROOF_TASK.json`、`docs/SlotSPE证明任务_服务器一键运行.txt`；新包为 `docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip`。服务器请解压新包到独立目录，以保持原版源码字节与回传验收一致。
- 本轮仅original：fold0/seed3、3轮、256真实patch、batch8、20次评估；配置 `configs/slotspe_clean_original_server.json`。服务器AI执行，本窗口验收；不索要SSH、不自动扩展预算。
- 未修改原版模型forward、槽模块、decoder和原生损失。诊断协议使用完整配对病例、固定真实patch与训练折分箱，原版全队列分箱另在副本中核查。
- 验收逐参数梯度、患者末轮固定选槽且冻结cross的干预、同权重重复预测与同RNG重放、补零敏感性及自然门控。受控补零/分数反例不作为患者发生率；未观察到则记not_observed。
- 本地实现/合成检查、Git发布、服务器实际运行、患者效果分别登记；没有原始回传不能说服务器已证明。新包和交付记录中的未发布字段是封包时快照，以实际Git提交与发布回执核实发布状态。
- 原版/修正版和A/B/C实现及历史包保留；B/C仅切换最终self汇总的选择梯度，辅助训练和梯度一致。患者修正归因属于后续阶段，其他研究线不因此改变。

"""+text[text.index(marker):], encoding="utf-8")
human = WORK/"docs/三条研究路线_工作总览_20261019.txt"
text = human.read_text(encoding="utf-8-sig")
a,b = text.index("2. SlotSPE问题线"),text.index("3. V313运输插拔线")
text = text[:a]+"""2. SlotSPE问题线
当前最高优先级为干净原版单臂服务器问题检查，SPE-07/SPE-09。
本地已有源码与随机权重/合成原始证据：self未选槽仍影响最终预测、eval随机性、主损失选择梯度断开、补零无mask和门控近零反例；不能当患者发生率或提分。
新包：docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip；原版10份文件按指定桌面仓库逐字节封存。
服务器解压新包后，使用configs/slotspe_clean_original_server.json，只训练original：fold0/seed3、3轮、256真实patch、batch8、20次同权重评估。
实现、完整34项合成检查及加固后4项关键复验已完成，重复覆盖不相加；服务器实际执行与原始回传仍待验收。
原两臂和A/B/C独立包继续保留；A/B/C代码及合成验收已完成，患者归因等待原版复现结果。

"""+text[b:]
a,b = text.index("三、SlotSPE下一步："),text.index("四、运输插拔下一步：")
text = text[:a]+"""三、SlotSPE下一步：原版问题复现优先

证据链：问题存在→影响什么→哪项修正有效→修正后还缺什么能力。
服务器AI先执行原版单臂短程任务，交接说明见docs/SlotSPE证明任务_服务器一键运行.txt。
回传配置/manifest、源码及checkpoint哈希、日志、曲线、逐患者预测、主/辅助梯度、患者槽干预、自然门控及受控检查原始JSON。
每项允许observed/not_observed；推送成功不表示服务器已运行，重复评估不算独立训练。
原模型和原生损失保持不变；固定真实patch及训练折分箱是诊断协议，不能当原论文完整基准配方。
已有A/B/C代码与合成检查用于后续：A原self汇总，B选中汇总且断开主选择梯度，C同前向接通；辅助梯度一致。
当前不自动扩展五折、多种子或30轮，不预先指定运输或新模块。

"""+text[b:]
text = text.replace("优先1：补齐SlotSPE A/B/C归因包及不混淆的比较协议。", "优先1：服务器先跑干净原版SlotSPE单臂问题检查，验收原始产物后进入已有修正归因。")
human.write_text(text, encoding="utf-8")

scope = frozen+extras+["README.md", "AGENTS.md", human.relative_to(WORK).as_posix()]
assert len(scope) == len(set(scope))
subprocess.run(["git", "add", "--", *scope], cwd=WORK, check=True)
changed = subprocess.check_output(["git", "diff", "--cached", "--name-only", "-z"], cwd=WORK).decode("utf-8").split("\0")
changed = [x for x in changed if x]
assert set(changed) == set(scope), (changed, scope)
plan = {"remote_base":BASE, "worktree":str(WORK), "scope":scope, "frozen_archive_sha256":expected_zip,
    "primary_head_before":main_head, "primary_index_semantic_sha256":index_sha,
    "frozen_source_files":frozen, "scope_sha256":{name:sha(WORK/name) for name in scope},
    "server_execution":"no_accessible_server_AI_channel_identified", "patient_jobs_run_by_this_window":False}
(AUDIT/"publication_plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"scoped_paths":len(scope),"base":BASE,"worktree":str(WORK),"archive_sha256":sha(WORK/ZIP.relative_to(ROOT))},ensure_ascii=False))
