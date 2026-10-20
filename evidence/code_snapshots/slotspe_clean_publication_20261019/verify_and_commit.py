import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

AUDIT = Path(__file__).parent
plan = json.loads((AUDIT/"publication_plan.json").read_text(encoding="utf-8"))
WORK, ROOT = Path(plan["worktree"]), Path(r"E:\DCT-Reg")
def git(*args):
    return subprocess.check_output(["git", *args], cwd=WORK)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
xml = AUDIT/"publication_tests.xml"
suite = ET.parse(xml).getroot().find("testsuite")
assert suite.attrib["tests"] == "2" and all(int(suite.attrib.get(k, 0)) == 0 for k in ("failures", "errors", "skipped"))
dest = WORK/"docs/research/slotspe_clean_original_20261019"
(dest/"publication_tests.xml").write_bytes(xml.read_bytes())
checks = {"status":"validated_ready_for_normal_push", "remote_base":plan["remote_base"], "scoped_delivery":True,
    "isolated_native_tests_passed":2, "counts_additive_with_previous_suites":False, "frozen_archive_sha256":plan["frozen_archive_sha256"],
    "server_execution_verified":False, "patient_jobs_run_by_this_window":False,
    "validation":"Native-only standalone package, safe default, native classes, raw-record acceptance and tamper rejection"}
(dest/"publication_checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
added = [(dest/name).relative_to(WORK).as_posix() for name in ("publication_tests.xml", "publication_checks.json")]
subprocess.run(["git", "add", "--", *added], cwd=WORK, check=True)
scope = plan["scope"]+added

def parse_tree(data, index=False):
    result = {}
    for row in data.split(b"\0"):
        if not row:
            continue
        metadata, path = row.split(b"\t", 1)
        values = metadata.decode().split()
        result[path.decode("utf-8")] = (values[0], values[1] if index else values[2])
    return result
base = parse_tree(git("ls-tree", "-rz", plan["remote_base"]))
index = parse_tree(git("ls-files", "--stage", "-z"), True)
assert set(base).issubset(index)
assert all(index[p] == value for p, value in base.items() if p not in scope)
changed = {p for p in index if p not in base or index[p] != base[p]}
assert changed == set(scope)
archive = WORK/"docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip"
assert sha(archive) == plan["frozen_archive_sha256"]
with zipfile.ZipFile(archive) as z:
    name = next(n for n in z.namelist() if n.endswith("/DELIVERY_MANIFEST.json"))
    m = json.loads(z.read(name))
    for relative in plan["frozen_source_files"]:
        expected = z.read(m["prefix"]+relative).replace(b"\r\n", b"\n")
        actual = git("show", ":"+relative)
        assert actual == expected, relative
proof = {"base_tracked_paths":len(base), "all_base_paths_preserved":True,
    "unrelated_base_blobs_identical":sum(index[p] == v for p, v in base.items()), "scoped_changed_paths":len(scope),
    "new_paths":len(set(index)-set(base)), "source_matches_frozen_delivery":True, "scope":scope}
(AUDIT/"tree_verification.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
subprocess.run(["git", "diff", "--cached", "--check"], cwd=WORK, check=True)
subprocess.run(["git", "commit", "-m", "feat: deliver clean original SlotSPE server audit"], cwd=WORK, check=True)
commit = git("rev-parse", "HEAD").decode().strip()
assert not git("status", "--porcelain").strip()
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip() == plan["primary_head_before"]
assert hashlib.sha256(subprocess.check_output(["git", "ls-files", "-s", "-z"], cwd=ROOT)).hexdigest() == plan["primary_index_semantic_sha256"]
(AUDIT/"commit.json").write_text(json.dumps({"commit":commit,"tree_proof":proof},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"commit":commit,"scoped_paths":len(scope),"base_paths_preserved":len(base),"new_paths":proof["new_paths"],"primary_head_and_index_preserved":True}))
