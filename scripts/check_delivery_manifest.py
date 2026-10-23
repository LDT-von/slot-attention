"""Maintain the current public-file inventory without reading private evidence.

The older PUBLICATION_MANIFEST.json remains a historical migration receipt.
This manifest excludes its own bytes to avoid a recursive self-hash; Git's
commit/tree identity covers that one file during delivery verification.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


REPO = Path(__file__).resolve().parents[1]
MANIFEST = "DELIVERY_MANIFEST.json"
SCHEMA = "slotspe_current_delivery_v1"
PRIVATE_ROOTS = {".git", ".codex", ".aws", ".agents", ".codegraph", ".cursor",
                 ".review", ".validation", "local_evidence", "tmp", "results",
                 "data", "prepared"}
PRIVATE_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".h5", ".hdf5", ".pkl",
                    ".pickle", ".ckpt", ".safetensors", ".zip"}
# These existing immutable archives contain source/documents only. Their member
# inventories were checked during delivery; no arbitrary archive is permitted.
HISTORICAL_SOURCE_ARCHIVES = {
    "docs/deliveries/slotspe_clean_original_audit_20261019_v1.zip": "a690578675d51c1328c92cac8d7dc41dc6494fbd179e53476763df06621007a4",
    "docs/deliveries/slotspe_original_questions_v1_20261021.zip": "ec8c663355d44225931397d2954e1dc80bbcea9ffbc3aef58d67f027cbcb533c",
    "docs/deliveries/slotspe_proof_abc_20261019_v2.zip": "18baac2c686307f0b89684f85768f60b75415830b009fde857ea1a55c75f89ec",
    "docs/deliveries/slotspe_server_proof_20261019.zip": "107e16baecddd5a318fa56a790d258b1b2c6de2868f34546e2281b12a7adf8c1",
}


def safe_paths(root, paths):
    """Reject private, ambiguous or escaping paths before any file is read."""
    root = Path(root).resolve()
    seen = set()
    result = []
    for text in sorted(paths):
        path = PurePosixPath(text)
        folded = text.casefold()
        if (not text or path.is_absolute() or path.as_posix() != text
                or ".." in path.parts or "\\" in text or ":" in text
                or path.parts[0].casefold() in PRIVATE_ROOTS
                or (path.suffix.casefold() in PRIVATE_SUFFIXES
                    and text not in HISTORICAL_SOURCE_ARCHIVES)
                or path.name == ".env" or path.name.startswith(".env.")
                or path.name == "LOCAL_MIGRATION_MANIFEST.json"):
            raise ValueError("Not a public source/document path: " + text)
        actual = root / text
        if not actual.resolve().is_relative_to(root) or actual.is_symlink():
            raise ValueError("Escaping or symbolic-link path: " + text)
        if text in HISTORICAL_SOURCE_ARCHIVES and actual.is_file():
            if hashlib.sha256(actual.read_bytes()).hexdigest() != HISTORICAL_SOURCE_ARCHIVES[text]:
                raise ValueError("Historical source archive changed: " + text)
        if folded in seen:
            raise ValueError("Duplicate or case-ambiguous path: " + text)
        seen.add(folded)
        result.append(text)
    return result


def public_paths(root, require_tracked=False):
    def names(*args):
        raw = subprocess.check_output(["git", *args], cwd=root)
        return [p for p in raw.decode("utf-8").split("\0") if p]
    tracked = names("ls-files", "-z")
    untracked = names("ls-files", "--others", "--exclude-standard", "-z")
    if require_tracked and untracked:
        raise ValueError("Unpublished public files remain: " + ", ".join(untracked))
    return safe_paths(root, sorted(set(tracked + untracked + [MANIFEST])))


def make_manifest(root, paths, base_commit, recorded_at=None):
    root = Path(root)
    paths = safe_paths(root, paths)
    files = []
    for name in paths:
        if name == MANIFEST:
            continue
        data = (root / name).read_bytes()
        files.append({"path": name, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    return {
        "schema": SCHEMA,
        "recorded_at": recorded_at or datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "base_commit_before_delivery": base_commit,
        "canonical_implementation": "slotspe_proof/",
        "legacy_import_shim": "survot_rank/research/slotspe_proof/__init__.py",
        "historical_manifest": "PUBLICATION_MANIFEST.json",
        "coverage": "All non-ignored Git tracked and untracked public files; require-tracked checks enforce publication completeness.",
        "excluded_self": [MANIFEST],
        "self_exclusion_reason": "A file cannot contain its own SHA256. The verified Git commit/tree covers the manifest itself.",
        "covered_file_count": len(files),
        "repository_file_count_including_manifest": len(files) + 1,
        "patient_data_included": False,
        "files": files,
    }


def validate_manifest(root, paths, manifest):
    root = Path(root)
    expected = set(safe_paths(root, paths)) - {MANIFEST}
    if manifest.get("schema") != SCHEMA or manifest.get("excluded_self") != [MANIFEST]:
        raise ValueError("Unsupported schema or incomplete self-exclusion contract")
    rows = manifest.get("files", [])
    names = safe_paths(root, [row["path"] for row in rows])
    actual = set(names)
    errors = []
    if MANIFEST in actual:
        errors.append("Manifest must exclude its own bytes")
    for path in sorted(expected - actual):
        errors.append("Not inventoried: " + path)
    for path in sorted(actual - expected):
        errors.append("Not in current public tree: " + path)
    if manifest.get("covered_file_count") != len(rows):
        errors.append("Incorrect covered_file_count")
    if manifest.get("repository_file_count_including_manifest") != len(rows) + 1:
        errors.append("Incorrect repository_file_count_including_manifest")
    for row in rows:
        path = root / row["path"]
        if not path.is_file():
            errors.append("Missing: " + row["path"])
            continue
        data = path.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            errors.append("Content changed: " + row["path"])
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--refresh", action="store_true", help="Rewrite the current manifest only")
    action.add_argument("--check", action="store_true", help="Read-only validation (default)")
    parser.add_argument("--require-tracked", action="store_true", help="Fail if any public file is not in Git")
    args = parser.parse_args(argv)
    try:
        paths = public_paths(REPO, args.require_tracked)
        target = REPO / MANIFEST
        if args.refresh:
            base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
            data = make_manifest(REPO, paths, base)
            target.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        else:
            data = json.loads(target.read_text(encoding="utf-8"))
        errors = validate_manifest(REPO, paths, data)
        print(json.dumps({"ok": not errors, "covered_files": data["covered_file_count"],
                          "self_excluded": MANIFEST, "findings": errors}, ensure_ascii=True))
        return 1 if errors else 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
