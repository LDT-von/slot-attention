"""Public source-inventory checks; no patient inputs or model execution."""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "delivery_manifest", Path(__file__).resolve().parents[1] / "scripts/check_delivery_manifest.py")
DELIVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DELIVERY)


@pytest.fixture
def release(tmp_path):
    (tmp_path / "README.md").write_bytes(b"readme\n")
    (tmp_path / "module.py").write_bytes(b"VALUE = 1\n")
    paths = ["README.md", "module.py", DELIVERY.MANIFEST]
    manifest = DELIVERY.make_manifest(tmp_path, paths, "fixture", "fixed")
    return tmp_path, paths, manifest


def test_public_source_inventory_roundtrip(release):
    root, paths, manifest = release
    assert DELIVERY.validate_manifest(root, paths, manifest) == []
    assert manifest["covered_file_count"] == 2


def test_same_length_content_change_is_detected(release):
    root, paths, manifest = release
    (root / "module.py").write_bytes(b"VALUE = 2\n")
    assert "Content changed: module.py" in DELIVERY.validate_manifest(root, paths, manifest)


def test_missing_file_is_detected(release):
    root, paths, manifest = release
    (root / "module.py").unlink()
    assert "Missing: module.py" in DELIVERY.validate_manifest(root, paths, manifest)


def test_added_file_cannot_be_silently_omitted(release):
    root, paths, manifest = release
    (root / "extra.py").write_bytes(b"VALUE = 3\n")
    assert "Not inventoried: extra.py" in DELIVERY.validate_manifest(root, paths + ["extra.py"], manifest)


def test_removed_tree_entry_is_detected(release):
    root, paths, manifest = release
    assert "Not in current public tree: module.py" in DELIVERY.validate_manifest(
        root, [p for p in paths if p != "module.py"], manifest)


@pytest.mark.parametrize("path", ["local_evidence/raw.json", "data/features.h5", "../escape.py", ".env", "checkpoint.pt", "arbitrary.zip"])
def test_private_or_escaping_path_is_rejected_before_reading(tmp_path, path):
    with pytest.raises(ValueError):
        DELIVERY.make_manifest(tmp_path, [path], "fixture")


def test_duplicate_inventory_entry_is_rejected(release):
    root, paths, manifest = release
    broken = copy.deepcopy(manifest)
    broken["files"].append(copy.deepcopy(broken["files"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        DELIVERY.validate_manifest(root, paths, broken)


def test_only_the_manifest_itself_may_be_excluded(release):
    root, paths, manifest = release
    broken = copy.deepcopy(manifest)
    broken["excluded_self"].append("module.py")
    with pytest.raises(ValueError):
        DELIVERY.validate_manifest(root, paths, broken)


def test_historical_archive_exception_is_pinned_to_original_bytes(tmp_path):
    name = next(iter(DELIVERY.HISTORICAL_SOURCE_ARCHIVES))
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"replacement content")
    with pytest.raises(ValueError, match="archive changed"):
        DELIVERY.make_manifest(tmp_path, [name], "fixture")
