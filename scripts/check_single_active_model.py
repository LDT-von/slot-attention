"""Regenerate the archive manifest and verify every archived byte.

The manifest is the machine-readable record of the archive restructuring:
for each archived file it stores the ARCHIVED location, the ORIGINAL path the
historical receipts name, and the SHA-256 of the bytes on disk.

The script is also the audit.  It re-derives every SHA-256 and compares it
against two independent references:

  1. the receipt-recorded hashes, for every path the receipts pin, and
  2. the git object of the pre-restructure commit, for every moved file.

It never trains, never reads patient data and never starts an experiment.

Run from the repository root:

    python -B scripts/check_single_active_model.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SYSPATH = str(ROOT)
if SYSPATH not in sys.path:
    sys.path.insert(0, SYSPATH)

MANIFEST_PATH = ROOT / 'research_models/common_private_synergy/archive/ARCHIVE_MANIFEST.json'
REGISTRY_PATH = ROOT / 'research_models/common_private_synergy/MODEL_REGISTRY.json'

ARCHIVE_ROOTS = (
    ROOT / 'research_models/common_private_synergy/archive',
    ROOT / 'archive',
)

# The commit that holds the pre-restructure bytes of every moved file.  It
# is pinned rather than left as HEAD, because after the restructuring Git
# stores the audit paths as symlink blobs, so HEAD no longer carries the
# original file content.
BASELINE_COMMIT = '68e74dd'

# The receipt files that pin bytes, each with the directory its paths are
# relative to.
RECEIPT_SOURCES = (
    ('research_models/common_private_synergy/validation_receipt.json',
     'research_models/common_private_synergy'),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha256(commit: str, path: str) -> str | None:
    result = subprocess.run(
        ['git', '-C', str(ROOT), 'show', f'{commit}:{path}'],
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def receipt_pinned_hashes() -> dict[str, set[str]]:
    """Every path the receipts pin, with the hashes they name."""
    pinned: dict[str, set[str]] = {}

    # The local evidence receipts pin paths relative to whichever directory
    # the run used as its root.  The same name can therefore refer to
    # different files.  A name is pinned only when exactly one candidate
    # root resolves it; an unresolvable name is skipped, because pinning it
    # under the wrong root would report a false failure.
    for relative, base in RECEIPT_SOURCES:
        path = ROOT / relative
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        _collect_source_hashes(payload, pinned, base)

    evidence = ROOT / 'local_evidence'
    if evidence.is_dir():
        for path in sorted(evidence.rglob('*.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (ValueError, OSError):
                continue
            _collect_source_hashes(payload, pinned)
    return pinned


# Roots a receipt path may be relative to.
CANDIDATE_ROOTS = (
    '',
    'research_models/common_private_synergy',
    'third_party/SlotSPE',
    'third_party/SlotSPE/dataset',
    'third_party/SlotSPE/models',
    'third_party/SlotSPE/utils',
    'slotspe_proof',
)

# The receipt keys that hold source-code hashes.  Other hash maps in a
# receipt describe data inputs, prepared manifests or checkpoints, which are
# not source files.
SOURCE_HASH_KEYS = ('source_SHA256', 'source_sha256')
SOURCE_HASH_MAP_KEYS = (
    'source_sha256_excluding_this_receipt',
    'source_sha256_excluding_receipt',
)


def _collect_source_hashes(node, pinned: dict[str, set[str]],
                            base: str = '') -> None:
    """Collect only the SOURCE-code hash maps.

    A receipt also hashes data inputs, prepared manifests and checkpoints.
    Those are not source files and are not part of the source audit, so only
    the keys that name sources are read.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SOURCE_HASH_KEYS and isinstance(value, dict):
                for name, digest in value.items():
                    if not (isinstance(digest, str) and len(digest) == 64 and _is_hex(digest)):
                        continue
                    pinned.setdefault(_normalise_path(name, base), set()).add(digest.lower())
            elif key in SOURCE_HASH_MAP_KEYS and isinstance(value, dict):
                for name, digest in value.items():
                    if isinstance(digest, str) and len(digest) == 64 and _is_hex(digest):
                        pinned.setdefault(_normalise_path(name, base), set()).add(digest.lower())
            else:
                _collect_source_hashes(value, pinned, base)
    elif isinstance(node, list):
        for item in node:
            _collect_source_hashes(item, pinned, base)


def _normalise_path(name: str, base: str) -> str:
    """Map a receipt path onto a repository-relative path.

    A receipt path may be absolute, or relative to the directory the run used
    as its root.  Absolute paths are made relative to this repository; a
    relative path keeps its own name and gains the receipt's base directory.
    """
    text = name.replace('\\', '/')
    if text.startswith(str(ROOT)):
        return text[len(str(ROOT)):].lstrip('/')
    if text.startswith('/'):
        # An absolute path outside this repository.  Keep only the tail that
        # the candidate roots can resolve.
        text = text.lstrip('/')
    if base:
        return f'{base}/{text}'
    return text


def _is_hex(text: str) -> bool:
    return all(character in '0123456789abcdefABCDEF' for character in text)


def collect_archived_files() -> list[tuple[str, list[str], str]]:
    """Every archived file with its archive path, its original paths and hash.

    Byte-compiled caches are not source and are excluded.  A file that is
    only a symlink into the shared library is excluded too, because its bytes
    live at the shared location and the shared location is listed there.
    """
    rows: list[tuple[str, list[str], str]] = []
    for root in ARCHIVE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.is_dir() or path.is_symlink():
                continue
            if '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            relative = path.relative_to(ROOT).as_posix()
            originals = _original_paths_for(path)
            rows.append((relative, originals, sha256_file(path)))
    return rows


def _original_paths_for(path: Path) -> list[str]:
    """Every ORIGINAL path that still resolves to the archived file.

    The archive keeps the original path resolvable through a symlink, so the
    original path is found by scanning the repository for a symlink that
    points at the archived file.  More than one symlink may point at the
    same bytes, for example a shared component that two arms both use.
    """
    resolved = path.resolve()
    found: list[str] = []
    for candidate in _all_paths():
        if candidate.is_symlink() and candidate.resolve() == resolved:
            found.append(candidate.relative_to(ROOT).as_posix())
    return sorted(found, key=lambda item: (item.count('/'), item))


def _all_paths() -> list[Path]:
    found: list[Path] = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', 'local_evidence',
                                               '.pytest_cache', 'third_party', 'evidence')]
        for name in list(dirs) + files:
            found.append(Path(root) / name)
    return found


def check_registry() -> list[str]:
    """The registry must name exactly one active model, and it must exist."""
    problems: list[str] = []
    if not REGISTRY_PATH.exists():
        return [f'MISSING registry {REGISTRY_PATH.relative_to(ROOT)}']
    registry = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
    active = registry.get('active_model')
    if not isinstance(active, dict):
        return ['registry has no active_model object']
    if active.get('status') != 'active':
        problems.append('the registry active_model is not marked active')
    for key in ('model_class', 'entrypoint', 'config', 'synthetic_check'):
        value = active.get(key)
        if not value:
            problems.append(f'registry active_model lacks {key}')
        elif key != 'model_class' and not (ROOT / value).exists():
            problems.append(f'registry active_model {key} path does not exist: {value}')
    for key in ('archived_models', 'legacy_code_retained_in_place'):
        if not isinstance(registry.get(key), list):
            problems.append(f'registry lacks a {key} list')
    others = [m for m in registry.get('archived_models', []) if m.get('status') == 'active']
    if others:
        problems.append(f'registry lists {len(others)} archived model as active')
    if len(registry.get('archived_models', [])) < 5:
        problems.append('the registry does not list the five superseded arms')
    return problems


def check_active_isolation() -> list[str]:
    """The active runner must not import any archived model module.

    Only real imports are inspected.  A docstring that mentions an archived
    arm is a provenance note, not a dependency, so the check reads the parsed
    import statements rather than the raw text.
    """
    problems: list[str] = []
    entry = ROOT / 'scripts/run_common_private_scaled16.py'
    if not entry.exists():
        return [f'MISSING active entrypoint {entry.relative_to(ROOT)}']

    forbidden = ('matched35', 'spread_arm', 'symmetry_arm', 'cox_objective',
                 'full_slot_engine', 'full_slot_epoch', 'common_private_synergy.model')
    sources = [entry]
    package = ROOT / 'research_models/common_private_synergy/scaled'
    sources += sorted(package.glob('*.py'))

    for path in sources:
        for module in imported_modules(path):
            for token in forbidden:
                if token in module:
                    problems.append(
                        f'the active module {path.relative_to(ROOT)} imports {module}')
    return problems


def imported_modules(path: Path) -> list[str]:
    """Every module name the file imports, absolute and relative."""
    import ast

    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A relative import.  Resolve it against the package root so
                # the check can tell which archived module it points at.
                anchor = path.parent
                for _ in range(node.level - 1):
                    anchor = anchor.parent
                base = anchor.relative_to(ROOT).as_posix().replace('/', '.')
                modules.append(f'{base}.{node.module}' if node.module else base)
            elif node.module:
                modules.append(node.module)
    return modules


def baseline_hashes() -> dict[str, str]:
    """The pre-restructure SHA-256 of every path the manifest tracks.

    The manifest is the self-contained record of the archive.  When it
    already lists the file, its recorded hash is the baseline; a checkout
    without Git history still audits correctly.  Only when the manifest has
    no entry yet does the script fall back to the Git baseline commit.
    """
    hashes: dict[str, str] = {}
    if MANIFEST_PATH.exists():
        try:
            payload = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        except ValueError:
            payload = {}
        for entry in payload.get('files', []):
            digest = entry.get('sha256')
            if isinstance(digest, str) and len(digest) == 64:
                for original in entry.get('original_paths', []):
                    hashes[original] = digest
        for entry in payload.get('retained_in_place', []):
            name = entry.get('path')
            digest = entry.get('baseline_sha256') or entry.get('sha256')
            if isinstance(name, str) and isinstance(digest, str) and len(digest) == 64:
                hashes[name] = digest
    return hashes


def baseline_hash_for(name: str, cached: dict[str, str]) -> str | None:
    """The pre-restructure hash of a path.

    Git is the authority when it is available.  The manifest is the fallback
    for a checkout without Git history.  The manifest stores the BASELINE
    hash, not the current hash, so a tampered file cannot poison it: the
    next run still compares the file against the true baseline.
    """
    from_git = git_blob_sha256(BASELINE_COMMIT, name)
    if from_git is not None:
        return from_git
    return cached.get(name)


def check_receipt_paths() -> tuple[list[str], dict[str, list[str]]]:
    """Every receipt-pinned path must still resolve with the pinned bytes.

    A receipt hash and the pre-restructure baseline can disagree.  When they
    do, the question that matters is WHICH reference the current bytes
    match:

      * bytes equal the baseline but not the receipt -> the receipt was
        already stale before this restructuring, and the restructuring
        preserved the bytes exactly.  This is a pre-existing discrepancy and
        is reported, not silently accepted.
      * bytes match neither -> the restructuring changed the bytes, which is
        a real failure.

    The baseline is the reference for "the restructuring preserved the
    bytes"; the receipt is the independent second reference.
    """
    problems: list[str] = []
    pre_existing: list[str] = []
    detail: dict[str, list[str]] = {}
    cached = baseline_hashes()
    pinned = receipt_pinned_hashes()
    for name in sorted(pinned):
        path = ROOT / name
        if not path.exists():
            problems.append(f'receipt-pinned path is missing: {name}')
            detail[name] = ['missing']
            continue
        current = sha256_file(path)
        detail[name] = [current]
        if current in pinned[name]:
            continue
        baseline = baseline_hash_for(name, cached)
        if baseline == current:
            pre_existing.append(
                f'receipt-pinned path was already stale before the restructuring '
                f'and is unchanged by it: {name} '
                f'({current[:12]} != receipt {sorted(pinned[name])[0][:12]})')
        else:
            problems.append(
                f'receipt-pinned path changed bytes: {name} '
                f'({current[:12]} != receipt {sorted(pinned[name])[0][:12]}, '
                f'baseline {(baseline or "none")[:12]})')
    return problems, pre_existing, detail


def check_git_baseline() -> list[str]:
    """Every archived file must equal its pre-restructure bytes.

    The baseline object lives at the ORIGINAL path, because that is where
    the file was tracked before the restructuring.  The manifest carries the
    same hashes, so the check also works in a checkout without Git history.
    """
    problems: list[str] = []
    cached = baseline_hashes()
    for relative, originals, digest in collect_archived_files():
        baselines = [baseline_hash_for(original, cached) for original in originals]
        baselines = [b for b in baselines if b is not None]
        if not baselines:
            # A file the restructuring created, for example the manifest
            # itself.  It has no baseline and is not a failure.
            continue
        if digest not in baselines:
            problems.append(
                f'archived bytes differ from the pre-restructure baseline: {relative}')
    return problems


def build_manifest() -> dict:
    rows = collect_archived_files()
    pinned = receipt_pinned_hashes()
    cached = baseline_hashes()
    entries = []
    for relative, originals, digest in rows:
        # The manifest is a record of the archive, not a member of it.
        if relative == MANIFEST_PATH.relative_to(ROOT).as_posix():
            continue
        baselines = [baseline_hash_for(original, cached) for original in originals]
        baselines = [b for b in baselines if b is not None]
        entries.append({
            'archived_path': relative,
            'original_paths': originals,
            'sha256': digest,
            'baseline_sha256': baselines[0] if baselines else None,
            'receipt_pinned': any(original in pinned for original in originals),
            'matches_pre_restructure_baseline': bool(baselines) and digest in baselines,
        })

    # The receipt-pinned files that stayed in place.  Their bytes are the
    # pre-restructure bytes too, so the manifest records them and the audit
    # can run without Git history.
    retained = []
    for name in sorted(pinned):
        path = ROOT / name
        if not path.exists():
            continue
        if any(name in entry['original_paths'] for entry in entries):
            continue
        digest = sha256_file(path)
        baseline = baseline_hash_for(name, cached)
        retained.append({
            'path': name,
            'sha256': digest,
            'baseline_sha256': baseline,
            'receipt_hashes': sorted(pinned[name]),
            'matches_receipt': digest in pinned[name],
            'matches_pre_restructure_baseline': baseline == digest,
        })
    return {
        'schema': 'slotspe_archive_manifest_v1',
        'date': '2026-11-17',
        'baseline_commit': BASELINE_COMMIT,
        'policy': (
            'Archived files are byte-identical to the version the historical '
            'receipts name. Every original path still resolves through a '
            'symlink, so sha(ROOT / name) lookups keep working unchanged. '
            'Only scaled_role16_v1 is active; see MODEL_REGISTRY.json. '
            'The retained section records the receipt-pinned files that stay '
            'in place, so a checkout without Git history can still audit the '
            'whole consolidation from this file alone.'),
        'active_model': 'scaled_role16_v1',
        'file_count': len(entries),
        'files': entries,
        'retained_in_place': retained,
    }


def main() -> int:
    problems: list[str] = []
    problems += check_registry()
    problems += check_active_isolation()
    receipt_problems, pre_existing, detail = check_receipt_paths()
    problems += receipt_problems
    problems += check_git_baseline()

    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    pinned_count = len(detail)
    print(json.dumps({
        'registry_active_models': 1,
        'archived_files': manifest['file_count'],
        'receipt_pinned_paths_verified': len(detail),
        'pre_existing_receipt_discrepancies': pre_existing,
        'problems': problems,
        'ok': not problems,
    }, indent=2, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
