#!/usr/bin/env python3
"""Recompute the anti-tamper path pins after an intentional code change.

The pins exist so that a queued run refuses to continue when an implementation
file changed underneath it. They must therefore be refreshed deliberately, and
this helper prints exactly what it changed so the refresh is reviewable.

It never invents a new pin entry and never touches evidence pins: only files
that are already pinned, whose current bytes differ, are rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/slotspe_redundancy_queue.json"
PIN_GROUPS = ("evidence_sha256", "protected_existing_queue_sha256", "candidate_implementation_sha256")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Report stale pins and exit 1 without writing anything.")
    args = parser.parse_args(argv)

    raw = CONFIG.read_text(encoding="utf-8-sig")
    config = json.loads(raw)
    stale = []
    for group in PIN_GROUPS:
        for name in config.get(group, {}):
            path = (ROOT / name).resolve()
            if not path.is_file():
                print(f"MISSING {group}: {name}")
                continue
            current = sha256(path)
            if current != config[group][name]:
                stale.append((name, config[group][name], current, group))

    if not stale:
        print("All pins already match the current bytes.")
        return 0
    print(f"{len(stale)} pinned file(s) differ from the configuration:\n")
    for name, old, new, group in stale:
        print(f"  [{group}] {name}\n      pinned {old}\n      current {new}")
    if args.check:
        print("\n--check only; nothing written.")
        return 1
    if group_is_evidence(stale):
        print("\nRefusing to rewrite an evidence pin. Evidence files are historical "
              "measurements and must never be re-pinned to new bytes.")
        return 2

    for name, _, new, _ in stale:
        pattern = re.compile(r'("%s"\s*:\s*)"[0-9a-f]{64}"' % re.escape(name))
        raw, count = pattern.subn(lambda m: m.group(1) + '"%s"' % new, raw, count=1)
        if count != 1:
            print(f"FAILED to rewrite the pin for {name}", file=sys.stderr)
            return 2
    CONFIG.write_text(raw, encoding="utf-8")
    print(f"\nRewrote {len(stale)} pin(s) in {CONFIG.relative_to(ROOT)}. Record this refresh in docs/PROGRESS.json.")
    return 0


def group_is_evidence(stale):
    return any(group == "evidence_sha256" for _, _, _, group in stale)


if __name__ == "__main__":
    raise SystemExit(main())