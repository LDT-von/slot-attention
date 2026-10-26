#!/usr/bin/env python3
"""Recompute the multi-fold queue's anti-tamper pins after an intentional change.

Mirrors scripts/refresh_slotspe_queue_pins.py for the single-fold queue, but
for configs/slotspe_redundancy_multifold_queue.json.  It never invents a new
pin entry and never touches an evidence pin: only files that are already
pinned, whose current bytes differ, are rewritten.  Original SlotSPE source
pins are refreshed too, so a change to third_party/SlotSPE cannot slip through
unnoticed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/slotspe_redundancy_multifold_queue.json"
PIN_GROUPS = ("candidate_implementation_sha256", "protected_original_source_sha256")


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

    raw = CONFIG.read_text(encoding="utf-8")
    config = json.loads(raw)
    stale = []
    for group in PIN_GROUPS:
        for name, expected in config.get(group, {}).items():
            path = (ROOT / name).resolve()
            if not path.is_file():
                print(f"MISSING {group}: {name}")
                continue
            current = sha256(path)
            if current != expected:
                stale.append((name, expected, current, group))

    if not stale:
        print("All pins already match the current bytes.")
        return 0
    print(f"{len(stale)} pinned file(s) differ from the configuration:\n")
    for name, old, new, group in stale:
        print(f"  [{group}] {name}\n      pinned {old}\n      current {new}")
    if args.check:
        print("\n--check only; nothing written.")
        return 1

    for name, _, new, _ in stale:
        pattern = re.compile(r'("%s"\s*:\s*)"[0-9a-f]{64}"' % re.escape(name))
        raw, count = pattern.subn(lambda m: m.group(1) + '"%s"' % new, raw, count=1)
        if count != 1:
            print(f"FAILED to rewrite the pin for {name}", file=__import__("sys").stderr)
            return 2
    CONFIG.write_text(raw, encoding="utf-8")
    print(f"\nRewrote {len(stale)} pin(s) in {CONFIG.relative_to(ROOT)}. Record this refresh in docs/PROGRESS.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
