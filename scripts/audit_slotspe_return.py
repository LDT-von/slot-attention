"""Validate an unpacked server return without its original server data paths."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.research.slotspe_proof.acceptance import audit_return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = audit_return(args.root, args.delivery)
        code = 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"status": "rejected", "error": f"{type(exc).__name__}: {exc}",
            "server_proof_accepted": False}
        code = 1
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
