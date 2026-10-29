"""Plan or analyze a private, explicitly bound ROI case; no model forward."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--synthetic-demo", action="store_true")
    parser.add_argument("--run-cohort", action="store_true")
    parser.add_argument("--cohort-manifest", type=Path)
    args = parser.parse_args(argv)
    if sum((args.plan, args.run, args.synthetic_demo, args.run_cohort)) > 1:
        parser.error("Choose one of --plan, --run, --run-cohort or --synthetic-demo")
    if args.run_cohort:
        if args.output is None or args.cohort_manifest is None or args.case is not None or args.trace is not None:
            parser.error("--run-cohort requires --cohort-manifest and --output")
        from slotspe_proof.roi_workflow import analyze_cohort
        try:
            result = analyze_cohort(args.cohort_manifest, args.output)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    if args.synthetic_demo:
        if args.output is None or args.case is not None or args.trace is not None or args.cohort_manifest is not None:
            parser.error("--synthetic-demo requires only --output")
        from slotspe_proof.roi_demo import run_demo
        try:
            result = run_demo(args.output)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    if not args.run:
        print(json.dumps({"action": "plan_only", "requires": ["case JSON", "bound attention NPZ", "new private output"],
                          "reads_patient_files": False, "model_forwards": 0, "training": 0}))
        return 0
    if args.case is None or args.trace is None or args.output is None or args.cohort_manifest is not None:
        parser.error("--run requires --case, --trace and --output")
    from slotspe_proof.roi_workflow import analyze_case
    try:
        result = analyze_case(args.case, args.trace, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
