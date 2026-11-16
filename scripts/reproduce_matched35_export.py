"""Reproduce the exact historical export from pinned JSON inputs, then audit it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_matched35_chain import EXPECTED_SHA256, verify_chain


def reproduce(run_root, original_exporter, output, audit_output):
    root, exporter, output, audit = map(lambda p: Path(p).resolve(),
                                       (run_root, original_exporter, output, audit_output))
    if output == audit or output.exists() or audit.exists() or root in output.parents or root in audit.parents:
        raise ValueError('fresh output and audit outside original run required')
    inputs = {name: root/name for name in EXPECTED_SHA256 if name.endswith('receipt.json')}
    inputs['original_exporter'] = exporter
    for name, path in inputs.items():
        if path in (output, audit) or hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256[name]:
            raise ValueError('pinned reproduction input differs')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-S', '-B', str(exporter), '--run-root', str(root), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('original exporter failed; preserve the fresh output directory')
    verified = verify_chain(root, exporter, output)
    verified['exporter_reexecuted'] = True
    verified['exporter_exit_code'] = result.returncode
    verified['reproduction_command'] = command
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(verified, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--original-exporter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.run_root, args.original_exporter, args.output, args.audit_output)
    print(json.dumps({'status': result['status'], 'exporter_reexecuted': True,
                      'total_epochs': 82, 'model_executed': False}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        raise SystemExit('Historical export reproduction failed ('+type(error).__name__+'); inputs unchanged.') from None
