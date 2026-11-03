"""Run ONLY standard-library contracts + syntax compilation, never model code.

Run from the repository root: python -B research_models/common_private_synergy/checks/run_static.py
This does not establish tensor shapes, gradients, numerical stability or efficacy.
"""
import ast
import json
from pathlib import Path
import sys
import unittest


def main():
    folder = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(folder.parents[1]))
    parsed = []
    for path in sorted(list((folder/'components').glob('*.py')) + list((folder/'deferred_checks').glob('*.py'))):
        source = path.read_text(encoding='utf-8')
        ast.parse(source, filename=str(path))
        compile(source, str(path), 'exec')  # compile only, never exec/import
        parsed.append(path.relative_to(folder).as_posix())
    suite = unittest.defaultTestLoader.discover(str(folder/'checks'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if 'torch' in sys.modules:
        raise RuntimeError('static check unexpectedly imported torch')
    print(json.dumps({'compiled_only': parsed, 'pure_python_tests_run': result.testsRun,
                      'ok': result.wasSuccessful(), 'torch_imported': False,
                      'model_forward_backward': 0, 'optimizer_steps': 0}))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
