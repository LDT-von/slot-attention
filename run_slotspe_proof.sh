#!/usr/bin/env bash
set -euo pipefail
PROOF_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROOF_REPO_ROOT"
exec "${PYTHON:-python}" -B -u scripts/run_slotspe_proof.py "$@"
