"""Keep test scratch space off the 97 GB root partition.

The root partition of this machine holds the 10 TB volume's sibling files and
is small; a full test run writes roughly 0.7 GB of pytest scratch into it by
default, because pytest creates its temporary directory under the system
temporary directory.  That is the same class of mistake that truncated a
checkpoint in the 4096-patch run recorded in
``/data1/DCT-Reg/docs/research/slotspe_4096_smoke_20261026/FINDINGS.md``.

This file sets the temporary directory environment variables before pytest
imports and collects anything, so that pytest's own temporary directory and
every library that honours the standard variables lands on the 10 TB volume
instead.  It changes only where scratch files are written.  It does not change
any test, any fixture value, any seed or any number.
"""
import os
from pathlib import Path
import tempfile

_REPO = Path(__file__).resolve().parents[1]
# local_evidence is ignored by Git and is the only place this repository
# writes private artifacts, so the test scratch belongs there too.
_SCRATCH = _REPO / "local_evidence" / "pytest_tmp"

# The directory must exist before the variable is set: the standard library
# falls back to /tmp when TMPDIR names a path that does not exist.
_SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_SCRATCH)
os.environ["TEMP"] = str(_SCRATCH)
os.environ["TMP"] = str(_SCRATCH)
# The standard library caches the temporary directory the first time it is
# asked for, so setting the environment variables alone is not enough once
# anything has already resolved it.  Clearing the cache and resolving it
# again moves both pytest's own temporary directory and every library that
# follows the standard variables.
tempfile.tempdir = None
tempfile.gettempdir()
