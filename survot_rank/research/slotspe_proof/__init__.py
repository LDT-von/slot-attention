"""Backward-compatible shim for the historical ``survot_rank`` import path.

The research harness now lives in the top-level :mod:`slotspe_proof` package.
This module keeps ``survot_rank.research.slotspe_proof.<module>`` importable so
that every historical script, test and diagnostic continues to work unchanged.

The shim copies no code and, importantly, imports nothing eagerly. Some
harness modules pull in torch/pandas at import time, and this repository treats
"``--plan`` must not import torch, pandas or any patient array" as a hard
contract. Eagerly aliasing every submodule here would break it. Instead a meta
path finder resolves a legacy name on first use, imports the single canonical
module under its own name and binds *that same module object* to the legacy
name. There is therefore exactly one copy of every module in the process, and
an unused legacy import costs nothing.

New code should import from :mod:`slotspe_proof` directly.
"""
import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

_LEGACY_PREFIX = "survot_rank.research.slotspe_proof"
_CANONICAL_ROOT = "slotspe_proof"


def _canonical_name(legacy_name):
    """Return the canonical module name for a legacy dotted name."""
    suffix = legacy_name[len(_LEGACY_PREFIX) + 1:]
    return _CANONICAL_ROOT + "." + suffix


class _AliasLoader(Loader):
    """Load a canonical module under a legacy name, without re-executing it."""

    def __init__(self, canonical_name):
        self._canonical_name = canonical_name

    def create_module(self, spec):
        # Returning the already imported canonical module makes the legacy name
        # an alias of it, so no code is executed a second time.
        return importlib.import_module(self._canonical_name)

    def exec_module(self, module):
        # The canonical module was executed under its own name already.
        return None


class _AliasSpec(ModuleSpec):
    """Module spec whose parent package matches the canonical module.

    A plain :class:`ModuleSpec` built for the legacy name would make
    ``__spec__.parent`` disagree with the canonical module's ``__package__``,
    and CPython emits a DeprecationWarning on every relative import. Reporting
    the canonical parent keeps relative imports silent while the module object
    itself still resolves to the single canonical implementation.

    The parent is derived from the name text only. Looking the canonical module
    up in ``sys.modules`` instead would recurse: importing it is what triggers
    this spec in the first place.
    """

    def __init__(self, legacy_name, canonical_name, loader):
        super().__init__(legacy_name, loader)
        self._canonical_parent = canonical_name.rpartition(".")[0]

    @property
    def parent(self):
        return self._canonical_parent


class _AliasFinder(MetaPathFinder):
    """Resolve ``survot_rank.research.slotspe_proof.*`` onto ``slotspe_proof.*``."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_LEGACY_PREFIX + "."):
            return None
        canonical = _canonical_name(fullname)
        if canonical == fullname:  # defensive: never alias onto itself
            return None
        return _AliasSpec(fullname, canonical, _AliasLoader(canonical))


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())


def __getattr__(name):
    """Expose canonical submodules as attributes of the legacy package."""
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        module = importlib.import_module(_canonical_name(name))
    except ImportError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} ({exc})"
        ) from exc
    globals()[name] = module
    return module


def __dir__():
    canonical = importlib.import_module(_CANONICAL_ROOT)
    return sorted(set(globals()) | set(dir(canonical)))