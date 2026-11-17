"""Pure-Python view *plans*, without data I/O, tensors or model execution.

Indices refer to the caller's one patient/feature ordering; plans contain no
patient identifiers. Reusing a seed replays the same plan. The caller must use
an explicit, audited per-patient/access seed schedule for fresh training draws.
Patch subsampling is not image augmentation. Noise is only specified here,
never sampled or applied to a feature value.
"""
from dataclasses import dataclass
import hashlib
import math
import random
from typing import Optional, Tuple


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _finite(name, value, minimum=0.0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} outside allowed range')


def _digest(name, value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')


def _child_seed(seed, label):
    _integer('seed', seed)
    return int.from_bytes(hashlib.sha256(f'view-plan-v1:{seed}:{label}'.encode('ascii')).digest()[:8], 'big')


def _mode(mode):
    if mode not in ('train', 'eval'):
        raise ValueError("mode must be 'train' or 'eval'")


@dataclass(frozen=True)
class PatchIndexView:
    indices: Tuple[Optional[int], ...]
    seed: int

    @property
    def real_mask(self):
        """Metadata for the adapter; does not authorize changing native masks."""
        return tuple(index is not None for index in self.indices)


@dataclass(frozen=True)
class PatchViewPair:
    first: PatchIndexView
    second: PatchIndexView
    mode: str
    real_count: int
    budget: int
    overlap_count: int
    overlap_fraction: float
    union_count: int
    jaccard: float
    identical_real_sets: bool


def plan_patch_views(real_count, budget, seed, *, mode='train'):
    """Select real patches without replacement; pad short bags with None.

    Train views use separate derived local RNG streams and may overlap or even
    coincide by chance. Short bags include all real rows in both views, so this
    operation alone supplies no distinct content view for such bags. Eval uses
    one fixed seed-derived subset twice. This is a new explicit eval policy,
    not a replacement for SlotSPE's original repeated-evaluation protocol.
    """
    _integer('real_count', real_count, 1)
    _integer('budget', budget, 1)
    _integer('seed', seed)
    _mode(mode)
    seeds = (_child_seed(seed, mode + ':0'), _child_seed(seed, mode + ':1'))

    def draw(view_seed):
        if real_count <= budget:
            selected = tuple(range(real_count))
        else:
            selected = tuple(random.Random(view_seed).sample(range(real_count), budget))
        return PatchIndexView(selected + (None,) * (budget - len(selected)), view_seed)

    first = draw(seeds[0])
    second = first if mode == 'eval' else draw(seeds[1])
    a = {i for i in first.indices if i is not None}
    b = {i for i in second.indices if i is not None}
    overlap, union = len(a & b), len(a | b)
    return PatchViewPair(first, second, mode, real_count, budget, overlap,
                         overlap / min(len(a), len(b)), union, overlap / union, a == b)


@dataclass(frozen=True)
class FeatureContext:
    """Expected split/fold/feature-order identity, not raw identifiers."""
    split_digest: str
    fold: int
    feature_schema_digest: str
    feature_count: int

    def __post_init__(self):
        _digest('split_digest', self.split_digest)
        _integer('fold', self.fold)
        _digest('feature_schema_digest', self.feature_schema_digest)
        _integer('feature_count', self.feature_count, 1)


@dataclass(frozen=True)
class TrainingStatisticsProvenance:
    """Attestation contract for already fitted training-fold statistics.

    Checking this metadata cannot prove that external statistics were fitted
    honestly. The future data adapter must verify the source manifest/hashes
    in ignored local evidence before using them. No values are loaded here.
    """
    context: FeatureContext
    fit_rows_digest: str
    statistics_digest: str
    transform_version: str
    fit_partition: str = 'train'
    uses_outcomes: bool = False

    def __post_init__(self):
        if not isinstance(self.context, FeatureContext):
            raise ValueError('context must be FeatureContext')
        _digest('fit_rows_digest', self.fit_rows_digest)
        _digest('statistics_digest', self.statistics_digest)
        if not isinstance(self.transform_version, str) or not self.transform_version.strip():
            raise ValueError('transform_version is required')
        if self.fit_partition != 'train' or self.uses_outcomes is not False:
            raise ValueError('statistics must be fitted on training rows without outcomes')

    def assert_compatible(self, context):
        if self.context != context:
            raise ValueError('training statistics do not match split/fold/feature schema')


@dataclass(frozen=True)
class OmicsMaskView:
    masked_indices: Tuple[int, ...]
    observed_indices: Tuple[int, ...]
    unavailable_indices: Tuple[int, ...]
    seed: int
    requested_mask_count: int
    noise_bound_standardized: float

    @property
    def actual_mask_count(self):
        return len(self.masked_indices)


@dataclass(frozen=True)
class OmicsViewPair:
    first: OmicsMaskView
    second: OmicsMaskView
    mode: str
    feature_kind: str
    context: FeatureContext
    standardization: Optional[TrainingStatisticsProvenance]


def plan_omics_views(context, seed, *, feature_kind, mask_fraction=0.0,
                     min_observed=1, unavailable_indices=(), protected_groups=(),
                     min_observed_per_group=1, noise_bound_standardized=0.0,
                     standardization=None, mode='train'):
    """Plan masks of observed continuous features; do not modify any values.

    Each protected group is a tuple of coordinate indices. Greedy masking may
    mask fewer than requested to preserve each group's and the whole vector's
    minimum observed count; requested and actual counts are explicit. This is
    not uniform sampling over all feasible masks. Missing coordinates remain
    unavailable and never count as observed. The adapter must carry both masks,
    not silently interpret every masked coordinate as biological zero.

    Discrete mutation inputs are identity-only in this initial component.
    Bounded continuous noise is a future operation in standardized units and
    requires compatible training-only statistics. Eval always produces two
    identical, unperturbed plans, while retaining original missingness.
    """
    if not isinstance(context, FeatureContext):
        raise ValueError('context must be FeatureContext')
    _integer('seed', seed)
    _mode(mode)
    if feature_kind not in ('continuous_expression', 'continuous_pathway', 'discrete_mutation'):
        raise ValueError('unsupported feature_kind')
    _finite('mask_fraction', mask_fraction, maximum=1.0)
    _finite('noise_bound_standardized', noise_bound_standardized)
    _integer('min_observed', min_observed, 1)
    _integer('min_observed_per_group', min_observed_per_group, 1)
    if feature_kind == 'discrete_mutation' and (mask_fraction != 0 or noise_bound_standardized != 0):
        raise ValueError('discrete mutation augmentation is not supported; continuous noise is invalid')
    if standardization is not None:
        if not isinstance(standardization, TrainingStatisticsProvenance):
            raise ValueError('standardization must be TrainingStatisticsProvenance')
        standardization.assert_compatible(context)
    if noise_bound_standardized and standardization is None:
        raise ValueError('continuous noise requires compatible training-only standardization provenance')

    def coordinate_set(values, name):
        values = tuple(values)
        for index in values:
            _integer(name, index)
            if index >= context.feature_count:
                raise ValueError(f'{name} out of range')
        if len(set(values)) != len(values):
            raise ValueError(f'{name} contains duplicate indices')
        return frozenset(values)

    unavailable = coordinate_set(unavailable_indices, 'unavailable index')
    available = frozenset(range(context.feature_count)) - unavailable
    if len(available) < min_observed:
        raise ValueError('insufficient originally observed features; no imputation/drop is authorized')
    groups = tuple(coordinate_set(group, 'group index') for group in protected_groups)
    if any(len(group & available) < min_observed_per_group for group in groups):
        raise ValueError('protected group has insufficient originally observed features')
    requested = math.floor(len(available) * mask_fraction) if mode == 'train' else 0

    def draw(view_seed):
        remaining = set(available)
        masked = []
        candidates = list(sorted(available))
        random.Random(view_seed).shuffle(candidates)
        for index in candidates:
            if len(masked) >= requested or len(remaining) <= min_observed:
                break
            if any(index in group and len(group & remaining) <= min_observed_per_group for group in groups):
                continue
            remaining.remove(index)
            masked.append(index)
        return OmicsMaskView(tuple(sorted(masked)), tuple(sorted(remaining)), tuple(sorted(unavailable)),
                             view_seed, requested, float(noise_bound_standardized) if mode == 'train' else 0.0)

    first = draw(_child_seed(seed, 'omics:' + mode + ':0'))
    second = first if mode == 'eval' else draw(_child_seed(seed, 'omics:' + mode + ':1'))
    return OmicsViewPair(first, second, mode, feature_kind, context, standardization)
