"""Pure metadata contracts for the future feature/data adapter.

These classes deliberately hold schema digests and counts rather than feature
values, patient identifiers, or files.  They prevent the model assembly from
silently guessing whether an omics table contains expression, pathway scores,
or discrete mutation indicators.  Constructing a contract is not a data
access operation and does not authorize patient processing.
"""
from dataclasses import dataclass
import math
from typing import Tuple

from .contracts import ArchitectureConfig, positive_int
from .views import FeatureContext


def _text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')


def _digest(name: str, value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')


def _finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')


@dataclass(frozen=True)
class FeatureTokenSchema:
    """One already-tokenized modality schema, without feature values."""

    modality: str
    feature_kind: str
    token_width: int
    token_count: int
    identity_encoding: str
    schema_digest: str
    missingness_explicit: bool = True

    def __post_init__(self) -> None:
        if self.modality not in ('wsi', 'omic'):
            raise ValueError("modality must be 'wsi' or 'omic'")
        if self.feature_kind not in ('wsi_embedding', 'continuous_expression', 'continuous_pathway', 'discrete_mutation'):
            raise ValueError('unsupported feature_kind')
        positive_int('token_width', self.token_width)
        positive_int('token_count', self.token_count)
        _text('identity_encoding', self.identity_encoding)
        _digest('schema_digest', self.schema_digest)
        if type(self.missingness_explicit) is not bool:
            raise ValueError('missingness_explicit must be boolean')
        if self.modality == 'wsi' and self.feature_kind != 'wsi_embedding':
            raise ValueError('wsi requires wsi_embedding feature_kind')
        if self.modality == 'omic' and self.feature_kind == 'wsi_embedding':
            raise ValueError('omic requires an omics feature_kind')
        if self.modality == 'omic' and self.feature_kind in ('continuous_expression', 'continuous_pathway', 'discrete_mutation'):
            # Every omics token must carry a stable gene/pathway identity.  A
            # scalar bag with no identity is not an admissible adapter input.
            if self.identity_encoding == 'none':
                raise ValueError('omics tokens require explicit gene/pathway identity encoding')


@dataclass(frozen=True)
class PatientPairContract:
    """Row-level pairing and split metadata; raw patient IDs stay external."""

    context: FeatureContext
    row_count: int
    wsi: FeatureTokenSchema
    omic: FeatureTokenSchema
    pairing_digest: str
    row_order_version: str
    partition: str = 'train'

    def __post_init__(self) -> None:
        if not isinstance(self.context, FeatureContext):
            raise ValueError('context must be FeatureContext')
        positive_int('row_count', self.row_count)
        if self.wsi.modality != 'wsi' or self.omic.modality != 'omic':
            raise ValueError('schemas must be supplied in wsi/omic roles')
        _digest('pairing_digest', self.pairing_digest)
        _text('row_order_version', self.row_order_version)
        if self.partition not in ('train', 'validation', 'test'):
            raise ValueError('partition must be train, validation, or test')
        if self.context.feature_count != self.omic.token_count:
            raise ValueError('context feature_count must equal omic token_count')


@dataclass(frozen=True)
class TimeBinContract:
    """Train-fitted discrete-time mapping metadata for the survival adapter."""

    bins: int
    finite_boundaries: Tuple[float, ...]
    boundary_rule: str
    fit_partition: str = 'train'
    mapping_version: str = 'completed-intervals-v1'

    def __post_init__(self) -> None:
        positive_int('bins', self.bins)
        if len(self.finite_boundaries) != self.bins - 1:
            raise ValueError('finite_boundaries must contain bins-1 values')
        for value in self.finite_boundaries:
            _finite_number('boundary', value)
        if tuple(self.finite_boundaries) != tuple(sorted(self.finite_boundaries)):
            raise ValueError('finite_boundaries must be strictly increasing')
        if len(set(self.finite_boundaries)) != len(self.finite_boundaries):
            raise ValueError('finite_boundaries must be strictly increasing')
        _text('boundary_rule', self.boundary_rule)
        if self.fit_partition != 'train':
            raise ValueError('time bins must be fitted on the training partition')
        _text('mapping_version', self.mapping_version)

    def assert_matches(self, *, bins: int, partition: str) -> None:
        if bins != self.bins:
            raise ValueError('time-bin count does not match model config')
        if partition not in ('train', 'validation', 'test'):
            raise ValueError('invalid evaluation partition')


@dataclass(frozen=True)
class AdapterContract:
    """Complete metadata gate before a future adapter may produce tensors."""

    pair: PatientPairContract
    time_bins: TimeBinContract
    model: ArchitectureConfig
    independent_patient_batch: int

    def __post_init__(self) -> None:
        if not isinstance(self.pair, PatientPairContract):
            raise ValueError('pair must be PatientPairContract')
        if not isinstance(self.time_bins, TimeBinContract):
            raise ValueError('time_bins must be TimeBinContract')
        if not isinstance(self.model, ArchitectureConfig):
            raise ValueError('model must be ArchitectureConfig')
        positive_int('independent_patient_batch', self.independent_patient_batch)
        if self.model.wsi_dim != self.pair.wsi.token_width:
            raise ValueError('model wsi_dim does not match the token schema')
        if self.model.omic_dim != self.pair.omic.token_width:
            raise ValueError('model omic_dim does not match the token schema')
        if self.model.bins != self.time_bins.bins:
            raise ValueError('model bins do not match the time-bin contract')
        if self.independent_patient_batch > self.pair.row_count:
            raise ValueError('independent batch cannot exceed paired row count')

    def assert_partition(self, partition: str) -> None:
        if partition not in ('train', 'validation', 'test'):
            raise ValueError('invalid partition')
        if partition == 'train' and self.pair.partition != 'train':
            raise ValueError('training adapter contract must use train partition')
