"""Standard-library configuration; no model construction or data access."""
from dataclasses import dataclass


def positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass(frozen=True)
class ArchitectureConfig:
    wsi_dim: int
    omic_dim: int
    width: int = 64
    slots: int = 8
    common_slots: int = 7
    bins: int = 4
    iterations: int = 3
    interaction_rank: int = 16
    use_interaction: bool = False

    def __post_init__(self):
        for name in ('wsi_dim', 'omic_dim', 'width', 'slots', 'bins', 'iterations', 'interaction_rank'):
            positive_int(name, getattr(self, name))
        if type(self.common_slots) is not int or not 0 <= self.common_slots <= self.slots:
            raise ValueError('common_slots must be an integer between zero and slots')
        if type(self.use_interaction) is not bool:
            raise ValueError('use_interaction must be boolean')

    @property
    def private_slots(self):
        return self.slots - self.common_slots


@dataclass(frozen=True)
class CorrelationContract:
    """Necessary rank check, NOT a sufficient statistical reliability criterion.

    Each modality uses [project(common), project(private)] for intra loss.
    Keep one independent row per patient; views and slots are never extra rows.
    """
    projection_dim: int = 2
    min_patients: int = 8

    def __post_init__(self):
        positive_int('projection_dim', self.projection_dim)
        positive_int('min_patients', self.min_patients)
        if self.min_patients < 2:
            raise ValueError('at least two patients required')

    def validate_batch(self, patients, *, has_common=True, has_private=True, all_role_terms=True):
        positive_int('patients', patients)
        if any(type(v) is not bool for v in (has_common, has_private, all_role_terms)):
            raise ValueError('role and objective flags must be boolean')
        if not (has_common or has_private):
            raise ValueError('at least one nonempty role group required')
        if patients < self.min_patients:
            raise ValueError('fewer independent patients than declared minimum')
        dim = (int(has_common) + int(has_private)) * self.projection_dim
        if dim > patients - 1:
            raise ValueError('intra correlation identity is rank-infeasible: D > B - 1')
        joint = (int(has_common) + 2*int(has_private))*self.projection_dim if all_role_terms else dim
        if joint > patients - 1:
            raise ValueError('simultaneous role targets need dc + 2dp <= B - 1 (absent roles count zero)')
        return {'patients': patients, 'intra_dimension': dim, 'joint_required_rank': joint,
                'rank_ceiling': patients - 1}
