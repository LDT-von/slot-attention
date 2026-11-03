"""Metadata-only adapter checks; never load features or construct a model."""
import unittest

from research_models.common_private_synergy.components.adapters import (
    AdapterContract, FeatureTokenSchema, PatientPairContract, TimeBinContract,
)
from research_models.common_private_synergy.components.contracts import ArchitectureConfig
from research_models.common_private_synergy.components.views import FeatureContext


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        context = FeatureContext('a' * 64, 0, 'b' * 64, 6)
        wsi = FeatureTokenSchema('wsi', 'wsi_embedding', 4, 12, 'patch-order-v1', 'c' * 64)
        omic = FeatureTokenSchema('omic', 'continuous_expression', 5, 6, 'gene-id-v1', 'd' * 64)
        pair = PatientPairContract(context, 16, wsi, omic, 'e' * 64, 'paired-row-v1')
        bins = TimeBinContract(4, (1.0, 2.0, 4.0), 'left-closed-right-open-v1')
        self.contract = AdapterContract(pair, bins, ArchitectureConfig(4, 5, bins=4), 8)

    def test_complete_contract_accepts_matching_shapes(self):
        self.contract.assert_partition('train')
        self.contract.time_bins.assert_matches(bins=4, partition='validation')

    def test_omics_identity_is_required(self):
        with self.assertRaisesRegex(ValueError, 'identity'):
            FeatureTokenSchema('omic', 'continuous_pathway', 5, 6, 'none', 'd' * 64)

    def test_schema_rejects_wrong_modality_kind(self):
        with self.assertRaises(ValueError):
            FeatureTokenSchema('wsi', 'continuous_expression', 4, 12, 'patch-order-v1', 'c' * 64)

    def test_pairing_context_must_match_omics_count(self):
        context = FeatureContext('a' * 64, 0, 'b' * 64, 7)
        wsi = FeatureTokenSchema('wsi', 'wsi_embedding', 4, 12, 'patch-order-v1', 'c' * 64)
        omic = FeatureTokenSchema('omic', 'continuous_expression', 5, 6, 'gene-id-v1', 'd' * 64)
        with self.assertRaisesRegex(ValueError, 'feature_count'):
            PatientPairContract(context, 16, wsi, omic, 'e' * 64, 'paired-row-v1')

    def test_time_bins_are_train_fitted_and_strict(self):
        with self.assertRaises(ValueError):
            TimeBinContract(4, (1.0, 1.0, 4.0), 'rule')
        with self.assertRaisesRegex(ValueError, 'training'):
            TimeBinContract(4, (1.0, 2.0, 4.0), 'rule', fit_partition='validation')

    def test_model_dimensions_are_checked_without_tensor_access(self):
        with self.assertRaisesRegex(ValueError, 'omic_dim'):
            AdapterContract(self.contract.pair, self.contract.time_bins,
                            ArchitectureConfig(4, 9, bins=4), 8)


if __name__ == '__main__':
    unittest.main()
