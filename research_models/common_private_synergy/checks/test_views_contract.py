"""No neural dependencies or model execution: pure index/provenance contracts."""
import random
import unittest

from research_models.common_private_synergy.components.views import (
    FeatureContext, TrainingStatisticsProvenance, plan_omics_views, plan_patch_views,
)


class ViewsContractTests(unittest.TestCase):
    def setUp(self):
        self.context = FeatureContext('a' * 64, 0, 'b' * 64, 10)
        self.stats = TrainingStatisticsProvenance(self.context, 'c' * 64, 'd' * 64, 'train-zscore-v1')

    def test_patch_plan_replays_without_touching_global_rng(self):
        before = random.getstate()
        plan = plan_patch_views(20, 7, 91)
        self.assertEqual(plan, plan_patch_views(20, 7, 91))
        self.assertEqual(before, random.getstate())
        for view in (plan.first, plan.second):
            self.assertEqual(len(view.indices), 7)
            self.assertEqual(len(set(view.indices)), 7)
            self.assertTrue(all(0 <= i < 20 for i in view.indices))
        a, b = set(plan.first.indices), set(plan.second.indices)
        self.assertEqual(plan.overlap_count, len(a & b))
        self.assertAlmostEqual(plan.jaccard, len(a & b) / len(a | b))

    def test_short_bag_retained_padded_and_not_falsely_new_view(self):
        plan = plan_patch_views(3, 8, 2)
        self.assertEqual(plan.first.indices, (0, 1, 2, None, None, None, None, None))
        self.assertEqual(plan.first.real_mask, (True, True, True, False, False, False, False, False))
        self.assertEqual(plan.second.indices, plan.first.indices)
        self.assertEqual(plan.overlap_fraction, 1.0)
        self.assertTrue(plan.identical_real_sets)

    def test_eval_patch_views_identical_and_rng_independent(self):
        plan = plan_patch_views(100, 8, 22, mode='eval')
        self.assertEqual(plan.first, plan.second)
        self.assertEqual(plan, plan_patch_views(100, 8, 22, mode='eval'))

    def test_patch_invalid_counts_and_modes(self):
        for args in ((0, 5, 2), (3, 0, 2), (True, 5, 2), (3, 5, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                plan_patch_views(*args)
        with self.assertRaises(ValueError):
            plan_patch_views(3, 5, 2, mode='test')

    def test_omics_missingness_groups_and_requested_count(self):
        pair = plan_omics_views(self.context, 3, feature_kind='continuous_expression', mask_fraction=1,
                                unavailable_indices=(9,), protected_groups=((0, 1), (1, 2, 3), (4, 5, 6)),
                                min_observed=2)
        for view in (pair.first, pair.second):
            observed = set(view.observed_indices)
            self.assertEqual(view.requested_mask_count, 9)
            self.assertLess(view.actual_mask_count, view.requested_mask_count)
            self.assertEqual(set(view.unavailable_indices), {9})
            self.assertTrue(set(view.masked_indices).isdisjoint(observed))
            self.assertEqual(observed | set(view.masked_indices), set(range(9)))
            self.assertGreaterEqual(len(observed), 2)
            for group in ((0, 1), (1, 2, 3), (4, 5, 6)):
                self.assertGreaterEqual(len(observed & set(group)), 1)

    def test_omics_plan_replay_and_global_rng(self):
        before = random.getstate()
        kwargs = dict(feature_kind='continuous_pathway', mask_fraction=.3)
        pair = plan_omics_views(self.context, 17, **kwargs)
        self.assertEqual(pair, plan_omics_views(self.context, 17, **kwargs))
        self.assertEqual(before, random.getstate())

    def test_eval_omics_retains_missingness_and_disables_perturbation(self):
        pair = plan_omics_views(self.context, 6, feature_kind='continuous_expression', mask_fraction=.9,
                                noise_bound_standardized=.1, standardization=self.stats,
                                unavailable_indices=(2,), mode='eval')
        self.assertEqual(pair.first, pair.second)
        self.assertEqual(pair.first.masked_indices, ())
        self.assertEqual(pair.first.noise_bound_standardized, 0.0)
        self.assertEqual(pair.first.unavailable_indices, (2,))

    def test_discrete_mutation_identity_only(self):
        pair = plan_omics_views(self.context, 1, feature_kind='discrete_mutation')
        self.assertEqual(pair.first.masked_indices, ())
        for kwargs in (dict(mask_fraction=.1), dict(noise_bound_standardized=.1, standardization=self.stats)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_omics_views(self.context, 1, feature_kind='discrete_mutation', **kwargs)

    def test_noise_requires_matching_training_provenance(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            plan_omics_views(self.context, 2, feature_kind='continuous_expression', noise_bound_standardized=.1)
        mismatched = FeatureContext('e' * 64, 0, 'b' * 64, 10)
        with self.assertRaisesRegex(ValueError, 'match'):
            plan_omics_views(mismatched, 2, feature_kind='continuous_expression', standardization=self.stats)
        for kwargs in (dict(fit_partition='validation'), dict(uses_outcomes=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingStatisticsProvenance(self.context, 'c' * 64, 'd' * 64, 'v1', **kwargs)

    def test_invalid_missingness_and_invalid_strengths(self):
        cases = (dict(unavailable_indices=(0, 0)), dict(unavailable_indices=(10,)),
                 dict(protected_groups=((1, 2),), unavailable_indices=(1, 2)),
                 dict(unavailable_indices=tuple(range(10))), dict(mask_fraction=float('nan')),
                 dict(noise_bound_standardized=-1), dict(mask_fraction=True))
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                plan_omics_views(self.context, 2, feature_kind='continuous_expression', **kwargs)


if __name__ == '__main__':
    unittest.main()
