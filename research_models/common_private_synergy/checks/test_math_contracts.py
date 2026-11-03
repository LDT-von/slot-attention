"""Exact small examples and configuration only: never import tensor modules."""
import math
import unittest
from collections import Counter
from itertools import product
from research_models.common_private_synergy.components.contracts import ArchitectureConfig, CorrelationContract


def entropy(rows):
    counts = Counter(rows)
    n = len(rows)
    return -sum((k/n)*math.log2(k/n) for k in counts.values())


def mi(x, y):
    return entropy(x) + entropy(y) - entropy(list(zip(x, y)))


class MathematicalContracts(unittest.TestCase):
    def test_ratio_and_rank_contract(self):
        self.assertEqual(ArchitectureConfig(4, 5).private_slots, 1)
        self.assertEqual(ArchitectureConfig(4, 5, common_slots=8).private_slots, 0)
        self.assertEqual(ArchitectureConfig(4, 5, common_slots=0).private_slots, 8)
        with self.assertRaises(ValueError):
            ArchitectureConfig(4, 5, common_slots=True)
        self.assertEqual(CorrelationContract().validate_batch(8)['intra_dimension'], 4)
        with self.assertRaises(ValueError):
            CorrelationContract(projection_dim=3).validate_batch(8)
        with self.assertRaises(ValueError):
            CorrelationContract(projection_dim=64).validate_batch(8)
        CorrelationContract(projection_dim=4).validate_batch(8, has_private=False)
        with self.assertRaises(ValueError):
            CorrelationContract(projection_dim=4).validate_batch(8, has_common=False)

    def test_pid_difference_known_answers(self):
        states = list(product((0, 1), repeat=2))
        x, y = [s[0] for s in states], [s[1] for s in states]
        def difference(a, b, target):
            return mi(list(zip(a, b)), target) - mi(a, target) - mi(b, target)
        self.assertAlmostEqual(difference(x, y, [a ^ b for a, b in states]), 1)
        self.assertAlmostEqual(difference(x, x, x), -1)
        self.assertAlmostEqual(difference(x, y, x), 0)
        self.assertAlmostEqual(difference(x, y, [0]*4), 0)

    def test_disabled_branch_must_remove_all_joint_nonlinearity(self):
        def mixed(f):
            return f(1, 1)-f(1, 0)-f(0, 1)+f(0, 0)
        self.assertEqual(mixed(lambda x, y: x*x+3*y), 0)
        self.assertEqual(mixed(lambda x, y: max(0, x+y-1)), 1)
        self.assertEqual(mixed(lambda x, y: x*y), 1)

    def test_pooled_zero_does_not_certify_slot_independence(self):
        # U1=G, U2=-G -> mean(U)=0, despite each slot fully carrying G.
        g = [-1, 1]
        self.assertEqual([(v-v)/2 for v in g], [0, 0])
        self.assertEqual(mi(g, g), 1)

    def test_patient_variance_misses_within_patient_slot_copies(self):
        slots = [[z]*8 for z in (-2, -1, 1, 2)]
        self.assertTrue(all(len(set(row)) == 1 for row in slots))
        self.assertGreater(sum(row[0]**2 for row in slots)/len(slots), 0)

    def test_completed_interval_likelihood_reference(self):
        # h=(.2,.3): event in bin1 = .8*.3; censored after1 = .8;
        # censored before first completed interval contributes likelihood1.
        h = (.2, .3)
        def nll(completed, event):
            return -sum(math.log1p(-v) for v in h[:completed]) - (math.log(h[completed]) if event else 0)
        self.assertAlmostEqual(nll(1, True), -math.log(.24))
        self.assertAlmostEqual(nll(1, False), -math.log(.8))
        self.assertEqual(nll(0, False), 0)
        self.assertAlmostEqual(nll(2, False), -math.log(.56))


if __name__ == '__main__':
    unittest.main()
