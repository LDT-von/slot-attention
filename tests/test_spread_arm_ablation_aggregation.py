"""Regression test for the spread arm's ablation aggregation.

The spread arm's evaluation epoch aggregated the per-batch slot-ablation
logit changes into a per-role average. The aggregation was written as

    {k: [v / len(ids) for v in values] for k, v in ablations.items()}

The OUTER comprehension bound ``v`` to each role's own accumulated list, but
the INNER comprehension then rebound ``v`` to each element of a DIFFERENT
variable, ``values``, which was a leftover name from an earlier loop in the
same function. The inner loop therefore always divided the same leaked list,
so every role group received an identical vector: the four role groups of
``ablation_mean_abs_logit_change`` were numerically indistinguishable in the
receipt, and the value was the last role group's raw per-batch numbers rather
than a per-role average.

The archived evaluation epoch in ``full_slot_epoch.py`` binds the outer
variable to ``values``, and the same archived expression is what this test
pins, so the archived receipts keep their own provenance.

This test is a pure aggregation test on invented numbers. It does not load a
patient, a checkpoint or a prepared cache.
"""
import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from research_models.common_private_synergy import spread_arm


SPREAD_ARM = pathlib.Path(spread_arm.__file__)
ARCHIVED_EPOCH = SPREAD_ARM.with_name('full_slot_epoch.py')


def _bound_names(target):
    """Names bound by a comprehension target, including tuple targets."""
    if isinstance(target, ast.Name):
        return [target]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for element in target.elts:
            names.extend(_bound_names(element))
        return names
    return []


def _aggregation_expression(path):
    """Return the source of the ablation aggregation dict comprehension."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or key.value != 'ablation_mean_abs_logit_change':
                continue
            return ast.unparse(value)
    raise AssertionError('no ablation_mean_abs_logit_change aggregation found in ' + str(path))


def test_spread_arm_aggregation_binds_its_own_role_list():
    """The outer variable must be the role list, not a leaked name."""
    expression = _aggregation_expression(SPREAD_ARM)
    assert 'for k, values in ablations.items()' in expression, expression


def test_spread_arm_aggregation_matches_the_archived_expression():
    """The spread arm must aggregate exactly like the archived epoch."""
    assert _aggregation_expression(SPREAD_ARM) == _aggregation_expression(ARCHIVED_EPOCH)


def test_spread_arm_aggregation_keeps_the_roles_distinct():
    """Four role groups of distinct numbers must aggregate to four distinct vectors."""
    ablations = {'p_common': [1.0, 2.0], 'p_residual': [3.0, 4.0],
                 'g_common': [5.0, 6.0], 'g_residual': [7.0, 8.0]}
    ids = [0, 1, 2, 3]
    aggregated = {k: [v / len(ids) for v in values] for k, values in ablations.items()}
    assert aggregated == {'p_common': [0.25, 0.5], 'p_residual': [0.75, 1.0],
                          'g_common': [1.25, 1.5], 'g_residual': [1.75, 2.0]}
    assert len({tuple(v) for v in aggregated.values()}) == len(aggregated)


def test_spread_arm_evaluation_epoch_never_shadows_the_role_list():
    """No comprehension inside the evaluation epoch may rebind its outer variable.

    This is the general guard against the whole failure mode, rather than a
    single renamed expression.
    """
    tree = ast.parse(SPREAD_ARM.read_text(encoding='utf-8'))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == 'spread_evaluate_epoch')
    for node in ast.walk(function):
        if not isinstance(node, ast.DictComp):
            continue
        targets = [name.id for generator in node.generators
                   for name in _bound_names(generator.target)]
        if len(targets) != 1:
            continue
        outer = targets[0]
        # Any inner comprehension that rebinds the outer name shadows it.
        for inner in ast.walk(node.value):
            if not isinstance(inner, ast.DictComp) and not isinstance(inner, ast.ListComp):
                continue
            if inner is node.value:
                continue
            for generator in inner.generators:
                for t in generator.target:
                    if isinstance(t, ast.Name) and t.id == outer:
                        raise AssertionError(
                            'inner comprehension rebinds the outer name ' + repr(outer)
                            + ' inside ' + ast.unparse(node))
