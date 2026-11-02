"""Held-out nonlinear predictive increment, NOT a mutual information estimator.

All preprocessing uses fit rows. Validation selects hyperparameters; test is
scored once. Common-only is nested in the joint model at private_weight=0.
Both arms use the same number of kernel centres (fit examples). Effective
degrees of freedom are reported, since equal centres do not equal capacity.
"""
import torch


def _matrix(x):
    x = torch.as_tensor(x).detach().to(device='cpu', dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError('finite nonempty matrix required')
    return x


def _scale(train, rest):
    mean, std = train.mean(0), train.std(0, unbiased=False).clamp_min(1e-8)
    return [(v-mean)/std for v in [train, *rest]]


def _kernel(a, b, family, bandwidth):
    if family == 'polynomial2':
        return (1 + a @ b.T / a.shape[1]).square()
    if family == 'linear':
        return 1 + a @ b.T / a.shape[1]
    distance = torch.cdist(a, b).square() / a.shape[1]
    return torch.exp(-distance/(2*bandwidth**2))


def incremental_probe(fit, validation, test):
    """Each partition is (common, private, target); no automatic data splitting.

    Caller must supply disjoint subjects, and representations frozen without
    probe-test involvement. Positive test gain is function-class-specific
    predictive evidence, not conditional MI or semantic identification.
    """
    parts = [[_matrix(x) for x in part] for part in (fit, validation, test)]
    for part in parts:
        if len(part) != 3 or len({x.shape[0] for x in part}) != 1:
            raise ValueError('aligned common/private/target rows required')
    for j in range(3):
        if len({part[j].shape[1] for part in parts}) != 1:
            raise ValueError('feature dimensions must match across partitions')
    common = _scale(parts[0][0], [parts[1][0], parts[2][0]])
    private = _scale(parts[0][1], [parts[1][1], parts[2][1]])
    ymean = parts[0][2].mean(0)
    yfit = parts[0][2]-ymean
    n = yfit.shape[0]
    best = {'common': None, 'joint': None}
    grid = [('linear', 1.), ('polynomial2', 1.), ('rbf', .5), ('rbf', 1.), ('rbf', 2.)]
    for family, bandwidth in grid:
        kc = [_kernel(v, common[0], family, bandwidth) for v in common]
        for weight in [0., .25, 1.]:
            # Common kernel remains available even when private noise is added.
            joint = [torch.cat([c, p], -1) for c,p in zip(common, private)]
            kernels = kc if weight == 0 else [k+weight*_kernel(v, joint[0], family, bandwidth) for k,v in zip(kc,joint)]
            eig, vec = torch.linalg.eigh(kernels[0])
            eig = eig.clamp_min(0)
            projected = vec.T @ yfit
            for regularization in [1e-6, 1e-4, .01, 1.]:
                alpha = vec @ (projected/(eig+n*regularization)[:,None])
                prediction = kernels[1] @ alpha+ymean
                loss = float((prediction-parts[1][2]).square().mean())
                for arm in (['common','joint'] if weight == 0 else ['joint']):
                    if best[arm] is None or loss < best[arm]['validation_mse']:
                        best[arm] = {'validation_mse':loss,'family':family,'bandwidth':bandwidth,
                                     'private_weight':weight,'regularization':regularization,
                                     'effective_df':float((eig/(eig+n*regularization)).sum()),
                                     '_prediction':kernels[2] @ alpha+ymean}
    test_variance = float((parts[2][2]-ymean).square().mean())
    for arm in best:
        prediction = best[arm].pop('_prediction')
        best[arm]['test_mse'] = float((prediction-parts[2][2]).square().mean())
    gain = best['common']['test_mse']-best['joint']['test_mse']
    return {'common':best['common'],'joint':best['joint'],'test_mse_gain':gain,
            'gain_relative_to_fit_mean_baseline':gain/max(test_variance,1e-12),
            'fit_rows':n,'validation_rows':parts[1][2].shape[0],'test_rows':parts[2][2].shape[0],
            'meaning':'finite probe predictive gain, not MI; negative is not proof of no information'}
