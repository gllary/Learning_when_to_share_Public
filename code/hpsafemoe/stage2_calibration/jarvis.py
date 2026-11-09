"""Stage2 safety calibration task-wise correction calibration, independent of the frozen proposer."""
import numpy as np

ALPHAS = (0., .25, .5, .75, 1.)
EPSILON_GAIN_PP = 1.
SCHEMA = 'jarvis-safety-calibration-v1'


def validate(a, r, y=None, classification=False):
    a, r = np.asarray(a, dtype=np.float64), np.asarray(r, dtype=np.float64)
    assert a.ndim == r.ndim == 1 and len(a) and a.shape == r.shape
    assert np.isfinite(a).all() and np.isfinite(r).all()
    if classification:
        assert ((0 <= a) & (a <= 1)).all() and ((0 <= r) & (r <= 1)).all()
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        assert y.shape == a.shape and np.isfinite(y).all()
        if classification:
            assert np.isin(y, [0., 1.]).all()
    return a, r, y


def apply_beta(a, r, beta):
    a, r = np.asarray(a, dtype=np.float64), np.asarray(r, dtype=np.float64)
    assert beta in ALPHAS and a.shape == r.shape
    assert np.isfinite(a).all()
    if beta == 0:
        return a.copy()
    assert np.isfinite(r).all()
    if beta == 1:
        return r.copy()
    return a + beta * (r - a)


def losses(y, p, classification):
    return ((p > .5) != y).astype(np.float64) if classification else np.abs(p - y)


def score(y, p, classification):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean((p > .5) == y)) if classification else float(np.abs(p-y).mean())


def select_alpha(a, r, y, classification):
    a, r, y = validate(a, r, y, classification)
    values = [float(losses(y, apply_beta(a, r, x), classification).mean()) for x in ALPHAS]
    # Ascending grid and strict comparison: exact metric ties choose smaller alpha.
    index = min(range(len(ALPHAS)), key=lambda i: (values[i], ALPHAS[i]))
    return ALPHAS[index], values


def meta_partitions(y, classification, seed):
    """Exactly the accepted parent's deterministic meta-calibration split."""
    y = np.asarray(y)
    assert len(y) >= 5
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(y), dtype=np.int8)
    if classification:
        for label in (0, 1):
            selected = np.flatnonzero(y == label)
            rng.shuffle(selected)
            assignment[selected] = np.arange(len(selected), dtype=np.int64) % 5
    else:
        selected = np.arange(len(y), dtype=np.int64)
        rng.shuffle(selected)
        assignment[selected] = np.arange(len(selected), dtype=np.int64) % 5
    assert set(assignment.tolist()) == set(range(5))
    return assignment


def normalized_gain(base_losses, new_losses):
    base = np.asarray(base_losses, dtype=np.float64)
    new = np.asarray(new_losses, dtype=np.float64)
    return 100. * (float(base.mean()) - float(new.mean())) / max(float(base.mean()), 1e-12)


def paired_bootstrap(y, base_losses, new_losses, classification, seed, resamples=5000):
    """Full-n paired percentile bootstrap, classification preserves class counts."""
    y, base, new = map(np.asarray, (y, base_losses, new_losses))
    assert len(y) == len(base) == len(new) and len(y)
    point = normalized_gain(base, new)
    if np.array_equal(base, new):
        values = np.zeros(resamples, dtype=np.float64)
        zero_denominator_draws = resamples if base.mean() == 0 else 0
    else:
        rng = np.random.default_rng(seed)
        values = np.empty(resamples, dtype=np.float64)
        zero_denominator_draws = 0
        groups = [np.flatnonzero(y == c) for c in (0, 1)] if classification else [np.arange(len(y))]
        for start in range(0, resamples, 32):
            size = min(32, resamples - start)
            sums_a, sums_q = np.zeros(size), np.zeros(size)
            for group in groups:
                if not len(group):
                    continue
                indices = group[rng.integers(0, len(group), size=(size, len(group)))]
                sums_a += base[indices].sum(axis=1)
                sums_q += new[indices].sum(axis=1)
            means_a, means_q = sums_a / len(y), sums_q / len(y)
            zero_denominator_draws += int((means_a == 0).sum())
            values[start:start+size] = 100. * (means_a-means_q) / np.maximum(means_a, 1e-12)
    assert np.isfinite(values).all()
    return dict(G=point, L=float(np.quantile(values, .05)), upper_diagnostic=float(np.quantile(values, .95)),
                confidence=.95, method='paired_percentile', resamples=resamples, resample_size=len(y),
                class_counts_preserved=bool(classification), seed=int(seed),
                zero_baseline_error_draws=zero_denominator_draws, denominator_floor=1e-12), values


def accept_liberal(G, L):
    assert np.isfinite(G) and np.isfinite(L)
    return bool(G > 0. and L >= -EPSILON_GAIN_PP)


def calibrate(a, r, y, classification, seed, resamples=5000):
    a, r, y = validate(a, r, y, classification)
    assignment = meta_partitions(y, classification, seed)
    q = np.empty_like(a)
    fold_records = []
    for held in range(5):
        fit, evaluate = assignment != held, assignment == held
        alpha, candidates = select_alpha(a[fit], r[fit], y[fit], classification)
        q[evaluate] = apply_beta(a[evaluate], r[evaluate], alpha)
        fold_records.append(dict(held_fold=held, fit_rows=int(fit.sum()), held_rows=int(evaluate.sum()),
                                 alpha=alpha, fit_candidate_mean_losses=candidates,
                                 held_gain_percent=normalized_gain(losses(y[evaluate], a[evaluate], classification),
                                                                   losses(y[evaluate], q[evaluate], classification))))
    certificate, bootstrap = paired_bootstrap(y, losses(y, a, classification), losses(y, q, classification),
                                              classification, seed+1777, resamples)
    alpha, final_candidates = select_alpha(a, r, y, classification)
    passed = accept_liberal(certificate['G'], certificate['L'])
    beta = alpha if passed else 0.
    result = dict(schema=SCHEMA, rule='fixed_liberal', epsilon_gain_pp=EPSILON_GAIN_PP,
                  alpha_grid=list(ALPHAS), alpha_all_oof=alpha, beta=beta, task_rule_passed=passed,
                  correction_enabled=bool(beta > 0), candidate_mean_losses_all_oof=final_candidates,
                  certificate=certificate, meta_folds=fold_records,
                  positive_fold_count_diagnostic=sum(f['held_gain_percent'] > 0 for f in fold_records),
                  worst_fold_gain_diagnostic=min(f['held_gain_percent'] for f in fold_records),
                  fold_diagnostics_used_for_acceptance=False, final_alpha_method='all_oof_grid_search',
                  classification_threshold=.5 if classification else None,
                  classification_operator='>' if classification else None,
                  final_oof_training_score_diagnostic=score(y, apply_beta(a, r, beta), classification),
                  calibration_seed=seed, train_labels_only=True, independent_risk_guarantee=False)
    return result, dict(meta_fold=assignment, crossfit_prediction=q, bootstrap_gains=bootstrap)
