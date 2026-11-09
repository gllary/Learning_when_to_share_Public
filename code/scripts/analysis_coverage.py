"""Correction coverage: the proportion of test records with beta > 0."""
import numpy as np


def accepted_count(beta, n):
    """Count accepted records for a scalar task/fold beta or per-record betas."""
    values = np.asarray(beta, dtype=float).reshape(-1)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Calibration coefficients must be finite and in [0, 1]')
    if values.size == 1:
        return n if values[0] > 0 else 0
    if values.size != n:
        raise ValueError('Expected one coefficient per task/fold or test record')
    return int(np.count_nonzero(values > 0))
