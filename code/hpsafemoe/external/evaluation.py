"""Frozen external evaluation mathematics; no model fitting or beta selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("stage1", "stage2")
DATASETS = ("Core-23", "DS2-248", "HEA-95", "B2-18")
COUNTS = {"Core-23": 23, "DS2-248": 248, "HEA-95": 95, "B2-18": 18}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def vector(value, n=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
        raise ValueError("Expected a nonempty finite vector")
    if n is not None and len(value) != n:
        raise ValueError("Row count mismatch")
    return value


def apply_frozen_beta(anchor, proposal, beta):
    a = vector(anchor)
    r = vector(proposal, len(a))
    if beta not in (0, 0.25, 0.5, 0.75, 1):
        raise ValueError("Beta is not an accepted frozen grid value")
    if beta == 0:
        return a.copy()  # Exact original anchor, not an inverse transform.
    if beta == 1:
        return r.copy()
    return a + beta * (r - a)


def log_youngs(log_k, log_g):
    """E=9KG/(3K+G), expressed stably as log10(E/GPa), per outer."""
    k = vector(log_k)
    g = vector(log_g, len(k))
    ln10 = np.log(10.0)
    denominator = np.logaddexp(np.log(3.0) + ln10 * k, ln10 * g) / ln10
    result = np.log10(9.0) + k + g - denominator
    return vector(result, len(k))


def ensemble_outer(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 5 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Exactly five finite outer predictions are required")
    return np.mean(x, axis=0)


def regression_metrics(prediction, truth):
    p = vector(prediction)
    y = vector(truth, len(p))
    error = p - y
    denominator = np.square(y - y.mean()).sum()
    return {"n": len(y), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "max_error": float(np.abs(error).max()),
            "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else None}


def validate_prediction(path, sample_ids):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"sample_ids", "stage1", "stage2", "outer_stage1", "outer_stage2"}:
            raise ValueError("Invalid prediction keys; labels and alternative experimental arms are forbidden")
        ids = np.asarray(data["sample_ids"]).astype(str).tolist()
        if ids != list(sample_ids) or len(set(ids)) != len(ids):
            raise ValueError("Prediction ID/order mismatch")
        for method in METHODS:
            p = vector(data[method], len(ids))
            expected = ensemble_outer(data[f"outer_{method}"])
            if not np.array_equal(p, expected):
                raise ValueError("Outer ensemble order mismatch")
