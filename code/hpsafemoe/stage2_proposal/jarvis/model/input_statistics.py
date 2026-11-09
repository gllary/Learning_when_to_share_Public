import json
import os
import sys
from pathlib import Path
import numpy as np
from ..common import ROOT, CONTEXT, atomic_json, sha256, fold
from .base_inputs import Inputs as BaseInputs
from ..common import require_zero_u
from ..common import RELEASE
POLICY = json.loads((RELEASE / 'checkpoint/jarvis/stage2_proposal/model/training_config.json').read_text())

def policy_hash():
    return sha256(RELEASE / 'checkpoint/jarvis/stage2_proposal/model/training_config.json')

def partition(ids, inner):
    assert inner in range(5)
    fit = [j for j in ids if fold(j) != inner]
    held = [j for j in ids if fold(j) == inner]
    assert fit and held and (not set(fit).intersection(held))
    assert len(fit) + len(held) == len(ids) == len(set(ids))
    return (fit, held)

def scalar_stats(values):
    a = np.asarray(values, dtype=np.float64)
    assert len(a) and np.isfinite(a).all()
    return (float(a.mean()), float(max(a.std(), 1e-08)))

class Inputs(BaseInputs):
    """JARVIS identity adapter; Matbench raw-token and anchor numerical contract."""

    def __init__(self):
        super().__init__()
        from ..common import output_dir, PUBLIC_ROOT
        self.kinds = {r['id']: 'classification' if r['metric'] == 'ACC' else 'regression' for r in self.records}
        self.inner = None
        self.token_stats = None
        self.uncertainty = {}
        self.availability = {}
        extra = []
        for source in self.tasks:
            self.uncertainty[source] = {}
            self.availability[source] = {}
            for f in [-1, 0, 1, 2, 3, 4]:
                out = PUBLIC_ROOT / source if f < 0 and source in self.public_overrides else output_dir(source, f)
                up = out / 'uncertainty.npy'
                ap = out / 'available.npy'
                require_zero_u(up, ap)
                u = np.load(up, mmap_mode='r') if up.is_file() else np.zeros(len(self.ids), dtype=np.float32)
                a = np.load(ap, mmap_mode='r') if ap.is_file() else np.ones(len(self.ids), dtype=bool)
                assert u.shape == a.shape == (len(self.ids),)
                assert a.dtype == np.bool_
                p = self.prediction[source][f]
                h = self.hidden[source][f]
                valid = np.asarray(a, dtype=bool) & np.isfinite(p) & np.isfinite(u) & (u >= 0) & np.isfinite(h).all(axis=1)
                if self.kinds[source] == 'classification':
                    valid &= (p >= 0) & (p <= 1)
                self.uncertainty[source][f] = u
                self.availability[source][f] = valid
                extra.append(dict(source=source, fold=f, uncertainty_provided=up.is_file(), uncertainty_sha256=sha256(up) if up.is_file() else None, availability_sha256=sha256(ap) if ap.is_file() else None, valid_rows=int(valid.sum()), rows=len(valid)))
        for t in self.tasks:
            for role in ['train', 'val', 'test']:
                ids = self.roles[t][role]
                rows = np.asarray([self.lookup[j] for j in ids])
                fs = self.assignment[rows] if role == 'train' else np.full(len(rows), -1)
                assert all((self.availability[t][int(f)][rows[fs == f]].all() for f in np.unique(fs))), ('Missing target anchor', t, role)
        atomic_json(ROOT / 'stage2/optional_input_audit.json', dict(records=extra, family_exposure_mask=False))

    def source_arrays(self, source, rows, groups):
        p = np.empty(len(rows), dtype=np.float64)
        u = np.empty_like(p)
        a = np.empty(len(rows), dtype=bool)
        h = np.empty((len(rows), self.dims[source]), dtype=np.float32)
        for f in np.unique(groups):
            take = groups == f
            r = rows[take]
            p[take] = self.prediction[source][int(f)][r]
            u[take] = self.uncertainty[source][int(f)][r]
            a[take] = self.availability[source][int(f)][r]
            h[take] = self.hidden[source][int(f)][r]
        return (p, u, a, h)

    def set_inner(self, inner, persist=True):
        self.inner = inner
        self.fit_ids = {}
        self.held_ids = {}
        self.token_stats = {}
        self.uncertainty_stats = {}
        for t in self.tasks:
            fit, held = partition(self.roles[t]['train'], inner)
            self.fit_ids[t] = fit
            self.held_ids[t] = held
            y = [self.labels[t]['train'][j] for j in fit]
            self.stats[t] = (0.0, 1.0) if self.kinds[t] == 'classification' else scalar_stats(y)
            if self.kinds[t] == 'classification':
                assert set(y) == {0, 1}
        for t in self.tasks:
            rows = np.asarray([self.lookup[j] for j in self.fit_ids[t]])
            groups = self.assignment[rows]
            self.token_stats[t] = {}
            self.uncertainty_stats[t] = {}
            for source in self.tasks:
                p, u, a, _ = self.source_arrays(source, rows, groups)
                for dest, v in [(self.token_stats, p), (self.uncertainty_stats, np.log1p(np.where(a, u, 0.0)))]:
                    mean, scale = scalar_stats(v[a]) if a.any() else (0.0, 1.0)
                    dest[t][source] = dict(mean=mean, scale=scale, fit_rows=int(a.sum()))
        receipt = dict(schema=POLICY['version'], inner=inner, fit_ids=self.fit_ids, held_ids=self.held_ids, label_stats={t: list(v) for t, v in self.stats.items()}, token_stats=self.token_stats, uncertainty_stats=self.uncertainty_stats, protocol_sha256=policy_hash(), val_labels_read=False, test_labels_read=False)
        if persist:
            path = ROOT / 'stage2/normalizations' / f'inner{inner}.json'
            if path.exists():
                assert json.loads(path.read_text()) == receipt
            else:
                atomic_json(path, receipt)
        self.normalization = receipt
        return receipt

    def standard_anchor(self, task, p):
        p = np.asarray(p, dtype=np.float64)
        if self.kinds[task] == 'classification':
            q = np.clip(p, 1e-06, 1.0 - 1e-06)
            return (np.log(q) - np.log1p(-q)).astype(np.float32)
        m, s = self.stats[task]
        return ((p - m) / s).astype(np.float32)

    def batch(self, task, jids, role, device, dropout=0, rng=None):
        import torch
        assert self.inner is not None and role in ['train', 'val', 'test']
        if role != 'train':
            assert (ROOT / 'stage2/selection_locked.json').is_file()
        rows = np.asarray([self.lookup[j] for j in jids])
        groups = self.assignment[rows] if role == 'train' else np.full(len(rows), -1)
        hidden = {}
        pred = {}
        unc = {}
        masks = []
        for source in self.tasks:
            p, u, a, h = self.source_arrays(source, rows, groups)
            h = np.where(a[:, None], h, 0.0).astype(np.float32)
            p = np.where(a, p, 0.0)
            u = np.log1p(np.where(a, u, 0.0))
            ps = self.token_stats[task][source]
            us = self.uncertainty_stats[task][source]
            hidden[source] = torch.as_tensor(h, device=device)
            pred[source] = torch.as_tensor(((p - ps['mean']) / ps['scale']).astype(np.float32), device=device)
            unc[source] = torch.as_tensor(((u - us['mean']) / us['scale']).astype(np.float32), device=device)
            masks.append(a)
        available = np.stack(masks, 1)
        index = self.tasks.index(task)
        assert available[:, index].all()
        if dropout:
            assert rng is not None
            available &= rng.random(available.shape) >= dropout
            available[:, index] = True
        anchor = self.anchor(task, jids, role)
        return dict(task_index=torch.full((len(rows),), index, dtype=torch.long, device=device), anchor_standard=torch.as_tensor(self.standard_anchor(task, anchor), device=device), hidden=hidden, prediction_standard=pred, uncertainty_standard=unc, available=torch.as_tensor(available, device=device))

    def restore(self, task, p):
        p = np.asarray(p, dtype=np.float64)
        if self.kinds[task] == 'classification':
            return 1.0 / (1.0 + np.exp(-np.clip(p, -40.0, 40.0)))
        m, s = self.stats[task]
        return p * s + m
