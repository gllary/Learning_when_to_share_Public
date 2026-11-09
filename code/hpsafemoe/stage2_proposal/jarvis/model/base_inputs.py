"""Frozen Stage-1 exports using role-specific raw-official-split inputs."""
import json
from pathlib import Path
import numpy as np
from ..common import ROOT, registry, fold, output_dir, manifest, atomic_json, sha256, validate_expert_protocol, SOURCE_ROOT, PUBLIC_TASKS, PUBLIC_ROOT

class Inputs:

    def __init__(self):
        self.records = registry()['tasks']
        self.tasks = tuple((r['id'] for r in self.records))
        self.ids = np.load(SOURCE_ROOT / 'data/universe_ids.npy').astype(str)
        self.lookup = {j: i for i, j in enumerate(self.ids)}
        self.assignment = np.asarray([fold(j) for j in self.ids])
        self.hidden = {}
        self.prediction = {}
        self.stats = {}
        self.trained = {}
        self.labels = {}
        self.roles = {}
        self.dims = {}
        self.public_overrides = set(PUBLIC_TASKS)
        for r in self.records:
            task = r['id']
            m = manifest(task)
            assert 'role_ids' in m, 'Refresh manifests before Stage 2'
            self.roles[task] = m['role_ids']
            self.labels[task] = {role: json.loads((SOURCE_ROOT / 'data/tasks' / task / (role + '.json')).read_text()) for role in ['train']}
            y = np.asarray(list(self.labels[task]['train'].values()), dtype=np.float64)
            classification = r['metric'] == 'ACC'
            self.stats[task] = (0.0, 1.0) if classification else (float(y.mean()), float(max(y.std(), 1e-08)))
            self.trained[task] = {}
            self.hidden[task] = {}
            self.prediction[task] = {}
            for held in [-1, 0, 1, 2, 3, 4]:
                if held == -1 and task in self.public_overrides:
                    out = PUBLIC_ROOT / task
                    done = json.loads((out / 'complete.json').read_text())
                    assert done['status'] == 'complete' and done['rows'] == len(self.ids)
                    assert done['official_split_membership_and_order_exact'] is True
                    assert done['trained_or_finetuned'] is False
                    assert done['universe_ids_sha256'] == sha256(SOURCE_ROOT / 'data/universe_ids.npy')
                    assert done['prediction_sha256'] == sha256(out / 'prediction.npy')
                    assert done['hidden_sha256'] == sha256(out / 'hidden.npy')
                    assert all((not ids for ids in m['within_task_role_overlap'].values()))
                    trained = set(m['role_ids']['train'])
                else:
                    out = output_dir(task, held)
                    done = json.loads((out / 'complete.json').read_text())
                    cfg = json.loads((out / 'config.json').read_text())
                    assert done['rows'] == len(self.ids) and cfg['test_labels_read'] is False
                    validate_expert_protocol(task, cfg)
                    assert cfg['fold'] == held
                    checkpoints = [out / name for name in ['model.pt', 'model.weights.h5', 'model.joblib'] if (out / name).is_file()]
                    assert len(checkpoints) == 1 and sha256(checkpoints[0]) == done['checkpoint_sha256']
                    trained = set(cfg['trained_ids'])
                    expected = set(m['role_ids']['train']) if held < 0 else {j for j in m['role_ids']['train'] if fold(j) != held}
                    assert trained == expected
                    if held >= 0:
                        assert all((fold(j) != held for j in trained))
                self.trained[task][held] = trained
                self.hidden[task][held] = np.load(out / 'hidden.npy', mmap_mode='r')
                self.prediction[task][held] = np.load(out / 'prediction.npy', mmap_mode='r')
                assert self.hidden[task][held].shape == (len(self.ids), done['hidden_dim'])
                assert self.prediction[task][held].shape == (len(self.ids),)
            self.dims[task] = self.hidden[task][-1].shape[1]

    def anchor(self, task, jids, role):
        """Exact unstandardized anchor; never substitute full-fit train values."""
        assert role in ['train', 'val', 'test']
        rows = np.asarray([self.lookup[j] for j in jids])
        held = self.assignment[rows] if role == 'train' else np.full(len(rows), -1)
        result = np.empty(len(rows), dtype=np.float64)
        for f in np.unique(held):
            mask = held == f
            result[mask] = self.prediction[task][int(f)][rows[mask]]
        return result
