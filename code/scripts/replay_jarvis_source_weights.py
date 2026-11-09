"""Frozen full-test JARVIS source diagnostics; no labels, training or release writes."""
from pathlib import Path
import sys, os, json, hashlib, time
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--device', default='cpu')
args = parser.parse_args()
RELEASE = Path(__file__).resolve().parents[2]
RUN = args.output.resolve()
if not RUN.is_relative_to(RELEASE / 'runs'):
    parser.error('Output must be under runs/')
RUN.mkdir(parents=True, exist_ok=False)
(RUN / 'source_data').mkdir()
import torch
ROOT = RELEASE
EXP = ROOT
NEW = RUN
REC = RUN
OUT = RUN / 'jarvis'
OUT.mkdir(exist_ok=True)
os.environ['HPSAFEMOE_RUN_DIR'] = str(OUT)
sys.path.insert(0, str(EXP / 'code'))
from hpsafemoe.stage2_proposal.jarvis.model.input_statistics import Inputs
from hpsafemoe.stage2_proposal.jarvis.model.predictor import predict
from hpsafemoe.stage2_proposal.jarvis.model.model import CrossExpertProposalModel, ProposalConfig
from hpsafemoe.stage2_proposal.jarvis.common import SOURCE_ROOT, PUBLIC_ROOT, PUBLIC_TASKS, registry, manifest, output_dir, require_zero_u
torch.set_num_threads(4)
records = json.loads((EXP / 'checkpoint/jarvis/stage2_proposal/model/manifest.json').read_text())['models']
records = sorted(records, key=lambda r: r['path'])
assert len(records) == 10
(OUT / 'stage2').mkdir(exist_ok=True)
(OUT / 'stage2/selection_locked.json').write_text(json.dumps(dict(frozen_weights=records, refitted=False, test_labels_read=False), indent=2))
inputs = object.__new__(Inputs)
inputs.records = registry()['tasks']
inputs.tasks = tuple((r['id'] for r in inputs.records))
inputs.ids = np.load(SOURCE_ROOT / 'data/universe_ids.npy').astype(str)
inputs.lookup = {j: i for i, j in enumerate(inputs.ids)}
inputs.assignment = np.full(len(inputs.ids), -1)
inputs.kinds = {r['id']: 'classification' if r['metric'] == 'ACC' else 'regression' for r in inputs.records}
inputs.roles = {}
inputs.hidden = {}
inputs.prediction = {}
inputs.uncertainty = {}
inputs.availability = {}
inputs.dims = {}
finals = {}
start = time.time()
for task in inputs.tasks:
    inputs.roles[task] = {'test': manifest(task)['role_ids']['test']}
    path = PUBLIC_ROOT / task if task in PUBLIC_TASKS else output_dir(task, -1)
    require_zero_u(path / 'uncertainty.npy', path / 'available.npy')
    h = np.load(path / 'hidden.npy', mmap_mode='r')
    p = np.load(path / 'prediction.npy', mmap_mode='r')
    done = json.loads((path / 'complete.json').read_text())
    assert len(p) == len(h) == len(inputs.ids) == done['rows']
    a = np.isfinite(p) & np.isfinite(h).all(axis=1)
    if inputs.kinds[task] == 'classification':
        a &= (p >= 0) & (p <= 1)
    inputs.hidden[task] = {-1: h}
    inputs.prediction[task] = {-1: p}
    inputs.uncertainty[task] = {-1: np.zeros(len(p), dtype=np.float32)}
    inputs.availability[task] = {-1: a}
    inputs.dims[task] = h.shape[1]
    z = np.load(EXP / f'results/jarvis/full/predictions/{task}.npz')
    finals[task] = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(finals[task]['sample_ids'], inputs.roles[task]['test'])
print('Full-test inputs ready', sum((len(f['sample_ids']) for f in finals.values())), 'records', flush=True)
for mi, rec in enumerate(records):
    dest = OUT / f'member_{mi}.npz'
    if dest.exists():
        print('Cached', mi, flush=True)
        continue
    path = EXP / rec['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256']
    q = torch.load(path, map_location='cpu', weights_only=False)
    cfg = q.get('model_config', q.get('config'))
    state = q.get('state_dict', q.get('model'))
    assert tuple(cfg['expert_ids']) == inputs.tasks
    model = CrossExpertProposalModel(ProposalConfig(**cfg)).to(args.device).eval()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    norm = q['normalization']
    inputs.inner = q['inner']
    inputs.stats = {t: tuple(v) for t, v in norm['label_stats'].items()}
    inputs.token_stats = norm['token_stats']
    inputs.uncertainty_stats = norm['uncertainty_stats']
    payload = {}
    for task in inputs.tasks:
        f = finals[task]
        beta = float(np.ravel(f['beta'])[0])
        if not beta:
            continue
        v = predict(inputs, model, task, 'test', args.device)
        np.testing.assert_array_equal(v['sample_ids'], f['sample_ids'])
        np.testing.assert_array_equal(v['anchor_prediction'], f['anchor_prediction'])
        w = v['route_weights'].astype(float)
        joint = w[:, 1, None] * v['data_attention'] + w[:, 2, None] * v['physics_attention']
        assert np.isfinite(joint).all() and np.all(joint >= 0) and (np.max(np.abs(joint[:, inputs.tasks.index(task)])) == 0)
        changed = f['prediction'] != f['anchor_prediction']
        payload[task + '_source_sum'] = beta * joint[changed].sum(axis=0)
        payload[task + '_raw'] = v['raw_prediction']
    np.savez_compressed(dest, **payload)
    print(f'Member {mi + 1}/10; {time.time() - start:.1f}s', flush=True)
rows = []
checks = []
for task in inputs.tasks:
    f = finals[task]
    beta = float(np.ravel(f['beta'])[0])
    total = np.zeros(len(inputs.tasks))
    raw = []
    if beta:
        for mi in range(10):
            with np.load(OUT / f'member_{mi}.npz') as z:
                total += z[task + '_source_sum'] / 10
                raw.append(z[task + '_raw'])
        replay = np.mean(raw, axis=0)
        np.testing.assert_allclose(replay, f['raw_prediction'], rtol=2e-05, atol=1e-05)
        maxerr = float(np.max(np.abs(replay - f['raw_prediction'])))
    else:
        maxerr = 0.0
    pc = 100 * total / total.sum() if total.sum() > 0 else np.full(len(total), np.nan)
    n = len(f['sample_ids'])
    changed = int(np.count_nonzero(f['prediction'] != f['anchor_prediction']))
    checks.append(dict(task=task, n=n, changed_n=changed, beta=beta, raw_replay_max_abs_difference=maxerr))
    for source, mass, value in zip(inputs.tasks, total, pc):
        rows.append(dict(target_task=task, source_task=source, diagnostic_n=n, changed_n=changed, calibrated_route_attention_mass=float(mass), normalized_source_percent=float(value)))
pd.DataFrame(rows).to_csv(NEW / 'source_data/jarvis_current_source_weights.csv', index=False)
protocol = dict(inference_only=True, labels_used=False, new_training=False, beta_refitted=False, sample_records=sum((len(f['sample_ids']) for f in finals.values())), checkpoint_hashes_checked=True, ids_and_anchors_exact=True, hidden_file_hashes_recomputed=False, checks=checks)
(NEW / 'source_data/jarvis_source_weight_protocol.json').write_text(json.dumps(protocol, indent=2))
print('COMPLETE', len(rows), flush=True)
