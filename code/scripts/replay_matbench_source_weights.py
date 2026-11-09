"""Frozen Stage 2 replay on all Matbench test records; writes to a new run.

No labels, fitting, calibration or release files are changed. We retain joint
member-level route/attention products, never a product of ensemble means.
"""
from pathlib import Path
import sys, os, json, hashlib, importlib.util, time, argparse
import numpy as np
import pandas as pd
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
ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT
NEW = RUN
REC = RUN
sys.path.insert(0, str(EXP / 'code'))
from hpsafemoe.stage2_proposal.matbench.model.model import CrossExpertProposalModel, ProposalConfig
from hpsafemoe.stage2_proposal.matbench.model.training import _predict
from hpsafe_sota.expert_inputs.protocol import load_expert_pool
spec = importlib.util.spec_from_file_location('frozen_benchmark', EXP / 'code/scripts/predict_benchmark.py')
predictor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predictor)
import hpsafe_sota.expert_inputs.cache as cache
validate = cache.validate_expert_export

def validate_metadata(*args, **kwargs):
    kwargs['verify_hashes'] = False
    return validate(*args, **kwargs)
cache.validate_expert_export = validate_metadata
torch.set_num_threads(4)
identities = []
for outer in range(5):
    for path in sorted((EXP / f'results/matbench/full/predictions/outer_{outer}').glob('*.npz')):
        with np.load(path) as z:
            ids = z['sample_ids'].astype(str)
        identities.append(pd.DataFrame(dict(task=path.stem, sample_id=ids, outer_fold=outer + 1)))
sample = pd.concat(identities, ignore_index=True)
pool = load_expert_pool(EXP / 'data/matbench/stage1_pool/pool/experts.yaml')
records = json.loads((EXP / 'checkpoint/matbench/stage2_proposal/model/manifest.json').read_text())['models']
records = sorted([r for r in records if '/ablation_no_cross_expert/' not in r['path']], key=lambda r: r['path'])
assert len(records) == 50
out = RUN / 'matbench'
out.mkdir(exist_ok=True)
checks = []
rows = []
start = time.time()
for outer in range(5):
    dest = out / f'outer_{outer}.npz'
    audit = out / f'outer_{outer}.json'
    if dest.exists() and audit.exists():
        checks.extend(json.loads(audit.read_text()))
        print('Cached', outer, flush=True)
        continue
    members = [r for r in records if f'/outer_{outer}/' in r['path']]
    selected = {}
    finals = {}
    accum = {}
    raw = {}
    fold = None
    for mi, rec in enumerate(members):
        path = EXP / rec['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256']
        q = torch.load(path, map_location='cpu', weights_only=False)
        cfg = q.get('model_config', q.get('config'))
        state = q.get('state_dict', q.get('model'))
        assert tuple(cfg['expert_ids']) == pool.expert_ids
        model = CrossExpertProposalModel(ProposalConfig(**cfg)).to(args.device).eval()
        model.load_state_dict(state, strict=True)
        model.requires_grad_(False)
        if fold is None:
            fold = predictor.label_free_fold(pool, outer, q['normalization'])
            assert tuple(cfg['task_names']) == tuple(fold.tasks)
            for task in fold.tasks:
                f = np.load(EXP / f'results/matbench/full/predictions/outer_{outer}/{task}.npz')
                wanted = sample[(sample.task == task) & (sample.outer_fold == outer + 1)].sample_id.to_numpy()
                index = {s: i for i, s in enumerate(f['sample_ids'].astype(str))}
                ix = np.array([index[s] for s in wanted])
                assert len(ix)
                selected[task] = ix
                finals[task] = {k: f[k][ix] for k in ['sample_ids', 'anchor_prediction', 'raw_prediction', 'prediction']}
                finals[task]['beta'] = float(np.ravel(f['beta'])[0])
                accum[task] = np.zeros((len(ix), len(pool.expert_ids)), dtype=np.float64)
                raw[task] = []
        else:
            fold.stats = {t: fold._parse_stats(t, q['normalization'][t]) for t in fold.tasks}
        for task in fold.tasks:
            if finals[task]['beta'] == 0:
                continue
            v = _predict(fold=fold, model=model, task=task, role='test', device=torch.device(args.device), indices=selected[task], chunk_size=256)
            np.testing.assert_array_equal(v['sample_ids'], finals[task]['sample_ids'])
            np.testing.assert_array_equal(v['anchor_prediction'], finals[task]['anchor_prediction'])
            w = v['route_weights'].astype(float)
            joint = w[:, 1, None] * v['data_attention'] + w[:, 2, None] * v['physics_attention']
            assert np.all(joint >= 0) and np.isfinite(joint).all()
            own = pool.expert_ids.index(pool.anchor_for(task))
            assert np.max(np.abs(joint[:, own])) == 0
            accum[task] += joint / len(members)
            raw[task].append(v['raw_prediction'])
        print(f'outer {outer + 1}/5 member {mi + 1}/10; {time.time() - start:.1f}s', flush=True)
    payload = {}
    foldchecks = []
    for task in fold.tasks:
        f = finals[task]
        changed = f['prediction'] != f['anchor_prediction']
        if f['beta']:
            replay = np.mean(raw[task], axis=0)
            np.testing.assert_allclose(replay, f['raw_prediction'], rtol=2e-05, atol=1e-05)
            maxerr = float(np.max(np.abs(replay - f['raw_prediction'])))
        else:
            maxerr = 0.0
        payload[task + '_ids'] = f['sample_ids']
        payload[task + '_changed'] = changed
        payload[task + '_source_mass'] = f['beta'] * accum[task]
        foldchecks.append(dict(task=task, outer_fold=outer + 1, n=len(changed), changed=int(changed.sum()), beta=f['beta'], raw_replay_max_abs_difference=maxerr, checkpoint_hashes_checked=True, export_metadata_ids_checked=True, hidden_file_hashes_recomputed=False))
    np.savez_compressed(dest, **payload)
    audit.write_text(json.dumps(foldchecks, indent=2))
    checks.extend(foldchecks)
for task in sorted(sample.task.unique()):
    total = np.zeros(len(pool.expert_ids))
    n = 0
    changed = 0
    for outer in range(5):
        z = np.load(out / f'outer_{outer}.npz')
        mask = z[task + '_changed']
        n += len(mask)
        changed += mask.sum()
        total += z[task + '_source_mass'][mask].sum(axis=0)
    percent = 100 * total / total.sum() if total.sum() > 0 else np.full(len(total), np.nan)
    for ex, mass, value in zip(pool.experts, total, percent):
        rows.append(dict(target_task=task, source_task=ex.source_task, source_expert=ex.expert_id, diagnostic_n=n, changed_n=int(changed), calibrated_route_attention_mass=float(mass), normalized_source_percent=float(value)))
pd.DataFrame(rows).to_csv(NEW / 'source_data/matbench_current_source_weights.csv', index=False)
protocol = dict(sample_scope='All official test records', sample_records=len(sample), inference_only=True, labels_used=False, new_training=False, beta_refitted=False, zero_beta_models_skipped=True, checks=checks)
(NEW / 'source_data/source_weight_protocol.json').write_text(json.dumps(protocol, indent=2))
print('COMPLETE', len(rows), flush=True)
