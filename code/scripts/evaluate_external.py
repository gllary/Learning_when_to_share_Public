"""Verify full-cohort metrics from committed frozen predictions; read-only."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from hpsafemoe.external.evaluation import COUNTS, regression_metrics, sha256, validate_prediction

COHORTS = ('core23', 'ds2_248', 'hea95', 'b2_18')
MODE = 'frozen_external_validation'

def read(path):return json.loads(path.read_text())
def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),(a.keys(),b.keys())
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,float):assert np.isclose(a,b,rtol=1e-13,atol=1e-14),(a,b)
    else:assert a==b,(a,b)

def validate_contract(cfg):
    required = {
        'evaluation_mode': MODE, 'external_calibration_samples': 0,
        'labels_used_for': 'evaluation_only', 'model_training': False,
        'beta_refitting': False, 'route_temperature': .75,
        'uncertainty': 'frozen_stage1_inputs',
    }
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f'Invalid frozen evaluation contract: {key}')
    n = COUNTS[cfg['dataset']]
    if cfg['n'] != n or cfg['evaluation_samples'] != n:
        raise ValueError('All cohort samples must be evaluated')
    if len(cfg['sample_ids']) != n or len(set(cfg['sample_ids'])) != n:
        raise ValueError('Sample count or identity mismatch')


def evaluate(slug):
    if slug not in COHORTS:
        raise ValueError('Unknown cohort')
    base = ROOT / 'results/external' / slug
    cfg = read(base / 'config.json')
    validate_contract(cfg)
    path = base / 'predictions/final/predictions.npz'
    validate_prediction(path, cfg['sample_ids'])
    with np.load(path, allow_pickle=False) as z:
        ids = z['sample_ids'].astype(str).tolist()
        pred = {k: z[k].copy() for k in ('stage1', 'stage2')}
    source = read(base / 'data/source_manifest.json')
    assert source['label_access'] == 'evaluation only'
    assert sha256(base / 'data/labels.csv') == source['labels_sha256']
    assert sha256(base / 'data/materials.csv') == source['materials_sha256']
    with (base / 'data/labels.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert [r['sample_id'] for r in rows] == ids == source['sample_id_order']
    key = 'youngs_modulus' if cfg['dataset'] == 'B2-18' else 'target'
    y = np.asarray([float(r[key]) for r in rows])
    metrics = {k: regression_metrics(p, y) for k, p in pred.items()}
    expected = read(base / 'results/metrics.json')
    assert expected['evaluation_mode'] == MODE and expected['external_calibration_samples'] == 0
    assert expected['n'] == expected['evaluation_samples'] == len(y)
    close(metrics, expected['external_validation_metrics'])
    with (base / 'results/per_sample_frozen.csv').open(newline='') as f:
        samples = list(csv.DictReader(f))
    assert [r['sample_id'] for r in samples] == ids
    np.testing.assert_array_equal([float(r['target']) for r in samples], y)
    for method, values in pred.items():
        np.testing.assert_array_equal([float(r[method]) for r in samples], values)
    assert all(r['unit'] == cfg['evaluation_unit'] for r in samples)
    table = read(ROOT / 'results/tables/external.json')
    assert table['evaluation_mode'] == MODE
    close(expected, table['datasets'][cfg['dataset']])
    model_refs = read(base / 'model_refs.json')
    beta_count = 0
    for outer, tasks in model_refs['frozen_beta'].items():
        for task, ref in tasks.items():
            certificate = ROOT / ref['file']
            assert sha256(certificate) == ref['sha256']
            value = read(certificate)
            assert value == read(ROOT / f'results/matbench/full/calibration/outer_{outer}/{task}.json')
            assert ref['beta'] == value['beta']
            beta_count += 1
    return {'dataset': cfg['dataset'], 'n': len(y), 'unit': cfg['evaluation_unit'],
            'evaluation_mode': MODE, 'external_calibration_samples': 0,
            'metrics': metrics, 'matches_published_result': True,
            'sample_ids_and_arrays_exact': True,
            'frozen_beta_references_verified': beta_count}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['all','core23','ds2_248','hea95','b2_18']);args=p.parse_args()
    selected=['core23','ds2_248','hea95','b2_18'] if args.dataset=='all' else [args.dataset]
    print(json.dumps([evaluate(s) for s in selected],indent=2))
