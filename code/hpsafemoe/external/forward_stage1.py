"""Fresh PRIMARY-only frozen Stage1 forwards. No old numeric export is an input."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from evaluation import sha256, write_json_new

if 'HPSAFEMOE_EXTERNAL_WORKSPACE' not in os.environ:
    raise RuntimeError('Prepare a new runs/ workspace with code/scripts/prepare_fresh_stage1.py first')
ROOT = Path(os.environ['HPSAFEMOE_EXTERNAL_WORKSPACE']).resolve()
SOURCES = json.loads((ROOT / 'stage1_sources.json').read_text())
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
sys.path.insert(0, SOURCES['project'] + '/code')
from frozen_forward_helpers import ProductionReplayCache, _materialize_external_jmp_cache, _predict_external_jmp


def material_rows(dataset):
    with (ROOT / 'inputs' / dataset / 'materials.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames) == {'sample_id', 'composition', 'structure_json'}
        rows = list(reader)
    assert [r['sample_id'] for r in rows] == PROTOCOL['materials'][dataset]['sample_ids']
    return rows


def check_source(job):
    paths = [job['checkpoint']] + [job[k] for k in ('preprocessing', 'preprocessor') if k in job]
    for p in paths:
        assert sha256(p) == SOURCES['protected_files'][p], f'Frozen source changed: {p}'


def export(dataset, job, output, detail):
    ids = PROTOCOL['materials'][dataset]['sample_ids']
    n = len(ids)
    for key in ('prediction', 'uncertainty', 'available'):
        assert np.asarray(output[key]).shape == (n,), key
    assert np.asarray(output['hidden']).shape == (n, job['hidden_dim'])
    for key in ('prediction', 'uncertainty', 'hidden'):
        assert np.isfinite(output[key]).all(), key
    assert (output['uncertainty'] >= 0).all()
    assert np.asarray(output['available']).dtype == np.bool_
    dest = ROOT / 'stage1_exports' / dataset / f"outer_{job['outer']}" / job['expert_id']
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for key, value in dict(sample_ids=np.asarray(ids), **{k: output[k] for k in ('hidden', 'prediction', 'uncertainty', 'available')}).items():
        path = dest / f'{key}.npy'
        with path.open('xb') as f:
            np.save(f, value, allow_pickle=False)
        files[path.name] = sha256(path)
    check_source(job)
    write_json_new(dest / 'receipt.json', dict(time=time.time(), dataset=dataset, **job, files=files,
        available_rows=int(output['available'].sum()), labels_accessed=False, weights_updated=False,
        fresh_forward=True, old_external_numeric_exports_used=False, detail=detail))
    print(json.dumps(dict(event='stage1_export_complete', dataset=dataset,
                         expert=job['expert_id'], outer=job['outer'], available=int(output['available'].sum()))), flush=True)


def mattervial(outer):
    import tensorflow as tf
    from hpsafe_sota.expert_inputs.mattervial_replay import load_ensemble_checkpoint, replay_loaded_ensemble
    gpu = tf.config.list_physical_devices('GPU')
    assert len(gpu) == 1
    tf.config.experimental.set_memory_growth(gpu[0], True)
    caches = {}
    for dataset in PROTOCOL['datasets']:
        source = ROOT / 'data_features' / dataset
        cache = ProductionReplayCache(source / 'production_features',
            matminer_overlay_root=source / 'frozen_matminer_overlay', sisso_overlay_root=source / 'frozen_sisso_overlay')
        ids = cache.sample_ids.astype(str).tolist()
        assert len(set(ids)) == len(ids)
        rows = np.asarray([ids.index(i) for i in PROTOCOL['materials'][dataset]['sample_ids']], dtype=np.int64)
        caches[dataset] = (cache, rows)
    for job in SOURCES['jobs']:
        if job['kind'] != 'mattervial' or job['outer'] != outer:
            continue
        check_source(job)
        ensemble = load_ensemble_checkpoint(Path(job['checkpoint']))
        members = list(ensemble.models)
        for dataset, (cache, rows) in caches.items():
            contracts = [tuple(str(v) for v in m.optimal_descriptors[:m.n_feat]) for m in members]
            missing = [i for i, names in enumerate(contracts) if cache.member_matrix(names, row_positions=rows) is None]
            if missing:
                # The only predeclared technical inability is structural providers on composition-only DS2.
                assert dataset == 'DS2-248' and job['source_task'] in ('dielectric','log_kvrh','log_gvrh','mp_is_metal')
                n = len(rows)
                output = dict(hidden=np.zeros((n,job['hidden_dim']),np.float32), prediction=np.zeros(n,np.float32),
                              uncertainty=np.zeros(n,np.float32), available=np.zeros(n,bool))
                detail = dict(used_members=0,total_members=len(members),missing_member_contracts=missing,
                              reason='required_structural_descriptors_unavailable_for_composition_only_input',
                              partial_ensemble_used=False)
            else:
                output = replay_loaded_ensemble(ensemble, cache, batch_size=256, row_positions=rows)
                assert output['used_members'] == output['total_members'] == len(members)
                assert not output['missing_members'] and output['available'].all()
                detail = dict(used_members=len(members), total_members=len(members),
                              uncertainty='frozen_member_prediction_std_ddof0', partial_ensemble_used=False)
            export(dataset, job, output, detail)
        del ensemble
        tf.keras.backend.clear_session()


def glass(outer):
    import joblib
    import torch
    from hpsafe_sota.enriched_features.features import enriched_descriptor
    from hpsafe_sota.stage1_anchorboost.anchor_replay import predict_checkpoint_payload
    torch.set_num_threads(4)
    job = next(j for j in SOURCES['jobs'] if j['outer']==outer and j['kind']=='glass')
    check_source(job)
    rows = material_rows('DS2-248')
    features = np.stack([enriched_descriptor(r['composition']) for r in rows]).astype(np.float32)
    assert features.shape == (248,330) and np.isfinite(features).all()
    pp = joblib.load(job['preprocessor'])
    for key in ('imputer','scaler'):
        assert pp[key].n_features_in_ == 330
    tree = pp['imputer'].transform(features).astype(np.float32)
    neural = pp['scaler'].transform(tree).astype(np.float32)
    p,u,h = predict_checkpoint_payload(joblib.load(job['checkpoint']), tree_features=tree,
        neural_features=neural, classification=True, batch_size=256, device=torch.device('cuda:0'))
    export('DS2-248',job,dict(prediction=p.astype(np.float32),uncertainty=u.astype(np.float32),
        hidden=h.astype(np.float32),available=np.ones(len(rows),bool)),
        dict(frozen_preprocessor=True, descriptor_width=330, classification_probability=True))


def jmp(outer, task):
    import torch
    torch.set_num_threads(4)
    job = next(j for j in SOURCES['jobs'] if j['outer']==outer and j['kind']=='jmp' and j['source_task']==task)
    check_source(job)
    protocol, epochs = None, 500
    if task in ('mp_gap','mp_e_form'):
        from hpsafe_sota.experts.jmp_l_fp32_compat import FP32_MAX_EPOCHS, FP32_PROTOCOL, activate_fp32_protocol
        os.environ['TORCH_BLAS_PREFER_CUBLASLT']='0'
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32=True
        activate_fp32_protocol()
        protocol, epochs = FP32_PROTOCOL, FP32_MAX_EPOCHS
    datasets = ('Core-23','HEA-95','B2-18')
    rows = [r for dataset in datasets for r in material_rows(dataset)]
    ids = np.asarray([r['sample_id'] for r in rows])
    assert len(rows)==136 and len(set(ids))==136
    work = ROOT/'work'/f'jmp_{task}_outer_{outer}'
    work.mkdir(parents=True, exist_ok=False)
    cache = _materialize_external_jmp_cache(runtime_root=Path(SOURCES['jmp_runtime']), work_root=work,
        rows=rows, sample_ids=ids)
    prediction, hidden = _predict_external_jmp(runtime_root=Path(SOURCES['jmp_runtime']), source_task=task,
        outer_fold=outer, source_artifact=Path(job['checkpoint']).parent, checkpoint=Path(job['checkpoint']),
        cache_dir=cache, work_root=work, batch_size=1, num_workers=0,
        finetune_protocol=protocol, maximum_epochs=epochs)
    offset=0
    for dataset in datasets:
        n=len(PROTOCOL['materials'][dataset]['sample_ids']); sl=slice(offset,offset+n); offset+=n
        export(dataset,job,dict(prediction=prediction[sl],hidden=hidden[sl],uncertainty=np.zeros(n,np.float32),
            available=np.ones(n,bool)),dict(uncertainty='original_JMP_unavailable_zero_sentinel',
                forward_only=True, lmdb_target_values=False, inference_batch_size=1))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['mattervial','glass','jmp'])
    parser.add_argument('outer',type=int,choices=range(5));parser.add_argument('--task')
    args=parser.parse_args()
    if args.kind=='jmp':jmp(args.outer,args.task)
    else:globals()[args.kind](args.outer)
