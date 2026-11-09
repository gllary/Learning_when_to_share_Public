import json

from pathlib import Path

import numpy as np

NEW_PROVIDERS = {
    'glass':'anchorboost_v2_own_task_inner_oof_20261028',
    'steels':'tpot_mat_own_task_inner_oof_20261028',
}

def check_records(producer, manifest, outer):
    assert producer.get('checkpoint_role')=='inner_oof'
    assert producer.get('checkpoint_fit_role')=='inner_train_crossfit'
    assert producer.get('inner_validation_partition_exact') is True
    assert producer.get('inner_fold_count')==5
    assert producer.get('oof_scope')=='source-own-task'
    assert producer.get('new_test_selection') is False
    records=producer.get('inner_checkpoint_records',[])
    assert len(records)==5 and {r['inner'] for r in records}==set(range(5))
    all_held=[]
    for r in records:
        i=r['inner'];fit=manifest.positions('inner_train',outer,i);held=manifest.positions('inner_val',outer,i)
        assert r['outer']==outer and r['task']==manifest.task_name
        assert r['manifest_digest']==manifest.digest
        assert r['outer_test_used_for_fit_or_selection'] is False
        assert np.array_equal(r['fit_positions'],fit) and np.array_equal(r['held_positions'],held)
        assert not np.intersect1d(fit,held).size
        assert np.array_equal(np.sort(np.concatenate([fit,held])),np.sort(manifest.positions('outer_train',outer)))
        all_held.extend(held.tolist())
        p=Path(r['checkpoint']);assert not p.is_absolute() and '..' not in p.parts
        assert len(r['checkpoint_sha256'])==64
    assert len(set(all_held))==len(all_held)
    assert np.array_equal(np.sort(all_held),np.sort(manifest.positions('outer_train',outer)))

def validate_provider(metadata, manifest, spec, outer, role, path):
    if spec.provider not in NEW_PROVIDERS.values():return
    assert NEW_PROVIDERS.get(spec.source_task)==spec.provider
    assert Path(path).resolve().is_relative_to(STAGE1/'exports')
    producer=metadata['producer']
    if spec.source_task==manifest.task_name and role in ('train','val'):
        check_records(producer,manifest,outer)
    else:
        assert producer['checkpoint_role']=='outer'
        assert producer['target_values_accessed_by_exporter'] is False

STAGE1 = Path(__file__).resolve().parents[3]/'data/matbench/stage1_pool'
