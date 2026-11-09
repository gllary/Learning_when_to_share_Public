"""Train one fresh inner-fit expert; never fit on its OOF prediction rows."""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from common import ROOT, PROJECT, RestrictedTargets, array_sha, write, sha
from hpsafe_sota.data.manifest import TaskManifest, _single_validation_positions

def glass_fit(manifest, fit, validation, predict, out, seed, extra):
    spec = importlib.util.spec_from_file_location('isolated_anchorboost', Path(__file__).with_name('anchorboost_oof.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    out.mkdir(parents=True, exist_ok=False)
    bridge = object.__new__(mod.AnchorBoostBridge)
    bridge.request = SimpleNamespace(project_path=PROJECT, task_name='glass', output_path=out, seed=seed, extra=extra)
    bridge.manifest = SimpleNamespace(targets=RestrictedTargets(manifest.targets, np.concatenate([fit, validation])))
    bridge.split = SimpleNamespace(train_positions=fit, validation_positions=validation, prediction_positions=predict)
    prepared = bridge.prepare()
    fitted = bridge.fit(prepared)
    prediction, uncertainty = bridge.predict(fitted, prepared)
    hidden, available = bridge.export_latent(fitted, prepared)
    import joblib, torch
    from hpsafe_sota.stage1_anchorboost.anchor_replay import predict_checkpoint_payload
    prep = joblib.load(out / 'feature_preprocessor.joblib')
    tree = prep['imputer'].transform(prepared.cache.rows(predict)).astype(np.float32)
    neural = prep['scaler'].transform(tree).astype(np.float32)
    reload = predict_checkpoint_payload(joblib.load(fitted.checkpoint), tree_features=tree, neural_features=neural, classification=True, device=torch.device('cuda'))
    deltas = []
    for a, b in zip((prediction, uncertainty, hidden), reload):
        assert np.allclose(a, b, rtol=1e-05, atol=1e-05)
        deltas.append(float(np.max(np.abs(a - b))))
    return (fitted.summary, (prediction, uncertainty, hidden, available))

def run(task, outer, inner):
    started = time.time()
    manifest = TaskManifest(PROJECT / 'data/manifests' / f'{task}.npz')
    manifest.require_valid()
    fit = manifest.positions('inner_train', outer, inner)
    held = manifest.positions('inner_val', outer, inner)
    test = manifest.positions('outer_test', outer)
    assert not set(fit) & set(held)
    assert not (set(fit) | set(held)) & set(test)
    out = ROOT / 'members' / task / f'outer_{outer}' / f'inner_{inner}'
    out.mkdir(parents=True, exist_ok=False)
    if task == 'glass':
        import torch
        torch.set_num_threads(4)
        assert torch.cuda.is_available()
        seed = 20260817 + outer * 1009 + inner
        definition = SimpleNamespace(task_type='classification')
        selection_val = _single_validation_positions(definition, RestrictedTargets(manifest.targets, fit), fit, fraction=0.1, seed=20260814 + outer * 1009 + inner)
        selected = set(selection_val.tolist())
        selection_fit = np.asarray([p for p in fit if p not in selected], dtype=np.int64)
        assert not (set(selection_val) | set(selection_fit)) & set(held)
        extra = {'ablation': 'full', 'trial_id': 'independent_inner_fit_selection'}
        summary, _ = glass_fit(manifest, selection_fit, selection_val, selection_val, out / 'selection_fit', seed, extra)
        extra.update(summary['selected_parameters'])
        summary, values = glass_fit(manifest, fit, np.empty(0, dtype=np.int64), held, out / 'fit', seed, extra)
        checkpoint = out / 'fit/anchorboost_v2.joblib'
    elif task == 'steels':
        import joblib
        from hpsafe_sota.stage1_tpot.features import load_feature_cache
        from hpsafe_sota.stage1_tpot.pipeline import fresh_official_pipeline, fit_pipeline, replay_pipeline
        cache = PROJECT / 'data/stage1_pool/archive/steels_features'
        x, ids, receipt = load_feature_cache(PROJECT, cache)
        seed = 18012019
        pipeline, updates = fresh_official_pipeline(PROJECT, seed=seed, threads=4)
        pipeline, recipe = fit_pipeline(pipeline, x[fit], RestrictedTargets(manifest.targets, fit)[fit])
        checkpoint = out / 'tpot_mat_pipeline.joblib'
        joblib.dump({'pipeline': pipeline, 'latent_recipe': recipe}, checkpoint, compress=3)
        result = replay_pipeline(pipeline, recipe, x[held])
        loaded = joblib.load(checkpoint)
        reload = replay_pipeline(loaded['pipeline'], loaded['latent_recipe'], x[held])
        for key in ['prediction', 'uncertainty', 'hidden']:
            assert np.allclose(getattr(result, key), getattr(reload, key), rtol=1e-14, atol=1e-12), key
        values = (result.prediction, result.uncertainty, result.hidden, np.ones(len(held), dtype=bool))
        write(out / 'preprocessing_manifest.json', {'fit_positions': fit.tolist(), 'seed': seed, 'feature_receipt': receipt, 'official_fitted_state_reused': False})
    else:
        raise ValueError(task)
    prediction, uncertainty, hidden, available = values
    assert prediction.shape == (len(held),)
    assert hidden.shape == (len(held), 512 if task == 'glass' else 103)
    assert all((np.all(np.isfinite(a)) for a in (prediction, uncertainty, hidden)))
    assert available.all()
    np.savez_compressed(out / 'oof.npz', sample_ids=manifest.sample_ids[held], positions=held, prediction=prediction, uncertainty=uncertainty, hidden=hidden, available=available)
    receipt = {'task': task, 'outer': outer, 'inner': inner, 'seed': seed, 'fit_positions': fit.tolist(), 'held_positions': held.tolist(), 'fit_positions_sha256': array_sha(fit), 'held_positions_sha256': array_sha(held), 'manifest_digest': manifest.digest, 'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha(checkpoint), 'oof_sha256': sha(out / 'oof.npz'), 'rows': len(held), 'elapsed_seconds': time.time() - started, 'outer_test_used_for_fit_or_selection': False}
    write(out / 'complete.json', receipt)
    print(receipt, flush=True)
if __name__ == '__main__':
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['glass', 'steels'])
    p.add_argument('outer', type=int)
    p.add_argument('inner', type=int)
    a = p.parse_args()
    run(a.task, a.outer, a.inner)
