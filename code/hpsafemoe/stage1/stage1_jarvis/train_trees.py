import argparse
import json
import math
import os
from pathlib import Path
import time
import joblib
import importlib.metadata
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from common import protocol_hash
from common import ROOT, atomic_json, registry, training_labels, validation_labels, output_dir, sha256

def hashed_leaves(leaves):
    leaves = np.asarray(leaves, dtype=np.uint64).reshape(len(leaves), -1)
    tree = np.arange(leaves.shape[1], dtype=np.uint64)
    hashed = (leaves + 1) * np.uint64(2654435761) + (tree + 1) * np.uint64(2246822519)
    cols = (hashed % 128).astype(np.int64)
    sign = np.where(hashed // 128 % 2, 1.0, -1.0).astype('float32')
    out = np.zeros((len(leaves), 128), dtype='float32')
    np.add.at(out, (np.repeat(np.arange(len(leaves)), leaves.shape[1]), cols.ravel()), sign.ravel() / math.sqrt(leaves.shape[1]))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--fold', type=int, default=-1)
    a = p.parse_args()
    r = next((x for x in registry()['tasks'] if x['id'] == a.task))
    out = output_dir(a.task, a.fold)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    ids = np.load(ROOT / 'data' / 'universe_ids.npy').astype(str)
    lookup = {j: i for i, j in enumerate(ids)}
    labels = training_labels(a.task, a.fold)
    train_ids = list(labels)
    positions = [lookup[j] for j in train_ids]
    y = np.asarray(list(labels.values()))
    raw = np.load(ROOT / 'data' / 'descriptors.npy', mmap_mode='r')
    assert raw.dtype == np.float64, 'Stage1 requires rebuilt official float64 descriptors'
    receipt = json.loads((ROOT / 'data/descriptors.receipt.json').read_text())
    assert receipt['protocol_sha256'] == protocol_hash()
    from cache_contract import verify
    verify(receipt, 'descriptors', [(ROOT / 'data/descriptors.npy', receipt['sha256'])])
    classification = r['metric'] == 'ACC'
    threads = int(os.environ.get('OMP_NUM_THREADS', '4'))
    from tree_reference import official_pipeline
    pipeline = official_pipeline(r['model_id'], classification)
    imputer = pipeline.named_steps['imputer']
    scaler = pipeline.named_steps['scaler']
    model = pipeline.named_steps['model']
    X = scaler.fit_transform(imputer.fit_transform(raw[positions]))
    model.set_params(n_jobs=threads)
    if r['model_id'] != 'matminer_rf':
        import xgboost as xgb
        assert xgb.__version__ == '1.7.6', 'Do not silently substitute a different gpu_hist implementation'
    started = time.monotonic()
    if r['model_id'] == 'matminer_xgboost':

        class Progress(xgb.callback.TrainingCallback):

            def after_iteration(self, booster, epoch, evals_log):
                if epoch < 5 or (epoch + 1) % 100 == 0:
                    atomic_json(out / 'progress.json', {'epoch': epoch + 1, 'epochs': 10000, 'unit': 'boosting_round', 'seconds': time.monotonic() - started, 'status': 'training'})
                return False
        model.fit(X, y, callbacks=[Progress()])
    else:
        model.fit(X, y)
    val = validation_labels(a.task)
    vx = scaler.transform(imputer.transform(raw[[lookup[j] for j in val]]))
    vy = np.asarray(list(val.values()))
    vp = model.predict(vx)
    atomic_json(out / 'validation.json', {'score': float(np.mean(vp == vy) if classification else np.mean(np.abs(vp - vy))), 'metric': r['metric'], 'checkpoint_rule': 'published fixed iterations', 'test_labels_read': False})
    checkpoint = out / 'model.joblib'
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'model': model}, checkpoint, compress=3)
    atomic_json(out / 'config.json', {'task': a.task, 'model_id': r['model_id'], 'fold': a.fold, 'trained_ids': train_ids, 'params': model.get_params(), 'test_labels_read': False, 'official_validation_labels_read': True, 'checkpoint_rule': 'published fixed iterations', 'protocol_sha256': protocol_hash(), 'retrained_from_scratch': True, 'descriptor_sha256': receipt['sha256'], 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scikit-learn', 'xgboost']}})
    pred = np.lib.format.open_memmap(out / 'prediction.npy', mode='w+', dtype='float64', shape=(len(ids),))
    native = np.lib.format.open_memmap(out / 'native_prediction.npy', mode='w+', dtype='float64', shape=(len(ids),))
    hidden = np.lib.format.open_memmap(out / 'hidden.npy', mode='w+', dtype='float32', shape=(len(ids), 128))
    for start in range(0, len(ids), 256):
        stop = min(start + 256, len(ids))
        xx = scaler.transform(imputer.transform(raw[start:stop]))
        pred[start:stop] = model.predict_proba(xx)[:, 1] if classification else model.predict(xx)
        native[start:stop] = model.predict(xx)
        if classification:
            assert np.array_equal(pred[start:stop] > 0.5, native[start:stop]), 'Native class labels and deployment threshold differ'
        if r['model_id'] == 'matminer_rf':
            leaves = model.apply(xx)
        else:
            import xgboost as xgb
            leaves = model.get_booster().predict(xgb.DMatrix(xx), pred_leaf=True, iteration_range=(0, 128), strict_shape=True)
        hidden[start:stop] = hashed_leaves(leaves)
    pred.flush()
    hidden.flush()
    native.flush()
    assert np.isfinite(pred).all() and np.isfinite(hidden).all()
    atomic_json(out / 'complete.json', {'task': a.task, 'fold': a.fold, 'rows': len(ids), 'hidden_dim': 128, 'checkpoint_sha256': sha256(checkpoint), 'seconds': time.monotonic() - started, 'test_labels_read': False})
    print('EXPERT_COMPLETE', a.task, a.fold, flush=True)
if __name__ == '__main__':
    main()
