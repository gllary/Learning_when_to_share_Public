from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
from common import protocol_hash
from common import ROOT, atomic_json, registry, training_labels, validation_labels, output_dir, sha256
sys.path.insert(0, str(ROOT / 'src'))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--fold', type=int, default=-1)
    p.add_argument('--batch', type=int, default=64)
    a = p.parse_args()
    if a.batch != 64:
        raise RuntimeError('Official kgcnn batch size is fixed at 64')
    r = next((x for x in registry()['tasks'] if x['id'] == a.task))
    out = output_dir(a.task, a.fold)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('ALREADY_COMPLETE')
        return
    import tensorflow as tf
    devices = tf.config.list_physical_devices('GPU')
    assert len(devices) == 1, devices
    tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ.get('JARVIS_GPU_LIMIT_MB', '36000')))])
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.mixed_precision.set_global_policy('float32')
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    from kgcnn.literature.coGN import make_model, model_default_nested
    from kgcnn.training.schedule import KerasPolynomialDecaySchedule
    from kgcnn.data.transform.scaler.standard import StandardScaler
    from hpsafe_cogn_stage1.model import OFFICIAL_MODEL_CONFIG
    from hpsafe_cogn_stage1.graph_store import load_graphs_by_ids
    from hpsafe_cogn_stage1.tensors import input_tensors
    nested = r['model_id'] == 'kgcnn_coNGN'
    kind = 'congn' if nested else 'cogn'
    from cache_contract import verify
    receipt = json.loads((ROOT / 'data' / (kind + '_graphs.receipt.json')).read_text())
    verify(receipt, kind, [(ROOT / 'data' / (kind + '_graphs.h5'), receipt['sha256'])])
    model = make_model(**deepcopy(model_default_nested if nested else OFFICIAL_MODEL_CONFIG))
    candidates = [l for l in model.layers if l.__class__.__name__ == 'SequentialGraphNetwork']
    if not candidates:
        raise RuntimeError('No official final graph-network latent state')
    node = candidates[-1].output[1]
    hidden = node['features']
    mult = tf.cast(node['multiplicity'], hidden.dtype)
    latent = tf.reduce_sum(hidden * tf.expand_dims(mult, -1), axis=1) / tf.maximum(tf.reduce_sum(mult, axis=1, keepdims=True), 1e-08)
    latent_model = tf.keras.Model(model.inputs, latent)
    ys = training_labels(a.task, a.fold)
    ids = np.asarray(list(ys), dtype=str)
    y = np.asarray(list(ys.values()), dtype=np.float64)
    graphs = load_graphs_by_ids(ROOT / 'data' / (kind + '_graphs.h5'), ids)
    x = input_tensors(model, graphs)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(y.reshape(-1, 1))
    mean = float(scaler.mean_[0])
    scale = float(scaler.scale_[0])
    val = validation_labels(a.task)
    val_ids = list(val)
    val_y = np.asarray([val[j] for j in val_ids], dtype=np.float64)
    val_graphs = load_graphs_by_ids(ROOT / 'data' / (kind + '_graphs.h5'), val_ids)
    val_x = input_tensors(model, val_graphs)
    epochs = 600 if nested else 800
    lr = KerasPolynomialDecaySchedule(dataset_size=len(y), batch_size=a.batch, epochs=epochs, lr_start=0.0005, lr_stop=1e-05)
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=['mean_absolute_error'])
    config = {'task': a.task, 'model_id': r['model_id'], 'fold': a.fold, 'seed': seed, 'epochs': epochs, 'batch': a.batch, 'mean': mean, 'scale': scale, 'trained_ids': ids.tolist(), 'trained_id_sha256': __import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest(), 'latent_dim': int(latent_model.output_shape[-1]), 'test_labels_read': False, 'official_validation_labels_read': True, 'validation_ids': val_ids, 'protocol_sha256': protocol_hash(), 'packages': {n: importlib.metadata.version(n) for n in ['tensorflow', 'kgcnn', 'numpy', 'pymatgen', 'pyxtal', 'jarvis-tools']}}
    config.update(input_adapter_sha256=sha256(ROOT / 'src/hpsafe_cogn_stage1/tensors.py'), input_semantics='official_POSCAR_CIF_primitive_True_then_kgcnn_reverse_kj_v6', retrained_from_scratch=True, preprocessing_sha256=sha256(ROOT / 'official_preprocessing.py'), graph_cache_sha256=json.loads((ROOT / 'data' / (kind + '_graphs.receipt.json')).read_text())['sha256'])
    atomic_json(out / 'config.json', config)
    started = time.monotonic()

    class Progress(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            atomic_json(out / 'progress.json', {'epoch': int(epoch) + 1, 'epochs': epochs, 'seconds': time.monotonic() - started, 'train_mae': float((logs or {}).get('loss', float('nan'))) * scale, 'val_mae': None if 'val_loss' not in (logs or {}) else float(logs['val_loss']) * scale, 'status': 'training'})
    callbacks = [Progress()]
    callbacks.append(tf.keras.callbacks.BackupAndRestore(backup_dir=str(out / 'resume'), save_freq='epoch', delete_checkpoint=False))
    model.fit(x, scaled, batch_size=a.batch, epochs=epochs, shuffle=True, callbacks=callbacks, validation_data=(val_x, scaler.transform(val_y.reshape(-1, 1))), validation_freq=10, verbose=2)
    checkpoint = out / 'model.weights.h5'
    model.save_weights(checkpoint)
    val_prediction = scaler.inverse_transform(model.predict(val_x, batch_size=32, verbose=0)).reshape(-1)
    atomic_json(out / 'validation.json', {'metric': 'MAE', 'score': float(np.mean(np.abs(val_y - val_prediction))), 'selected_epoch': epochs, 'test_labels_read': False})
    universe = np.load(ROOT / 'data' / 'universe_ids.npy').astype(str)
    hidden_dim = int(latent_model.output_shape[-1])
    prediction = np.lib.format.open_memmap(out / 'prediction.npy', mode='w+', dtype='float32', shape=(len(universe),))
    hs = np.lib.format.open_memmap(out / 'hidden.npy', mode='w+', dtype='float32', shape=(len(universe), hidden_dim))
    for start in range(0, len(universe), 512):
        stop = min(start + 512, len(universe))
        gg = load_graphs_by_ids(ROOT / 'data' / (kind + '_graphs.h5'), universe[start:stop])
        xx = input_tensors(model, gg)
        prediction[start:stop] = scaler.inverse_transform(model.predict(xx, batch_size=32, verbose=0)).reshape(-1)
        hs[start:stop] = latent_model.predict(xx, batch_size=min(a.batch, 32), verbose=0)
    prediction.flush()
    hs.flush()
    if not np.isfinite(prediction).all() or not np.isfinite(hs).all():
        raise RuntimeError('Nonfinite expert export')
    atomic_json(out / 'complete.json', {'task': a.task, 'fold': a.fold, 'checkpoint_sha256': sha256(checkpoint), 'rows': len(universe), 'hidden_dim': hidden_dim, 'test_labels_read': False, 'seconds': time.monotonic() - started})
    print('EXPERT_COMPLETE', a.task, a.fold, flush=True)
if __name__ == '__main__':
    main()
