"""Use the pinned official scalar ALIGNN training engine, not a custom loop."""
import argparse
import importlib.metadata
import json
import os
import time
import numpy as np
from common import protocol_hash
from common import ROOT, atomic_json, output_dir, training_labels, validation_labels, registry, sha256
from alignn_reference import bind, configuration, COMMIT

class LabeledGraphs:

    def __init__(self, store, labels):
        self.store = store
        self.ids = list(labels)
        self.labels = list(labels.values())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        g, lg = self.store.get(self.ids[i])
        return (g, lg, torch.tensor(self.labels[i], dtype=torch.float32))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--fold', type=int, default=-1)
    a = p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    bind()
    import torch
    import dgl
    from torch.utils.data import DataLoader
    import alignn.train as official_train
    from alignn.graphs import StructureDataset, prepare_line_graph_batch
    from alignn.models.alignn import ALIGNN
    from alignn_store import GraphStore
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_per_process_memory_fraction(0.65, 0)
    row = next((r for r in registry()['tasks'] if r['id'] == a.task))
    assert row['model_id'] == 'alignn_model'
    out = output_dir(a.task, a.fold)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    if (out / 'official_run/history_val.json').exists():
        raise RuntimeError('Interrupted official ALIGNN engine: inspect its saved checkpoints before any retry; no silent restart')
    labels = training_labels(a.task, a.fold)
    val = validation_labels(a.task)
    config = configuration(row['prop'])
    reference = config.dict()
    config.output_dir = str(out / 'official_run')
    config.write_predictions = False
    config.progress = False
    store = GraphStore()
    store.preload(list(labels) + list(val))
    train_ds = LabeledGraphs(store, labels)
    val_ds = LabeledGraphs(store, val)
    opts = dict(batch_size=config.batch_size, collate_fn=StructureDataset.collate_line_graph, drop_last=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, **opts)
    val_loader = DataLoader(val_ds, shuffle=False, **opts)
    assert len(train_loader) > 0 and len(val_loader) > 0
    latent = []
    constructed = []

    def hook(layer, args):
        latent.clear()
        latent.append(args[0])

    def captured_constructor(model_config):
        model = ALIGNN(model_config)
        model.fc.register_forward_pre_hook(hook)
        constructed.append(model)
        return model
    started = time.time()
    original_constructor = official_train.ALIGNN
    official_train.ALIGNN = captured_constructor
    try:
        official_train.train_dgl(config, model=None, train_val_test_loaders=[train_loader, val_loader, None, prepare_line_graph_batch])
    finally:
        official_train.ALIGNN = original_constructor
    assert len(constructed) == 1
    model = constructed[0]
    model.eval()
    history = json.loads((out / 'official_run/history_val.json').read_text())
    atomic_json(out / 'validation.json', dict(metric='MAE', score=float(history['mae'][-1]), selected_epoch=config.epochs, checkpoint_rule='official_train_dgl_final_epoch', test_labels_read=False))
    metadata = dict(task=a.task, model_id=row['model_id'], fold=a.fold, trained_ids=list(labels), validation_ids=list(val), test_labels_read=False, protocol_sha256=protocol_hash(), retrained_from_scratch=True, reference_commit=COMMIT, reference_config=reference, effective_config=config.dict(), checkpoint_rule='official_train_dgl_final_epoch', packages={n: importlib.metadata.version(n) for n in ['torch', 'dgl', 'jarvis-tools', 'pydantic', 'numpy', 'pytorch-ignite']})
    atomic_json(out / 'config.json', metadata)

    def forward(js):
        pairs = [store.get(j) for j in js]
        g = dgl.batch([x[0] for x in pairs]).to('cuda')
        lg = dgl.batch([x[1] for x in pairs]).to('cuda')
        pred = model((g, lg)).reshape(-1)
        assert len(latent) == 1
        return (pred, latent[0])
    with torch.inference_mode():
        weights = out / 'model.pt'
        torch.save(model.state_dict(), weights)
        ids = store.ids
        predmap = np.lib.format.open_memmap(out / 'prediction.npy', mode='w+', dtype='float32', shape=(len(ids),))
        hidden = np.lib.format.open_memmap(out / 'hidden.npy', mode='w+', dtype='float32', shape=(len(ids), 256))
        for i, j in enumerate(ids):
            pred, h = forward([j])
            predmap[i] = pred.cpu().numpy()[0]
            hidden[i] = h.cpu().numpy()[0]
        predmap.flush()
        hidden.flush()
        assert np.isfinite(predmap).all() and np.isfinite(hidden).all()
    atomic_json(out / 'complete.json', dict(task=a.task, fold=a.fold, rows=len(ids), hidden_dim=256, checkpoint_sha256=sha256(weights), test_labels_read=False, seconds=time.time() - started))
    print('EXPERT_COMPLETE', a.task, a.fold, flush=True)
if __name__ == '__main__':
    main()
