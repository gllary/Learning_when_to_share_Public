"""Stage1 full/OOF experts. Per-architecture official checkpoint rules; gated."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from common import protocol_hash
from common import ROOT, atomic_json, output_dir, registry, sha256, training_labels, validation_labels
from strong_backend import RECIPES, build_model, load_graph_dataset, receipt_path

class LabeledView:

    def __init__(self, base, indices, labels):
        self.base = base
        self.indices = list(indices)
        self.labels = np.asarray(labels, dtype=np.float32)
        assert len(self.indices) == len(self.labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        import torch
        value = self.base[self.indices[index]].clone()
        value.y = torch.tensor(self.labels[index], dtype=torch.float32)
        return value

def package_versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'not-distribution'
    return result

def forward_with_hidden(model, batch, architecture, capture):
    capture.clear()
    result = model(batch)
    prediction = result[0] if isinstance(result, tuple) else result
    prediction = prediction.reshape(-1)
    if len(capture) != 1:
        raise RuntimeError(f'Expected one latent capture, got {len(capture)}')
    return (prediction, capture[0])

def training_epoch(model, loader, architecture, optimizer, scheduler, capture, device):
    """The production loop is also exercised by the independent author parity test."""
    import torch
    import itertools
    model.train()
    if architecture == 'prism_model':
        optimizer.train()
    losses = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        pred, _ = forward_with_hidden(model, batch, architecture, capture)
        loss = torch.nn.functional.l1_loss(pred, batch.y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite Stage 1 loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach()))
    if architecture == 'prism_model':
        optimizer.eval()
        model.train()
        with torch.no_grad():
            for batch in itertools.islice(loader, 50):
                forward_with_hidden(model, batch.to(device), architecture, capture)
    model.eval()
    return losses

def optimizer_bundle(model, recipe_name, n_batches):
    import torch
    recipe = RECIPES[recipe_name]
    architecture = recipe['architecture']
    if architecture == 'atomnet_model':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.98), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=recipe['epochs'] * n_batches, pct_start=0.01)
        name = 'AdamW+OneCycleLR'
    elif architecture == 'prism_model':
        import schedulefree
        optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-08, warmup_steps=int(recipe['epochs'] * n_batches * 0.01))
        scheduler = None
        name = 'schedulefree.AdamWScheduleFree'
    else:
        decay = []
        no_decay = []
        for key, param in model.named_parameters():
            (no_decay if any((k in key for k in ['bias', 'bn', 'norm'])) else decay).append(param)
        optimizer = torch.optim.AdamW([{'params': decay}, {'params': no_decay, 'weight_decay': 0}], lr=0.001, weight_decay=1e-05)
        pct_start = 2000 / (recipe['epochs'] * n_batches)
        assert 0 < pct_start < 1
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=recipe['epochs'] * n_batches, pct_start=pct_start)
        name = 'AdamW+OneCycleLR'
    return (optimizer, scheduler, name)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--fold', type=int, default=-1)
    p.add_argument('--batch', type=int)
    a = p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import global_mean_pool
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    fraction = float(os.environ.get('JARVIS_GPU_FRACTION', '0.78'))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    device = torch.device('cuda:0')
    row = next((x for x in registry()['tasks'] if x['id'] == a.task))
    assert row.get('replacement') and row['metric'] == 'MAE'
    recipe_name = row['strong_recipe']
    recipe = RECIPES[recipe_name]
    architecture = recipe['architecture']
    assert row['model_id'] == architecture
    out = output_dir(a.task, a.fold)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('ALREADY_COMPLETE', a.task, a.fold, flush=True)
        return
    seed = recipe['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if architecture == 'gsmnet_model':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    labels = training_labels(a.task, a.fold)
    trained_ids = list(labels)
    trained_y = np.asarray(list(labels.values()), dtype=np.float32)
    if not receipt_path(recipe_name).exists():
        raise RuntimeError('Missing target-free graph receipt: ' + str(receipt_path(recipe_name)))
    base = load_graph_dataset(recipe_name, build=False)
    universe = np.load(ROOT / 'data' / 'universe_ids.npy').astype(str)
    index = {j: i for i, j in enumerate(universe)}
    assert set(trained_ids) <= set(index)
    val = validation_labels(a.task)
    val_ids = list(val)
    view = LabeledView(base, [index[j] for j in trained_ids], trained_y)
    batch_size = int(a.batch or recipe['batch'])
    if batch_size != recipe['batch']:
        raise RuntimeError('Formal training cannot override the verified official batch size')
    workers = recipe['workers']
    loader_options = dict(num_workers=workers, pin_memory=recipe['pin_memory'], persistent_workers=bool(workers and recipe['persistent_workers']), drop_last=recipe['drop_last'])
    loader = DataLoader(view, batch_size=min(batch_size, len(view)), shuffle=True, **loader_options)
    val_loader = DataLoader(LabeledView(base, [index[j] for j in val_ids], [val[j] for j in val_ids]), batch_size=batch_size, shuffle=False, **loader_options)
    model = build_model(recipe_name, device)
    capture = []
    if architecture in {'atomnet_model', 'prism_model'}:

        def head_hook(module, args):
            graph = args[0]
            capture.append(global_mean_pool(graph.x, graph.batch))
        hook = model.head.register_forward_pre_hook(head_hook)
    else:
        hook = model.fc_out.register_forward_pre_hook(lambda module, args: capture.append(args[0]))
    optimizer, scheduler, optimizer_name = optimizer_bundle(model, recipe_name, len(loader))
    epochs = int(recipe['epochs'])
    initial = 0
    resume = out / 'resume.pt'
    best = float('inf')
    best_epoch = 0
    val_mae = None
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        assert state['seed'] == seed and state['batch'] == batch_size and (state['recipe'] == recipe_name)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(state['scheduler'])
        initial = int(state['epoch'])
        torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        best = state['best']
        best_epoch = state['best_epoch']
        val_mae = state['val_mae']
    config = {'task': a.task, 'model_id': architecture, 'model': row['model'], 'recipe': recipe_name, 'fold': a.fold, 'seed': seed, 'epochs': epochs, 'batch': batch_size, 'trained_ids': trained_ids, 'trained_id_sha256': hashlib.sha256('\n'.join(trained_ids).encode()).hexdigest(), 'latent_dim': 256, 'test_labels_read': False, 'official_validation_labels_read': True, 'selection': recipe['checkpoint_rule'], 'data_loader_options': loader_options, 'retrained_from_scratch': True, 'validation_ids': val_ids, 'protocol_sha256': protocol_hash(), 'loss': 'raw-unit MAE', 'optimizer': optimizer_name, 'recipe_config': recipe, 'vendor_provenance': json.loads((ROOT / 'vendor_ext' / 'provenance.json').read_text()), 'packages': package_versions(['torch', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'jarvis-tools', 'schedulefree', 'mamba-ssm'])}
    atomic_json(out / 'config.json', config)
    started = time.monotonic()
    for epoch in range(initial, epochs):
        losses = training_epoch(model, loader, architecture, optimizer, scheduler, capture, device)
        error = 0.0
        count = 0
        with torch.no_grad():
            for vb in val_loader:
                vb = vb.to(device)
                vp, _ = forward_with_hidden(model, vb, architecture, capture)
                error += float((vp - vb.y.reshape(-1)).abs().sum())
                count += len(vp)
        val_mae = error / count
        if not math.isfinite(val_mae):
            raise RuntimeError('Nonfinite validation score')
        if val_mae < best:
            best = val_mae
            best_epoch = epoch + 1
            tmp = out / 'best.tmp'
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, out / 'best.pt')
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': None if scheduler is None else scheduler.state_dict(), 'epoch': epoch + 1, 'seed': seed, 'batch': batch_size, 'recipe': recipe_name, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'numpy_rng': np.random.get_state(), 'best': best, 'best_epoch': best_epoch, 'val_mae': val_mae}
        tmp = resume.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, resume)
        atomic_json(out / 'progress.json', {'epoch': epoch + 1, 'epochs': epochs, 'seconds': time.monotonic() - started, 'train_mae': float(np.mean(losses)), 'val_mae': val_mae, 'best_val_mae': best, 'best_epoch': best_epoch, 'status': 'training', 'architecture': architecture})
        print('EPOCH', epoch + 1, epochs, 'MAE', float(np.mean(losses)), flush=True)
    model.eval()
    if architecture == 'prism_model':
        optimizer.eval()
    if recipe['checkpoint_rule'] == 'best_validation_mae':
        model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
        selected_score, selected_epoch = (best, best_epoch)
    elif recipe['checkpoint_rule'] == 'official_default_final_epoch':
        selected_score, selected_epoch = (val_mae, epochs)
    else:
        raise RuntimeError('Unverified checkpoint rule')
    atomic_json(out / 'validation.json', {'score': selected_score, 'metric': 'MAE', 'selected_epoch': selected_epoch, 'checkpoint_rule': recipe['checkpoint_rule'], 'test_labels_read': False})
    weights = out / 'model.pt'
    torch.save(model.state_dict(), weights)
    all_ids = np.load(ROOT / 'data' / 'universe_ids.npy').astype(str)
    assert len(base) == len(all_ids)
    predmap = np.lib.format.open_memmap(out / 'prediction.npy', mode='w+', dtype='float32', shape=(len(all_ids),))
    hiddenmap = np.lib.format.open_memmap(out / 'hidden.npy', mode='w+', dtype='float32', shape=(len(all_ids), 256))
    infer = DataLoader(base, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    cursor = 0
    with torch.no_grad():
        for batch in infer:
            batch = batch.to(device, non_blocking=True)
            pred, hidden = forward_with_hidden(model, batch, architecture, capture)
            stop = cursor + len(pred)
            predmap[cursor:stop] = pred.float().cpu().numpy()
            hiddenmap[cursor:stop] = hidden.float().cpu().numpy()
            cursor = stop
    assert cursor == len(all_ids)
    predmap.flush()
    hiddenmap.flush()
    if not np.isfinite(predmap).all() or not np.isfinite(hiddenmap).all():
        raise RuntimeError('Non-finite expert export')
    atomic_json(out / 'complete.json', {'task': a.task, 'fold': a.fold, 'checkpoint_file': 'model.pt', 'checkpoint_sha256': sha256(weights), 'rows': len(all_ids), 'hidden_dim': 256, 'test_labels_read': False, 'official_validation_labels_read': True, 'selected_epoch': selected_epoch, 'validation_mae': selected_score, 'seconds': time.monotonic() - started, 'architecture': architecture, 'recipe': recipe_name})
    print('EXPERT_COMPLETE', a.task, a.fold, flush=True)
if __name__ == '__main__':
    main()
