"""Adapters for frozen AtomNet, PRISM and GSMNet vendor implementations.

The adapters deliberately keep labels outside structural graph caches.  Each
worker injects only the labels named by ``training_labels`` into a lightweight
dataset view. Validation labels are injected separately for checkpoint
selection; test labels are never used by Stage 1.
"""
from __future__ import annotations
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from common import ROOT, atomic_json, sha256
from cache_contract import contract, verify
VENDOR = ROOT / 'vendor_ext'
CACHE = ROOT / 'data' / 'strong_cache'

def cache_key(recipe_name):
    r = RECIPES[recipe_name]
    kind = r['architecture'].replace('_model', '')
    return r['cache'] + '_' + contract(kind)[:16]
RECIPES = {'formation': dict(architecture='atomnet_model', cache='atomnet_eneg_native_empty_v6', batch=128, epochs=300, atom_init='atom_features(116d)_update01', use_electronegativity=True, eneg_type='newRBF', envelope_type='simply', seed=206, workers=5, drop_last=False, pin_memory=True, persistent_workers=True, checkpoint_rule='best_validation_mae'), 'total_energy': dict(architecture='atomnet_model', cache='atomnet_eneg_native_empty_v6', batch=128, epochs=300, atom_init='atom_features(116d)_update01', use_electronegativity=True, eneg_type='newRBF', envelope_type='cubic', seed=306, workers=5, drop_last=False, pin_memory=True, persistent_workers=True, checkpoint_rule='best_validation_mae'), 'opt_bandgap': dict(architecture='atomnet_model', cache='atomnet_plain_native_empty_v6', batch=128, epochs=300, atom_init='atom_features(116d)_update01', use_electronegativity=False, eneg_type=None, envelope_type='cubic', seed=448, workers=5, drop_last=False, pin_memory=True, persistent_workers=True, checkpoint_rule='best_validation_mae'), 'ehull': dict(architecture='prism_model', cache='prism', batch=64, epochs=500, seed=0, workers=5, drop_last=False, pin_memory=True, persistent_workers=True, checkpoint_rule='best_validation_mae'), 'mbj_bandgap': dict(architecture='gsmnet_model', cache='gsmnet', batch=64, epochs=500, seed=123, workers=8, drop_last=True, pin_memory=False, persistent_workers=False, checkpoint_rule='official_default_final_epoch')}

def recipe(name):
    return dict(RECIPES[name])

def universe():
    ids = np.load(ROOT / 'data' / 'universe_ids.npy').astype(str).tolist()
    structures = json.loads((ROOT / 'data' / 'structures.json').read_text())
    assert set(ids) == set(structures)
    return (ids, structures)

def _records(ids, structures):
    return [{'jid': j, 'atoms': structures[j], 'target': 0.0} for j in ids]

def configure_atomnet(recipe_name):
    source = VENDOR / 'atomnet' / 'AtomNet'
    sys.path.insert(0, str(source))
    from strong_reference import pyg_configuration
    cfg, _, _, _ = pyg_configuration(recipe_name)
    r = RECIPES[recipe_name]
    assert (cfg.seed, cfg.batch, cfg.optim.max_epoch, cfg.workers) == (r['seed'], r['batch'], r['epochs'], r['workers'])
    assert cfg.envelope_type == r['envelope_type'] and cfg.useElectronegativity == r['use_electronegativity']
    return cfg

def configure_prism():
    source = VENDOR / 'prism'
    sys.path.insert(0, str(source))
    from strong_reference import pyg_configuration
    cfg, _, _, _ = pyg_configuration('ehull')
    return cfg

def _atomnet_dataset(recipe_name, ids, structures):
    cfg = configure_atomnet(recipe_name)
    from dataset.figshare_dataset import Figshare_Dataset
    key = cache_key(recipe_name) + ''
    root = CACHE / key
    data = _records(ids, structures)
    ds = Figshare_Dataset(root=str(root), data=data, targets=[0.0] * len(data), name='target_free_universe', radius=cfg.radius, max_neigh=cfg.max_neighbours, atom_init=cfg.atom_init)
    if len(ds) != len(ids):
        raise RuntimeError(f'AtomNet graph builder dropped {len(ids) - len(ds)} structures')
    return ds

def _prism_dataset(ids, structures):
    cfg = configure_prism()
    from dataset.figshare_dataset import Figshare_Dataset
    key = cache_key('ehull') + ''
    data = _records(ids, structures)
    root = CACHE / key
    ds = Figshare_Dataset(root=str(root), data=data, targets=[0.0] * len(data), name='target_free_universe', radius=cfg.radius, max_neigh=-1, augment=False, n_hops=-1)
    if len(ds) != len(ids):
        raise RuntimeError('PRISM graph count changed')
    return ds

def _gsmnet_dataset(ids, structures, build):
    source = VENDOR / 'gsmnet'
    sys.path.insert(0, str(source))
    from data import get_torch_dataset
    key = cache_key('mbj_bandgap') + ''
    root = CACHE / key
    data = _records(ids, structures)
    ds = get_torch_dataset(dataset=data, root=str(root), cachedir='raw', processdir='processed', name='target_free_universe', id_tag='jid', target='target', atom_features='cgcnn', normalize=False, euclidean=False, cutoff=4.0, max_neighbors=16, infinite_funcs=['zeta', 'zeta', 'exp'], infinite_params=[0.5, 3.0, 3.0], R=3)
    marker = root / 'rrwp16.complete.json'
    if not marker.exists():
        if not build:
            raise RuntimeError('GSMNet RRWP cache is not prepared')
        from grit.transform.posenc_stats import compute_posenc_stats
        from grit.transform.transforms import pre_transform_in_memory
        pre_transform_in_memory(ds, partial(compute_posenc_stats, pe_types=['RRWP'], ksteps=16), show_progress=True, posenc_mode=True)
        torch_path = Path(ds.processed_paths[0])
        tmp = torch_path.with_suffix('.rrwp.tmp')
        import torch
        torch.save((ds.data, ds.slices), tmp)
        tmp.replace(torch_path)
        atomic_json(marker, {'rows': len(ds), 'ksteps': 16, 'targets': 'constant_zero_only'})
    elif not hasattr(ds[0], 'rrwp_index'):
        raise RuntimeError('GSMNet RRWP marker exists but tensors are absent')
    if len(ds) != len(ids):
        raise RuntimeError('GSMNet graph count changed')
    return ds

def load_graph_dataset(recipe_name, ids=None, build=False):
    if not build:
        verify_receipt(recipe_name)
    all_ids, structures = universe()
    if ids is None:
        ids = all_ids
    else:
        ids = list(ids)
        structures = {j: structures[j] for j in ids}
    arch = RECIPES[recipe_name]['architecture']
    if arch == 'atomnet_model':
        return _atomnet_dataset(recipe_name, ids, structures)
    if arch == 'prism_model':
        return _prism_dataset(ids, structures)
    if arch == 'gsmnet_model':
        return _gsmnet_dataset(ids, structures, build)
    raise KeyError(arch)

def build_model(recipe_name, device):
    import torch
    r = RECIPES[recipe_name]
    arch = r['architecture']
    if arch == 'atomnet_model':
        cfg = configure_atomnet(recipe_name)
        from models.atomnet import AtomNet
        return AtomNet(dim_in=cfg.dim_in, dim_rbf=cfg.dim_rbf, num_layers=cfg.num_layers, radius=cfg.radius, invariant=cfg.invariant, use_envelope=cfg.envelope, atom_init=cfg.atom_init).to(device)
    if arch == 'prism_model':
        cfg = configure_prism()
        from models.prism import Prism
        return Prism(dim_in=cfg.dim_in, dim_rbf=cfg.dim_rbf, num_layers=cfg.num_layers, radius=cfg.radius, invariant=cfg.invariant, use_envelope=cfg.envelope).to(device)
    if arch == 'gsmnet_model':
        source = VENDOR / 'gsmnet'
        sys.path.insert(0, str(source))
        from models.gsmnet import GSMNet
        from strong_reference import gsm_configuration
        return GSMNet(gsm_configuration().model).to(device)
    raise KeyError(arch)

def receipt_path(recipe_name):
    return CACHE / 'receipts' / (cache_key(recipe_name) + '.json')

def verify_receipt(recipe_name):
    item = json.loads(receipt_path(recipe_name).read_text())
    verify(item, RECIPES[recipe_name]['architecture'].replace('_model', ''), [(ROOT / p, h) for p, h in item['files'].items()])
    return item

def write_receipt(recipe_name, ds, ids):
    path = receipt_path(recipe_name)
    payload = {'cache_contract': contract(RECIPES[recipe_name]['architecture'].replace('_model', '')), 'files': {str(Path(p).relative_to(ROOT)): sha256(p) for p in ds.processed_paths}, 'recipe': recipe_name, 'cache': cache_key(recipe_name), 'rows': len(ds), 'universe_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(), 'target_free': True, 'physical_labels_present': False}
    atomic_json(path, payload)
    return path
