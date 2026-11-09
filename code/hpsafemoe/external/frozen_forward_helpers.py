from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from evaluation import write_json_new
SCHEMA_VERSION = 'v8-youngs-case-study-1.3'

def read_json(path):
    return json.loads(Path(path).read_text())

def atomic_json(path, value):
    write_json_new(path, value)

def atomic_npy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.save(f, value, allow_pickle=False)

class ProductionReplayCache:
    """Name-aligned production cache plus frozen-SISSO overlay for MODNet replay."""

    def __init__(self, root: Path, *, matminer_overlay_root: Path, sisso_overlay_root: Path) -> None:
        from hpsafe_features.core import ProductionFeatureCache, _feature_catalog, _select_named_features
        self._cache = ProductionFeatureCache.open(root.expanduser().resolve(strict=True), verify_hashes=False)
        self.root = self._cache.root
        self.receipt = self._cache.receipt
        self.sample_ids = self._cache.sample_ids
        self._catalog = _feature_catalog(self.root, self.receipt)
        self._select = _select_named_features
        base_overlay_root = matminer_overlay_root.expanduser().resolve(strict=True)
        base_overlay_receipt = read_json(base_overlay_root / 'receipt.json')
        base_required = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'cache_kind': 'frozen_matminer_sparse_schema_zero_overlay_v1', 'case_target_values_read': False, 'fit_performed': False, 'feature_selection_performed': False, 'checkpoint_parameters_updated': False}
        for key, expected in base_required.items():
            if base_overlay_receipt.get(key) != expected:
                raise ValueError(f'Frozen MatMiner schema overlay violates {key}')
        base_ids = np.load(base_overlay_root / 'sample_ids.npy', allow_pickle=False).astype(str)
        if not np.array_equal(base_ids, self.sample_ids.astype(str)):
            raise ValueError('Frozen MatMiner schema overlay sample order differs from cache')
        base_names = list(map(str, base_overlay_receipt.get('feature_names', [])))
        base_values = np.load(base_overlay_root / 'matminer_schema_zeros.npy', mmap_mode='r', allow_pickle=False)
        if base_values.shape != (len(self.sample_ids), len(base_names)):
            raise ValueError('Frozen MatMiner schema overlay shape/schema mismatch')
        if set(base_names).intersection(self._catalog):
            raise ValueError('Frozen MatMiner schema overlay duplicates production columns')
        self._base_overlay_root = base_overlay_root
        self._base_overlay_names = {value: index for index, value in enumerate(base_names)}
        self._base_overlay_values = base_values
        overlay_root = sisso_overlay_root.expanduser().resolve(strict=True)
        overlay_receipt = read_json(overlay_root / 'receipt.json')
        required = {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'cache_kind': 'frozen_mattervial_sisso_v1_overlay', 'case_target_values_read': False, 'formula_fit_performed': False, 'feature_selection_performed': False, 'checkpoint_parameters_updated': False}
        for key, expected in required.items():
            if overlay_receipt.get(key) != expected:
                raise ValueError(f'Frozen SISSO overlay violates {key}: {overlay_root}')
        overlay_ids = np.load(overlay_root / 'sample_ids.npy', allow_pickle=False).astype(str)
        if not np.array_equal(overlay_ids, self.sample_ids.astype(str)):
            raise ValueError('Frozen SISSO overlay sample order differs from production cache')
        overlay_names = [str(value) for value in overlay_receipt.get('feature_names', [])]
        overlay_values = np.load(overlay_root / 'sisso_task.npy', mmap_mode='r', allow_pickle=False)
        if overlay_values.shape != (len(self.sample_ids), len(overlay_names)):
            raise ValueError('Frozen SISSO overlay shape/schema mismatch')
        if len(set(overlay_names)) != len(overlay_names):
            raise ValueError('Frozen SISSO overlay has duplicate feature names')
        duplicate = set(overlay_names).intersection(self._catalog)
        if duplicate:
            raise ValueError(f'Frozen SISSO overlay duplicates production columns: {sorted(duplicate)[:5]}')
        self._overlay_root = overlay_root
        self._overlay_receipt = overlay_receipt
        self._overlay_names = {value: index for index, value in enumerate(overlay_names)}
        self._overlay_values = overlay_values

    def member_matrix(self, descriptors: Iterable[str], *, row_positions: np.ndarray | None=None) -> np.ndarray | None:
        names = [str(value) for value in descriptors]
        if any((name not in self._catalog and name not in self._base_overlay_names and (name not in self._overlay_names) for name in names)):
            return None
        rows = np.arange(len(self.sample_ids), dtype=np.int64) if row_positions is None else np.asarray(row_positions, dtype=np.int64).reshape(-1)
        matrix = np.empty((len(rows), len(names)), dtype=np.float32)
        primary_positions = [index for index, name in enumerate(names) if name in self._catalog]
        if primary_positions:
            primary_names = [names[index] for index in primary_positions]
            matrix[:, primary_positions] = self._select(self.root, self.receipt, self._catalog, primary_names, rows)
        base_positions = [index for index, name in enumerate(names) if name in self._base_overlay_names]
        if base_positions:
            columns = [self._base_overlay_names[names[index]] for index in base_positions]
            matrix[:, base_positions] = self._base_overlay_values[np.ix_(rows, columns)]
        overlay_positions = [index for index, name in enumerate(names) if name in self._overlay_names]
        if overlay_positions:
            columns = [self._overlay_names[names[index]] for index in overlay_positions]
            matrix[:, overlay_positions] = self._overlay_values[np.ix_(rows, columns)]
        return matrix

def _materialize_external_jmp_cache(*, runtime_root: Path, work_root: Path, rows: list[dict[str, str]], sample_ids: np.ndarray) -> Path:
    """Create a target-free JMP LMDB directly from the case structures."""
    import torch
    from pymatgen.core import Structure
    from torch_geometric.data import Data
    source_root = runtime_root.expanduser().resolve(strict=True) / 'external/JMP'
    source_path = str((source_root / 'src').resolve(strict=True))
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from jmp.datasets.finetune.base import LmdbDataset
    root = work_root / 'external_jmp_input'
    receipt_path = root / 'receipt.json'
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        cached_ids = np.load(root / 'sample_ids.npy', allow_pickle=False).astype(str)
        if receipt.get('status') != 'complete' or receipt.get('target_values_stored') is not False or (not np.array_equal(cached_ids, sample_ids.astype(str))):
            raise ValueError(f'Existing external JMP cache is incompatible: {root}')
        return root / 'lmdb'
    if root.exists():
        raise ValueError(f'Incomplete external JMP cache already exists: {root}')
    lmdb_root = root / 'lmdb'
    lmdb_root.mkdir(parents=True)

    def records() -> Iterable[Any]:
        for row in rows:
            structure = Structure.from_dict(json.loads(row['structure_json']))
            yield Data(atomic_numbers=torch.tensor([site.specie.number for site in structure], dtype=torch.long), pos=torch.tensor(np.asarray(structure.cart_coords), dtype=torch.float32), cell=torch.tensor(np.asarray(structure.lattice.matrix), dtype=torch.float32).unsqueeze(0))
    print(json.dumps({'event': 'case_jmp_target_free_lmdb_build_started', 'sample_count': len(rows), 'target_values_stored': False}, sort_keys=True), flush=True)
    LmdbDataset.dump_data(records(), count=len(rows), path=lmdb_root)
    atomic_npy(root / 'sample_ids.npy', sample_ids.astype(str))
    atomic_json(receipt_path, {'schema_version': SCHEMA_VERSION, 'status': 'complete', 'representation': 'target_free_pyg_atomic_numbers_cartesian_positions_cell', 'n_samples': len(rows), 'sample_ids_path': 'sample_ids.npy', 'target_values_stored': False, 'case_labels_read': False, 'training_manifest_created': False, 'case_target_array_materialized': False, 'row_order': 'exact_case_manifest_order'})
    return lmdb_root

def _predict_external_jmp(*, runtime_root: Path, source_task: str, outer_fold: int, source_artifact: Path, checkpoint: Path, cache_dir: Path, work_root: Path, batch_size: int, num_workers: int, finetune_protocol: str | None, maximum_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    """Instantiate one frozen source checkpoint and forward the external LMDB."""
    import torch
    from torch.utils.data import DataLoader
    from torch_scatter import scatter
    from hpsafe_sota.experts.interface import ExpertRequest
    from hpsafe_sota.experts.jmp_l_bridge import LATENT_DIM, TASK_SETTINGS, JMPLPrepared, JMPLStage1Bridge
    preprocessing = read_json(source_artifact / 'preprocessing_manifest.json')
    if preprocessing.get('task') != source_task or int(preprocessing.get('outer_fold', -1)) != outer_fold:
        raise ValueError(f'JMP preprocessing task/fold mismatch: {source_artifact}')
    split_root = work_root / 'splits'
    split_root.mkdir(parents=True, exist_ok=True)
    train_split = split_root / 'train_unused.npz'
    prediction_split = split_root / 'test_all.npz'
    np.savez_compressed(train_split, positions=np.empty(0, dtype=np.int64))
    with np.load(cache_dir / 'metadata.npz', allow_pickle=False) as metadata:
        n_samples = len(np.asarray(metadata['natoms']))
    np.savez_compressed(prediction_split, positions=np.arange(n_samples, dtype=np.int64))
    extra: dict[str, Any] = {'maximum_epochs': maximum_epochs, 'num_workers': num_workers, 'hyperparameter_search': False, 'refit_branch': False, 'stage2_enabled': False}
    if finetune_protocol is not None:
        extra['finetune_protocol'] = finetune_protocol
    request = ExpertRequest(schema_version=1, project_root=str(runtime_root), expert_name='jmp_l', task_name=source_task, outer_fold=outer_fold, phase='artifact_import', output_dir=str(work_root / 'model_context'), seed=42, extra=extra)
    prepared = JMPLPrepared(cache_dir=cache_dir, train_split=train_split, prediction_split=prediction_split, preprocessing_manifest=source_artifact / 'preprocessing_manifest.json', target_mean=float(preprocessing['target_mean']), target_std=float(preprocessing['target_std']), source_root=runtime_root / 'external/JMP', checkpoint=runtime_root / 'checkpoints/jmp-l.pt', checkpoint_sha256='not_recomputed_for_case_study')
    bridge = object.__new__(JMPLStage1Bridge)
    bridge.request = request
    print(json.dumps({'event': 'case_jmp_configuration_started', 'source_task': source_task, 'outer_fold': outer_fold}, sort_keys=True), flush=True)
    config, model_class, api = bridge._configuration(prepared)
    print(json.dumps({'event': 'case_jmp_configuration_complete', 'source_task': source_task, 'outer_fold': outer_fold}, sort_keys=True), flush=True)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    with api['Trainer'].runner_init(config):
        with api['Trainer'].context(config):
            print(json.dumps({'event': 'case_jmp_model_construction_started', 'source_task': source_task, 'outer_fold': outer_fold}, sort_keys=True), flush=True)
            model = model_class(config)
            print(json.dumps({'event': 'case_jmp_model_construction_complete', 'source_task': source_task, 'outer_fold': outer_fold}, sort_keys=True), flush=True)
            payload = torch.load(checkpoint, map_location='cpu')
            state = payload.get('state_dict', payload)
            incompatible = model.load_state_dict(state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ValueError(f'Frozen JMP state mismatch: missing={list(incompatible.missing_keys)[:10]} unexpected={list(incompatible.unexpected_keys)[:10]}')
            model.requires_grad_(False)
            if any((parameter.requires_grad for parameter in model.parameters())):
                raise RuntimeError('Frozen JMP parameter gate failed')
            dataset = model.test_dataset()
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=model.collate_fn)
            device = torch.device('cuda:0')
            model.to(device).eval()
            predictions: list[np.ndarray] = []
            hidden_values: list[np.ndarray] = []
            captured: dict[str, Any] = {}

            def capture(_module: Any, _inputs: Any, output: Any) -> None:
                captured['energy'] = output['energy']
            hook = model.backbone.register_forward_hook(capture)
            try:
                with torch.inference_mode():
                    for batch_index, batch in enumerate(loader):
                        print(json.dumps({'event': 'case_jmp_batch_started', 'source_task': source_task, 'outer_fold': outer_fold, 'batch_index': batch_index}, sort_keys=True), flush=True)
                        batch = batch.to(device)
                        captured.clear()
                        normalized = model(batch)['y']
                        prediction = normalized * prepared.target_std + prepared.target_mean
                        predictions.append(prediction.detach().cpu().numpy().reshape(-1))
                        atom_hidden = captured.get('energy')
                        if atom_hidden is None or atom_hidden.ndim != 2 or atom_hidden.shape[1] != LATENT_DIM:
                            raise ValueError('JMP hidden hook returned an invalid representation')
                        n_graphs = int(torch.max(batch.batch).item()) + 1
                        hidden_values.append(scatter(atom_hidden, batch.batch, dim=0, dim_size=n_graphs, reduce=TASK_SETTINGS[source_task]['reduction']).float().cpu().numpy())
                        print(json.dumps({'event': 'case_jmp_batch_complete', 'source_task': source_task, 'outer_fold': outer_fold, 'batch_index': batch_index}, sort_keys=True), flush=True)
            finally:
                hook.remove()
            prediction_array = np.concatenate(predictions).astype(np.float32, copy=False)
            hidden_array = np.concatenate(hidden_values).astype(np.float32, copy=False)
            model.to('cpu')
            del model
    if not np.isfinite(prediction_array).all() or not np.isfinite(hidden_array).all():
        raise ValueError('Frozen JMP external output contains NaN/Inf')
    return (prediction_array, hidden_array)
