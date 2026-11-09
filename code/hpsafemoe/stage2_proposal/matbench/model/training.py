from __future__ import annotations
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from hpsafe_sota.metrics import score_predictions
from .model import CrossExpertProposalModel, ProposalConfig
from hpsafe_sota.expert_inputs.cache import ARRAY_FILES, ExpertExport, TaskRoleCache, export_directory
from hpsafe_sota.expert_inputs.data import FoldInputs, ScalarStats, TaskStats
from hpsafe_sota.expert_inputs.io import atomic_json, read_json
from hpsafe_sota.expert_inputs.protocol import CANONICAL_TASKS, ExpertPool, load_expert_pool
SCHEMA_VERSION = 'matbench-proposal-v1'
OUTER_FOLDS = (0, 1, 2, 3, 4)
PRIMARY_PROPOSAL = 'proposal_full'
ARCHITECTURE_CANDIDATES = {'proposal_full': {'role': 'Full', 'use_cross_expert_input': True, 'use_data_branch': True, 'use_physics_branch': True, 'use_learned_task_source_relation': True, 'use_shared_low_rank_basis': False, 'use_task_private_correction': True}, 'ablation_no_cross_expert': {'role': 'Strict_NoCross', 'use_cross_expert_input': False, 'use_data_branch': True, 'use_physics_branch': True, 'use_learned_task_source_relation': False, 'use_shared_low_rank_basis': False, 'use_task_private_correction': True}}

def _candidate_variant(candidate: str) -> str:
    prefix = 'final' if candidate == PRIMARY_PROPOSAL else candidate
    return f'{prefix}__oof_locked'

def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

def _scalar_stats(values: np.ndarray) -> ScalarStats:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return ScalarStats(0.0, 1.0)
    return ScalarStats(float(array.mean()), max(float(array.std()), 1e-08))

def _inner_partitions(fold: FoldInputs, partitions: int, *, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Create deterministic, balanced partitions inside each official outer-train."""
    result: dict[str, dict[str, np.ndarray]] = {}
    for task_number, task in enumerate(CANONICAL_TASKS):
        role_sizes = [fold.roles[task][role].n_samples for role in ('train', 'val')]
        labels = np.concatenate([np.asarray(fold.roles[task][role].y_true) for role in ('train', 'val')])
        assignment = np.empty(len(labels), dtype=np.int8)
        rng = np.random.default_rng(seed + task_number * 1009)
        if fold.task_types[task] == 'classification':
            for label in (0, 1):
                selected = np.flatnonzero(labels == label)
                if len(selected) < partitions:
                    raise ValueError(f'{task} has too few class-{label} rows for inner OOF')
                rng.shuffle(selected)
                assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
        else:
            selected = np.arange(len(labels), dtype=np.int64)
            rng.shuffle(selected)
            assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
        if set(assignment.tolist()) != set(range(partitions)):
            raise ValueError(f'Incomplete inner partitions for {task}')
        result[task] = {}
        offset = 0
        for role, size in zip(('train', 'val'), role_sizes, strict=True):
            result[task][role] = assignment[offset:offset + size]
            offset += size
    return result

def _partition_indices(fold: FoldInputs, assignments: dict[str, dict[str, np.ndarray]], held_out: int, *, training: bool) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for task in CANONICAL_TASKS:
        result[task] = {}
        for role in ('train', 'val'):
            selected = assignments[task][role] != held_out if training else assignments[task][role] == held_out
            indices = np.flatnonzero(selected).astype(np.int64)
            if training and (not len(indices)):
                raise ValueError(f'Empty inner training slice for {task}/{role}')
            result[task][role] = indices
    return result

def _masked_stats(fold: FoldInputs, eligible: dict[str, dict[str, np.ndarray]]) -> dict[str, TaskStats]:
    result: dict[str, TaskStats] = {}
    for task in CANONICAL_TASKS:
        targets = np.concatenate([np.asarray(fold.roles[task][role].y_true)[eligible[task][role]] for role in ('train', 'val')])
        target = ScalarStats(0.0, 1.0) if fold.task_types[task] == 'classification' else _scalar_stats(targets)
        prediction: dict[str, ScalarStats] = {}
        uncertainty: dict[str, ScalarStats] = {}
        for expert_id in fold.pool.expert_ids:
            prediction_parts, uncertainty_parts = ([], [])
            for role in ('train', 'val'):
                export = fold.roles[task][role].exports[expert_id]
                if export is None:
                    continue
                indices = eligible[task][role]
                available = np.asarray(export.available, dtype=np.bool_)[indices]
                if np.any(available):
                    prediction_parts.append(np.asarray(export.prediction)[indices][available])
                    uncertainty_parts.append(np.log1p(np.asarray(export.uncertainty, dtype=np.float64)[indices][available]))
            prediction[expert_id] = _scalar_stats(np.concatenate(prediction_parts) if prediction_parts else np.asarray([]))
            uncertainty[expert_id] = _scalar_stats(np.concatenate(uncertainty_parts) if uncertainty_parts else np.asarray([]))
        result[task] = TaskStats(target, prediction, uncertainty)
    return result

def _model_config(pool: ExpertPool, config: dict[str, Any], candidate: str) -> ProposalConfig:
    spec = ARCHITECTURE_CANDIDATES[candidate]
    return ProposalConfig(task_names=CANONICAL_TASKS, expert_ids=pool.expert_ids, hidden_dims=pool.hidden_dims, anchor_indices=tuple((pool.expert_ids.index(pool.anchor_for(task)) for task in CANONICAL_TASKS)), token_dim=int(config['token_dim']), task_dim=int(config['task_dim']), residual_hidden=int(config['residual_hidden']), residual_rank=int(config['residual_rank']), dropout=float(config['dropout']), attention_top_k=int(config['attention_top_k']), residual_limit=float(config['residual_limit']), route_temperature=float(config['route_temperature']), use_cross_expert_input=bool(spec['use_cross_expert_input']), use_data_branch=bool(spec['use_data_branch']), use_physics_branch=bool(spec['use_physics_branch']), use_learned_task_source_relation=bool(spec['use_learned_task_source_relation']), use_shared_low_rank_basis=bool(spec['use_shared_low_rank_basis']), use_task_private_correction=bool(spec['use_task_private_correction']), use_sample_gate=True)

def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)

def _sample_indices(*, fold: FoldInputs, task: str, eligible: dict[str, dict[str, np.ndarray]], config: dict[str, Any], rng: np.random.Generator) -> tuple[str, np.ndarray]:
    roles = ('train', 'val')
    choices = [eligible[task][role] for role in roles]
    sizes = np.asarray([len(value) for value in choices], dtype=np.float64)
    if fold.task_types[task] == 'classification':
        for role_number, (role, valid) in enumerate(zip(roles, choices, strict=True)):
            y_role = np.asarray(fold.roles[task][role].y_true)
            if any((not np.any(y_role[valid] == label) for label in (0, 1))):
                sizes[role_number] = 0.0
    if sizes.sum() <= 0.0:
        raise ValueError(f'No eligible training shard for {task}')
    role_number = int(rng.choice(len(roles), p=sizes / sizes.sum()))
    role, valid = (roles[role_number], choices[role_number])
    y = np.asarray(fold.roles[task][role].y_true)
    if fold.task_types[task] == 'classification':
        count = int(config['classification_per_class'])
        pieces = []
        for label in (0, 1):
            label_rows = valid[y[valid] == label]
            if not len(label_rows):
                raise ValueError(f'Inner training slice lost class {label} for {task}/{role}')
            pieces.append(rng.choice(label_rows, size=count, replace=len(label_rows) < count))
        indices = np.concatenate(pieces).astype(np.int64)
        rng.shuffle(indices)
        return (role, indices)
    count = int(config['regression_batch_size'])
    return (role, rng.choice(valid, size=count, replace=len(valid) < count).astype(np.int64))

def _prediction_loss(prediction: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor, task_type: str, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    if task_type == 'classification':
        positive = target > 0.5
        difference = prediction[positive][:, None] - prediction[~positive][None, :]
        anchor_difference = anchor[positive][:, None] - anchor[~positive][None, :]
        loss = F.softplus(-difference / temperature).mean()
        anchor_loss = F.softplus(-anchor_difference / temperature).mean().detach()
    else:
        loss = torch.abs(prediction - target).mean()
        anchor_loss = torch.abs(anchor - target).mean().detach()
    return (loss, anchor_loss)

def _fit_one_model(*, fold: FoldInputs, candidate: str, config: dict[str, Any], seed: int, eligible: dict[str, dict[str, np.ndarray]], device: torch.device, progress_path: Path, progress_offset: int, progress_total: int, phase: str) -> CrossExpertProposalModel:
    rng = _seed_everything(seed)
    model = CrossExpertProposalModel(_model_config(fold.pool, config, candidate)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
    steps, started = (int(config['proposal_training_steps']), time.time())
    history: list[float] = []
    spec = ARCHITECTURE_CANDIDATES[candidate]
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses, anchor_losses, branch_auxiliary, entropies = ([], [], [], [])
        for task in CANONICAL_TASKS:
            role, indices = _sample_indices(fold=fold, task=task, eligible=eligible, config=config, rng=rng)
            batch = fold.batch(task=task, role=role, indices=indices, device=device, expert_dropout=float(config['expert_dropout']), generator=rng)
            output = model(task_index=batch['task_index'], anchor_standard=batch['anchor_standard'], hidden=batch['hidden'], prediction_standard=batch['prediction_standard'], uncertainty_standard=batch['uncertainty_standard'], available=batch['available'])
            loss, anchor_loss = _prediction_loss(output.raw_standard, batch['anchor_standard'], batch['y_standard'], fold.task_types[task], float(config['pairwise_auc_temperature']))
            losses.append(loss)
            anchor_losses.append(anchor_loss)
            branch_losses = []
            if spec['use_data_branch']:
                branch_losses.append(_prediction_loss(output.data_standard, batch['anchor_standard'], batch['y_standard'], fold.task_types[task], float(config['pairwise_auc_temperature']))[0])
            if spec['use_physics_branch']:
                branch_losses.append(_prediction_loss(output.physics_standard, batch['anchor_standard'], batch['y_standard'], fold.task_types[task], float(config['pairwise_auc_temperature']))[0])
            branch_auxiliary.append(torch.stack(branch_losses).mean())
            probability = output.route_probabilities[:, 1:].clamp_min(1e-08)
            entropies.append(-(probability * probability.log()).sum(dim=1).mean())
        task_vector = torch.stack(losses)
        anchor_vector = torch.stack(anchor_losses)
        positive_regret = F.relu(task_vector - anchor_vector)
        tail_count = max(1, int(math.ceil(float(config['task_cvar_fraction']) * len(CANONICAL_TASKS))))
        cvar = torch.topk(positive_regret, k=tail_count).values.mean()
        entropy = torch.stack(entropies).mean()
        entropy_target = float(config['route_entropy_target'])
        objective = task_vector.mean() + float(config['mean_positive_task_regret_weight']) * positive_regret.mean() + float(config['task_cvar_weight']) * cvar + float(config['branch_auxiliary_weight']) * torch.stack(branch_auxiliary).mean() + float(config['route_entropy_weight']) * (entropy - entropy_target).square()
        if not torch.isfinite(objective):
            raise FloatingPointError(f'Non-finite proposal objective at step {step}')
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['gradient_clip']))
        optimizer.step()
        history.append(float(objective.detach().cpu()))
        if step == 0 or (step + 1) % int(config['progress_every']) == 0 or step + 1 == steps:
            completed = progress_offset + step + 1
            elapsed = time.time() - started
            atomic_json(progress_path, {'stage': phase, 'candidate': candidate, 'seed': seed, 'completed': completed, 'total': progress_total, 'percent': 100.0 * completed / progress_total, 'elapsed_seconds_current_model': elapsed, 'eta_seconds': elapsed / (step + 1) * (progress_total - completed), 'objective': float(np.mean(history[-10:])), 'positive_task_regret': float(positive_regret.mean().detach().cpu()), 'task_cvar': float(cvar.detach().cpu()), 'progress_unit_label': 'joint-13-task-step'})
    atomic_json(progress_path.parent / f'training_history_seed_{seed}.json', {'fit_seed': seed, 'steps': steps, 'objective': history, 'checkpoint_rule': 'fixed_final_step', 'test_labels_used': False})
    return model

def _predict(*, fold: FoldInputs, model: CrossExpertProposalModel, task: str, role: str, device: torch.device, indices: np.ndarray | None=None, chunk_size: int=4096, shuffle_relation: bool=False, shuffle_seed: int=0) -> dict[str, np.ndarray]:
    cache = fold.roles[task][role]
    selected = np.arange(cache.n_samples, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or not len(selected):
        raise ValueError('Prediction indices must be a non-empty vector')
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ('raw', 'data', 'physics', 'route_probabilities', 'route_weights', 'routing_components', 'route_index', 'data_attention', 'physics_attention', 'reliability', 'anchor')}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(selected), chunk_size):
            rows = selected[start:start + chunk_size]
            batch = fold.batch(task=task, role=role, indices=rows, device=device)
            output = model(task_index=batch['task_index'], anchor_standard=batch['anchor_standard'], hidden=batch['hidden'], prediction_standard=batch['prediction_standard'], uncertainty_standard=batch['uncertainty_standard'], available=batch['available'])
            physics_standard = output.physics_standard
            if shuffle_relation and len(rows) > 1:
                rng = np.random.default_rng(shuffle_seed + start)
                order = torch.as_tensor(rng.permutation(len(rows)), device=device)
                physics_standard = physics_standard[order]
                route = output.route_weights
                raw = route[:, 0] * batch['anchor_standard'] + route[:, 1] * output.data_standard + route[:, 2] * physics_standard
            else:
                raw = output.raw_standard
            parts['raw'].append(raw.detach().cpu().numpy())
            parts['data'].append(output.data_standard.detach().cpu().numpy())
            parts['physics'].append(physics_standard.detach().cpu().numpy())
            parts['route_probabilities'].append(output.route_probabilities.detach().cpu().numpy())
            parts['route_weights'].append(output.route_weights.detach().cpu().numpy())
            exported_physics_residual = physics_standard - batch['anchor_standard'] if shuffle_relation and len(rows) > 1 else output.physics_residual
            parts['routing_components'].append(torch.stack([batch['anchor_standard'], output.data_residual, exported_physics_residual, raw], dim=1).detach().cpu().numpy())
            parts['route_index'].append(output.route_one_hot.argmax(dim=1).detach().cpu().numpy())
            parts['data_attention'].append(output.data_attention.detach().cpu().numpy())
            parts['physics_attention'].append(output.physics_attention.detach().cpu().numpy())
            parts['reliability'].append(output.reliability.detach().cpu().numpy())
            parts['anchor'].append(np.asarray(batch['anchor_prediction']))
    return {'sample_ids': np.asarray(cache.sample_ids[selected]).astype(str), 'raw_prediction': fold.restore_prediction(task, np.concatenate(parts['raw'])), 'data_prediction': fold.restore_prediction(task, np.concatenate(parts['data'])), 'physics_prediction': fold.restore_prediction(task, np.concatenate(parts['physics'])), 'route_probabilities': np.concatenate(parts['route_probabilities']).astype(np.float32), 'route_weights': np.concatenate(parts['route_weights']).astype(np.float32), 'routing_components': np.concatenate(parts['routing_components']).astype(np.float32), 'route_index': np.concatenate(parts['route_index']).astype(np.int8), 'data_attention': np.concatenate(parts['data_attention']).astype(np.float32), 'physics_attention': np.concatenate(parts['physics_attention']).astype(np.float32), 'reliability': np.concatenate(parts['reliability']).astype(np.float32), 'anchor_prediction': np.concatenate(parts['anchor']).astype(np.float64)}

def _empty_oof_buffers(fold: FoldInputs) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    experts = len(fold.pool.expert_ids)
    for task in CANONICAL_TASKS:
        size = sum((fold.roles[task][role].n_samples for role in ('train', 'val')))
        result[task] = {'sample_ids': np.empty(size, dtype='U256'), 'y_true': np.full(size, np.nan, dtype=np.float64), 'raw_prediction': np.full(size, np.nan, dtype=np.float64), 'data_prediction': np.full(size, np.nan, dtype=np.float64), 'physics_prediction': np.full(size, np.nan, dtype=np.float64), 'route_probabilities': np.full((size, 3), np.nan, dtype=np.float32), 'route_weights': np.full((size, 3), np.nan, dtype=np.float32), 'routing_components': np.full((size, 4), np.nan, dtype=np.float32), 'route_index': np.full(size, -1, dtype=np.int8), 'data_attention': np.full((size, experts), np.nan, dtype=np.float32), 'physics_attention': np.full((size, experts), np.nan, dtype=np.float32), 'reliability': np.full((size, CrossExpertProposalModel.RELIABILITY_DIM), np.nan, dtype=np.float32), 'anchor_prediction': np.full(size, np.nan, dtype=np.float64)}
    return result

def _write_oof_rows(*, destination: dict[str, np.ndarray], prediction: dict[str, np.ndarray], positions: np.ndarray, y_true: np.ndarray) -> None:
    destination['sample_ids'][positions] = prediction['sample_ids']
    destination['y_true'][positions] = np.asarray(y_true, dtype=np.float64)
    for key in PREDICTION_KEYS:
        destination[key][positions] = prediction[key]

def _check_oof_complete(candidate: str, task: str, arrays: dict[str, np.ndarray]) -> None:
    for key, values in arrays.items():
        if key == 'sample_ids':
            if np.any(np.asarray(values).astype(str) == ''):
                raise RuntimeError(f'Incomplete OOF sample IDs for {candidate}/{task}')
        elif key == 'route_index':
            if np.any(np.asarray(values) < 0):
                raise RuntimeError(f'Incomplete OOF routes for {candidate}/{task}')
        elif not np.isfinite(np.asarray(values)).all():
            raise RuntimeError(f'Non-finite OOF {key} for {candidate}/{task}')

def _ensemble_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError('At least one proposal record is required')
    if any((not np.array_equal(records[0]['sample_ids'], record['sample_ids']) for record in records[1:])):
        raise ValueError('Proposal ensemble sample IDs differ')
    anchor = np.asarray(records[0]['anchor_prediction'], dtype=np.float64)
    if any((not np.array_equal(anchor, r['anchor_prediction']) for r in records[1:])):
        raise ValueError('Ensemble anchors differ')
    raw_members = np.stack([np.asarray(record['raw_prediction'], dtype=np.float64) for record in records])
    residual_members = raw_members - anchor[None, :]
    denominator = np.mean(np.abs(residual_members), axis=0)
    directional = np.divide(np.abs(np.mean(residual_members, axis=0)), denominator, out=np.ones_like(denominator), where=denominator > 1e-12)
    route_members = np.stack([record['route_index'] for record in records])
    route_consensus = np.max(np.stack([(route_members == route).mean(axis=0) for route in range(3)]), axis=0)
    return {'sample_ids': np.asarray(records[0]['sample_ids']).astype(str), 'anchor_prediction': anchor, 'raw_prediction': raw_members.mean(axis=0), 'member_raw_prediction': raw_members, 'route_weights': np.mean([r['route_weights'] for r in records], axis=0).astype(np.float32), 'member_route_weights': np.stack([r['route_weights'] for r in records]), 'member_routing_components': np.stack([r['routing_components'] for r in records]), 'data_prediction': np.mean([record['data_prediction'] for record in records], axis=0), 'physics_prediction': np.mean([record['physics_prediction'] for record in records], axis=0), 'route_probabilities': np.mean([record['route_probabilities'] for record in records], axis=0).astype(np.float32), 'route_index': np.asarray(np.mean([record['route_probabilities'] for record in records], axis=0).argmax(axis=1), dtype=np.int8), 'route_consensus': route_consensus.astype(np.float32), 'directional_agreement': directional.astype(np.float32), 'data_attention': np.mean([record['data_attention'] for record in records], axis=0), 'physics_attention': np.mean([record['physics_attention'] for record in records], axis=0), 'reliability': np.mean([record['reliability'] for record in records], axis=0)}
PREDICTION_KEYS = ('raw_prediction', 'data_prediction', 'physics_prediction', 'route_probabilities', 'route_weights', 'routing_components', 'route_index', 'data_attention', 'physics_attention', 'reliability', 'anchor_prediction')
