import random, time
import numpy as np
import torch
from .model import CrossExpertProposalModel
from .objective_step import train_step
from .model import ProposalConfig
POLICY = {'version': 'jarvis-proposal-v1', 'variants': ['full'], 'seeds': [20261071, 20261072], 'steps': 1400, 'source_dropout': 0.05, 'learning_rate': 0.0003, 'weight_decay': 0.0001, 'architecture': 'model', 'regression_batch_size': 128, 'classification_batch_size': 128}
CONTRACT = {'batch_per_task': 128, 'updates': 1400}

def model_config(tasks, dims):
    from .model import ProposalConfig
    return ProposalConfig(task_names=tasks, expert_ids=tasks, hidden_dims=dims, anchor_indices=tuple(range(len(tasks))), token_dim=128, task_dim=32, residual_hidden=192, residual_rank=16, dropout=0.08, attention_top_k=4, residual_limit=1.25, route_temperature=0.75, use_cross_expert_input=True, use_data_branch=True, use_physics_branch=True, use_learned_task_source_relation=True, use_shared_low_rank_basis=False, use_task_private_correction=True, use_sample_gate=True)

def fit(inputs, variant, seed, inner, steps, progress=None):
    if variant != 'full':
        raise ValueError('Only the Full main model is released')
    fit_seed = seed + inner * 1000003
    random.seed(fit_seed)
    np.random.seed(fit_seed)
    torch.manual_seed(fit_seed)
    torch.cuda.manual_seed_all(fit_seed)
    rng = np.random.default_rng(fit_seed)
    model = CrossExpertProposalModel(model_config(inputs.tasks, inputs.dims)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=POLICY['learning_rate'], weight_decay=POLICY['weight_decay'])
    history = []
    started = time.time()
    for step in range(1, steps + 1):
        value, regret = train_step(inputs, variant, model, opt, rng, inputs.fit_ids, POLICY, 'cuda')
        if step == 1 or step % 10 == 0 or step == steps:
            rec = dict(time=time.time(), phase='training', inner=inner, variant=variant, seed=seed, fit_seed=fit_seed, step=step, total=steps, elapsed_seconds=time.time() - started, objective=value, regret=regret, optimizer_updates=step, batch_contract=CONTRACT, checkpoint_selection='fixed_final', val_labels_read=False, test_labels_read=False)
            history.append(rec)
            if progress:
                progress(rec)
    return (model, history)
