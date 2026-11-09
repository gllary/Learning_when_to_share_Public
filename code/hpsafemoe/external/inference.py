"""Label-free frozen Full inference; applies benchmark-fitted beta without fitting."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from evaluation import apply_frozen_beta, ensemble_outer, log_youngs, sha256, vector

MODEL_SCHEMA_SHA256 = "aa34c9a2073de0aa989eabef8e986f87d306f38f8c88a2161e71a70bd04f4a82"
EXTERNAL_TASKS = {"dielectric", "expt_gap", "log_kvrh", "log_gvrh"}


def standardize_inputs(model_config, normalization, task, exports, rows):
    """Same raw-prediction/log1p(U)/anchor transforms as the frozen parent."""
    if task not in EXTERNAL_TASKS:
        raise ValueError("This entry point only permits the four regression target tasks")
    names = model_config["expert_ids"]
    target = normalization[task]["target"]
    for section in ("expert_prediction", "expert_uncertainty"):
        if set(normalization[task][section]) != set(names):
            raise ValueError("Frozen normalization source set differs")
    stats = [target] + list(normalization[task]["expert_prediction"].values()) + list(normalization[task]["expert_uncertainty"].values())
    if any(not np.isfinite([s["mean"], s["scale"]]).all() or s["scale"] <= 0 for s in stats):
        raise ValueError("Invalid frozen statistics")
    if set(exports) != set(names):
        raise ValueError("Explicit entries, including unavailable sources, are required for all experts")
    hidden, pred, unc, available = {}, {}, {}, []
    for name in names:
        export = exports[name]
        if export is None:
            h = np.zeros((len(rows), model_config["hidden_dims"][name]), dtype=np.float32)
            p = np.zeros(len(rows), dtype=np.float32)
            u = np.zeros(len(rows), dtype=np.float32)
            a = np.zeros(len(rows), dtype=bool)
        else:
            h = np.asarray(export["hidden"][rows], dtype=np.float32)
            p_raw = np.asarray(export["prediction"][rows], dtype=np.float64)
            u_raw = np.log1p(np.asarray(export["uncertainty"][rows], dtype=np.float64))
            ps = normalization[task]["expert_prediction"][name]
            us = normalization[task]["expert_uncertainty"][name]
            p = ((p_raw - ps["mean"]) / ps["scale"]).astype(np.float32)
            u = ((u_raw - us["mean"]) / us["scale"]).astype(np.float32)
            a = np.asarray(export["available"][rows], dtype=bool)
        hidden[name], pred[name], unc[name] = h, p, u
        available.append(a)
    task_index = model_config["task_names"].index(task)
    anchor_id = names[model_config["anchor_indices"][task_index]]
    anchor_export = exports[anchor_id]
    if anchor_export is None or not np.asarray(anchor_export["available"])[rows].all():
        raise ValueError("A finite available anchor is required for every external sample")
    anchor = vector(anchor_export["prediction"][rows], len(rows))
    anchor_standard = ((anchor - target["mean"]) / target["scale"]).astype(np.float32)
    return {"hidden": hidden, "prediction_standard": pred, "uncertainty_standard": unc,
            "available": np.stack(available, axis=1), "anchor": anchor,
            "anchor_standard": anchor_standard, "task_index": np.full(len(rows), task_index)}


def load_checkpoint(record, outer, inner, seed, device):
    import torch
    from frozen_model import ProposalConfig, CrossExpertProposalModel

    if sha256(record["path"]) != record["sha256"]:
        raise ValueError("Frozen checkpoint hash changed")
    payload = torch.load(record["path"], map_location="cpu", weights_only=False)
    for key in ("schema_version", "feature_schema"):
        value = payload.get(key)
        if not isinstance(value, str) or hashlib.sha256(value.encode()).hexdigest() != MODEL_SCHEMA_SHA256:
            raise ValueError(f"Wrong frozen checkpoint format identity: {key}")
    expected = {"routing_mode": "softmax_weighted_residual", "candidate": "proposal_full",
                "training_steps": 1400, "outer_fold": outer, "inner_fold": inner,
                "seed": seed, "fit_seed": seed + inner * 1000003, "outer_test_labels_accessed": False}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Wrong frozen checkpoint identity: {key}")
    config = payload["model_config"]
    if config["route_temperature"] != .75 or not all(config[k] for k in (
            "use_data_branch", "use_physics_branch", "use_cross_expert_input", "use_learned_task_source_relation")):
        raise ValueError("Only the frozen Full model configuration is allowed")
    model = CrossExpertProposalModel(ProposalConfig(**config)).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 3549181:
        raise ValueError("Full model parameter count differs")
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite frozen state")
    model.requires_grad_(False)
    model.eval()
    return model, config, payload["normalization"]


def predict_member(model, config, normalization, task, exports, n_rows, device, chunk=384):
    import torch
    predictions, anchors, routes = [], [], []
    with torch.inference_mode():
        for start in range(0, n_rows, chunk):
            rows = np.arange(start, min(start + chunk, n_rows), dtype=np.int64)
            x = standardize_inputs(config, normalization, task, exports, rows)
            tensor_inputs = {key: {name: torch.as_tensor(value, device=device) for name, value in x[key].items()}
                             for key in ("hidden", "prediction_standard", "uncertainty_standard")}
            output = model(task_index=torch.as_tensor(x["task_index"], dtype=torch.long, device=device),
                           anchor_standard=torch.as_tensor(x["anchor_standard"], device=device),
                           available=torch.as_tensor(x["available"], dtype=torch.bool, device=device), **tensor_inputs)
            weights = output.route_weights
            mixture = (torch.as_tensor(x["anchor_standard"], device=device)
                       + weights[:, 1] * output.data_residual + weights[:, 2] * output.physics_residual)
            if not torch.allclose(output.raw_standard, mixture, atol=1e-6, rtol=1e-6):
                raise ValueError("Member output is not the expected weighted correction")
            if not torch.allclose(weights.sum(1), torch.ones(len(rows), device=device), atol=1e-6, rtol=0):
                raise ValueError("Route weights are not normalized")
            target = normalization[task]["target"]
            restored = output.raw_standard.detach().cpu().numpy().astype(np.float64) * target["scale"] + target["mean"]
            predictions.append(vector(restored, len(rows)))
            anchors.append(x["anchor"])
            routes.append(weights.detach().cpu().numpy())
    return {"raw_prediction": np.concatenate(predictions), "anchor_prediction": np.concatenate(anchors),
            "route_weights": np.concatenate(routes)}


def combine_task_members(records, beta):
    if len(records) != 10:
        raise ValueError("Exactly 10 independent frozen members are required per outer/task")
    anchor = vector(records[0]["anchor_prediction"])
    if any(not np.array_equal(anchor, record["anchor_prediction"]) for record in records):
        raise ValueError("Members do not share the exact same outer Stage1 anchor")
    proposal = np.mean(np.stack([vector(r["raw_prediction"], len(anchor)) for r in records]), axis=0)
    return {"stage1": anchor, "stage2": apply_frozen_beta(anchor, proposal, beta),
            "proposal_internal": proposal}


def combine_dataset(dataset, outer_task_predictions, sample_ids):
    stage1, stage2 = [], []
    if set(outer_task_predictions) != set(range(5)):
        raise ValueError("All five outer models are required")
    for outer in range(5):
        tasks = outer_task_predictions[outer]
        for method, destination in (("stage1", stage1), ("stage2", stage2)):
            if dataset in ("HEA-95", "B2-18"):
                p = log_youngs(tasks["log_kvrh"][method], tasks["log_gvrh"][method])
            else:
                task = "dielectric" if dataset == "Core-23" else "expt_gap" if dataset == "DS2-248" else None
                if task is None:
                    raise ValueError("Unknown dataset")
                p = vector(tasks[task][method], len(sample_ids))
            destination.append(vector(p, len(sample_ids)))
    return {"sample_ids": np.asarray(sample_ids).astype(str), "outer_stage1": np.stack(stage1),
            "outer_stage2": np.stack(stage2), "stage1": ensemble_outer(stage1), "stage2": ensemble_outer(stage2)}


def require_stage1_replay_ready(path, expected_dataset, expected_ids, expected_pool_sha):
    """A blocking contract, not a bypass for missing feature provenance."""
    receipt = json.loads(Path(path).read_text())
    required = {"dataset": expected_dataset, "sample_ids": list(expected_ids),
                "pool_sha256": expected_pool_sha, "labels_accessed": False,
                "weights_updated": False, "providers_and_checkpoint_hashes_verified": True,
                "all_required_descriptors_verified": True, "real_U_preserved": True,
                "target_source_availability_contract_verified": True,
                "fresh_current_stage1_forward": True,
                "old_external_outputs_used": False,
                "current_input_validation_passed": True}
    for key, value in required.items():
        if receipt.get(key) != value:
            raise ValueError(f"External Stage1 replay is not validated: {key}")
    if not receipt.get("files") or not receipt.get("checkpoint_hashes"):
        raise ValueError("Missing Stage1 export/checkpoint inventory")
    for name, digest in receipt["files"].items():
        if sha256(Path(path).parent / name) != digest:
            raise ValueError("Stage1 export changed")
    return receipt
