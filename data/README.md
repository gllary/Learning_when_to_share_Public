# Data and evaluation artifacts

The repository and the [v1.0.0 release](https://github.com/gllary/Learning_when_to_share_Public/releases/tag/v1.0.0) together provide the benchmark datasets, external evaluation cohorts, saved expert inputs, prediction results and analysis data listed below. Small metadata and summary files are stored directly in Git, while larger datasets and numerical arrays are included in the same release archive as the model weights. The download command in the [main README](../README.md#download-the-complete-repository) restores both models and data in one step.

All paths below are relative to the repository root after extraction.

| Data included | Location after extraction |
|---|---|
| **Matbench — 13 tasks:** task datasets, target labels, sample identifiers and saved fold assignments | `data/matbench/raw_inputs/` and `data/matbench/manifests/` |
| **JARVIS — 34 tasks:** source dataset snapshot, crystal structures, descriptors, graph inputs, task labels and split assignments | `data/jarvis/stage1_pool/data/`; task definitions are listed in `data/jarvis/registry.json` |
| **Benchmark Stage1 exports:** saved expert predictions, hidden representations and associated metadata used as Stage2 inputs | `data/matbench/stage1_pool/exports/` and `data/jarvis/stage1_pool/artifacts/` |
| **External evaluation cohorts:** Core-23, DS2-248, HEA-95 and B2-18, containing 23, 248, 95 and 18 samples, respectively; material descriptions, target labels and feature arrays are included | `data/external/core23/`, `data/external/ds2_248/`, `data/external/hea95/` and `data/external/b2_18/` |
| **Stage2 training and calibration data:** out-of-fold predictions, normalization parameters, calibration inputs and fitted coefficients | `records/proposal_training/` and `records/calibration/` |
| **Expert and routing outputs:** numerical arrays supporting expert-level and routing analyses | `records/expert_and_routing/` |
| **Benchmark predictions:** saved outputs for the full Matbench and JARVIS models and the Matbench ablation variants | `results/matbench/full/`, `results/matbench/ablations/` and `results/jarvis/full/` |
| **External predictions and evaluation results:** cohort predictions, metrics and associated evaluation records | `results/external/<cohort>/`; frozen calibration coefficients are also available under `records/external/<cohort>/` |
| **Figure and table data:** representation descriptors, PCA coordinates, permutation results, source weights, bootstrap distributions and numerical figure inputs | `data/analysis/representation/` and `records/analysis/`; derived tables and figures are stored under `results/tables/` and `results/analysis/` |

These files support the evaluation, Stage2 prediction, calibration and figure-reproduction commands documented in the [main README](../README.md). Sample identifiers and saved split assignments provide the correspondence between dataset entries, expert inputs and prediction outputs.
