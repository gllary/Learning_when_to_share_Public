# Model weights

See the [main README](../README.md#download-the-complete-repository) for download and setup instructions.

The [v1.0.0 release](https://github.com/gllary/Learning_when_to_share_Public/releases/tag/v1.0.0) contains the trained expert models and HP-SafeMoE Stage2 checkpoints listed below, together with their supporting data and prediction results. Files named `hpsafemoe-artifacts.tar.gz.part-000` through `part-049` are consecutive pieces of **one compressed archive**; they do not correspond to individual models. From the repository root, run `python3 code/scripts/download_artifacts.py` to download all 50 parts and restore the files to their original locations.

All paths below are relative to the repository root after extraction.

| Model weights included | Location after extraction |
|---|---|
| **Matbench Stage1 — JMP-L:** experts for `jdft2d`, `perovskites`, `phonons`, `mp_gap` and `mp_e_form` | `checkpoint/matbench/stage1/jmp_l_*_anchor/` |
| **Matbench Stage1 — MatterVial/MODNet:** experts for `dielectric`, `expt_gap`, `expt_is_metal`, `log_gvrh`, `log_kvrh` and `mp_is_metal` | `checkpoint/matbench/stage1/mattervial_*_anchor/` |
| **Matbench Stage1 — AnchorBoost and TPOT-Mat:** experts for `glass` and `steels`, respectively | `checkpoint/matbench/stage1/anchorboost_glass_anchor/` and `checkpoint/matbench/stage1/tpot_mat_steels_anchor/` |
| **Matbench Stage1 — inner-fold models:** MatterVial, AnchorBoost and TPOT-Mat models used to generate out-of-fold predictions | `checkpoint/matbench/stage1/inner_oof/` |
| **JARVIS Stage1:** task-specific AtomNet, PRISM, GSMNet, coGN, coNGN, ALIGNN, and matminer-based Random Forest/XGBoost experts | `data/jarvis/stage1_pool/artifacts/<task>/`; PyTorch and scikit-learn model files are also available under `checkpoint/jarvis/stage1/members/<task>/` |
| **JARVIS Stage1 — public ALIGNN checkpoints** | `checkpoint/jarvis/stage1/public_alignn/<task>/` |
| **Matbench Stage2 — HP-SafeMoE:** full and no-cross-expert variants, organized by outer fold, inner fold and ensemble member | `checkpoint/matbench/stage2_proposal/model/output/proposal_training/` |
| **JARVIS Stage2 — HP-SafeMoE:** full-model ensemble, organized by inner fold and ensemble member | `checkpoint/jarvis/stage2_proposal/model/stage2/full/` |

Model files use `.ckpt`, `.pt`, `.pkl`, `.joblib` or `.weights.h5`, depending on the model implementation. The JSON files visible in the GitHub `checkpoint/` directory contain configurations and metadata; the model files appear locally after downloading and extracting the release. Calibration coefficients and records are provided separately under `records/calibration/` and `records/external/`.
