# Learning when to share: calibrated residual corrections for materials property prediction

Code, data, prediction results and model weights for “Learning when to share: calibrated residual corrections for materials property prediction”  （HP-SafeMoE） are available in this repository.

## Download the complete repository

Code, configurations, prediction arrays, summary tables and figures are stored in Git.
Published results can be browsed or downloaded directly from [results/](results/)
without downloading the release archive.

Model weights and larger input datasets are available in the
[v1.0.0 release](https://github.com/gllary/Learning_when_to_share_Public/releases/tag/v1.0.0).
The download command below restores models and supporting data for training,
inference and full result verification.

See [Model weights](checkpoint/README.md) for the released models and their
locations, and [Data and evaluation artifacts](data/README.md) for the datasets,
saved predictions and analysis files.

```bash
gh repo clone gllary/Learning_when_to_share_Public
cd Learning_when_to_share_Public
python3 code/scripts/download_artifacts.py
```

The downloader requires Python 3.10 or later and uses only its standard library.
It downloads the numbered archive parts, restores the original directory structure,
and removes the downloaded parts after successful extraction. Allow approximately
130 GB of free disk space during download and extraction. A complete checkout
occupies approximately 64 GB. Downloaded parts can be reused if a download is interrupted.
The internal symbolic links require a filesystem that supports symbolic links.

## Repository contents

| Path | Contents |
|---|---|
| `code/hpsafemoe/stage1/` | Expert training and export implementations |
| `code/hpsafemoe/stage2_proposal/` | Matbench and JARVIS models and training |
| `code/hpsafemoe/stage2_calibration/` | Calibration implementations |
| `code/hpsafemoe/external/` | External inference and evaluation |
| `code/hpsafe_sota/`, `code/hpsafe_features/` | Data loaders, features and checkpoint adapters |
| `code/scripts/` | Command-line entry points listed below |
| `code/tests/` | Unit and artifact-contract tests |
| `code/environments/` | Expert environment YAML files and recorded package versions |
| `code/vendor/` | Bundled third-party runtime source |
| `checkpoint/matbench/`, `checkpoint/jarvis/` | Stage1/Stage2 weights, preprocessing artifacts and training configurations |
| `data/matbench/`, `data/jarvis/` | Benchmark identities, splits, labels and saved expert inputs |
| `data/external/` | Core-23, DS2-248, HEA-95 and B2-18 materials and labels |
| `data/analysis/representation/` | Complete task IDs, folds, descriptors, reduced representations, PCA coordinates and permutation draws; original frozen exports are reused |
| `records/proposal_training/` | OOF arrays, normalization parameters and per-member numerical outputs |
| `records/calibration/` | Calibration records and saved coefficients |
| `records/expert_and_routing/` | Expert and routing arrays |
| `records/external/` | Frozen calibration coefficients for external evaluation |
| `records/analysis/` | Analysis seeds, source-weight replay arrays, Supplementary Table 15 bootstrap draws and group assignments, figure source data and analysis configurations |
| `results/matbench/`, `results/jarvis/` | Experiment configurations and benchmark predictions |
| `results/external/` | Cohort inputs, predictions and metrics |
| `results/tables/` | Summary results in JSON and CSV |
| `results/analysis/` | Full-precision representation, source-weight and Supplementary Table 15 tables, and regenerated analysis figures |
| `manifests/` | Experiment registry, artifact indexes |
| `runs/` | New execution outputs; created by the commands below |

Benchmark prediction accepts these experiment IDs:

| Experiment ID | Result directory |
|---|---|
| `matbench_full` | `results/matbench/full/` |
| `matbench_no_cross` | `results/matbench/ablations/no_cross/` |
| `matbench_no_safety_calibration` | `results/matbench/ablations/no_safety_calibration/` |
| `jarvis_full` | `results/jarvis/full/` |

External entry points accept `core23`, `ds2_248`, `hea95` and `b2_18`.

## Working directory and installation

Clone or unpack the repository, preserving its internal relative symbolic
links. Run every command below from the repository root, which contains this
README and `pyproject.toml`:

```bash
cd /path/to/Learning_when_to_share_Public
conda create -n hpsafemoe python=3.11 -y
conda activate hpsafemoe
python -m pip install .
```

The base dependencies are NumPy, PyYAML, scikit-learn, orjson and ijson. Evaluating
saved results requires only this environment. For Stage2 inference, model tests
and training, also install PyTorch; the recorded Stage2 version is 2.5.1:

```bash
python -m pip install "torch==2.5.1" "numpy==1.26.4" "scikit-learn==1.5.2"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Stage2 prediction supports `--device cpu` or `--device cuda:0`. Member training
uses CUDA and requires an NVIDIA GPU with a compatible driver. Recorded package
versions are in `code/environments/*.versions.json`.

For direct module imports from the repository root, use `PYTHONPATH="$PWD/code"`.
Each new execution needs a new output directory under `runs/`; the entry points
reject an existing output directory.

## Evaluate saved predictions

With the `hpsafemoe` environment active, run:

```bash
python code/scripts/evaluate_benchmarks.py
python code/scripts/verify_calibration_records.py
python code/scripts/evaluate_external.py all
```

These commands print checks or metrics to the terminal.
To evaluate one external cohort or print a summary table:

```bash
python code/scripts/evaluate_external.py core23
python code/scripts/rebuild_paper_tables.py matbench
python code/scripts/rebuild_paper_tables.py jarvis
python code/scripts/rebuild_paper_tables.py external
```

Run the complete test suite after installing PyTorch and the analysis dependencies:

```bash
python -m pip install '.[analysis]'
python -m unittest discover -s code/tests -v
```

## Reproduce the numerical figures

```bash
python -m pip install '.[analysis]'
python code/scripts/plot_analysis_figures.py --figures all --output runs/figures
```

Use `fig2`, `fig3` or `fig4` for the main numerical figures. Supplementary figure
keys are `s1` (Matbench fold boxplots), `s2` (JARVIS source weights and coverage),
`s3` (descriptor CKA), `s4` (PCA) and `s5` (permutation reference).
The file `manifests/analysis_artifacts.json` lists the corresponding figure assets.

Correction coverage is the proportion of test records with a fixed calibration
coefficient greater than zero. Counts are pooled across outer test folds before
calculating the percentage. Source-weight aggregation uses the records whose
final predictions differ numerically from their target anchors.

## Run Stage2 prediction with the included expert inputs

```bash
python code/scripts/predict_benchmark.py --experiment jarvis_full --device cuda:0 --output runs/jarvis_prediction
python code/scripts/predict_benchmark.py --experiment matbench_full --outer 0 --device cuda:0 --output runs/matbench_outer0_prediction
```

Omit `--outer` to run all five Matbench outer folds. Substitute any Matbench
experiment ID from the table above as needed. Outputs are written to
`<output>/predictions/` and `<output>/predictions_committed.json`.

For an external cohort using the included Stage1 exports:

```bash
python code/scripts/predict_external.py core23 --device cpu --output runs/core23_prediction
```

Substitute another cohort ID as needed. The output contains `predictions.npz`,
`members/` and `predictions_committed.json`.

## Train a new Stage2 member

Run in the GPU-enabled `hpsafemoe` environment, from the repository root:

```bash
python code/scripts/train_proposal_member.py --dataset jarvis --inner 0 --seed 20261071 --output runs/jarvis_member_i0_s71
python code/scripts/train_proposal_member.py --dataset matbench --outer 0 --inner 0 --seed 20261071 --output runs/matbench_member_o0_i0_s71
python code/scripts/train_proposal_member.py --dataset matbench --no-cross --outer 0 --inner 0 --seed 20261071 --output runs/matbench_no_cross_o0_i0_s71
```

Each command saves `final.pt` and `run_config.json`, together with training history
or progress files. Use inner indices `0` through `4` and seeds `20261071` and
`20261072`; Matbench additionally uses outer indices `0` through `4`. Give every
combination a separate output directory. Training configurations are in
`checkpoint/<dataset>/stage2_proposal/model/training_config.json`.

## Run calibration

The input NPZ must contain `anchor`, `raw` and either `y_oof` for Matbench or
`y_train_oof` for JARVIS. Add `--classification` for a classification task.
These commands use included records:

```bash
python code/scripts/calibrate_task.py --dataset matbench --oof results/matbench/full/calibration_records/outer_0/expt_gap.npz --seed 20261171 --output runs/matbench_expt_gap_calibration
python code/scripts/calibrate_task.py --dataset jarvis --oof results/jarvis/full/calibration_records/reg_formation_energy_peratom.npz --seed 20261074 --output runs/jarvis_formation_calibration
```

Outputs are `calibration.json` and `calibration_records.npz`. For newly trained
members, supply their OOF predictions in this schema. The member-training command
produces a member checkpoint; it does not assemble the full OOF file.

## Generate fresh Stage1 exports for external prediction

This workflow uses Linux, an NVIDIA GPU and separate expert environments.
Install the MatterVial and JMP environments from the repository root:

```bash
conda env create -f code/environments/mattervial.yml
conda run -n hpsafe-mattervial python -m pip install --no-deps "modnet==0.4.5"
conda env create -f code/environments/jmp_l.yml
```

For the glass expert, add the recorded feature and estimator dependencies to the
`hpsafemoe` environment, then prepare a workspace:

```bash
conda activate hpsafemoe
python -m pip install "pymatgen==2024.8.9" "pandas==2.2.3" "scipy==1.14.1" "xgboost==2.1.4" "joblib==1.4.2"
python code/scripts/prepare_fresh_stage1.py --output runs/fresh_stage1
```

Preparation creates `job_commands.json`, `stage1_sources.json`, `protocol.json`
and input links in `runs/fresh_stage1/`. Glass jobs use the `hpsafemoe`
environment prepared above.
Execute all outer indices and JMP tasks from the repository root:

```bash
export HPSAFEMOE_EXTERNAL_WORKSPACE="$PWD/runs/fresh_stage1"
export CUDA_VISIBLE_DEVICES=0

for outer in 0 1 2 3 4; do
  conda run --no-capture-output -n hpsafe-mattervial python code/hpsafemoe/external/forward_stage1.py mattervial "$outer"
  conda run --no-capture-output -n hpsafemoe python code/hpsafemoe/external/forward_stage1.py glass "$outer"
  for task in jdft2d perovskites phonons mp_gap mp_e_form; do
    conda run --no-capture-output -n hpsafe-jmp-l python code/hpsafemoe/external/forward_stage1.py jmp "$outer" --task "$task"
  done
done
```

Exports are written under `runs/fresh_stage1/stage1_exports/<dataset>/`.
Run Stage2 against that new directory:

```bash
conda activate hpsafemoe
python code/scripts/predict_external.py core23 --stage1-root runs/fresh_stage1/stage1_exports --device cuda:0 --output runs/core23_fresh_prediction
```

Additional Stage1 environments are defined in `code/environments/anchorboost.yml`,
`tpot_mat.yml`, `mattervial_features.yml` and `mattervial_orb.yml`. Expert-specific
training/export source is under `code/hpsafemoe/stage1/`; these modules use their
own configuration and environment inputs. The commands above provide the result
evaluation, Stage2 training/prediction and external Stage1 export entry points.

## License

The MIT License below applies to original HP-SafeMoE code. Third-party code retains
its own copyright notices and license terms.

### HP-SafeMoE code

MIT License

Copyright (c) 2026 HP-SafeMoE authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Third-party code

`code/vendor/jmp/` includes an adapted copy of [JMP](https://github.com/facebookresearch/JMP),
Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Most JMP source is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en);
see the [upstream license](https://github.com/facebookresearch/JMP/blob/main/LICENSE.md)
for the full terms and disclaimer.

Components covered by separate licenses retain those terms, as listed in
[JMP's licensing notes](https://github.com/facebookresearch/JMP#license) and the source-file notices.
Third-party components identified as MIT-licensed use the MIT terms above
with their respective copyright holders.
