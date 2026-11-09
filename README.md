# Learning when to share

Code, data, prediction results and model weights for HP-SafeMoE are available in this repository.

## Download the complete repository

Code, configurations, summary results and figures are stored in the Git repository.
Model weights, prediction arrays and larger datasets are provided in the
[v1.0.0 release](https://github.com/gllary/Learning_when_to_share_Public/releases/tag/v1.0.0).
Restore these files before running the evaluation, prediction or training commands:

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

Copy or unpack the complete repository, preserving its internal relative symbolic
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

## Bundled JMP license

The bundled JMP source is subject to the following license. Copyright notices in its source files apply.

Attribution-NonCommercial 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
	wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More_considerations
     for the public: 
	wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution-NonCommercial 4.0 International Public
License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution-NonCommercial 4.0 International Public License ("Public
License"). To the extent this Public License may be interpreted as a
contract, You are granted the Licensed Rights in consideration of Your
acceptance of these terms and conditions, and the Licensor grants You
such rights in consideration of benefits the Licensor receives from
making the Licensed Material available under these terms and
conditions.

Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.
  d. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  e. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  f. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  g. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  h. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  i. NonCommercial means not primarily intended for or directed towards
     commercial advantage or monetary compensation. For purposes of
     this Public License, the exchange of the Licensed Material for
     other material subject to Copyright and Similar Rights by digital
     file-sharing or similar means is NonCommercial provided there is
     no payment of monetary compensation in connection with the
     exchange.

  j. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  k. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  l. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.

Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part, for NonCommercial purposes only; and

            b. produce, reproduce, and Share Adapted Material for
               NonCommercial purposes only.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties, including when
          the Licensed Material is used other than for NonCommercial
          purposes.

Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.

       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

       4. If You Share Adapted Material You produce, the Adapter's
          License You apply must not prevent recipients of the Adapted
          Material from complying with this Public License.

Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database for NonCommercial purposes
     only;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material; and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.

Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.

Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.

Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.

Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.

=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.
