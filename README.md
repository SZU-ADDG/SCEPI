# SCEPI reproducibility

SCEPI is Single-mutant-Conditioned Epistasis Prediction. It predicts combination effects in a held-out protein from a shared source-trained model and permitted target single-mutant measurements. The primary S-all protocol supplies no target double-mutant labels for fitting or configuration selection.

## Contents

- `epistasis/`: model, likelihood, feature construction, metrics, and baseline interfaces.
- `scripts/`: source training, target adaptation, evaluation, aggregation, and plotting.
- `configs/environment.yml`: the recorded Linux Conda environment, without a machine-specific prefix.
- `reproducibility/environment.json`: recorded interpreter and package versions.
- `reproducibility/source_repositories.json`: exact upstream repositories and commits.
- `reproducibility/source_models.json`: selected source configurations, fitting permissions, and checkpoint epochs.
- `reproducibility/source_jobs.json`: portable commands for the 50 primary source fits and the 100 matched-control fits, including their single-effect initialization dependencies.
- `evidence/`: saved per-protein results, independently reconstructed table values, random expectations, and supplementary summaries.
- `paper/`: current manuscript, supplementary material, editable figure sources, and references.

## Environment

The recorded calculations use Python 3.11.17, PyTorch 2.3.1, NumPy 1.26.4, pandas 2.2.3, SciPy 1.14.1, and transformers 4.48.1. The full baseline environment also includes GPyTorch, MAVE-NN, MoCHI, TensorFlow, R, HMMER, and MMseqs2. It was checked with `pip check` in the system Conda environment.

Create one environment in the system Conda `envs/` directory. Keep environments outside the project checkout.

```bash
conda env create --name recomb-3-epistasis --file configs/environment.yml
conda activate recomb-3-epistasis
python -m pip check
```

MoCHI is installed from the recorded v1.1 source commit in the environment file. Third-party repositories are downloaded separately at the commits in `source_repositories.json`. Their licenses and pretrained-model terms remain applicable. The `fetch_sources.py` script fetches the historical primary repositories; the source manifest also identifies the additional functional and higher-order sources.

For the original server, use the existing system environment `/data1/doomx/anaconda3/envs/recomb-3-epistasis`. No new environment is needed there. On another Linux host, set the absolute path to its system-managed environment:

```bash
export SCEPI_CONDA_ENV=/absolute/system/conda/envs/recomb-3-epistasis
source scripts/runtime.sh
```

The release runtime configures `work/tmp/`, `cache/`, and the existing environment. It does not configure a proxy. Set a proxy explicitly if the host needs one. Historical queue launchers retain the original server paths; the portable source-job runner below uses the active Python interpreter.

## Reconstruct the reported tables without training

The compact evidence files contain real saved predictions aggregated within protein and repetition. The following command independently reconstructs the two 26-row primary tables and all 1,248 random-enrichment comparisons.

```bash
python reproduce.py --output work/reproduced
```

The command checks every reconstructed value against the saved reference summaries. It verifies five repetitions per eligible protein, candidate-pool equality, hypergeometric expectations, and enrichment denominators. It writes CSV tables and a verification manifest. This route requires the numerical Python dependencies and no GPU or sequence encoder.

The release also includes one trained checkpoint and 32 real, held-out quartets with their recorded features. Add the model check to reproduce their saved phenotype and residual predictions on CPU:

```bash
python reproduce.py --model-check --output work/reproduced
```

This check requires PyTorch. It compares CPU predictions with the previously saved evaluation predictions at a tolerance of 1e-5. The example contains no target combination training labels. The verification record is described in `reproducibility/VERIFICATION.md`.

The original experiment base commit is `875d04d6df818110001f3889758cf998321a96f6`. The release contains the source snapshot used for the current method, matched controls, reporting, and manuscript revisions. Its release manifest distinguishes the base commit from that snapshot.

## Data and frozen representations

Data source metadata and download manifests retain the original URLs and file sizes. Tsuboyama stability data originate from Zenodo record 7992926. ProteinGym and FLIP2 source metadata are supplied with the release. Raw datasets and pretrained encoders are downloaded from their original providers.

```bash
python scripts/download_data.py --groups tsuboyama
python scripts/extract_data.py data/raw/tsuboyama/Processed_K50_dG_datasets.zip data/raw/tsuboyama/AlphaFold_model_PDBs.zip data/raw/tsuboyama/Data_tables_for_figs.zip
python scripts/prepare_tsuboyama.py
python scripts/prepare_structures.py
python scripts/download_esm2.py
```

Frozen family partitions are supplied in `data/splits/families_30.parquet` and `families_25.parquet`. The source split seed is 20261009. The original scripts regenerate combination splits and single-label order:

```bash
python scripts/prepare_combination_splits.py
```

To regenerate the homology partitions independently, use `download_pfam.py` and `prepare_family_splits.py` with Pfam 37.0, HMMER 3.4, and the recorded MMseqs2 version. Supplied family files preserve the actual evaluation assignments.

Cache the ESM-2 reference and single-mutant representations for ranks 0 through 7. Each rank is an independent command on its assigned GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/cache_esm2.py --rank 0 --world-size 8 --stage singles --dataset tsuboyama
```

Run the same command with each remaining rank and assigned device. After all eight manifests complete:

```bash
python scripts/build_training_features.py
```

The encoder revision is `08e4846e537177426273712802403f7ba8261b6c` for `facebook/esm2_t33_650M_UR50D`. Single-mutant residue states are reused across candidate pairs. The main pair model does not encode every double-mutant sequence.

## Refit the selected source models

The portable job manifest records actual selected hyperparameters and single-effect initialization dependencies. Listing jobs does not start training:

```bash
python reproducibility/run_source_jobs.py --list
```

Refit one selected primary configuration with its exact prerequisite single model:

```bash
CUDA_VISIBLE_DEVICES=0 python reproducibility/run_source_jobs.py --arm interaction --threshold 30 --fold 0 --seed 11
```

The supported arms are `interaction`, `direct_residual`, and `no_pair_measured`. Thresholds are 30 and 25; folds are 0–4; seeds are 11, 23, 47, 89, and 131. Repeating the command across the recorded grid reproduces 50 primary and 100 control fits. Existing completed fits are retained and their manifests checked before reuse. Source validation determines the checkpoint and calibration weight. All target homology groups are excluded from source fitting and selection.

Source-model selection can also be regenerated with the original single and pair grid scripts. The main model compares projection dimensions 64/128, learning rates 1e-4/3e-4/1e-3, and auxiliary phenotype weights 0.1/1. Each fit uses at most 100 epochs and patience 10. The two matched controls reuse the primary model's selected configuration. They retain initialization, source rows, optimizer, objectives, stopping rules, and target permissions.

`direct_residual` keeps measured pair inputs and predicts the symmetric head output directly. `no_pair_measured` supplies predicted single effects and zero availability indicators inside the pair network while retaining external measured-single calibration.

## Full evaluation and supplementary methods

`run_sall_full_target.py` reconstructs the full-target truth and evaluates saved source/target predictions. `paper_full_target_evidence.py`, `integrate_key_ablations.py`, and `summarize_full_target_evidence.py` aggregate completed arms. `analyze_random_enrichment.py` computes exact random expectations from the frozen candidate pools. The compact evidence route above reproduces their reported primary summaries without rebuilding embeddings.

| Comparison | Entry points and retained settings |
| --- | --- |
| ThermoMPNN-D | `prepare_thermo_training.py`, `train_thermo.py`, `run_thermo_targets.py`; source-held-out refitting, native loss and symmetry penalty, permitted target singles only in S-all |
| Kermut | `run_kermut_targets.py`; full composite kernel and permitted target labels |
| StabilityArc + Kermut | `cache_stabilityarc.py`, `train_stabilityarc.py`, `run_stabilityarc_targets.py`; 50 held-out source models, four decoder layers, 256 hidden dimensions, 40-epoch budget, target GP calibration |
| Matched controls | `run_key_ablations.py`, `verify_key_ablations.py`; 100 source fits and identical primary evaluation permissions |
| Functional/FLIP2 | `run_functional_adaptation.py`, `run_supplementary_baselines.py`, `run_mean_flip.py`; predicted-single two-stage adaptation, evaluated separately from S-all |
| RFA | `fit_rfa.R`, `configs/rfa_fixed.json`; lambda=0, 500-iteration caps and recorded stopping criteria; 284 fixed-default and 66 previously validation-selected fits retained |
| ESM-MSR | Available epoch-10 checkpoint, family30/fold0/seed11 only; its additional training was discontinued and is outside the complete repeated main comparison |

Mutate Everything is included in the information-use comparison. It is not a row in the current matched full-target empirical comparison. No comparison score is inferred from its published benchmark.

The complete original-main and compensation-intersection summaries are retained separately from the primary S-all tables. Functional results use equal-assay averaging, including within-assay averaging of the three KRAS conditions. Twelve undefined RFA GFP Spearman values remain excluded only from common finite rank intersections; their residual values remain included.

## Statistical and selection definitions

The main phenotype/residual/selection cohorts contain 159/155/120 proteins. Selection requires two measured single effects at most -0.5 kcal/mol, an in-range complete quartet, reference stability at most 4.5 kcal/mol, and at least 100 candidates. A hit recovers at least 0.5 kcal/mol above the better constituent single. It can remain below the wild type. Candidate budgets are 1, 3, 10, and 30. Measurement margins are 0, 0.25, and 0.5 kcal/mol.

Each protein contributes one value after averaging its five fitting repetitions. Homology groups are resampled together. The primary paired analysis uses 10,000 group bootstrap draws with seed 20261013; the added random analysis uses seed 20261019. Enrichment divides the equal-protein model hit rate by the equal-protein random expectation. It does not divide pooled hit counts or average undefined per-protein ratios.

The standard random expectation is 2.6290798%, based on 79,564 candidates and 1,718 measured hits. SCEPI native top-10 hit rate is 11.2166667%, giving 4.266385-fold enrichment. Saved predictions and candidate pools are unchanged by the added analysis.

## License and upstream materials

The SCEPI source is provided under the repository license. Original datasets, upstream repositories, and pretrained models retain their own terms. Download third-party code and data from the recorded original sources. The manuscript and supplementary tables identify implementation choices required by incomplete upstream training specifications.
