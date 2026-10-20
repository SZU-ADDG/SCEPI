# SCEPI evidence ledger

Snapshot date: 2026-10-21. The manuscript's primary results use source-selected full-target S-all evaluation. Fixed unseen-site-pair intersections and the initial comparison-design results are recorded separately. Dated entries below retain their original experimental scope.

The formal method name is SCEPI, Single-mutant-Conditioned Epistasis Prediction. The existing `interaction` experiment identifier denotes the primary model. Target-combination fitting variants are labeled SCEPI adaptation and retain their separately documented input and training conditions. Author and affiliation details follow the supplied author list, with Jiayu Zhang affiliated with EasternDawn, Shenzhen, China.

## Final numerical-input and availability-indicator evidence

The final source-selection program reads 240 source-fit records and their source-validation histories. Its score is 0.5 Z + 0.25 S128 + 0.25 S-all residual RMSE; omega is selected with source-validation S128. Nine outer partitions select the combined objective and family30/fold3 selects phenotype-only. Those decisions are fixed across five fitting repetitions. No target prediction or metric is read by this selection program. The fifty full-model fits are reused. Two hundred matched controls contain ninety reusable fits and 110 newly completed fits; all 300 associated evaluations are complete.

The four input cells independently vary calibrated numerical values and observed indicators inside the pair function. They retain the external calibration rule, WT gating, parameter count, source initialization, training/validation rows, objective, and stopping criterion. Under family30, full/values-only/indicators-only/predicted-single residual RMSE is 0.590416/0.594816/0.771990/0.767195 kcal/mol; native hit rate is 11.2167/11.4333/8.1000/8.2333 percent. Under family25, residual RMSE is 0.604329/0.620695/0.817131/0.813878 and native hit rate is 11.1500/11.1167/7.9500/8.1500 percent. Numerical values retain most of the residual gain without indicators. The indicator increment is smaller and its native selection effect varies. Direct-head residual RMSE is 0.632127/0.630207, with native hit-rate differences of +0.6833/−0.2333 percentage points for cycle subtraction.

Final evidence is `runs/conditioning_factorial_20261021/`. Its 32-arm matrix contains 1,600 evaluations, with 14,446 paired comparisons independently reconstructed under both thresholds. Maximum mean difference is 2.22e-16. All 200 fitting conditions and 1,460 real implementation-check cycles are verified. The analysis contains 120 factorial contrasts, 220 group-sensitivity comparisons, all four source-selected single-label budgets, and 150 designated site-pair-subset evaluations. The latter gives 0.545048 residual RMSE on 70 proteins and 5.87566 percent native compensation hits on 63 proteins.

Random enrichment includes 1,536 comparisons. On the unchanged 79,564-candidate pool, budget 1/3/10/30 enrichments are 5.261664/4.796778/4.266385/3.277446. Top-10 recovery gain over random is 0.637196 kcal/mol. Equal-group, maximum-group removal, and all leave-one-group-out analyses retain positive mean input-removal gains in residual and native hit rate. Recovery comparisons with external methods and mean ridge include paired intervals and group-weight sensitivity.

Protocol verification reads all family edges, combination splits, raw-data field definitions, Kermut optimization traces, and ThermoMPNN-D source manifests. Numerical source inputs use the stated WT-relative scale. The four-state/direct-head CPU profile records equal parameter count and different call counts; measured head-only median times are 6.880/3.388 ms on 32 real samples. The new generic prediction CLI reads no target double-mutant outcome.

The final source passes the built-in LaTeX compiler. The submission has ten main pages and one reference page; the supplement has eighteen pages. All 29 source pages were visually inspected, all fonts are embedded, internal links have valid destinations, and exported pages match their source rasterization at 110 dpi. Manuscript checks cover 64 full-table rows, five main conditioning rows, 29 paired-interval rows, source-selected budgets, the site-pair subset, 485 original-main cells, and all 1,536 random comparisons. The compact release reconstructs 384 primary values, 10,752 random values, and 2,400 conditioning/sensitivity values. Candidate-pool matching covers 908,800 records. Thirty-two real held-out examples pass both CPU model inference and the public prediction CLI at tolerance 1e-5. The release records 250 portable source jobs and passes Python compilation. Publication and downloaded-archive verification retain their own release manifests.

## 2026-10-19 information-use and random-enrichment analysis

Table 1 distinguishes source single-label supervision, target calibration or posterior fitting, and measured inputs inside the pair function. The three comparison sources are ThermoMPNN-D, Mutate Everything, and StabilityArc + Kermut. Mutate Everything remains a methodological comparison; no published score is substituted for a matched experiment.

The added analysis uses unchanged candidate pools and saved predictions. The primary 120-protein pool contains 79,564 candidates and 1,718 measured hits. Its equal-protein random hit expectation is 0.02629079787470596, and its pooled candidate fraction is 0.021592680106580865. Reported enrichment uses the former. SCEPI's native top-10 rate of 0.11216666666666668 gives 4.266385037-fold enrichment, an 8.587586879-percentage-point hit-rate gain, and a 0.642699674 kcal/mol recovery gain. Budget 1/3/10/30 enrichments are 5.515237716/4.902433524/4.266385037/3.256314682. At margin 0.5, native enrichments are 4.147006 and 4.192506 for the 30% and 25% groupings.

Source: `runs/reviewer_revision_20261019/random_enrichment/manifest.json`, `summary.json`, and per-protein Parquet files on the original server. Exact hypergeometric moments, unchanged pool sizes, five repetitions, and all 738,400 paired records were checked. There are 1,248 complete comparisons. Added manuscript cells and the embedded Figure 2 fragment are checked by `scripts/check_random_enrichment_paper.py`.

The initial compact release independently reconstructs 312 primary values and 8,736 random-analysis values at tolerance 1e-12. Its CPU example reproduces 32 actual held-out cycles at tolerance 1e-5. Source code, 150 selected source-job configurations, frozen assignments, recorded environment and upstream versions, compact evidence, and the example were published at `https://github.com/SZU-ADDG/SCEPI`, commit `f88a24ba69c977cdc3b73bff03e8d02ff32693f1`. The downloaded public archive passed these checks and Python compilation with its portable runtime and the existing system Conda environment. At that release, its manuscript and PDFs matched the formal files byte for byte, with ten main pages, one reference page, and fourteen supplement pages. The source experiment base commit remains `875d04d6df818110001f3889758cf998321a96f6`. That publication record is preserved in `runs/reviewer_revision_20261019/`; the current manuscript and delivery record are described below.

## 2026-10-18 to 2026-10-20 external-baseline evaluation and layout

The final 50 RFA evaluations and their statistical aggregation are complete. The current authorized scopes contain 12,622 original-main evaluations, 2,251 compensation-intersection evaluations, and 1,151 full-target S-all evaluations. All three scopes have complete saved results. Additional ESM-MSR training was discontinued on user instruction. Its available epoch-10 checkpoint is reported on the matched single-fold, single-repetition subset. The original task lists retain the discontinued scope.

RFA supplies 350 eligible target fits from 380 task records. Of these fits, 284 use the fixed default configuration, comprising 264 new fits and 20 reused default checkpoints. The other 66 retain their earlier validation-selected configurations. Thirty task records lack the required budget or target labels. The complete queue has no active, queued, or failed task. The final verification checks prediction coverage, label permissions, optimizer budgets, both saved weight files, and logs for all 284 fixed-default results. It also verifies the 1,199 preserved files. The old compatibility links are absent, and all computations use the existing system Conda environment.

The manuscript reports complete combination-adaptation, functional-assay, and FLIP2 comparisons. The functional cohort contains six assays and eight measurement conditions. KRAS conditions are averaged within their assay. Functional and higher-order tests use the predicted-single, two-stage adaptation variant. The primary S-all comparison retains measured inputs within the shared interaction function. RFA has twelve undefined GFP rank correlations from constant raw predictions. The common rank table discloses the finite-record intersection, while residual comparisons retain those cases. Complete per-evaluation records and the constant-prediction checks remain available.

The final analysis independently reconstructs 3,506 original-main paired comparisons with zero maximum mean difference. The manuscript check verifies 485 expanded-table values, the 350 eligible RFA fits, and the twelve undefined correlations. The primary-evidence check separately verifies all 11,650 paired comparisons, 52 complete model rows, and four native figure fragments, with maximum mean difference 1.11e-16. The current source compiles successfully with the built-in editor compiler. The authoritative records are `runs/final_main_reporting_20261018/manifest.json`, `manuscript_verification.json`, and `runtime_completion.json` on the existing remote project. The base experiment commit is `875d04d6df818110001f3889758cf998321a96f6`; recorded working-tree sources contain the current analysis and manuscript revisions.

The current layout follows the user-specified RECOMB-ProtDLM conversation and its LLNCS v2.25 source. The title, authors, affiliations, abstract, and Introduction use a continuous layout without keywords. The main document contains ten pages including the title and abstract, followed by one reference page. The supplement contains fifteen pages. Supplementary tables retain their section order and use short method names. Current delivery records are in `runs/writing_revision_20261020/`. Terminology evidence covers sixty concepts, with five independent-paper uses per concept and 300 records from 120 distinct works in `paper/term_evidence.csv`. The scientific evaluation scopes and saved predictions remain those described above.

## 2026-10-20 publication thesis and manuscript revision

The manuscript centers on conditioning the cross-protein interaction function on permitted target single-mutant measurements. Those measurements retain their external single-effect calibration role. Matched removal of the values and availability indicators from the pair network identifies the additional prediction and selection benefit. The four-state construction supplies exact reference constraints and average residual gains; its native hit-rate differences remain +0.52 and −0.23 percentage points across the two groupings. Representation branches, the phenotype map, and fitting objectives are model choices with their complete results preserved.

The title, abstract, Introduction, Related Work, Methods, Results, Discussion, and Conclusion use this thesis. Equation `eq:conditioning` explicitly records the single-effect values and availability indicators entering the pair network. Results begin with the matched controls, and the main table `tab:conditioning` displays both groupings. Figure 1 shows the conditioning and external-sum paths. Figure 4's revised caption identifies the input-removal control while retaining all plotted data. Full objective details and the representation and training comparisons remain in the appendix. Existing model weights, predictions, and logs are preserved.

## Added matched pair-function controls

The two new controls are defined in `/data3/doomx/RECOMB-3-Epistasis/runs/key_ablations_20261015/protocol.json`. Each has 50 source fits across two homology thresholds, five outer folds, and five training repetitions. They reuse the full model's selected hyperparameters and each repetition's pretrained single-effect checkpoint. Source training rows, validation rows, initial parameter tensors, losses, optimizer, maximum epoch budget, stopping rule, and checkpoint criterion are matched. Fitted weights and realized stopping epochs may differ. This comparison was designed after the existing main results.

The direct control uses the same symmetric network output as the pair residual without four-state subtraction. The other control supplies predicted single effects and zero availability flags to the pair network. The external measured-single calibration formula, permitted labels, and source-validation selection procedure remain unchanged. Its output is exactly preserved at fixed weights; independently refitted parameters are allowed to differ. Each control is evaluated against the existing full model using identical full-target S-all cohorts, native and common-main scores, four candidate budgets, and three boundary margins. The dedicated comparison has 150 evaluations, including the 50 existing reference-model evaluations.

Implementation verification has completed on 256 real source-validation quartets across projection dimensions 64 and 128. The corresponding model parameter counts are 836,927 and 984,383 for all three variants. Default forward behavior, common initialization, external single calibration, endpoint symmetry, direct-head output, and finite forward and backward passes are verified. Exchanging real measured single values or hiding their availability does not affect the removed-input pair output. It continues to affect external single calibration. The small verification artifact is `implementation_verification.json`; it contains no performance result.

All 100 source fits and 150 matched evaluations are complete. The 100 source runs pass condition matching, including parameter counts, pretrained single-effect weights, training and validation rows, hyperparameters, and selection rules. Independent recomputation checks 932 paired comparisons across the two thresholds, with maximum mean difference 5.55e-17. Records are `runs/key_ablations_20261015/final_manifest.json`, `matched_conditions_verification.json`, and `primary_statistics.json`.

Each threshold completes 75 evaluations across the full model and both controls. Under family30, residual RMSE is 0.5949217 for the full model, 0.6354888 for the direct head, and 0.7666211 for the removed-input control. Their native top-10 hit rates are 11.2167%, 10.7000%, and 8.1833%; mean selected recovery is −0.2569211, −0.3019407, and −0.3961306 kcal/mol. Common-main-effect hit rates are 11.2667%, 10.8000%, and 8.1833%. Under family25, residual RMSE is 0.6043286, 0.6302069, and 0.8138782 in the same order. Native hit rates are 11.1500%, 11.3833%, and 8.1500%; recovery is −0.2560606, −0.2572138, and −0.3863527 kcal/mol. Common-main-effect hit rates are 11.1667%, 11.4500%, and 8.1500%. Independent tables are in `evidence/verified_family30/` and `evidence/verified_family25/` under the new run directory.

Measured pair inputs improve average residual accuracy and native hit rate in all five repetitions under both groupings. With common measured main effects, their hit-rate gains are 3.0833 and 3.0167 percentage points, rising to 3.1130 and 3.1304 points at margin 0.5. The reference contrast improves mean residual accuracy in all five repetitions under both groupings. Its native hit-rate gains are +0.5167 and −0.2333 points. The manuscript therefore assigns measured conditioning a prediction-and-selection contribution and the contrast a reference-constraint and residual-prediction contribution. It reports all four paired intervals and the repetition directions in the appendix.

Additional source queues executed the same configurations under a per-run file lock; the completed experiment count remains 100 source fits. Cross-site context, the monotone map, and the residual objective remain model or training choices. Phenotype-only training improves residual RMSE and native hit rate under both groupings, as reported in the component analysis. Figure 4 contains both new controls and all five primary-group repetitions. Its 650-evaluation input exactly matches the final integrated primary-group phenotype, residual, and selection tables at tolerance 1e-12. This check is saved in `figure_input_verification.json`. The current saved source compiled successfully with the built-in editor compiler on 2026-10-18.

## Primary full-target method-paper evidence

The source is `/data3/doomx/RECOMB-3-Epistasis/runs/key_ablations_20261015/paper_evidence/`, source experiment commit `875d04d6df818110001f3889758cf998321a96f6`. The new controls use the modified working-tree source recorded by their manifests. The integrated complete-arm scope contains 1,300 evaluations: 26 predictors or controls, two homology thresholds, five outer folds, and five training repetitions. The expanded plan contains 1,350 evaluations, including 50 ESM-MSR slots. Its available single-fold record is reported separately; the other 49 slots were discontinued with additional ESM-MSR training. Geometry-rewired retraining, all 50 StabilityArc source-and-target groups, and the 100 new control fits are complete. Every completed arm has all 25 fold-by-repetition evaluations at each threshold. The earlier 24-arm evidence remains unchanged in `runs/sall_full_target/paper_evidence/`. Integration verifies that all original reference-model metrics are preserved.

S-all excludes all target double-mutant labels from source fitting, validation, and target adaptation. Every accepted target double mutant is scored. Phenotype evaluation contains 159 proteins, 51 homology groups, and 795 protein-repetition records. Residual evaluation contains 155 proteins, 51 groups, and 775 records. Compensation contains 120 proteins, 44 groups, and 600 records. The 0.5-kcal/mol margin cohort contains 115 proteins, 43 groups, and 575 records. Each protein has all five repetitions. Metrics average repetitions within protein and then weight proteins equally.

Compensation eligibility requires two measured single effects at most −0.5 kcal/mol, four point measurements inside the assay range, reference stability at most 4.5 kcal/mol, and at least 100 candidates per protein. Candidate budgets are 1, 3, 10, and 30. A hit requires measured recovery of at least 0.5 kcal/mol above the more stable constituent single mutant. Native rankings retain each predictor's phenotype output. Common-main-effect rankings retain the measured single-effect sum and each predictor's own four-term contrast.

| Predictor | Phenotype RMSE | Residual RMSE | Native top-10 compensation hit rate | Native selected recovery |
|---|---:|---:|---:|---:|
| SCEPI | 0.6568063 | 0.5949217 | 0.1121667 | −0.2569211 |
| ThermoMPNN-D | 0.7770976 | 0.8049410 | 0.0706667 | −0.4399899 |
| Kermut | 0.7984050 | 0.8472280 | 0.0540000 | −0.4578538 |
| StabilityArc + Kermut | 0.8154110 | 0.8533217 | 0.0875000 | −0.3442761 |
| Global nonlinear | 0.9105326 | 0.6353884 | 0.0445000 | −0.4371078 |
| Additive | 0.8926547 | 1.0909510 | 0.0638333 | −0.3871665 |
| Mean MLP | 1.4051350 | 0.9657895 | 0.1006667 | −0.4133627 |
| Mean ridge | 1.3280766 | 1.0017164 | 0.1141667 | −0.3544487 |
| Direct residual head | 0.6995900 | 0.6354888 | 0.1070000 | −0.3019407 |
| Measured pair inputs removed | 0.8101096 | 0.7666211 | 0.0818333 | −0.3961306 |

All RMSE and recovery values use kcal/mol. The shared predictor reduces residual RMSE by 26.1% against ThermoMPNN-D and 29.8% against Kermut, with 136/155 protein-level wins in each comparison. Its native selected recovery is higher in 86/120 and 93/120 proteins, respectively. Against StabilityArc + Kermut, residual error falls by 30.3%, with 130/155 protein-level wins; native hit rate improves by 2.4667 percentage points, and mean recovery improves by 0.087355 kcal/mol. Every training repetition has a positive mean native hit-rate difference in this comparison. The shared predictor has the highest mean selected recovery among the seven baselines at every candidate budget. Mean ridge has the highest native top-10 hit rate in this primary table. StabilityArc + Kermut has phenotype Spearman 0.7464551, compared with 0.7420913 for the shared model. Both results are retained.

Common-main-effect top-10 hit rates in the same row order are 11.2667%, 5.3000%, 5.4667%, 8.1667%, 4.7000%, 4.4167%, 7.1667%, and 7.3333%. At margin 0.5, they are 11.1478%, 5.2696%, 5.2870%, 8.1217%, 4.8174%, 4.5217%, 7.3043%, and 7.4783%. Mean effects, intervals, full distributions, and all repetitions are retained in the primary statistics. Bootstrap-positive fractions describe resampling support for the cohort mean.

The StabilityArc + Kermut common-main-effect hit rates are 8.1667% in the standard cohort and 8.1217% at margin 0.5. The independently recomputed native top-10 hit-rate gain has a 95% interval of [−1.17, 5.55] percentage points and a bootstrap positive fraction of 0.9165. At common-main-effect margin 0.5, the gain is 3.0261 points with interval [−0.64, 6.41]. These intervals describe the paired cohort mean under group resampling.

Inference-time pair removal gives residual RMSE 1.072328 and native hit rate 4.4167%. No-cross-site-context training gives 0.574979 and 11.4667%; identity-feature training gives 0.556803 and 9.1167%; phenotype-only training gives 0.561853 and 11.9500%. These results support the predictive pair model and identify which individual design choices have separate evidence.

The stricter family25 analysis independently completes 650 evaluations across the same 26 arms. It retains 159 phenotype proteins and 155 residual proteins in 45 homology groups, and 120 compensation proteins in 39 groups. Shared-model phenotype RMSE, residual RMSE, native hit rate, and recovery are 0.676502, 0.604329, 11.1500%, and −0.256061. Common-main-effect hit rate is 11.1667%; at margin 0.5 it is 11.2348% across 115 proteins and 38 groups. StabilityArc + Kermut has residual RMSE 0.873954 and native hit rate 8.2333%. Its common-main-effect hit rates are 8.0167% and 7.9478% in the standard and margin cohorts. The shared predictor's average native recovery exceeds every primary comparator at top-10. The direct head has a higher native hit rate of 11.3833%, and this result is retained in the component analysis and complete table.

Independent recomputation uses `scripts/summarize_full_target_evidence.py`. It reconstructs 11,650 paired comparisons from the integrated two-threshold per-protein Parquet records. Each threshold has 650 complete model evaluations and 5,825 paired comparisons. Wins, losses, and ties use a declared `1e-12` floating-point tolerance. Group bootstrap uses 10,000 replicates with seed 20261013. Full independent results remain in the formal remote `paper_evidence/verified_family30/` and `verified_family25/` directories. Small new tables and verification records are retained locally in `../work/key_ablations_20261015/`; the original-budget records remain in `../work/method_reporting_20261015/`. The manuscript check covers all 11,650 paired means and cohort records, 52 complete model rows, three main conditioning-table rows, four matched-control interval rows, eight original-budget rows, and the four embedded figure fragments. It also verifies the reported 22.4% and 25.7% relative residual reductions from the input-removal control. The largest mean difference is 1.11e-16. Its record is `paper_evidence/manuscript_verification.json`.

The manuscript includes the complete method, primary results, discussion, mathematical properties, optimization details, all 26 completed arms at both thresholds, target-label budgets, combination adaptation, functional assays, FLIP2, the ESM-MSR subset, and the earlier intersection. Its four figures retain the supplied T-REX organization and consistently name SCEPI. All four canonical figure fragments are unchanged in the LLNCS revision. The main method retains measured conditioning and reference constraints. Complete cross-site features and the global/pair phenotype decomposition are placed in Appendix A. Abstract, Introduction, and Discussion remove repeated statements while retaining the central claim and material boundaries. The current saved source passes the built-in compiler. All 24 pages are visually inspected at final size. The ten-page main text includes the title and abstract; references occupy one page and the supplement occupies thirteen pages. All ordinary text remains 10pt with Letter pages and one-inch margins. The exported PDFs render pixel-identically to the inspected source pages at 110 dpi, all fonts are embedded, and retained internal links resolve within each file. Latest sources and verification reports are recorded in `runs/scepi_lncs_20261018/`.

Method-and-ablation evaluations are complete in all three matrices: 7,940/7,940 in the original main matrix, 1,000/1,000 in the compensation intersection, and 800/800 in full-target S-all. The record is `/data3/doomx/RECOMB-3-Epistasis/runs/method_reporting_20261015/manifest.json`. All currently authorized external-baseline dependencies are also complete. All eight ESM-MSR source training processes and the automatic grid controllers were stopped on user instruction; all existing best checkpoints were loaded successfully and preserved with their logs. The default run is family30/fold0/seed11 at learning rate 2e-4 and weight decay 0. Its tenth-round checkpoint has source-validation residual RMSE 1.07368099 kcal/mol. Training completed 13 rounds and was stopped during the fourteenth round. The source manifest distinguishes the ready checkpoint from unfinished training, with `training_complete=false` and `user_requested_stop=true`. The available source and S-all target evaluations from this checkpoint are complete. This partial default evaluation remains separate from the full five-fold, five-repetition main comparison. StabilityArc's released pooled checkpoint remains excluded from the matched source holdouts; all reported StabilityArc results use the 50 independently retrained source models.

## Fixed unseen-site-pair intersection evidence

The source is `/data3/doomx/RECOMB-3-Epistasis/runs/compensation_main/paired_summary.json`, with source commit `875d04d6df818110001f3889758cf998321a96f6`. Each comparison reported below has a complete 25-evaluation grid. These numbers retain the existing unseen-pair and family30 evaluation intersection and appear separately in the manuscript appendix.

`scripts/paper_method_evidence.py` independently recalculated 90 selected comparisons from `selection_per_protein.parquet` and `residual_per_protein.parquet`. Means match at absolute tolerance `1e-12`; paired record counts, protein counts, and wins/losses/ties also match. The small local record is `../work/paper_method_evidence.json`. Each protein is averaged over eligible repetitions before equal-protein averaging. The main residual comparisons contain 70 proteins, 34 homology groups, and 339 protein-repetition records. Compensation contains 63 proteins, 32 groups, and 269 records.

| Predictor | Residual RMSE (kcal/mol) | Native compensation hit rate | Native selected recovery (kcal/mol) |
|---|---:|---:|---:|
| SCEPI | 0.5485176 | 0.0574868 | −0.4416138 |
| ThermoMPNN-D | 0.7775977 | 0.0487566 | −0.5112426 |
| Kermut | 0.8333403 | 0.0329101 | −0.5619355 |
| Global nonlinear | 0.5904278 | 0.0355556 | −0.5284684 |
| Additive | 1.0803415 | 0.0470899 | −0.4926291 |

The same measured-main-effect score yields hit rates 0.0578042, 0.0329101, 0.0306878, 0.0391270, and 0.0361905 in that row order. At margin 0.5, they are 0.0538172, 0.0311828, 0.0267204, 0.0363710, and 0.0333871. The margin cohort contains 62 proteins, 32 groups, and 263 paired protein-repetition records.

Inference-time pair removal gives residual RMSE 1.0615939 and native hit rate 0.0361905. Removing cross-site context gives 0.5514564 and 0.0562698. The identity-feature variant gives 0.5309399 and 0.0562963. Phenotype-only training gives 0.5222058 and 0.0623280. These results support the pair-model class and bound component-specific claims. The manuscript describes the full implemented predictor and retains these comparisons.

Source-trained methods receive their permitted source supervision and target singles. Kermut is a target-supervised comparator. All use zero target double-mutant training labels in S-all. Source labels and structural feature access are separate comparison dimensions.

## Initial comparison-design evidence

The following single-split records concern the completed exploratory comparison-design analysis. Their model choices were retained from source validation. They are distinct from the repeated values above.

## Provenance

| Item | Record |
|---|---|
| Remote run | `/data3/doomx/RECOMB-3-Epistasis/runs/comparison_design_diagnostic/` |
| Run manifest | `manifest.json`, `complete=true`, source commit `b3a3e31885c6def6823fc0a94c81a60374d26125` |
| Data inputs | `data/splits/combination_splits.parquet`; `data/processed/tsuboyama/variants.parquet` |
| Diagnostic scope | Seed11 unseen-pair test intersected with family30 held-out proteins |
| Selection protocol | Existing source-validation model choices; no model selection on candidate yield |
| Independent verification | `verification.json`, `complete=true`; 4,320 selected candidates rechecked; maximum recovery recomputation difference `4.44e-16`; eight original comparisons matched |
| Current evidence type | Retrospective test-set diagnostic; one test partition; no repeated-test uncertainty estimate |
| Formal evaluation status checked 2026-10-13 15:52 CST | `/data3/doomx/RECOMB-3-Epistasis/runs/compensation_main/current.json`: 2,182/2,400 evaluations complete, 218 waiting on prediction dependencies; seven completed evaluations await the next aggregation; source commit `875d04d6df818110001f3889758cf998321a96f6` |

## Residual prediction

Residuals are calculated as the four-term contrast for each model prediction. For each protein, the diagnostic computes RMSE on complete, in-range quartets with at least 30 eligible observations. The summary averages protein-level RMSE equally.

| Condition | Model | Mean per-protein RMSE (kcal/mol) | Mean advantage (kcal/mol) | Proteins with lower RMSE / total |
|---|---|---:|---:|---:|
| S-all | Additive | 1.0101 | — | — |
| S-all | Global nonlinear | 0.5873 | — | — |
| S-all | SCEPI | 0.5461 | +0.4639 vs additive; +0.0412 vs global nonlinear | 59/68 vs additive; 39/68 vs global nonlinear |
| S-all | Interaction removed at inference | 0.9944 | — | — |

The residual cohort contains 68 proteins across 34 families. Relative RMSE reductions for the interaction model are 45.9% against additive and 7.0% against global nonlinear. The interaction-removed score has 59/68 protein-level losses relative to the full interaction score.

## Fixed-budget compensatory selection

Candidate eligibility requires two measured single substitutions, each reducing stability by at least 0.5 kcal/mol relative to wild type; all four measurements must be point estimates inside the assay range. Each eligible protein must have at least 100 candidates. All methods use the same measured single-mutant main-effect sum. They add their own predicted residual and select the top 10 double mutants per protein. A measured hit is a double mutant whose stability is at least 0.5 kcal/mol higher than that of the more stable constituent single mutant.

| Analysis | SCEPI | Additive | Global nonlinear | Interaction removed |
|---|---:|---:|---:|---:|
| Standard pool: measured hits / 540 selections | 27 (5.00%) | 12 (2.22%) | 13 (2.41%) | 12 (2.22%) |
| Margin ≥0.5 kcal/mol: measured hits / 540 selections | 26 (4.81%) | 11 (2.04%) | 12 (2.22%) | 11 (2.04%) |
| Mean selected recovery advantage vs additive | +0.1336 kcal/mol | — | — | — |
| Mean selected recovery advantage vs global nonlinear | +0.1186 kcal/mol | — | — | — |
| Margin analysis: recovery advantage vs additive | +0.1351 kcal/mol | — | — | — |
| Margin analysis: recovery advantage vs global nonlinear | +0.1214 kcal/mol | — | — | — |

The standard candidate pool includes 14,079 combinations from 54 proteins and 27 families. In per-protein hit rate, the interaction score wins/ties/loses 9/45/0 against additive and 8/46/0 against global nonlinear. Mean selected recovery is higher for the interaction model in 34/54 proteins against either comparator. The margin is the minimum distance of any of the four quartet measurements from either assay boundary, −1 or 5 kcal/mol.

For the broader continuous candidate pool, the mean stability change of the selected ten candidates is lower by 0.0018 kcal/mol against additive and by 0.0054 kcal/mol against global nonlinear across 65 proteins. The current evidence therefore supports a task-specific compensation claim, not a general improvement in stability screening.

## Current claim and completed scope

**Supported now:** Measured single effects within the shared pair function improve average residual prediction and native compensation selection beyond external calibration under both homology groupings. The complete predictor improves those endpoints against ThermoMPNN-D, Kermut, calibrated StabilityArc, additive, and global nonlinear controls on full-target S-all evaluation. Its mean selected recovery exceeds all seven primary comparators at every evaluated budget. Common-main-effect gains survive measurement-boundary analysis and stricter homology grouping. The reference contrast improves average residual accuracy, and its native hit-rate gains vary across groupings. The earlier held-out intersection remains separately reported.

**Scope boundary:** All currently authorized evaluation scopes are complete. The available ESM-MSR result covers one fold and one repetition; its additional training and remaining repeated evaluations were discontinued. Current results establish retrospective compensation performance for the reported cohort. Molecular mechanism and general stability screening have distinct evidence requirements.

## Local source checks

- `../work/comparison_design_diagnostic/manifest.json`
- `../work/comparison_design_diagnostic/verification.json`
- `../work/comparison_design_diagnostic/summary.json`
- `../work/remote_current/scripts/evaluate_comparison_design.py`
- `../work/verify_comparison_design.py`
- `../work/remote_current/scripts/prepare_tsuboyama.py`
- `../work/remote_current/README.md`
