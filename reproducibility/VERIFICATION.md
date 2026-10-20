# Release verification

The release was checked with the recorded system Conda environment on the original Linux host. The checks use saved experimental results and actual held-out examples.

| Check | Verified result |
| --- | --- |
| Primary tables | 64 model rows across both homology groupings; 384 reconstructed means agree within 1e-12 |
| Random-enrichment analysis | 1,536 comparisons; 10,752 reconstructed values agree within 1e-12 |
| Candidate-pool matching | 908,800 paired protein/repetition/condition records |
| Numerical inputs and indicators | 120 contrasts and 220 group-sensitivity comparisons; 2,400 descriptive values agree within 1e-12 |
| CPU prediction example | 32 real held-out quartets; phenotype and residual predictions match the recorded evaluation within 1e-5 |
| Public prediction command | The same actual examples pass the documented CLI, saved-prediction comparison, and within-protein rank checks |
| Selected source configurations | 250 portable job entries covering 50 primary and 200 matched-control fits |
| Python source | All released Python modules compile successfully |

Version `recomb2027-20261021` contains these results. Run `python reproduce.py --model-check --output work/reproduced` from the release root to repeat the numerical checks. It writes its own `verification.json` and reconstructed CSV tables. The command reconstructs reported means, numerical-input contrasts, and group-sensitivity summaries; it retains the saved bootstrap intervals. The full analysis programs regenerate those intervals from the original per-protein records and candidate pools.

The environment specification records the working environment and pinned upstream commits. This revision reused that existing environment. It did not create a separate environment or perform a new installation from the exported specification. Complete refitting uses the supplied source-job manifest and downloaded source data; the compact verification command does not refit models.
