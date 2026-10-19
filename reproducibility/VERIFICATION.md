# Release verification

The release was checked with the recorded system Conda environment on the original Linux host. The checks use saved experimental results and actual held-out examples.

| Check | Verified result |
| --- | --- |
| Primary tables | 52 model rows across both homology groupings; 312 reconstructed means agree within 1e-12 |
| Random-enrichment analysis | 1,248 comparisons; 8,736 reconstructed values agree within 1e-12 |
| Candidate-pool matching | 738,400 paired protein/repetition/condition records |
| CPU prediction example | 32 real held-out quartets; phenotype and residual predictions match the recorded evaluation within 1e-5 |
| Selected source configurations | 150 portable job entries covering 50 primary and 100 matched-control fits |
| Python source | All released Python modules compile successfully |

Run `python reproduce.py --model-check --output work/reproduced` from the release root to repeat the numerical checks. It writes its own `verification.json` and both reconstructed CSV tables. The command reconstructs reported means; it retains the saved bootstrap intervals. The full random-analysis program regenerates those intervals from the original candidate pools.

The environment specification records the working environment and pinned upstream commits. This revision reused that existing environment. It did not create a separate environment or perform a new installation from the exported specification. Complete refitting uses the supplied source-job manifest and downloaded source data; the compact verification command does not refit models.
