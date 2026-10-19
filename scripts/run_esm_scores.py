from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs

jobs = []
for threshold in [30, 25]:
    for fold in range(5):
        for seed in [11, 23, 47, 89, 131]:
            directory = f"runs/e3/family{threshold}/fold{fold}/esm_scores/seed{seed}"
            jobs.append({
                "name": f"family{threshold}/fold{fold}/seed{seed}",
                "stages": [{
                    "name": "direct_scores",
                    "command": [sys.executable, "scripts/evaluate_esm_scores.py", "--threshold", str(threshold), "--fold", str(fold), "--seed", str(seed)],
                    "manifest": directory + "/manifest.json",
                }],
            })
run_jobs(ROOT, jobs, ROOT / "runs/e3/esm_scores", use_gpu=False)
