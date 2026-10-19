import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.jobs import run_jobs


def main():
    for name, arms in [("original_main", {"mean_ridge_target_C", "mean_mlp_target_C", "mean_ridge", "mean_mlp"}),
                       ("compensation_main", {"mean_ridge_target_C", "mean_mlp_target_C"})]:
        jobs = [job for job in json.loads((ROOT / "runs" / name / "jobs.json").read_text())
                if job["arm"] in arms and not (ROOT / job["stages"][0]["manifest"]).exists()]
        # 原有控制程序等待旧入口；本次仅补齐目录修正对应的缺失评估。
        run_jobs(ROOT, jobs, ROOT / "runs" / (name + "_mean_path_completion_20261013"), use_gpu=False)


if __name__ == "__main__":
    main()
