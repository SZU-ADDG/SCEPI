import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.main_evaluation import OUTPUT, functional_entries, stability_definitions, stability_entries
from epistasis.jobs import run_jobs
from epistasis.evaluation_scope import authorized_jobs, scope_status
from epistasis.async_summary import AsyncSummary


def main():
    all_jobs = json.loads((OUTPUT / "jobs.json").read_text())
    jobs, discontinued = authorized_jobs(ROOT, "original_main", all_jobs)
    split = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet", columns=["variant_id", "scheme", "seed", "set"])
    truth = {(threshold, fold): pd.read_parquet(OUTPUT / "truth" / f"family{threshold}/fold{fold}.parquet",
                                               columns=["variant_id", "protein_id", "structure_available"])
             for threshold in [30, 25] for fold in range(5)}
    protein_cache = {}
    for job in jobs:
        if job["dataset"] != "tsuboyama":
            continue
        structure = stability_definitions()[job["arm"]]["requires_structure"]
        key = (job["threshold"], job["fold"], job["scheme"], job["seed"], structure)
        if key in protein_cache:
            continue
        frame = truth[(job["threshold"], job["fold"])]
        if structure:
            frame = frame.loc[frame.structure_available]
        if job["scheme"] != "family":
            ids = split.loc[split.scheme.eq(job["scheme"]) & split.seed.eq(job["seed"]) & split["set"].eq("test"), "variant_id"]
            frame = frame.loc[frame.variant_id.isin(ids)]
        protein_cache[key] = sorted(frame.protein_id.unique())
    batches = ROOT / "work/original_batches"
    batch = len(list(batches.glob("batch*")))
    summary = AsyncSummary(ROOT, OUTPUT, "scripts/aggregate_original_main.py")
    while True:
        ready, waiting, completed = [], [], 0
        for job in jobs:
            path = ROOT / job["stages"][0]["manifest"]
            if path.exists():
                assert json.loads(path.read_text())["complete"]
                completed += 1
                continue
            if job["dataset"] == "tsuboyama":
                structure = stability_definitions()[job["arm"]]["requires_structure"]
                key = (job["threshold"], job["fold"], job["scheme"], job["seed"], structure)
                _, missing = stability_entries(job, protein_cache[key])
            else:
                _, missing = functional_entries(job)
            if missing:
                waiting.append({"job": job["name"], "dependencies": missing})
            else:
                ready.append(job)
        state = {**scope_status(len(all_jobs), discontinued, completed), "completed_evaluations": completed,
                 "ready_evaluations": len(ready), "waiting_evaluations": len(waiting),
                 "waiting_dependencies": waiting, "state": "evaluating" if ready else "waiting_for_predictions"}
        state["active_scope_complete"] = False
        (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
        print(json.dumps({key: value for key, value in state.items() if key != "waiting_dependencies"}), flush=True)
        if ready:
            batch += 1
            run_jobs(ROOT, ready, batches / f"batch{batch:04d}", use_gpu=False)
        completed_after_batch = completed + len(ready)
        reporting_complete = summary.update(completed_after_batch)
        state.update(scope_status(len(all_jobs), discontinued, completed_after_batch))
        state.update(completed_evaluations=completed_after_batch, ready_evaluations=0,
                     aggregated_evaluations=summary.published,
                     summary_pending_evaluations=completed_after_batch - summary.published,
                     reporting_complete=reporting_complete,
                     active_scope_complete=completed_after_batch == len(jobs) and reporting_complete,
                     state="waiting_for_predictions" if waiting else "aggregating")
        (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
        if completed_after_batch == len(jobs) and reporting_complete:
            state.update(state="complete" if not discontinued else "authorized_scope_complete")
            (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
