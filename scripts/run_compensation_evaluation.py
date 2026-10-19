import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import OUTPUT, definitions, frozen_table, input_records
from epistasis.jobs import run_jobs
from epistasis.evaluation_scope import authorized_jobs, scope_status
from epistasis.async_summary import AsyncSummary


def main():
    all_jobs = json.loads((OUTPUT / "jobs.json").read_text())
    jobs, discontinued = authorized_jobs(ROOT, "compensation_main", all_jobs)
    cache = {(threshold, seed, fold): frozen_table(threshold, seed, fold)
             for threshold in [30, 25] for seed in [11, 23, 47, 89, 131] for fold in range(5)}
    batch = len(list((ROOT / "work/compensation_batches").glob("batch*")))
    summary = AsyncSummary(ROOT, OUTPUT, "scripts/aggregate_compensation_main.py")
    while True:
        ready, waiting, finished = [], [], 0
        for job in jobs:
            manifest = ROOT / job["stages"][0]["manifest"]
            if manifest.exists():
                assert json.loads(manifest.read_text())["complete"]
                finished += 1
                continue
            truth = cache[(job["threshold"], job["seed"], job["fold"])]
            if definitions()[job["arm"]]["requires_structure"]:
                truth = truth.loc[truth.structure_available]
            entries, missing = input_records(job["arm"], job["threshold"], job["seed"], job["fold"], job["condition"], truth)
            if missing:
                waiting.append({"job": job["name"], "dependencies": missing})
            else:
                ready.append(job)
        state = {**scope_status(len(all_jobs), discontinued, finished), "completed_evaluations": finished,
                 "ready_evaluations": len(ready), "waiting_evaluations": len(waiting),
                 "aggregated_evaluations": summary.published,
                 "summary_pending_evaluations": max(0, finished - summary.published),
                 "waiting_dependencies": waiting, "state": "evaluating" if ready else "waiting_for_predictions"}
        state["active_scope_complete"] = False
        (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
        print(json.dumps({key: value for key, value in state.items() if key != "waiting_dependencies"}), flush=True)
        if ready:
            batch += 1
            run_jobs(ROOT, ready, ROOT / "work/compensation_batches" / f"batch{batch:04d}", use_gpu=False)
            subprocess.run([sys.executable, "scripts/verify_compensation_main.py"], cwd=ROOT, check=True)
        completed_after_batch = finished + len(ready)
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
            state.update(state="complete" if not discontinued else "authorized_scope_complete", completed_evaluations=completed_after_batch,
                         aggregated_evaluations=summary.published, summary_pending_evaluations=0)
            (OUTPUT / "current.json").write_text(json.dumps(state, indent=2))
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
