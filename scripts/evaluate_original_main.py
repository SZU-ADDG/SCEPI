import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.main_evaluation import OUTPUT, evaluation_directory, frozen_truth, functional_entries, functional_metrics, normalized_stability, stability_entries, stability_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    jobs = json.loads((OUTPUT / "jobs.json").read_text())
    matches = [job for job in jobs if job["name"] == args.job]
    assert len(matches) == 1
    job = matches[0]
    directory = evaluation_directory(job)
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / "manifest.json").exists()
    truth = frozen_truth(job)
    strata = selections = None
    if job["dataset"] == "tsuboyama":
        entries, missing = stability_entries(job, sorted(truth.protein_id.unique()))
        assert not missing, missing
        frame, excluded, provenance = normalized_stability(job, truth, entries)
        summary, strata, selections = stability_metrics(frame)
        normalized_variants = len(frame)
    else:
        entries, missing = functional_entries(job)
        assert not missing, missing
        summary, excluded, provenance = functional_metrics(job, truth, entries)
        normalized_variants = int(summary.loc[summary.mutation_order.eq("all"), "variants"].sum()) if len(summary) else 0
    outputs = []
    for table, filename in [(summary, "per_protein.parquet"), (strata, "strata_per_protein.parquet"), (selections, "selected_candidates.parquet")]:
        if table is None:
            continue
        for field in ["dataset", "arm", "threshold", "fold", "seed", "scheme", "condition"]:
            table[field] = job[field]
        table.to_parquet(directory / filename, index=False)
        outputs.append(filename)
    information = {"complete": True, "smoke": False, "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   **{key: value for key, value in job.items() if key != "stages"}, "run_path": str(directory),
                   "protocol": "runs/original_main/protocol.json", "outputs": outputs, "input_predictions": provenance,
                   "excluded_proteins": excluded, "normalized_variants": normalized_variants,
                   "frozen_test_ids_verified": True, "four_genotype_residuals_verified": True}
    (directory / "manifest.json").write_text(json.dumps(information, indent=2))
    print(json.dumps({key: value for key, value in information.items() if key not in ["input_predictions", "excluded_proteins"]}), flush=True)


if __name__ == "__main__":
    main()
