from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paper_full_target_evidence as evidence
from scripts.run_sall_full_target import paired_summary

STUDY = ROOT / "runs/conditioning_factorial_20261021"
OUTPUT = STUDY / "primary_evidence"
BASE = ROOT / "runs/key_ablations_20261015/paper_evidence"
ARMS = {"interaction", "additive", "global_nonlinear", "thermo_epistatic", "kermut", "stabilityarc_kermut", "mean_mlp", "mean_ridge"}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in ["phenotype", "residual", "selection"]:
        base = pd.read_parquet(BASE / (name + "_per_protein.parquet"))
        chunks = [base.loc[base.arm.isin(ARMS - {"interaction"})].copy()]
        for threshold in [30, 25]:
            for fold in range(5):
                for seed in [11, 23, 47, 89, 131]:
                    directory = STUDY / "evidence/evaluations/final_interaction" / f"family{threshold}/seed{seed}/fold{fold}"
                    assert json.loads((directory / "manifest.json").read_text())["complete"]
                    table = pd.read_parquet(directory / (name + ".parquet"))
                    table["arm"] = "interaction"
                    chunks.append(table)
        tables[name] = pd.concat(chunks, ignore_index=True)
        if name == "phenotype":
            tables[name] = tables[name].loc[tables[name].phenotype_count.ge(30)].copy()
        tables[name].to_parquet(OUTPUT / (name + "_per_protein.parquet"), index=False)
    grid = tables["phenotype"][["threshold", "arm", "seed", "fold"]].drop_duplicates()
    coverage = Counter((int(row.threshold), row.arm) for row in grid.itertuples())
    assert set(grid.arm) == ARMS and len(grid) == 400 and all(count == 25 for count in coverage.values())
    summary = []
    for metric in ["compensation_hit_rate", "selected_mean_recovery", "selected_mean_stability", "precision_lower", "precision_upper"]:
        summary += paired_summary(tables["selection"], ["threshold", "task", "margin", "unit", "scoring", "budget"], metric, True, coverage)
    for metric in ["epsilon_rmse", "pair_macro_epsilon_rmse"]:
        summary += paired_summary(tables["residual"], ["threshold", "margin"], metric, False, coverage)
    for metric, higher in [("phenotype_rmse", False), ("phenotype_spearman", True), ("phenotype_pearson", True)]:
        summary += paired_summary(tables["phenotype"], ["threshold"], metric, higher, coverage)
    evidence.bootstrap(summary, tables)
    (OUTPUT / "paired_summary.json").write_text(json.dumps(summary, indent=2))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"complete": True, "available_jobs_complete": True, "scope": "Eight complete primary comparisons; separate factorial training continues.",
                "completed_evaluations": 400, "expected_evaluations": 400, "source_commit": commit,
                "source_selection": str(STUDY / "source_selection.json"),
                "completed_grids": [{"threshold": threshold, "arm": arm, "evaluations": count} for (threshold, arm), count in sorted(coverage.items())]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for threshold in [30, 25]:
        with (STUDY / f"primary_summary_family{threshold}.log").open("w") as stream:
            subprocess.run([sys.executable, "scripts/summarize_full_target_evidence.py", "--input", str(OUTPUT), "--output", str(OUTPUT / f"verified_family{threshold}"),
                            "--threshold", str(threshold), "--source-run-path", str(OUTPUT), "--source-commit", commit],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    (OUTPUT / "integration_verification.json").write_text(json.dumps({"complete": True, "completed_arms": 8, "completed_evaluations": 400}, indent=2))
    with (STUDY / "primary_random_enrichment.log").open("w") as stream:
        subprocess.run([sys.executable, "scripts/analyze_random_enrichment.py", "--evidence", str(OUTPUT), "--output", str(STUDY / "primary_random_enrichment")],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"complete": True, "source_selected_primary_arms": 8, "complete_evaluations": 400,
                      "factorial_results_not_included": True, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
