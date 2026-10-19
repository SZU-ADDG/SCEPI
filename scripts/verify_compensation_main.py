import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import OUTPUT


def main():
    points = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet", columns=["variant_id", "point"]).set_index("variant_id").point
    quartets = pd.read_parquet(ROOT / "data/processed/tsuboyama/quartets.parquet").set_index("ab_id")
    assert points.index.is_unique and quartets.index.is_unique
    splits = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet")
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    diagnostic = pd.read_parquet(ROOT / "runs/comparison_design_diagnostic/selection_per_decision.parquet")
    families = {threshold: pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet").set_index("protein_id")
                for threshold in [30, 25]}
    jobs = json.loads((OUTPUT / "jobs.json").read_text())
    records = []
    for job in jobs:
        path = ROOT / job["stages"][0]["manifest"]
        if not path.exists():
            continue
        directory = path.parent
        proof_path = directory / "verification.json"
        if proof_path.exists():
            proof = json.loads(proof_path.read_text())
            assert proof["complete"]
            records.append(proof)
            continue
        information = json.loads(path.read_text())
        selection = pd.read_parquet(directory / "selection_per_decision.parquet")
        selection = selection.loc[selection.task.eq("compensation") & selection.scoring.eq("observed_main") & selection.budget.eq(10)].copy()
        primary = selection.loc[selection.unit.eq("protein") & selection.margin.eq(0)]
        assert int(primary.compensation_hits.sum()) == information["primary_compensation_hits"]
        maximum_difference = 0.0
        if len(selection):
            selection["decision_id"] = np.arange(len(selection))
            exploded = selection.explode(["selected_variant_ids", "selected_recovery", "selected_scores"], ignore_index=True)
            source = quartets.loc[exploded.selected_variant_ids].reset_index(drop=True)
            a = source.a_id.map(points).to_numpy()
            b = source.b_id.map(points).to_numpy()
            wt = source.wt_id.map(points).to_numpy()
            ab_points = exploded.selected_variant_ids.map(points).to_numpy()
            actual = ab_points - np.maximum(a, b)
            assert source.main.all()
            assert np.all(a - wt <= -0.5) and np.all(b - wt <= -0.5) and np.all(wt <= 4.5)
            assert np.all(source.margin.to_numpy() >= exploded.margin.to_numpy())
            permitted = set(order.loc[order.seed.eq(job["seed"]), "variant_id"])
            assert set(source.a_id) | set(source.b_id) <= permitted
            test = splits.loc[splits.seed.eq(job["seed"]) & splits.scheme.eq("unseen_pair") & splits["set"].eq("test")]
            assert set(exploded.selected_variant_ids) <= set(test.variant_id)
            assert families[job["threshold"]].loc[exploded.protein_id, "outer_group"].eq(job["fold"]).all()
            recorded = exploded.selected_recovery.to_numpy(dtype=float)
            np.testing.assert_allclose(actual, recorded, rtol=1e-8, atol=1e-8)
            maximum_difference = float(np.max(np.abs(actual - recorded)))
            counts = pd.Series(actual >= 0.5, index=exploded.decision_id).groupby(level=0).sum()
            np.testing.assert_array_equal(counts.to_numpy(), selection.compensation_hits.to_numpy())
            normalized = pd.read_parquet(directory / "predictions.parquet", columns=["variant_id", "prediction_epsilon"]).set_index("variant_id")
            expected_score = a + b - 2 * wt + exploded.selected_variant_ids.map(normalized.prediction_epsilon).to_numpy()
            np.testing.assert_allclose(expected_score, exploded.selected_scores.to_numpy(dtype=float), rtol=1e-6, atol=1e-6)
        matched = 0
        if job["arm"] in ["interaction", "additive", "global_nonlinear", "remove_interaction"] and job["threshold"] == 30 and job["seed"] == 11 and job["condition"] == "Sall":
            current = primary.set_index("protein_id")
            expected = diagnostic.loc[diagnostic.arm.eq(job["arm"]) & diagnostic.condition.eq("Sall")
                                      & diagnostic.task.eq("compensation") & diagnostic.unit.eq("protein")
                                      & diagnostic.scoring.eq("observed_main") & diagnostic.budget.eq(10)].set_index("protein_id")
            expected = expected.loc[current.index]
            np.testing.assert_allclose(current.compensation_hit_rate, expected.compensation_hit_rate, atol=1e-12)
            for protein in current.index:
                assert list(current.loc[protein, "selected_variant_ids"]) == list(expected.loc[protein, "selected_variant_ids"])
            matched = len(current)
        proof = {"complete": True, "evaluation": str(path.relative_to(ROOT)), "raw_selected_candidates_checked": int(len(selection) * 10),
                 "maximum_recovery_difference": maximum_difference, "diagnostic_primary_targets_matched": matched,
                 "primary_compensation_hits": information["primary_compensation_hits"],
                 "checks": ["raw_measurements", "permitted_single_labels", "unseen_pair_test", "held_out_family_fold",
                            "measurement_boundary", "compensation_counts", "fixed_single_main_scores"]}
        proof_path.write_text(json.dumps(proof, indent=2))
        records.append(proof)
    assert records
    proof = {"complete": True, "verified_scope": "All currently completed evaluations.",
             "all_protocol_evaluations_complete": len(records) == len(jobs), "expected_evaluations": len(jobs),
             "verified_evaluations": len(records), "raw_selected_candidates_checked": sum(item["raw_selected_candidates_checked"] for item in records),
             "diagnostic_primary_targets_matched": sum(item["diagnostic_primary_targets_matched"] for item in records),
             "maximum_recovery_difference": max(item["maximum_recovery_difference"] for item in records),
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "per_evaluation_records": records}
    (OUTPUT / "verification.json").write_text(json.dumps(proof, indent=2))
    print(json.dumps({key: value for key, value in proof.items() if key != "per_evaluation_records"}), flush=True)


if __name__ == "__main__":
    main()
