import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import BUDGETS, MARGINS, OUTPUT, SEEDS, definitions, destination


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (OUTPUT / "protocol.json").exists()
    columns = ["variant_id", "protein_id", "mutation_key", "pair_id", "a_id", "b_id", "wt_variant_id", "main", "margin",
               "epsilon", "relative_point", "relative_lower", "relative_upper", "wt_point", "wt_all_point", "wt_point_in_range", "structure_available"]
    doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet", columns=columns)
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet", columns=["variant_id", "point"]).set_index("variant_id")
    splits = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet")
    order = pd.read_parquet(ROOT / "data/splits/single_label_order.parquet")
    assert doubles.variant_id.is_unique and variants.index.is_unique
    doubles["single_a"] = doubles.a_id.map(variants.point) - doubles.wt_point
    doubles["single_b"] = doubles.b_id.map(variants.point) - doubles.wt_point
    doubles["common_observed_main"] = doubles.single_a + doubles.single_b
    point = doubles.loc[doubles.main.fillna(False)]
    np.testing.assert_allclose(point.common_observed_main + point.epsilon, point.relative_point, atol=1e-6)
    coverage, jobs = [], []
    for threshold in [30, 25]:
        families = pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet")
        for seed in SEEDS:
            test = splits.loc[splits.seed.eq(seed) & splits.scheme.eq("unseen_pair") & splits["set"].eq("test")]
            assert test.variant_id.is_unique
            frame = doubles.loc[doubles.variant_id.isin(test.variant_id)].merge(families, on="protein_id", validate="many_to_one")
            available = set(order.loc[order.seed.eq(seed), "variant_id"])
            frame["known_single_a"] = frame.a_id.isin(available)
            frame["known_single_b"] = frame.b_id.isin(available)
            frame["selection_pool"] = frame.wt_all_point & frame.wt_point_in_range & frame.wt_point.le(4.5)
            frame["compensation_pool"] = (frame.selection_pool & frame.main.fillna(False) & frame.known_single_a & frame.known_single_b
                                          & frame.single_a.le(-0.5) & frame.single_b.le(-0.5))
            for fold in range(5):
                current = frame.loc[frame.outer_group.eq(fold)].sort_values("variant_id").copy()
                assert not current.duplicated("variant_id").any()
                path = OUTPUT / "candidates" / f"family{threshold}" / f"seed{seed}" / f"fold{fold}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                current.to_parquet(path, index=False)
                counts = current.loc[current.compensation_pool].groupby("protein_id").size()
                eligible = counts.index[counts.ge(100)]
                selected = current.loc[current.compensation_pool & current.protein_id.isin(eligible)]
                coverage.append({"threshold": threshold, "seed": seed, "fold": fold, "variants": len(current),
                                 "test_proteins": current.protein_id.nunique(), "compensation_proteins": len(eligible),
                                 "compensation_family_groups": selected.family_group.nunique(), "compensation_candidates": len(selected),
                                 "candidates": str(path.relative_to(ROOT))})
                for arm, specification in definitions().items():
                    if threshold not in specification["thresholds"]:
                        continue
                    for condition in specification["conditions"]:
                        directory = destination(arm, threshold, seed, fold, condition)
                        command = [sys.executable, "scripts/evaluate_compensation_main.py", "--arm", arm,
                                   "--threshold", str(threshold), "--seed", str(seed), "--fold", str(fold), "--condition", condition]
                        jobs.append({"name": f"{arm}/family{threshold}/seed{seed}/fold{fold}/{condition}",
                                     "arm": arm, "threshold": threshold, "seed": seed, "fold": fold, "condition": condition,
                                     "stages": [{"name": "compensation_evaluation", "command": command,
                                                 "manifest": str((directory / "manifest.json").relative_to(ROOT))}]})
    (OUTPUT / "jobs.json").write_text(json.dumps(jobs, indent=2))
    (OUTPUT / "candidate_coverage.json").write_text(json.dumps(coverage, indent=2))
    protocol = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "scientific_claim": "Known single effects plus transferred interactions improve compensatory double-mutant discovery in unseen site pairs and held-out families.",
                "main_model": "Source-validation selected flexible-mapping, jointly optimized interaction model used in the motivation diagnostic.",
                "primary_condition": "Sall", "target_double_labels_primary": 0,
                "auxiliary_conditions": ["C128", "C512", "C2048", "Call"], "auxiliary_family_threshold": 30,
                "seeds": SEEDS, "family_thresholds": [30, 25], "folds": list(range(5)), "budgets": BUDGETS, "margins": MARGINS,
                "single_loss_threshold": -0.5, "compensation_recovery_threshold": 0.5, "candidate_minimum": 100,
                "primary_endpoint": "Protein-macro compensation hit rate at budget10, common observed single main effect, margin0.",
                "secondary_endpoints": ["selected_mean_recovery", "selected_best_recovery", "selected_mean_stability", "selected_best_stability", "epsilon_rmse", "pair_macro_epsilon_rmse"],
                "selection": "All models retain source-validation choices. Target combination labels never select the primary model or define compensation pools.",
                "pool": "Both permitted measured single effects <= -0.5 kcal/mol; all four measurements continuous in range; reference <= 4.5 kcal/mol.",
                "hit": "Measured double stability >= max(measured component single stabilities) + 0.5 kcal/mol.",
                "scoring": "Common observed single sum + each model's four-genotype residual; native predictions also retained.",
                "aggregation": "Equal protein weights; pair results first averaged within protein; five repeats and family25 retained separately.",
                "statistics": "Paired mean gains, repeat direction, protein win/loss/tie and 10000 family-cluster bootstrap samples; intervals descriptive.",
                "coverage": "All eligible targets retained including targets with zero compensatory hits. Baseline ineligibility recorded; paired comparisons use complete common protein pools.",
                "definitions": definitions(), "expected_evaluations": len(jobs), "inputs": [
                    "data/features/training/tsuboyama/doubles/metadata.parquet", "data/processed/tsuboyama/variants.parquet",
                    "data/splits/combination_splits.parquet", "data/splits/single_label_order.parquet",
                    "data/splits/families_30.parquet", "data/splits/families_25.parquet"],
                "outputs": ["candidate_coverage.json", "jobs.json", "evaluations", "selection_per_protein.parquet", "residual_per_protein.parquet", "paired_summary.json", "manifest.json"]}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps({"prepared": True, "expected_evaluations": len(jobs), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
