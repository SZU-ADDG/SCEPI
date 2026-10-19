import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from epistasis.main_evaluation import OUTPUT, SEEDS, SCHEMES, evaluation_directory, functional_definitions, selected_source, stability_definitions
from run_functional_adaptation import jobs_for
from run_mochi_targets import tasks_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permissions-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    permissions = OUTPUT / "tsuboyama_C_tasks.parquet"
    assert not permissions.exists()
    if args.permissions_only:
        assert (OUTPUT / "protocol.json").exists()
        pd.DataFrame(tasks_for("tsuboyama", "C")).to_parquet(permissions, index=False)
        print(json.dumps({"frozen_C_permissions": str(permissions.relative_to(ROOT))}), flush=True)
        return
    assert not (OUTPUT / "protocol.json").exists()
    pd.DataFrame(tasks_for("tsuboyama", "C")).to_parquet(permissions, index=False)
    columns = ["matrix_row", "variant_id", "protein_id", "mutation_key", "pair_id", "a_id", "b_id", "wt_variant_id",
               "main", "margin", "epsilon", "relative_point", "relative_lower", "relative_upper", "wt_point",
               "wt_all_point", "wt_point_in_range", "structure_available"]
    doubles = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet", columns=columns)
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet", columns=["variant_id", "point", "ci_width_max"]).set_index("variant_id")
    doubles["observed_additive"] = doubles.a_id.map(variants.point) + doubles.b_id.map(variants.point) - 2 * doubles.wt_point
    widths = np.column_stack([doubles[column].map(variants.ci_width_max).to_numpy() for column in ["variant_id", "a_id", "b_id", "wt_variant_id"]])
    doubles["quartet_ci_width_max"] = np.max(widths, axis=1)
    distance = np.load(ROOT / "data/features/training/tsuboyama/doubles/ca_distance.npy", mmap_mode="r")
    gaps = np.load(ROOT / "data/features/training/tsuboyama/doubles/sequence_gap.npy", mmap_mode="r")
    doubles["ca_distance"] = distance[doubles.matrix_row.to_numpy()]
    doubles["sequence_gap"] = gaps[doubles.matrix_row.to_numpy()]
    doubles["distance_stratum"] = pd.cut(doubles.ca_distance.where(doubles.structure_available), [-np.inf, 8, 12, np.inf], labels=["within8A", "8to12A", "above12A"]).astype(str)
    doubles["reference_stratum"] = pd.cut(doubles.wt_point, [-np.inf, 0, 2, 4, np.inf], labels=["below0", "0to2", "2to4", "above4"]).astype(str)
    doubles["additive_stratum"] = pd.cut(doubles.observed_additive, [-np.inf, -2, 0, 2, np.inf], labels=["below-2", "-2to0", "0to2", "above2"]).astype(str)
    point = doubles.loc[doubles.main.fillna(False)]
    np.testing.assert_allclose(point.observed_additive + point.epsilon, point.relative_point, atol=1e-6)
    boundaries, coverage, jobs = [], [], []
    specifications = stability_definitions()
    for threshold in [30, 25]:
        family = pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet")
        frame = doubles.merge(family, on="protein_id", validate="many_to_one")
        for fold in range(5):
            source = selected_source("runs/e3", f"pair_repeats_full_g0_f0_j0_family{threshold}/selected_hyperparameters.json", threshold, fold, 11)
            assert source is not None
            permission = json.loads((source / "permissions.json").read_text())
            training = frame.loc[frame.protein_id.isin(permission["train"]) & frame.main.fillna(False)]
            testing = frame.loc[frame.outer_group.eq(fold)].copy()
            assert set(testing.protein_id) <= set(permission["test"])
            assert not set(testing.protein_id) & (set(permission["train"]) | set(permission["validation"]))
            thresholds = {}
            for field, destination in [("sequence_gap", "sequence_gap_stratum"), ("quartet_ci_width_max", "precision_stratum")]:
                values = training[field].dropna()
                assert len(values) and np.isfinite(values).all()
                cuts = np.unique(values.quantile([1 / 3, 2 / 3]).to_numpy())
                testing[destination] = pd.cut(testing[field], [-np.inf, *cuts.tolist(), np.inf],
                                               labels=[f"source_group{index + 1}" for index in range(len(cuts) + 1)]).astype(str)
                thresholds[field] = cuts.tolist()
            directory = OUTPUT / "truth" / f"family{threshold}"
            directory.mkdir(parents=True, exist_ok=True)
            testing.sort_values("variant_id").to_parquet(directory / f"fold{fold}.parquet", index=False)
            boundaries.append({"threshold": threshold, "fold": fold, "training_proteins": permission["train"],
                               "training_quartets": len(training), "source_permissions": str((source / "permissions.json").relative_to(ROOT)),
                               "cutoffs": thresholds})
            coverage.append({"threshold": threshold, "fold": fold, "variants": len(testing), "proteins": testing.protein_id.nunique(),
                             "family_groups": testing.family_group.nunique(), "main_quartets": int(testing.main.sum())})
            for seed in SEEDS:
                for arm, item in specifications.items():
                    if threshold not in item["thresholds"]:
                        continue
                    for scheme in (["family"] if item["scope"] == "family" else SCHEMES):
                        for condition in item["conditions"]:
                            jobs.append({"dataset": "tsuboyama", "arm": arm, "threshold": threshold, "fold": fold,
                                         "seed": seed, "scheme": scheme, "condition": condition})
    for dataset in ["functional_doubles", "flip_primary"]:
        directory = OUTPUT / "functional_truth"
        directory.mkdir(parents=True, exist_ok=True)
        variants = pd.read_parquet(ROOT / "data/processed" / dataset / "variants.parquet")
        variants.to_parquet(directory / (dataset + ".parquet"), index=False)
        quartet_records = []
        for protein, table in variants.groupby("protein_id", sort=True):
            reference = table.loc[table.n_mutations.eq(0) & table.point.notna()]
            assert len(reference) <= 1
            if not len(reference):
                continue
            alleles = table.set_index("mutation_key")
            assert alleles.index.is_unique
            for row in table.loc[table.n_mutations.eq(2) & table.point.notna()].itertuples():
                names = [f"{wild}{int(position) + 1}{mutant}" for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)]
                if not all(name in alleles.index and np.isfinite(alleles.loc[name, "point"]) for name in names):
                    continue
                first, second = [alleles.loc[name] for name in names]
                quartet_records.append({"protein_id": protein, "ab_id": row.variant_id, "a_id": first.variant_id,
                                        "b_id": second.variant_id, "wt_id": reference.variant_id.iloc[0],
                                        "pair_id": f"{protein}:{int(row.positions[0])}:{int(row.positions[1])}",
                                        "epsilon": row.point - first.point - second.point + float(reference.point.iloc[0])})
        quartets = pd.DataFrame(quartet_records, columns=["protein_id", "ab_id", "a_id", "b_id", "wt_id", "pair_id", "epsilon"])
        if dataset == "functional_doubles":
            original = pd.read_parquet(ROOT / "data/processed/functional/quartets.parquet").set_index("ab_id")
            assert set(quartets.ab_id) == set(original.index)
            np.testing.assert_allclose(quartets.epsilon, original.epsilon.reindex(quartets.ab_id), atol=1e-8)
        quartets.to_parquet(directory / (dataset + "_quartets.parquet"), index=False)
        shared, local = pd.DataFrame(jobs_for(dataset)), pd.DataFrame(tasks_for(dataset, "C" if dataset == "functional_doubles" else "flip"))
        for name, tasks in [("shared", shared), ("local", local)]:
            assert not tasks.duplicated(["protein_id", "seed", "scheme", "condition"]).any()
            tasks.to_parquet(directory / (dataset + "_" + name + "_tasks.parquet"), index=False)
        for seed, scheme, condition in shared[["seed", "scheme", "condition"]].drop_duplicates().itertuples(index=False, name=None):
            for arm in functional_definitions(dataset):
                jobs.append({"dataset": dataset, "arm": arm, "threshold": 0, "fold": 0, "seed": int(seed),
                             "scheme": scheme, "condition": condition})
    for job in jobs:
        job["name"] = "/".join(str(job[key]) for key in ["dataset", "arm", "threshold", "seed", "scheme", "fold", "condition"])
        job["stages"] = [{"name": "original_evaluation", "manifest": str((evaluation_directory(job) / "manifest.json").relative_to(ROOT)),
                          "command": [sys.executable, "scripts/evaluate_original_main.py", "--job", job["name"]]}]
    assert len({job["name"] for job in jobs}) == len(jobs)
    (OUTPUT / "jobs.json").write_text(json.dumps(jobs, indent=2))
    (OUTPUT / "stratum_boundaries.json").write_text(json.dumps(boundaries, indent=2))
    (OUTPUT / "truth_coverage.json").write_text(json.dumps(coverage, indent=2))
    protocol = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "run_path": str(OUTPUT), "expected_evaluations": len(jobs), "seeds": SEEDS, "family_thresholds": [30, 25],
                "primary_family_model": "original_interaction", "compensation_model_family_comparison": "interaction",
                "conditions": ["Z", "S32", "S128", "Sall", "C128", "C512", "C2048", "Call"],
                "family_test": "Every held-out family double, retaining the original full-family test separately from the compensation intersection.",
                "combination_test": "Frozen random, unseen_pair and two_pair_secondary test IDs. C128/512/2048 use matched combination budgets.",
                "measurements": "Main continuous quartets, all identifiable point quartets, margins0.25/0.5 and censored interval distance.",
                "strata": "CA distance<=8/8to12/>12 A, source-training thirds of sequence gap and maximum four-measurement CI width, reference stability, observed additive stability, natural/design origin.",
                "statistics": "Equal-protein means after repeat averaging, paired differences, repeat direction, protein wins/losses/ties; 10000 paired family/protein bootstrap replicates for the predefined primary comparisons. Intervals describe uncertainty.",
                "stability_definitions": specifications,
                "functional_definitions": {dataset: functional_definitions(dataset) for dataset in ["functional_doubles", "flip_primary"]},
                "coverage_policy": "All eligible cases remain; ineligible configurations retain explicit reasons. Paired results require identical sample pools.",
                "outputs": ["jobs.json", "truth_coverage.json", "stratum_boundaries.json", "evaluations", "per_protein.parquet", "strata_per_protein.parquet", "paired_summary.json", "manifest.json"]}
    (OUTPUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps({"prepared": True, "expected_evaluations": len(jobs), "coverage": coverage}), flush=True)


if __name__ == "__main__":
    main()
