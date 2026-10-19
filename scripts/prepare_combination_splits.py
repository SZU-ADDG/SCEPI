import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/tsuboyama"
OUTPUT = ROOT / "data/splits"
SEEDS = [11, 23, 47, 89, 131]


def component_audit(variants, quartets):
    lookup = variants.set_index("variant_id")
    values = {}
    for part in ["wt", "a", "b", "ab"]:
        values[part] = lookup.loc[quartets[part + "_id"]].reset_index(drop=True)
        assert values[part].protein_id.equals(quartets.protein_id)
    for column, formula in [
        ("lower", values["ab"].lower - values["a"].upper - values["b"].upper + values["wt"].lower),
        ("upper", values["ab"].upper - values["a"].lower - values["b"].lower + values["wt"].upper),
        ("epsilon", values["ab"].point - values["a"].point - values["b"].point + values["wt"].point),
    ]:
        np.testing.assert_allclose(quartets[column], formula, rtol=0, atol=1e-12, equal_nan=True)
    assert all(values[part].point_in_range.loc[quartets.main].all() for part in values)
    assert values["wt"].n_mutations.eq(0).all()
    assert values["a"].n_mutations.eq(1).all() and values["b"].n_mutations.eq(1).all()
    assert values["ab"].n_mutations.eq(2).all()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    variants = pd.read_parquet(DATA / "variants.parquet")
    quartets = pd.read_parquet(DATA / "quartets.parquet").reset_index(drop=True)
    proteins = pd.read_parquet(DATA / "proteins.parquet")
    component_audit(variants, quartets)
    complete = quartets.loc[quartets.main]
    valid_pairs = complete.groupby("protein_id").pair_id.unique().to_dict()
    doubles = variants.loc[variants.n_mutations.eq(2)].copy()
    doubles["pair_id"] = doubles.apply(lambda row: f"{row.protein_id}:{row.positions[0]}:{row.positions[1]}", axis=1)
    wt = variants.loc[variants.n_mutations.eq(0)].set_index("protein_id")
    singles = variants.loc[variants.n_mutations.eq(1) & variants.point_in_range].copy()
    singles = singles.loc[singles.protein_id.isin(wt.index[wt.point_in_range])]
    singles["relative_point"] = singles.point - singles.protein_id.map(wt.point)
    single_rows, combination_rows, summaries = [], [], []
    for seed in SEEDS:
        for protein_index, protein in enumerate(proteins.protein_id.sort_values()):
            rng = np.random.default_rng(np.random.SeedSequence([seed, protein_index, 0]))
            single = singles.loc[singles.protein_id.eq(protein)].sort_values("mutation_key")
            permutation = rng.permutation(len(single))
            ordered = single.iloc[permutation]
            for rank, row in enumerate(ordered.itertuples()):
                single_rows.append({"seed": seed, "protein_id": protein, "variant_id": row.variant_id, "rank": rank, "relative_point": row.relative_point})
            group = doubles.loc[doubles.protein_id.eq(protein)].sort_values("mutation_key").copy()
            if len(group) == 0:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, protein_index, 1]))
            ordered_indices = rng.permutation(len(group))
            n_train = int(0.8 * len(group))
            n_val = int(0.1 * len(group))
            labels = np.full(len(group), "test", dtype=object)
            labels[ordered_indices[:n_train]] = "train"
            labels[ordered_indices[n_train:n_train + n_val]] = "validation"
            schemes = [("random", labels, True)]
            pairs = np.array(sorted(valid_pairs.get(protein, [])), dtype=object)
            if len(pairs) >= 2:
                rng = np.random.default_rng(np.random.SeedSequence([seed, protein_index, 2]))
                pairs = pairs[rng.permutation(len(pairs))]
                if len(pairs) == 2:
                    train_pairs, validation_pairs, test_pairs = pairs[:1], [], pairs[1:]
                else:
                    n_validation = max(1, int(round(len(pairs) * 0.2)))
                    n_test = max(1, int(round(len(pairs) * 0.2)))
                    train_pairs = pairs[:len(pairs) - n_validation - n_test]
                    validation_pairs = pairs[len(train_pairs):len(train_pairs) + n_validation]
                    test_pairs = pairs[len(train_pairs) + n_validation:]
                partition = {pair: category for category, entries in [("train", train_pairs), ("validation", validation_pairs), ("test", test_pairs)] for pair in entries}
                labels = group.pair_id.map(partition).fillna("excluded_pair_without_main_quartet").to_numpy()
                schemes.append(("unseen_pair" if len(pairs) >= 3 else "two_pair_secondary", labels, len(pairs) >= 3))
            for scheme, labels, primary in schemes:
                frame = group[["variant_id", "protein_id", "mutation_key", "pair_id"]].copy()
                frame["set"] = labels
                frame["seed"] = seed
                frame["scheme"] = scheme
                rng = np.random.default_rng(np.random.SeedSequence([seed, protein_index, 3]))
                train = frame.loc[frame["set"].eq("train")].variant_id.to_numpy()
                rng.shuffle(train)
                train_rank = {value: rank for rank, value in enumerate(train)}
                frame["train_rank"] = frame.variant_id.map(train_rank).fillna(-1).astype(int)
                combination_rows.append(frame)
                if scheme != "random":
                    assert frame.loc[frame["set"].isin(["train", "validation", "test"])].groupby("pair_id")["set"].nunique().eq(1).all()
                tested = complete.loc[complete.ab_id.isin(frame.loc[frame["set"].eq("test"), "variant_id"])]
                summaries.append({
                    "seed": seed, "protein_id": protein, "scheme": scheme,
                    "training_combinations": len(train), "validation_combinations": int(frame["set"].eq("validation").sum()),
                    "test_combinations": int(frame["set"].eq("test").sum()), "main_test_quartets": len(tested),
                    "main_test_eligible": bool(primary and len(tested) >= 30),
                    "C128_available": len(train) >= 128, "C512_available": len(train) >= 512,
                    "C2048_available": len(train) >= 2048, "available_single_labels": len(single),
                })
    singles_out = pd.DataFrame(single_rows)
    combinations_out = pd.concat(combination_rows, ignore_index=True)
    assert not singles_out.duplicated(["seed", "variant_id"]).any()
    assert not combinations_out.duplicated(["seed", "scheme", "variant_id"]).any()
    singles_out.to_parquet(OUTPUT / "single_label_order.parquet", index=False)
    combinations_out.to_parquet(OUTPUT / "combination_splits.parquet", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "combination_split_counts.csv", index=False)
    result = {
        "stage": "E2_split", "seeds": SEEDS, "dataset": "tsuboyama",
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "data_manifest": str((DATA / "manifest.json").relative_to(ROOT)),
        "quartet_components_and_interval_arithmetic": "passed_all_182398_quartets",
        "C_budget": "First 128/512/2048 training combinations or all training combinations. Validation labels are separately counted in the total supervised-label ledger.",
        "S_budget": "First 32/128/all point single variants in each fixed sequence-based permutation. WT observations are separately counted.",
        "two_pair_secondary": "One training pair, one test pair; no target validation labels. Hyperparameters must come from source validation.",
        "test_pair_overlap_with_training": 0,
        "counts": summary.groupby(["seed", "scheme"]).agg(proteins=("protein_id", "nunique"), main_test_proteins=("main_test_eligible", "sum"), C128=("C128_available", "sum")).reset_index().to_dict(orient="records"),
        "outputs": ["data/splits/single_label_order.parquet", "data/splits/combination_splits.parquet", "data/splits/combination_split_counts.csv"],
    }
    (OUTPUT / "combination_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
