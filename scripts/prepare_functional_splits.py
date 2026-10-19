import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [11, 23, 47, 89, 131]


def main():
    data = ROOT / "data/processed/functional_doubles"
    output = ROOT / "data/splits/functional"
    output.mkdir(parents=True, exist_ok=True)
    variants = pd.read_parquet(data / "variants.parquet")
    quartets = pd.read_parquet(ROOT / "data/processed/functional/quartets.parquet")
    doubles = variants.loc[variants.n_mutations.eq(2)].sort_values(["protein_id", "mutation_key"]).copy()
    doubles["pair_id"] = [f"{row.protein_id}:{row.positions[0]}:{row.positions[1]}" for row in doubles.itertuples()]
    singles = variants.loc[variants.n_mutations.eq(1) & variants.all_point & variants.point.notna()]
    records, single_records, counts = [], [], []
    for protein_index, (protein, group) in enumerate(doubles.groupby("protein_id", sort=True)):
        group = group.reset_index(drop=True)
        main_ids = set(quartets.loc[quartets.protein_id.eq(protein), "ab_id"])
        pairs = sorted(group.loc[group.variant_id.isin(main_ids), "pair_id"].unique())
        phenotype_only = not len(pairs)
        if phenotype_only:
            pairs = sorted(group.pair_id.unique())
        for seed in SEEDS:
            single = singles.loc[singles.protein_id.eq(protein)].sort_values("mutation_key")
            single = single.iloc[np.random.default_rng(seed).permutation(len(single))]
            single_records.append(single[["variant_id", "protein_id", "mutation_key"]].assign(seed=seed, rank=np.arange(len(single))))
            rng = np.random.default_rng(np.random.SeedSequence([seed, protein_index, 1]))
            order = rng.permutation(len(group))
            labels = np.full(len(group), "test", dtype=object)
            n_train, n_valid = int(len(group) * 0.8), int(len(group) * 0.1)
            labels[order[:n_train]], labels[order[n_train:n_train + n_valid]] = "train", "validation"
            schemes = [("random", labels)]
            if len(pairs) >= 3:
                rng = np.random.default_rng(np.random.SeedSequence([seed, protein_index, 2]))
                ordered = np.asarray(pairs, dtype=object)[rng.permutation(len(pairs))]
                n_valid = n_test = max(1, int(round(len(pairs) * 0.2)))
                n_train = len(pairs) - n_valid - n_test
                partition = {pair: label for label, entries in [
                    ("train", ordered[:n_train]), ("validation", ordered[n_train:n_train + n_valid]),
                    ("test", ordered[n_train + n_valid:])] for pair in entries}
                schemes.append(("unseen_pair", group.pair_id.map(partition).fillna("excluded_pair_without_quartet").to_numpy()))
            for scheme, assigned in schemes:
                frame = group[["variant_id", "protein_id", "mutation_key", "pair_id"]].copy()
                frame["set"], frame["seed"], frame["scheme"] = assigned, seed, scheme
                train_ids = frame.loc[frame["set"].eq("train"), "variant_id"].to_numpy()
                np.random.default_rng(np.random.SeedSequence([seed, protein_index, 3])).shuffle(train_ids)
                ranks = pd.Series(np.arange(len(train_ids)), index=train_ids)
                frame["train_rank"] = frame.variant_id.map(ranks).fillna(-1).astype(int)
                assert not frame.variant_id.duplicated().any()
                if scheme == "unseen_pair":
                    assert frame.loc[~frame["set"].str.startswith("excluded")].groupby("pair_id")["set"].nunique().eq(1).all()
                records.append(frame)
                test = frame.loc[frame["set"].eq("test")]
                counts.append({"protein_id": protein, "seed": seed, "scheme": scheme,
                               "train": len(train_ids), "validation": int(frame["set"].eq("validation").sum()),
                               "test": len(test), "test_quartets": len(set(test.variant_id) & main_ids),
                               "phenotype_only": phenotype_only, "singles": len(single)})
    combined = pd.concat(records, ignore_index=True)
    assert not combined.duplicated(["scheme", "seed", "variant_id"]).any()
    combined.to_parquet(output / "combination_splits.parquet", index=False)
    pd.concat(single_records, ignore_index=True).to_parquet(output / "single_label_order.parquet", index=False)
    pd.DataFrame(counts).to_csv(output / "counts.csv", index=False)
    manifest = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "seeds": SEEDS, "rows": len(combined), "counts": counts,
                "test_pair_overlap": 0, "source": "data/processed/functional_doubles/manifest.json",
                "phenotype_only_rule": "Assays without complete quartets retain sequence-based pair separation and phenotype evaluation."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"complete": True, "proteins": doubles.protein_id.nunique(), "split_rows": len(combined)}))


if __name__ == "__main__":
    main()
