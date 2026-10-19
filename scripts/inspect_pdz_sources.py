from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
original = pd.read_csv(ROOT / "data/extracted/flip_original_pdz/pdz3b1b2_norm_for_mochi.tsv", sep="\t", low_memory=False)
original["expressed"] = original.aa_seq.str.partition("*")[0]
first = pd.read_csv(ROOT / "data/raw/flip2/pdz3/single_to_double.csv.gz")
second = pd.read_csv(ROOT / "data/raw/flip2/pdz3/rand_split.csv.gz")
joined = first.merge(second, on="sequence", suffixes=("_first", "_second"), validate="one_to_one")
print({"official_same_inputs": len(joined) == len(first) == len(second),
       "official_label_difference": float((joined.target_first - joined.target_second).abs().max())})
source = original.groupby("expressed").fitness.mean()
comparison = first.copy()
comparison["expressed"] = comparison.sequence.str.replace(":", "", regex=False)
comparison["original"] = comparison.expressed.map(source)
comparison["difference"] = comparison.original - comparison.target
for keep in ["first", "last"]:
    exact = original.drop_duplicates("expressed", keep=keep).set_index("expressed").fitness.reindex(comparison.expressed).to_numpy()
    print({"duplicate_selection": keep, "round9_exact": int(np.equal(np.round(exact, 9), comparison.target).sum()),
           "maximum_difference": float(np.abs(exact - comparison.target).max())})
print(original.loc[original.expressed.eq(comparison.expressed.iloc[1]), ["aa_seq", "fitness"]].to_string(index=False))
comparison["receptor"] = comparison.sequence.str.partition(":")[0]
comparison["peptide"] = comparison.sequence.str.partition(":")[2]
print({"original_rows": len(original), "original_unique_sequences": original.aa_seq.nunique(),
       "maximum_difference": comparison.difference.abs().max(),
       "above_1e_7": int(comparison.difference.abs().gt(1e-7).sum()),
       "above_1e_5": int(comparison.difference.abs().gt(1e-5).sum()),
       "zero_mutation_reference": original.loc[original.Nham_aa.eq(0), ["aa_seq", "fitness"]].to_dict("records")})
print(comparison.groupby(["set", "validation"]).difference.agg(["count", "min", "max"]).to_string())
worst = comparison.nlargest(12, "difference", keep="all").index.union(comparison.difference.abs().nlargest(12).index)
print(comparison.loc[worst, ["peptide", "target", "original", "difference", "set", "validation"]].to_string(index=False))
keys = comparison.loc[comparison.difference.abs().gt(1e-5), "expressed"]
relevant = original.loc[original.expressed.isin(keys)].copy()
print(relevant[["aa_seq", "Nham_aa", "fitness", "growthrate", "sigma"]].head(8).to_string(index=False))
output = ROOT / "work/pdz_source_audit"
output.mkdir(exist_ok=True)
comparison.to_parquet(output / "comparison.parquet", index=False)
