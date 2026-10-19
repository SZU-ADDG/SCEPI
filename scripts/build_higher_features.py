import argparse
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.features import pair_features, single_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--double-proteins-only", action="store_true")
    args = parser.parse_args()
    data = ROOT / "data/processed" / args.dataset
    cache = ROOT / "data/features/esm2" / args.dataset / "full"
    output = ROOT / "data/features/higher" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_parquet(data / "proteins.parquet").sort_values("protein_id")
    variants = pd.read_parquet(data / "variants.parquet")
    if args.double_proteins_only:
        proteins = proteins.loc[proteins.protein_id.isin(variants.loc[variants.n_mutations.eq(2), "protein_id"])]
        variants = variants.loc[variants.protein_id.isin(proteins.protein_id)].copy()
    results = []
    for protein in proteins.itertuples():
        directory = output / protein.protein_id
        directory.mkdir(parents=True, exist_ok=True)
        group = variants.loc[variants.protein_id.eq(protein.protein_id)].reset_index(drop=True)
        with h5py.File(cache / f"{protein.protein_id}_singles.h5", "r") as handle:
            assert handle.attrs["reference_sequence"] == protein.reference_sequence
            allele_keys = handle["keys"].asstr()[:].tolist()
        allele_index = {key: i for i, key in enumerate(allele_keys)}
        parts = [tuple(key for key in mutation.split(":") if key and key != "WT") for mutation in group.mutation_key]
        pair_keys = sorted({tuple(pair) for values in parts for pair in combinations(values, 2)})
        pair_index = {key: i for i, key in enumerate(pair_keys)}
        allele_rows, allele_columns, pair_rows, pair_columns = [], [], [], []
        for row, values in enumerate(parts):
            assert len(values) == group.n_mutations.iloc[row]
            assert all(protein.reference_sequence[int(key[1:-1]) - 1] == key[0] for key in values)
            allele_rows.extend([row] * len(values))
            allele_columns.extend(allele_index[key] for key in values)
            pairs = list(combinations(values, 2))
            pair_rows.extend([row] * len(pairs))
            pair_columns.extend(pair_index[key] for key in pairs)
        single_incidence = sparse.coo_matrix((np.ones(len(allele_rows), dtype=np.float32),
                                              (allele_rows, allele_columns)), shape=(len(group), len(allele_keys))).tocsr()
        pair_incidence = sparse.coo_matrix((np.ones(len(pair_rows), dtype=np.float32),
                                            (pair_rows, pair_columns)), shape=(len(group), len(pair_keys))).tocsr()
        np.testing.assert_array_equal(np.asarray(single_incidence.sum(axis=1)).ravel(), group.n_mutations)
        np.testing.assert_array_equal(np.asarray(pair_incidence.sum(axis=1)).ravel(), group.n_mutations * (group.n_mutations - 1) // 2)
        alleles = pd.DataFrame({
            "feature_id": [f"{protein.protein_id}_allele_{i:05d}" for i in range(len(allele_keys))],
            "mutation_key": allele_keys, "positions": [[int(key[1:-1]) - 1] for key in allele_keys],
            "wt_residues": [key[0] for key in allele_keys], "mut_residues": [key[-1] for key in allele_keys],
        })
        pairs = pd.DataFrame({
            "feature_id": [f"{protein.protein_id}_pair_{i:06d}" for i in range(len(pair_keys))],
            "mutation_key": [":".join(value) for value in pair_keys],
            "positions": [[int(key[1:-1]) - 1 for key in value] for value in pair_keys],
            "wt_residues": ["".join(key[0] for key in value) for value in pair_keys],
            "mut_residues": ["".join(key[-1] for key in value) for value in pair_keys],
            "allele_indices": [[allele_index[key] for key in value] for value in pair_keys],
        })
        assert all(positions[0] < positions[1] for positions in pairs.positions)
        for kind, table, values in [
            ("single", alleles, single_features(protein, alleles, cache)),
            ("pair", pairs, pair_features(protein, pairs, cache)),
        ]:
            folder = directory / kind
            folder.mkdir(parents=True, exist_ok=True)
            table.to_parquet(folder / "metadata.parquet", index=False)
            for name, array in values.items():
                assert np.isfinite(array).all()
                np.save(folder / f"{name}.npy", array)
        sparse.save_npz(directory / "single_incidence.npz", single_incidence)
        sparse.save_npz(directory / "pair_incidence.npz", pair_incidence)
        group.to_parquet(directory / "variants.parquet", index=False)
        result = {"protein_id": protein.protein_id, "variants": len(group), "alleles": len(alleles),
                  "pairs": len(pairs), "maximum_mutation_count": int(group.n_mutations.max()),
                  "single_terms": int(single_incidence.nnz), "pair_terms": int(pair_incidence.nnz)}
        results.append(result)
        print(json.dumps(result), flush=True)
    manifest = {
        "complete": True, "dataset": args.dataset, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "proteins": len(proteins), "variants": len(variants), "features": results,
        "mechanism": "Per-variant sums of reference-relative allele effects and unordered pair interactions, followed by the assay mapping.",
        "labels_used_to_construct_features": 0, "geometry": False,
        "counterfactual_features": "Alleles and pairs occurring in actual variants; feature objects do not imply that the corresponding genotypes were experimentally measured.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
