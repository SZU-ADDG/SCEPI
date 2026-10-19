import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.features import pair_features, single_features

DATA = ROOT / "data/processed/tsuboyama"
CACHE = ROOT / "data/features/esm2/tsuboyama/full"
GEOMETRY = ROOT / "data/features/structures/tsuboyama"
OUTPUT = ROOT / "data/features/training/tsuboyama"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((CACHE / f"singles_manifest_rank{rank}.json").read_text()) for rank in range(8)]
    assert all(item["complete"] for item in manifests)
    proteins = pd.read_parquet(DATA / "proteins.parquet")
    variants = pd.read_parquet(DATA / "variants.parquet")
    quartets = pd.read_parquet(DATA / "quartets.parquet")
    wt = variants.loc[variants.n_mutations.eq(0)].set_index("protein_id")
    assert wt.index.is_unique
    counts = {}
    for kind, number in [("singles", 1), ("doubles", 2)]:
        table = variants.loc[variants.n_mutations.eq(number)].sort_values(["protein_id", "mutation_key"]).reset_index(drop=True)
        table.insert(0, "matrix_row", np.arange(len(table)))
        for column in ["variant_id", "point", "lower", "upper", "point_in_range", "all_point"]:
            table["wt_" + column] = table.protein_id.map(wt[column])
        table["relative_point"] = table.point - table.wt_point
        table["relative_lower"] = table.lower - table.wt_upper
        table["relative_upper"] = table.upper - table.wt_lower
        table["structure_available"] = table.protein_id.map(lambda protein: (GEOMETRY / f"{protein}.npz").exists())
        if number == 2:
            attached = quartets[["quartet_id", "ab_id", "a_id", "b_id", "pair_id", "epsilon", "lower", "upper", "main", "margin"]].rename(columns={"lower": "epsilon_lower", "upper": "epsilon_upper"})
            table = table.merge(attached, left_on="variant_id", right_on="ab_id", how="left", validate="one_to_one")
            table["main"] = table.main.fillna(False).astype(bool)
        directory = OUTPUT / kind
        directory.mkdir(parents=True, exist_ok=True)
        matrices = {}
        for protein in proteins.itertuples():
            group = table.loc[table.protein_id.eq(protein.protein_id)]
            if len(group) == 0:
                continue
            if number == 1:
                features = single_features(protein, group, CACHE)
            else:
                geometry = GEOMETRY if group.structure_available.all() else None
                features = pair_features(protein, group, CACHE, geometry)
                if geometry is None:
                    features["ca_distance"] = np.full(len(group), np.nan, dtype=np.float32)
                    features["confidence"] = np.full((len(group), 2), np.nan, dtype=np.float32)
            for name, values in features.items():
                if name not in matrices:
                    matrices[name] = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=values.dtype, shape=(len(table), *values.shape[1:]))
                matrices[name][group.matrix_row.to_numpy()] = values
                if name not in {"confidence", "ca_distance"} or group.structure_available.all():
                    assert np.isfinite(values).all(), (protein.protein_id, name)
            print(json.dumps({"kind": kind, "protein": protein.protein_id, "rows": len(group)}), flush=True)
        for matrix in matrices.values():
            matrix.flush()
        assert table.matrix_row.equals(pd.Series(np.arange(len(table)), name="matrix_row"))
        table.to_parquet(directory / "metadata.parquet", index=False)
        counts[kind] = {"rows": len(table), "shapes": {name: list(matrix.shape) for name, matrix in matrices.items()}, "structure_rows": int(table.structure_available.sum())}
    observations = pd.read_parquet(DATA / "observations.parquet")
    observations = observations.loc[observations.accepted, ["observation_id", "variant_id", "protein_id", "n_mutations", "point", "lower", "upper", "measurement_status", "point_in_range"]]
    observations.to_parquet(OUTPUT / "observation_targets.parquet", index=False)
    wt.reset_index().to_parquet(OUTPUT / "reference_targets.parquet", index=False)
    manifest = {
        "stage": "training_features", "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": manifests[0]["revision"], "frozen_feature_manifests": [str((CACHE / f"singles_manifest_rank{rank}.json").relative_to(ROOT)) for rank in range(8)],
        "data_manifest": str((DATA / "manifest.json").relative_to(ROOT)), "counts": counts,
        "raw_observations": len(observations),
        "labels": "Phenotypes and quartet components remain in metadata. Runtime manifests determine which labels may enter each model.",
        "array_precision": "float32 features derived from the fixed float16 per-residue cache",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
