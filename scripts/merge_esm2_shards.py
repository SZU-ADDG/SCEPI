import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "data/features/esm2" / args.dataset / ("smoke" if args.smoke else "full")
    manifests = [json.loads((directory / f"variants_manifest_rank{rank}.json").read_text()) for rank in range(8)]
    assert all(item["complete"] and item["shard_variants"] and item["smoke"] == args.smoke for item in manifests)
    assert all(item["world_size"] == 8 for item in manifests)
    data = ROOT / "data/processed" / args.dataset
    variants = pd.read_parquet(data / "variants.parquet")
    proteins = pd.read_parquet(data / "proteins.parquet")
    outputs = []
    for protein in proteins.itertuples():
        current = variants.loc[variants.protein_id.eq(protein.protein_id)]
        identifiers, means, additive, context = [], [], [], []
        for rank in range(8):
            subset = current.iloc[rank::8]
            if args.smoke:
                subset = subset.iloc[:max(1, int(len(subset) * manifests[rank]["smoke_fraction_per_selected_protein"]))]
            with h5py.File(directory / f"{protein.protein_id}_variants_rank{rank}.h5", "r") as handle:
                assert handle.attrs["reference_sequence"] == protein.reference_sequence
                names = handle["variant_ids"].asstr()[:]
                assert names.tolist() == subset.variant_id.tolist()
                identifiers.extend(names)
                means.append(handle["mean"][:])
                additive.append(handle["additive_score"][:])
                context.append(handle["context_score"][:])
        assert len(identifiers) == len(set(identifiers))
        if not args.smoke:
            assert set(identifiers) == set(current.variant_id)
        mean_array = np.concatenate(means)
        additive_array, context_array = np.concatenate(additive), np.concatenate(context)
        assert np.isfinite(mean_array).all() and np.isfinite(additive_array).all() and np.isfinite(context_array).all()
        path = directory / f"{protein.protein_id}_variants.h5"
        assert not path.exists()
        partial = path.with_suffix(".h5.part")
        with h5py.File(partial, "w") as handle:
            handle.attrs["protein_id"], handle.attrs["reference_sequence"] = protein.protein_id, protein.reference_sequence
            handle.attrs["model_revision"], handle.attrs["stage"] = manifests[0]["revision"], "variants"
            handle.create_dataset("variant_ids", data=np.asarray(identifiers, dtype=h5py.string_dtype()))
            handle.create_dataset("mean", data=mean_array, compression="lzf")
            handle.create_dataset("additive_score", data=additive_array)
            handle.create_dataset("context_score", data=context_array)
        os.replace(partial, path)
        outputs.append({"protein_id": protein.protein_id, "variants": len(identifiers), "path": str(path.relative_to(ROOT))})
    result = {"complete": True, "config": vars(args), "input_ranks": 8,
              "variants": sum(item["variants"] for item in outputs), "outputs": outputs,
              "shard_identifiers_and_finite_outputs_verified": True}
    (directory / "variants_merged_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
