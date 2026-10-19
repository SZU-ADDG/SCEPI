import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    proteins = pd.read_parquet(ROOT / "data/processed/tsuboyama/proteins.parquet")
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet", columns=["protein_id", "n_mutations"])
    structures = pd.read_parquet(ROOT / "data/features/structures/tsuboyama/availability.parquet")
    selected = proteins.loc[proteins.protein_id.isin(variants.loc[variants.n_mutations.eq(2), "protein_id"])]
    selected = selected.merge(structures, on="protein_id", validate="one_to_one").sort_values("protein_id")
    assert selected.available.all()
    total = len(selected)
    if args.smoke:
        selected = selected.iloc[:max(1, round(total * 0.01))]
    selected = selected.iloc[args.rank::args.world_size]
    output = ROOT / "data/features/proteinmpnn/tsuboyama" / ("smoke" if args.smoke else "full")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = []
    for protein in selected.itertuples():
        current = output / protein.protein_id
        current.mkdir(parents=True, exist_ok=True)
        manifest_path = current / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            assert previous["complete"] and previous["reference_sequence"] == protein.reference_sequence
            results.append(previous)
            continue
        command = [
            sys.executable, str(ROOT / "external/ProteinMPNN/protein_mpnn_run.py"),
            "--pdb_path", str(ROOT / protein.structure),
            "--save_score", "1", "--conditional_probs_only", "1",
            "--num_seq_per_target", "10", "--batch_size", "1", "--seed", "37",
            "--model_name", "v_48_020", "--out_folder", str(current / "author_output"),
        ]
        with (current / "author.log").open("w") as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        raw_path = current / "author_output/conditional_probs_only" / (Path(protein.structure).stem + ".npz")
        raw = np.load(raw_path)
        sequence = "".join(ALPHABET[int(value)] for value in raw["S"])
        assert sequence == protein.reference_sequence
        assert raw["log_p"].shape == (10, len(sequence), 21)
        assert np.isfinite(raw["log_p"]).all()
        assert np.all(raw["mask"] == 1) and np.all(raw["design_mask"] == 1)
        probabilities = np.exp(raw["log_p"].mean(axis=0))[:, :20]
        assert np.isfinite(probabilities).all() and (probabilities > 0).all()
        coordinates = np.load(ROOT / protein.features)["coordinates"]
        assert coordinates.shape == (len(sequence), 3)
        np.save(current / "conditional_probs.npy", probabilities)
        result = {
            "complete": True, "protein_id": protein.protein_id, "reference_sequence": sequence,
            "structure": protein.structure, "coordinates": protein.features,
            "probabilities": str((current / "conditional_probs.npy").relative_to(ROOT)),
            "author_output": str(raw_path.relative_to(ROOT)),
            "model": "v_48_020", "seed": 37, "samples": 10,
            "transformation": "exp(mean(log_p, axis=0))[:, :20], without renormalization",
            "command": command,
        }
        manifest_path.write_text(json.dumps(result, indent=2))
        results.append(result)
        print(json.dumps({"protein": protein.protein_id, "length": len(sequence), "complete": True}), flush=True)
    manifest = {
        "complete": True, "smoke": args.smoke, "rank": args.rank, "world_size": args.world_size,
        "proteins": len(results), "total_target_proteins": total,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "elapsed_seconds": time.monotonic() - started,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/ProteinMPNN"), "rev-parse", "HEAD"], text=True).strip(),
        "outputs": [str((output / item["protein_id"] / "manifest.json").relative_to(ROOT)) for item in results],
    }
    (output / f"manifest_rank{args.rank}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
