import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from Bio.Seq import MutableSeq
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/ThermoMPNN-D"))
from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2

OUTPUT = ROOT / "data/features/thermo/tsuboyama"
ROSETTA = ROOT / "data/extracted/thermompnn/FINAL_results/scans"


def direct_index():
    path = OUTPUT / "direct_index.parquet"
    if path.exists():
        return pd.read_parquet(path)
    proteins = pd.read_parquet(ROOT / "data/processed/tsuboyama/proteins.parquet").set_index("protein_id")
    structures = pd.read_parquet(ROOT / "data/features/structures/tsuboyama/availability.parquet")
    available = structures.loc[structures.available].set_index("protein_id").structure
    metadata = pd.concat([pd.read_parquet(ROOT / "data/features/training/tsuboyama" / kind / "metadata.parquet")
                          for kind in ["singles", "doubles"]], ignore_index=True)
    metadata = metadata.loc[metadata.protein_id.isin(available.index)].copy()
    metadata["structure"] = metadata.protein_id.map(available)
    metadata["background_sequence"] = metadata.protein_id.map(proteins.reference_sequence)
    metadata["native_ddg"] = -metadata.relative_point
    metadata["observed_variant"] = True
    metadata["training_point_eligible"] = metadata.point_in_range & metadata.wt_point_in_range
    counterfactuals = []
    for protein, group in metadata.groupby("protein_id"):
        reference = proteins.loc[protein, "reference_sequence"]
        known = set(group.mutation_key)
        needed = {(int(position), wild, mutant) for row in group.loc[group.n_mutations.eq(2)].itertuples()
                  for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)}
        for position, wild, mutant in sorted(needed):
            key = f"{wild}{position + 1}{mutant}"
            if key in known:
                continue
            sequence = MutableSeq(reference)
            assert sequence[position] == wild
            sequence[position] = mutant
            counterfactuals.append({"variant_id": f"thermo_cf_{protein}_{key}", "protein_id": protein, "mutation_key": key,
                                    "aa_seq": str(sequence), "positions": [position], "wt_residues": wild, "mut_residues": mutant,
                                    "n_mutations": 1, "observed_variant": False, "training_point_eligible": False,
                                    "native_ddg": np.nan, "structure": available[protein], "background_sequence": reference})
    metadata = pd.concat([metadata, pd.DataFrame(counterfactuals)], ignore_index=True)
    metadata = metadata.sort_values(["protein_id", "n_mutations", "mutation_key"]).reset_index(drop=True)
    metadata["matrix_row"] = np.arange(len(metadata))
    assert metadata.variant_id.is_unique and not metadata.duplicated(["protein_id", "mutation_key"]).any()
    metadata.to_parquet(path, index=False)
    result = {"complete": True, "rows": len(metadata), "observed_variants": int(metadata.observed_variant.sum()),
              "counterfactual_singles": len(counterfactuals), "proteins": metadata.protein_id.nunique(),
              "label_policy": "Original accepted variants; component singles without observations are prediction-only and have no labels.",
              "empty_genotype": "For the double head, both post-attention endpoints are zero. The single head uses equal wild-type and mutant outputs.",
              "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    (OUTPUT / "direct_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=int, choices=[25, 30], default=[30, 25])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 131])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = direct_index()
    if args.index_only:
        return
    proteins = pd.read_parquet(ROOT / "data/processed/tsuboyama/proteins.parquet").set_index("protein_id")
    singles = metadata.loc[metadata.n_mutations.eq(1) & metadata.training_point_eligible].copy()
    singles["WT_name"] = singles.protein_id
    singles["mut_type"] = singles.mutation_key
    singles["ddG_ML"] = -singles.native_ddg
    singles["rosetta_background"] = singles.protein_id.map(proteins.source_reference).str.removesuffix(".pdb").str.replace("|", ":", regex=False)
    singles["augmentation_structure"] = [
        str((ROSETTA / background / "pt_models" / f"A[{mutation}].pt").relative_to(ROOT))
        for background, mutation in zip(singles.rosetta_background, singles.mutation_key)]
    singles["augmentation_structure_available"] = singles.augmentation_structure.map(lambda value: (ROOT / value).is_file())
    singles[["variant_id", "protein_id", "augmentation_structure", "augmentation_structure_available"]].to_parquet(OUTPUT / "augmentation_structure_coverage.parquet", index=False)
    for threshold in args.thresholds:
        family = pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet").set_index("protein_id")
        for fold in args.folds:
            train_proteins = set(family.index[~family.outer_group.isin([fold, (fold + 1) % 5])])
            for seed in args.seeds:
                directory = OUTPUT / f"family{threshold}" / f"fold{fold}" / f"seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                manifest_path = directory / "augmentation_manifest.json"
                if manifest_path.exists():
                    assert json.loads(manifest_path.read_text())["complete"]
                    continue
                current = singles.loc[singles.protein_id.isin(train_proteins)].copy()
                assert len(current) and current.protein_id.isin(train_proteins).all()
                configuration = OmegaConf.create({"data": {"range": 3, "seed": seed}})
                augmented = MegaScaleDatasetv2._augment_double_mutants(SimpleNamespace(cfg=configuration), current, c=1)
                original_count = len(augmented)
                augmented = augmented.loc[augmented.augmentation_structure_available].copy()
                second_keys = augmented.mut_type.str.split(":").str[1]
                by_key = current.set_index(["protein_id", "mutation_key"])
                second = by_key.reindex(pd.MultiIndex.from_arrays([augmented.protein_id, second_keys]))
                assert second.variant_id.notna().all()
                augmented["source_a_id"] = augmented.variant_id
                augmented["source_b_id"] = second.variant_id.to_numpy()
                source_a_positions = [int(value[0]) for value in augmented.positions]
                source_b_positions = [int(value[0]) for value in second.positions]
                assert all(a != b for a, b in zip(source_a_positions, source_b_positions))
                augmented["positions"] = [[a, b] for a, b in zip(source_a_positions, source_b_positions)]
                augmented["background_sequence"] = augmented.aa_seq
                augmented["aa_seq"] = second.aa_seq.to_numpy()
                first_wild, first_mutant = augmented.wt_residues.to_numpy(), augmented.mut_residues.to_numpy()
                augmented["wt_residues"] = first_mutant + second.wt_residues.to_numpy()
                augmented["mut_residues"] = first_wild + second.mut_residues.to_numpy()
                augmented["structure"] = augmented.augmentation_structure
                augmented["native_ddg"] = -augmented.ddG_ML
                native = second.native_ddg.to_numpy() - current.set_index("variant_id").native_ddg.reindex(augmented.source_a_id).to_numpy()
                np.testing.assert_allclose(augmented.native_ddg, native, rtol=0, atol=1e-10)
                augmented["n_mutations"] = 2
                augmented["variant_id"] = [f"thermo_aug_f{threshold}_{fold}_s{seed}_{index:07d}" for index in range(len(augmented))]
                augmented["matrix_row"] = np.arange(len(augmented))
                assert set(augmented.source_a_id) | set(augmented.source_b_id) <= set(current.variant_id)
                fields = ["variant_id", "protein_id", "positions", "wt_residues", "mut_residues", "n_mutations", "aa_seq",
                          "background_sequence", "structure", "native_ddg", "source_a_id", "source_b_id", "matrix_row"]
                augmented[fields].to_parquet(directory / "augmentation_index.parquet", index=False)
                manifest = {"complete": True, "threshold": threshold, "fold": fold, "seed": seed,
                            "author_function": "MegaScaleDatasetv2._augment_double_mutants(c=1)", "range_exponent": 3,
                            "generated_transitions": original_count, "available_mutant_structure_transitions": len(augmented),
                            "unavailable_mutant_structures": original_count - len(augmented),
                            "source_training_proteins": sorted(train_proteins),
                            "target_definition": "Measured destabilizing free-energy difference from source single A to source single B.",
                            "label_permissions_verified": True, "phenotype_difference_verified": True,
                            "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(json.dumps({key: value for key, value in manifest.items() if key != "source_training_proteins"}), flush=True)


if __name__ == "__main__":
    main()
