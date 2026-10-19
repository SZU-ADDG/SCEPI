import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data/extracted/kermut/data"
OUTPUT = ROOT / "runs/e1_kermut_workspace"
ASSAY_ALIASES = {
    "F7YBW7_MESOW_Ding_2023": "F7YBW8_MESOW_Ding_2023",
    "PSAE_SYNP2_Tsuboyama_2023_1PSE": "PSAE_PICP2_Tsuboyama_2023_1PSE",
    "Q6WV13_9MAXI_Somermeyer_2022": "Q6WV12_9MAXI_Somermeyer_2022",
}


def main():
    data = OUTPUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name in ["conditional_probs", "structures", "embeddings", "zero_shot_fitness_predictions"]:
        destination = data / name
        if destination.exists():
            assert destination.resolve() == (RESOURCES / name).resolve()
        else:
            destination.symlink_to(RESOURCES / name, target_is_directory=True)
    reference_path = ROOT / "external/ProteinGym/reference_files/DMS_substitutions.csv"
    reference = pd.read_csv(reference_path)
    assert len(reference) == 217 and reference.DMS_id.is_unique
    reference["current_DMS_id"] = reference.DMS_id
    reference["DMS_id"] = reference.DMS_id.replace({value: key for key, value in ASSAY_ALIASES.items()})
    reference["DMS_filename"] = reference.DMS_id + ".csv"
    reference.to_csv(data / "DMS_substitutions.csv", index=False)
    (OUTPUT / "assay_aliases.json").write_text(json.dumps(ASSAY_ALIASES, indent=2))
    reference_sequences = reference.set_index("DMS_id").target_seq.to_dict()
    rows = []
    for kind in ["singles", "multiples"]:
        directory = data / f"substitutions_{kind}"
        directory.mkdir(parents=True, exist_ok=True)
        for path in sorted((RESOURCES / "embeddings" / f"substitutions_{kind}/ESM2").glob("*.h5")):
            assay = path.stem
            current_assay = ASSAY_ALIASES.get(assay, assay)
            original_path = ROOT / "data/extracted/proteingym" / f"cv_folds_{kind}_substitutions" / f"{current_assay}.csv"
            original = pd.read_csv(original_path)
            assert original.mutant.is_unique
            sequence = reference_sequences[assay]
            for record in original.itertuples():
                expected = list(sequence)
                for mutation in record.mutant.split(":"):
                    position = int(mutation[1:-1]) - 1
                    assert sequence[position] == mutation[0], (assay, mutation)
                    expected[position] = mutation[-1]
                assert "".join(expected) == record.mutated_sequence, (assay, record.mutant)
            with h5py.File(path, "r") as handle:
                names = handle["mutants"].asstr()[:].tolist()
                shape = handle["embeddings"].shape
                assert len(names) == len(set(names)) and shape == (len(names), 1280)
            assert set(original.mutant) == set(names), (assay, kind, len(original), len(names))
            order_changed = names != original.mutant.tolist()
            reordered = original.set_index("mutant").loc[names].reset_index()
            assert reordered.mutant.tolist() == names
            assert len(reordered) == len(original)
            np.testing.assert_allclose(reordered.set_index("mutant").DMS_score.reindex(original.mutant), original.DMS_score, rtol=0, atol=0)
            fold_changes = {}
            if kind == "singles":
                for split in ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]:
                    published_path = ROOT / "data/extracted/kermut/results/predictions" / assay / f"kermut_{split}.csv"
                    published = pd.read_csv(published_path).set_index("mutant").reindex(names)
                    assert len(published) == len(names) and published["fold"].notna().all()
                    fold_changes[split] = int((reordered[split].to_numpy() != published["fold"].to_numpy()).sum())
                    reordered[split] = published["fold"].to_numpy()
                    labels = torch.tensor(reordered.DMS_score.to_numpy(), dtype=torch.float32)
                    for fold in sorted(reordered[split].unique()):
                        testing = reordered[split].eq(fold).to_numpy()
                        training_values = labels[~testing]
                        expected = ((labels[testing] - training_values.mean()) / training_values.std()).numpy()
                        np.testing.assert_allclose(expected, published.y.to_numpy()[testing], rtol=1e-5, atol=2e-5,
                                                   err_msg=f"Published target mismatch: {assay} {split} {fold}")
            reordered.to_csv(directory / f"{assay}.csv", index=False)
            source_row = pd.Series(np.arange(len(original)), index=original.mutant).reindex(names).to_numpy()
            pd.DataFrame({"mutant": names, "source_row": source_row}).to_parquet(directory / f"{assay}_source_rows.parquet", index=False)
            rows.append({"assay": assay, "current_assay": current_assay, "kind": kind, "rows": len(original), "input_order_changed": order_changed,
                         "original": str(original_path.relative_to(ROOT)), "embedding": str(path.relative_to(ROOT)),
                         "current_vs_published_fold_changes": json.dumps(fold_changes)})
    audit = pd.DataFrame(rows)
    assert audit.loc[audit.kind.eq("singles"), "assay"].nunique() == 217
    assert set(audit.loc[audit.kind.eq("singles"), "assay"]) == set(reference.DMS_id)
    audit.to_csv(OUTPUT / "input_audit.csv", index=False)
    result = {
        "complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/kermut"), "rev-parse", "HEAD"], text=True).strip(),
        "reference": str(reference_path.relative_to(ROOT)), "counts": audit.groupby("kind").agg(assays=("assay", "size"), variants=("rows", "sum")).reset_index().to_dict("records"),
        "alignment": "Each assay is ordered by its author's HDF5 mutant identifiers. Sequence, target and CV memberships remain attached to the original mutant. The unmodified author loader therefore uses an identity embedding permutation.",
        "reordered_inputs": int(audit.input_order_changed.sum()),
        "published_single_benchmark_folds_used": True, "published_standardized_targets_verified": True,
        "official_multiple_benchmark_folds_preserved": True,
        "reference_and_mutated_sequences_verified": True, "assay_aliases": ASSAY_ALIASES,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
