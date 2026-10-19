import json
from pathlib import Path
import subprocess

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/flip_primary"
SPLITS = {
    "trpb": ["one_to_many", "two_to_many", "by_position"],
    "amylase": ["one_to_many"],
    "ired": ["two_to_many"],
    "hydro": ["three_to_many", "to_P06241", "to_P01053", "to_P0A9X9"],
}
LIMITS = {"one_to_many": 1, "two_to_many": 2, "three_to_many": 3}
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def references():
    records = []
    trpb_path = ROOT / "data/raw/flip_original_trpb/reference_files/tm9d8s_AAs.fasta"
    full = str(SeqIO.read(trpb_path, "fasta").seq)
    assert full.endswith("LEHHHHHH")
    records.append({"dataset": "trpb", "background": "Tm9D8", "reference_sequence": full[:-8],
                    "reference_source": str(trpb_path.relative_to(ROOT)),
                    "reference_processing": "Remove the terminal LEHHHHHH expression tag; the resulting 389-residue sequence is the official FLIP2 WT with target 1."})
    amy_path = ROOT / "external/mutation-predictability/data/AMY_BACSU/reference.fasta"
    records.append({"dataset": "amylase", "background": "AMY_BACSU",
                    "reference_sequence": str(SeqIO.read(amy_path, "fasta").seq),
                    "reference_source": str(amy_path.relative_to(ROOT)), "reference_processing": "Unmodified author reference FASTA."})
    ired_path = ROOT / "data/raw/flip_original_ired/srired_active_data.csv"
    ired = pd.read_csv(ired_path, low_memory=False)
    native = ired.loc[ired.Nham_aa.eq(0), "aa_seq"].unique()
    assert len(native) == 1 and native[0].endswith("*") and native[0].count("*") == 1
    records.append({"dataset": "ired", "background": "SrIRED", "reference_sequence": native[0][:-1],
                    "reference_source": str(ired_path.relative_to(ROOT)), "reference_processing": "Author Nham_aa=0 reference; remove only the terminal stop symbol."})
    hydro_root = ROOT / "data/extracted/flip_original_hydro/zenodo"
    native_path = hydro_root / "Fig1/Sparse_DTS_Cores_fitness_replicates.txt"
    hydro_raw = pd.read_csv(native_path, sep="\t", low_memory=False)
    synonymous_path = hydro_root / "Fig1/DiMSUM_Sparse_cores_DTS_synonymSequences.txt"
    control_dna = pd.read_csv(synonymous_path, sep="\t", header=None)[0]
    controls = [str(Seq(sequence).translate()) for sequence in control_dna]
    hydro = pd.read_csv(ROOT / "data/raw/flip2/hydro/three_to_many.csv.gz")
    for identifier in ["P06241", "P01053", "P0A9X9"]:
        official = pd.read_csv(ROOT / "data/raw/flip2/hydro" / f"to_{identifier}.csv.gz")
        lengths = official.loc[official["set"].eq("test"), "sequence"].str.len().unique()
        assert len(lengths) == 1
        length = int(lengths[0])
        candidates = [sequence for sequence in controls if len(sequence) == length]
        assert len(candidates) == 1
        sequence = candidates[0]
        reference_source = synonymous_path
        processing = "Translated author reference control sequence; background assignment follows the official held-out-background split."
        assert len(sequence) == length
        group = hydro.loc[hydro.sequence.str.len().eq(length)]
        array = np.array([list(value) for value in group.sequence])
        variable = np.flatnonzero(np.any(array != array[0], axis=0))
        assert len(variable) == 7
        fixed = np.ones(length, dtype=bool)
        fixed[variable] = False
        assert (array[:, fixed] == np.array(list(sequence))[fixed]).all()
        if identifier in ["P06241", "P01053"]:
            source = hydro_root / ("Fig3/FYN-SH3/FYN_fitness_replicates.txt" if identifier == "P06241" else "Fig3/CI-2A/CI2A_fitness_replicates.txt")
            core = pd.read_csv(source, usecols=["wt_core"]).wt_core.unique()
            assert len(core) == 1 and "".join(sequence[position] for position in variable) == core[0]
            processing += " The seven reference core residues exactly match the author's wt_core field."
        else:
            uniprot = str(SeqIO.read(ROOT / "work/sources/P0A9X9.fasta", "fasta").seq)
            assert sequence in uniprot
            processing += " The reference construct is an exact contiguous segment of UniProt P0A9X9."
        records.append({"dataset": "hydro", "background": identifier, "reference_sequence": sequence,
                        "reference_source": str(reference_source.relative_to(ROOT)), "reference_processing": processing})
    proteins = pd.DataFrame(records)
    proteins.insert(0, "protein_id", [f"flip_{i:04d}" for i in range(len(proteins))])
    proteins["length"] = proteins.reference_sequence.str.len()
    proteins["source_reference"] = proteins.background
    proteins["assay_id"] = proteins.dataset.map(lambda name: "FLIP2_v3_" + name)
    proteins["condition_id"] = proteins.assay_id
    proteins["origin"] = "natural_reference_construct"
    return proteins, ired, hydro_raw


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    proteins, ired_raw, hydro_raw = references()
    variants, observations, split_frames, counts = [], [], [], []
    variant_counter = 0
    for dataset, names in SPLITS.items():
        tables = []
        for name in names:
            path = ROOT / "data/raw/flip2" / dataset / f"{name}.csv.gz"
            table = pd.read_csv(path, low_memory=False)
            assert table.sequence.is_unique and table.target.notna().all()
            assert table["set"].isin(["train", "test"]).all() and table.validation.isin([True, False]).all()
            assert not (table["set"].eq("test") & table.validation).any()
            table["source_row"] = np.arange(len(table))
            table["official_split"] = name
            table["source_file"] = str(path.relative_to(ROOT))
            tables.append(table)
        all_rows = pd.concat(tables, ignore_index=True)
        variation = all_rows.groupby("sequence").target.agg(["min", "max"])
        assert np.allclose(variation["min"], variation["max"], rtol=0, atol=1e-12)
        unique = all_rows.drop_duplicates("sequence").reset_index(drop=True)
        refs = proteins.loc[proteins.dataset.eq(dataset)]
        length_mapping = refs.set_index("length").protein_id.to_dict()
        assert len(length_mapping) == len(refs)
        unique["protein_id"] = unique.sequence.str.len().map(length_mapping)
        assert unique.protein_id.notna().all()
        reference_map = refs.set_index("protein_id").reference_sequence.to_dict()
        current = []
        for row in unique.itertuples():
            sequence = row.sequence
            reference = reference_map[row.protein_id]
            assert len(reference) == len(sequence) and set(sequence) <= ALPHABET
            positions = [i for i, (wt, mt) in enumerate(zip(reference, sequence)) if wt != mt]
            wild = "".join(reference[position] for position in positions)
            mutant = "".join(sequence[position] for position in positions)
            mutation = ":".join(f"{wt}{position+1}{mt}" for position, wt, mt in zip(positions, wild, mutant))
            identifier = f"flip_var_{variant_counter:07d}"
            variant_counter += 1
            current.append({"variant_id": identifier, "protein_id": row.protein_id,
                            "aa_seq": sequence, "mutation_key": mutation, "n_mutations": len(positions),
                            "positions": positions, "wt_residues": wild, "mut_residues": mutant,
                            "point": float(row.target), "lower": float(row.target), "upper": float(row.target),
                            "all_point": True, "dataset": dataset})
        current = pd.DataFrame(current)
        variants.append(current)
        if dataset in ["ired", "hydro"]:
            raw = ired_raw.copy() if dataset == "ired" else hydro_raw.copy()
            raw["source_row"] = np.arange(len(raw))
            raw["canonical_sequence"] = raw.aa_seq.str.removesuffix("*") if dataset == "ired" else raw.aa_seq
            relevant = raw.loc[raw.canonical_sequence.isin(current.aa_seq)].copy()
            observed = relevant.groupby("canonical_sequence").fitness.mean().reindex(current.aa_seq)
            assert observed.notna().all()
            np.testing.assert_allclose(observed.to_numpy(), current.point, rtol=0, atol=1e-12)
            relevant = relevant.merge(current[["variant_id", "protein_id", "aa_seq"]].rename(columns={"aa_seq": "canonical_sequence"}),
                                      on="canonical_sequence", validate="many_to_one")
            relevant.to_parquet(OUTPUT / f"original_observations_{dataset}.parquet", index=False)
        mapping = current.set_index("aa_seq")
        for table in tables:
            frame = table.copy()
            frame["variant_id"] = frame.sequence.map(mapping.variant_id)
            frame["protein_id"] = frame.sequence.map(mapping.protein_id)
            frame["n_mutations"] = frame.sequence.map(mapping.n_mutations)
            frame["effective_set"] = np.where(frame.validation, "validation", frame["set"])
            frame["strict_set"] = frame.effective_set
            name = frame.official_split.iloc[0]
            if name in LIMITS:
                bound = LIMITS[name]
                invalid = frame.n_mutations.gt(bound) & frame.effective_set.isin(["train", "validation"])
                frame.loc[invalid, "strict_set"] = "excluded_distance"
            elif name == "by_position":
                train_ids = set(frame.loc[frame.effective_set.eq("train"), "variant_id"])
                train_positions = set(position for row in current.loc[current.variant_id.isin(train_ids)].itertuples() for position in row.positions)
                new_positions = current.set_index("variant_id").positions.map(lambda value: bool(set(value) - train_positions))
                invalid = frame.effective_set.eq("validation") & frame.variant_id.map(new_positions)
                frame.loc[invalid, "strict_set"] = "excluded_position"
            else:
                background = name.removeprefix("to_")
                target_protein = refs.loc[refs.background.eq(background), "protein_id"].item()
                assert frame.loc[frame["set"].eq("test"), "protein_id"].eq(target_protein).all()
                assert frame.loc[frame.effective_set.eq("train"), "protein_id"].ne(target_protein).all()
                invalid = frame.effective_set.eq("validation") & frame.protein_id.eq(target_protein)
                frame.loc[invalid, "strict_set"] = "excluded_background"
            assert frame.loc[frame.effective_set.eq("test"), "strict_set"].eq("test").all()
            assert frame.loc[frame["set"].eq("test"), "variant_id"].tolist() == frame.loc[frame.effective_set.eq("test"), "variant_id"].tolist()
            frame.drop(columns=["sequence", "target"]).to_parquet(OUTPUT / f"split_{dataset}_{name}.parquet", index=False)
            split_frames.append(frame.drop(columns=["sequence", "target"]))
            for protocol, column in [("official", "effective_set"), ("strict", "strict_set")]:
                for (partition, protein_id), subset in frame.groupby([column, "protein_id"], sort=True):
                    counts.append({"dataset": dataset, "split": name, "protocol": protocol, "partition": partition,
                                   "protein_id": protein_id, "rows": len(subset), "max_mutations": int(subset.n_mutations.max()),
                                   "orders": json.dumps(subset.n_mutations.value_counts().sort_index().to_dict())})
            observations.append(frame[["variant_id", "protein_id", "target", "source_file", "source_row", "official_split"]])
    variants = pd.concat(variants, ignore_index=True)
    assert variants.variant_id.is_unique
    assert not variants.duplicated(["protein_id", "aa_seq"]).any()
    variants.to_parquet(OUTPUT / "variants.parquet", index=False)
    proteins.to_parquet(OUTPUT / "proteins.parquet", index=False)
    pd.concat(observations, ignore_index=True).to_parquet(OUTPUT / "official_observations.parquet", index=False)
    pd.concat(split_frames, ignore_index=True).to_parquet(OUTPUT / "splits.parquet", index=False)
    pd.DataFrame(counts).to_csv(OUTPUT / "split_counts.csv", index=False)
    SeqIO.write([SeqRecord(Seq(row.reference_sequence), id=row.protein_id, description=row.background) for row in proteins.itertuples()],
                OUTPUT / "references.fasta", "fasta")
    manifest = {
        "complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source": "FLIP2 v3 fixed official files; original author references",
        "proteins": len(proteins), "variants": len(variants),
        "counts": variants.groupby("dataset").size().to_dict(), "official_splits": sum(len(names) for names in SPLITS.values()),
        "official_test_preserved": True,
        "strict_validation": "Labels outside the stated training distance, training positions, or source backgrounds are excluded. Official test membership remains exact.",
        "label_scale": "Original official FLIP2 target; IRED and Hydro values additionally matched to original experimental fitness records.",
        "wild_types": variants.loc[variants.n_mutations.eq(0), ["protein_id", "variant_id", "point"]].to_dict("records"),
        "original_measurements_vs_split_copies": "Repeated appearances across official split files are provenance records, not independent measurements.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
