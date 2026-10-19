import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/flip_supplementary"
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "external/nuclease_design"))
from nuclease_design.constants import FULL_REFERENCE_SEQ

ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
SPLITS = {"nucb": ["random", "two_to_many"], "pdz3": ["rand_split", "single_to_double"], "rhomax": ["by_wild_type"]}
frames = []
for dataset, splits in SPLITS.items():
    for split in splits:
        path = ROOT / "data/raw/flip2" / dataset / (split + ".csv.gz")
        table = pd.read_csv(path)
        assert table.target.notna().all() and np.isfinite(table.target).all()
        assert table["set"].isin(["train", "test"]).all() and table.validation.isin([True, False]).all()
        assert not (table["set"].eq("test") & table.validation).any()
        table["source_row"] = np.arange(len(table))
        table["dataset"], table["official_split"] = dataset, split
        table["source_file"] = str(path.relative_to(ROOT))
        table["effective_set"] = np.where(table.validation, "validation", table["set"])
        table["strict_set"] = table.effective_set
        table["input_key"] = table.sequence
        frames.append(table)
records = pd.concat(frames, ignore_index=True)
inputs, references, quartets, incomplete_quartets = [], [], [], []

nucb = pd.read_csv(ROOT / "external/nuclease_design/data/landscape.csv")
assert nucb.sequence.is_unique and len(nucb) == 55760
mapping = {"non-functional": 0, "activity > 0": 1, "activity > WT": 2, "activity > A73R": 3}
nucb["class_id"] = nucb.activity_level.map(mapping)
assert nucb.class_id.notna().all()
nucb["verified_mutations"] = nucb.sequence.map(lambda sequence: sum(a != b for a, b in zip(sequence, FULL_REFERENCE_SEQ)))
assert nucb.sequence.str.len().eq(len(FULL_REFERENCE_SEQ)).all()
np.testing.assert_array_equal(nucb.num_mutations, nucb.verified_mutations)
lookup = nucb.set_index("sequence")
selected = records.dataset.eq("nucb")
np.testing.assert_array_equal(records.loc[selected, "target"], lookup.loc[records.loc[selected, "sequence"], "class_id"])
records.loc[selected, "n_mutations"] = lookup.loc[records.loc[selected, "sequence"], "verified_mutations"].to_numpy()
records.loc[selected, "background"] = "NucB"
excluded = selected & records.official_split.eq("two_to_many") & records.effective_set.ne("test") & records.n_mutations.gt(2)
records.loc[excluded, "strict_set"] = "excluded_distance"
for sequence in sorted(records.loc[selected, "sequence"].unique()):
    inputs.append({"dataset": "nucb", "input_key": sequence, "sequence_a": sequence, "sequence_b": "",
                   "background": "NucB", "n_mutations": int(lookup.loc[sequence, "verified_mutations"]), "substitutions_only": True})
references.append({"dataset": "nucb", "background": "NucB", "sequence_a": FULL_REFERENCE_SEQ, "sequence_b": "",
                   "source": "external/nuclease_design/nuclease_design/constants.py", "official_reference_label_available": False})
nucb.loc[nucb.sequence.isin(records.loc[selected, "sequence"])].to_parquet(OUTPUT / "original_nucb.parquet", index=False)

rhomax_path = ROOT / "data/raw/flip_original_rhomax/42003_2021_1878_MOESM3_ESM.xlsx"
rhomax = pd.read_excel(rhomax_path, sheet_name="配列&波長", header=1)
rh = records.loc[records.dataset.eq("rhomax")].copy()
np.testing.assert_array_equal(rhomax.Sequence, rh.sequence)
np.testing.assert_array_equal(rhomax.lmax, rh.target)
records.loc[rh.index, "background"] = rhomax.Wildtype.to_numpy()
rhomax["partition"] = rh.effective_set.to_numpy()
assert rhomax.groupby("Wildtype").partition.nunique().eq(1).all()
assert rhomax.groupby("Sequence").partition.nunique().eq(1).all()
rhomax["source_excel_row"] = np.arange(len(rhomax)) + 3
rhomax[["Name", "Wildtype", "Sequence", "lmax", "Method", "partition", "source_excel_row"]].to_parquet(
    OUTPUT / "original_rhomax.parquet", index=False)
for sequence, group in rhomax.groupby("Sequence", sort=True):
    assert group.Wildtype.nunique() == 1
    inputs.append({"dataset": "rhomax", "input_key": sequence, "sequence_a": sequence, "sequence_b": "",
                   "background": group.Wildtype.iloc[0], "n_mutations": None, "substitutions_only": None})
duplicate = rhomax.groupby("Sequence").agg(observations=("Name", "size"), target_count=("lmax", "nunique"),
                                         minimum=("lmax", "min"), maximum=("lmax", "max"), partitions=("partition", "nunique"))
duplicate.loc[duplicate.observations.gt(1)].to_csv(OUTPUT / "rhomax_repeated_sequences.csv")

pdz_path = ROOT / "data/extracted/flip_original_pdz/pdz3b1b2_norm_for_mochi.tsv"
if not pdz_path.exists():
    pdz_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ROOT / "data/raw/flip_original_pdz/fuzzy_specificity_data.tar.gz", mode="r|gz") as archive:
        found = False
        for member in archive:
            if member.name == "./fuzzy_specificity_data/pdz3b1b2_norm_for_mochi.tsv":
                assert member.isfile() and member.size == 110624600
                partial = pdz_path.with_suffix(".tsv.part")
                with archive.extractfile(member) as source, partial.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                assert partial.stat().st_size == member.size
                os.replace(partial, pdz_path)
                found = True
                break
        assert found
assert pdz_path.stat().st_size == 110624600
pdz = pd.read_csv(pdz_path, sep="\t", low_memory=False)
pdz["source_row"] = np.arange(len(pdz))
pdz["expressed_sequence"] = pdz.aa_seq.str.partition("*")[0]
native = pdz.loc[pdz.Nham_aa.eq(0), "aa_seq"].unique()
assert len(native) == 1 and len(native[0]) == 108
receptor, peptide = native[0][:100], native[0][100:]
assert peptide == "KNYKQTSV"
pdz_first = pdz.drop_duplicates("expressed_sequence", keep="first").set_index("expressed_sequence")
truth = pdz_first.fitness
selected = records.dataset.eq("pdz3")
keys = records.loc[selected, "sequence"].str.replace(":", "", regex=False)
assert keys.isin(truth.index).all()
pdz_label_difference = float(np.max(np.abs(truth.loc[keys].to_numpy() - records.loc[selected, "target"].to_numpy())))
np.testing.assert_allclose(truth.loc[keys], records.loc[selected, "target"], rtol=0, atol=1e-7)
records.loc[selected, "original_source_row"] = pdz_first.loc[keys, "source_row"].to_numpy()
records.loc[selected, "original_fitness"] = truth.loc[keys].to_numpy()
records.loc[selected, "background"] = "PDZ3_CRIPT"
keyset = set(records.loc[selected, "input_key"])
reference_key = receptor + ":" + peptide
for key in sorted(keyset):
    chains = key.split(":")
    assert len(chains) == 2 and len(chains[0]) == 100
    if len(chains[1]) != 8:
        continue
    differences = [(chain, position) for chain in [0, 1] for position, (wt, mt) in
                   enumerate(zip((receptor, peptide)[chain], chains[chain])) if wt != mt]
    if len(differences) != 2:
        continue
    components = []
    for active in differences:
        values = [list(receptor), list(peptide)]
        chain, position = active
        values[chain][position] = chains[chain][position]
        components.append("".join(values[0]) + ":" + "".join(values[1]))
    requested = [key, *components, reference_key]
    flat = [value.replace(":", "") for value in requested]
    keyset.update(components)
    keyset.add(reference_key)
    missing = [value for value in flat if value not in truth.index]
    if missing:
        incomplete_quartets.append({"input_key": key, "missing_measured_components": missing})
        continue
    observed = truth.loc[flat].to_numpy()
    quartets.append({"input_key": key, "a_key": components[0], "b_key": components[1], "reference_key": reference_key,
                     "phenotype_ab": observed[0], "phenotype_a": observed[1], "phenotype_b": observed[2],
                     "phenotype_wt": observed[3], "epsilon": observed[0] - observed[1] - observed[2] + observed[3]})
for key in sorted(keyset):
    first, second = key.split(":")
    ordinary = len(second) == len(peptide)
    count = sum(a != b for a, b in zip(first + second, receptor + peptide)) if ordinary else None
    inputs.append({"dataset": "pdz3", "input_key": key, "sequence_a": first, "sequence_b": second,
                   "background": "PDZ3_CRIPT", "n_mutations": count, "substitutions_only": ordinary})
references.append({"dataset": "pdz3", "background": "PDZ3_CRIPT", "sequence_a": receptor, "sequence_b": peptide,
                   "source": str(pdz_path.relative_to(ROOT)), "official_reference_label_available": reference_key in set(records.loc[selected, "input_key"])})
pdz.loc[pdz.expressed_sequence.isin([key.replace(":", "") for key in keyset])].to_parquet(OUTPUT / "original_pdz3.parquet", index=False)

variants = pd.DataFrame(inputs).sort_values(["dataset", "input_key"]).reset_index(drop=True)
variants["variant_id"] = [f"supp_var_{index:06d}" for index in range(len(variants))]
variants["input_components"] = np.where(variants.dataset.eq("pdz3"), 2, 1)
variants["official_input"] = pd.MultiIndex.from_frame(variants[["dataset", "input_key"]]).isin(
    pd.MultiIndex.from_frame(records[["dataset", "input_key"]]))
assert not variants.duplicated(["dataset", "input_key"]).any()
assert variants.sequence_a.map(lambda value: bool(value) and set(value) <= ALPHABET).all()
assert variants.sequence_b.map(lambda value: set(value) <= ALPHABET).all()
records = records.drop(columns=["n_mutations"]).merge(
    variants[["dataset", "input_key", "variant_id", "n_mutations"]], on=["dataset", "input_key"], validate="many_to_one")
records["record_id"] = records.dataset + "_" + records.official_split + "_" + records.source_row.astype(str)
assert records.record_id.is_unique
records["target_kind"] = records.dataset.map({"nucb": "ordered_activity_class", "pdz3": "binding_fitness", "rhomax": "absorption_peak_nm"})
counts = []
for (dataset, split), group in records.groupby(["dataset", "official_split"], sort=True):
    assert group.loc[group.effective_set.eq("test"), "strict_set"].eq("test").all()
    for column in ["effective_set", "strict_set"]:
        memberships = group.loc[group[column].isin(["train", "validation", "test"])].groupby("variant_id")[column].nunique()
        assert memberships.eq(1).all()
        for partition, subset in group.groupby(column):
            counts.append({"dataset": dataset, "official_split": split, "protocol": column, "partition": partition,
                           "records": len(subset), "inputs": subset.variant_id.nunique()})
pdz_ids = variants.loc[variants.dataset.eq("pdz3")].set_index("input_key").variant_id
q = pd.DataFrame(quartets)
for column, target in [("input_key", "variant_id"), ("a_key", "a_id"), ("b_key", "b_id"), ("reference_key", "wt_id")]:
    q[target] = q[column].map(pdz_ids)
assert q[["variant_id", "a_id", "b_id", "wt_id"]].notna().all().all()
q.to_parquet(OUTPUT / "pdz3_quartets.parquet", index=False)
incomplete = pd.DataFrame(incomplete_quartets)
incomplete["variant_id"] = incomplete.input_key.map(pdz_ids)
assert incomplete.variant_id.notna().all()
incomplete.to_parquet(OUTPUT / "pdz3_incomplete_quartets.parquet", index=False)
variants.to_parquet(OUTPUT / "variants.parquet", index=False)
records.to_parquet(OUTPUT / "records.parquet", index=False)
pd.DataFrame(references).to_parquet(OUTPUT / "references.parquet", index=False)
pd.DataFrame(counts).to_csv(OUTPUT / "split_counts.csv", index=False)
manifest = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "inputs": len(variants), "official_inputs": int(variants.official_input.sum()), "records_across_splits": len(records),
            "official_splits": sum(map(len, SPLITS.values())), "counts": records.groupby("dataset").size().to_dict(),
            "official_tests_preserved": True, "original_labels_verified": True,
            "nucb_classes": mapping, "nucb_strict_distance_limit": 2,
            "rhomax_backgrounds": int(rhomax.Wildtype.nunique()), "rhomax_repeated_sequence_groups": int(duplicate.observations.gt(1).sum()),
            "rhomax_conflicting_label_groups": int(duplicate.target_count.gt(1).sum()), "rhomax_cross_partition_duplicate_groups": 0,
            "rhomax_evaluation_unit": "All 884 original measurement rows, including repeated sequence observations.",
            "pdz3_components": ["PDZ3 receptor", "CRIPT peptide"], "pdz3_stop_truncations": int((variants.dataset.eq("pdz3") & ~variants.substitutions_only.fillna(True)).sum()),
            "pdz3_complete_quartets": len(q), "pdz3_incomplete_quartets": len(incomplete),
            "pdz3_quartet_eligibility": "Complete measured WT, A, B and AB records in the original assay; official phenotype tests remain intact.",
            "counterfactual_labels_are_training_inputs": False,
            "pdz3_maximum_original_label_difference": pdz_label_difference,
            "pdz3_original_record_selection": "The first retained expressed-sequence record matches all official labels within 1e-7; original block observations remain intact.",
            "pdz3_test_selection": "Official specificity-changing diagnostic subset; interprotein substitutions retain separate components.",
            "supplementary_endpoints": "NucB classification, PDZ3 binding fitness and quartet residuals, Rhomax wavelength regression."}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
