import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/processed/flip_supplementary"
manifest = json.loads((source / "manifest.json").read_text())
assert manifest["complete"]
records = pd.read_parquet(source / "records.parquet")
variants = pd.read_parquet(source / "variants.parquet").set_index("variant_id")
references = pd.read_parquet(source / "references.parquet").set_index("dataset")
assert records.record_id.is_unique and variants.index.is_unique
assert set(records.variant_id) <= set(variants.index)
counts = []
for (dataset, split), group in records.groupby(["dataset", "official_split"], sort=True):
    official = pd.read_csv(ROOT / "data/raw/flip2" / dataset / (split + ".csv.gz"))
    group = group.sort_values("source_row")
    np.testing.assert_array_equal(group.source_row, np.arange(len(official)))
    np.testing.assert_array_equal(group.sequence, official.sequence)
    np.testing.assert_array_equal(group.target, official.target)
    expected = np.where(official.validation, "validation", official["set"])
    np.testing.assert_array_equal(group.effective_set, expected)
    np.testing.assert_array_equal(group.loc[group.effective_set.eq("test"), "strict_set"], "test")
    actual_inputs = variants.loc[group.variant_id]
    if dataset == "pdz3":
        np.testing.assert_array_equal(actual_inputs.sequence_a + ":" + actual_inputs.sequence_b, official.sequence)
    else:
        np.testing.assert_array_equal(actual_inputs.sequence_a, official.sequence)
        assert actual_inputs.sequence_b.eq("").all()
    for column in ["effective_set", "strict_set"]:
        eligible = group.loc[group[column].isin(["train", "validation", "test"])]
        assert eligible.groupby("variant_id")[column].nunique().eq(1).all()
    counts.append({"dataset": dataset, "split": split, "records": len(group),
                   "test_records": int(group.effective_set.eq("test").sum())})

original_nucb = pd.read_csv(ROOT / "external/nuclease_design/data/landscape.csv").set_index("sequence")
nucb = records.loc[records.dataset.eq("nucb")]
ordered = ["non-functional", "activity > 0", "activity > WT", "activity > A73R"]
expected_class = pd.Categorical(original_nucb.loc[nucb.sequence, "activity_level"], categories=ordered, ordered=True).codes
np.testing.assert_array_equal(nucb.target, expected_class)
np.testing.assert_array_equal(nucb.n_mutations, original_nucb.loc[nucb.sequence, "num_mutations"])
strict = nucb.loc[nucb.official_split.eq("two_to_many")]
eligible = strict.effective_set.eq("test") | strict.n_mutations.le(2)
np.testing.assert_array_equal(strict.strict_set.ne("excluded_distance"), eligible)

workbook = pd.read_excel(ROOT / "data/raw/flip_original_rhomax/42003_2021_1878_MOESM3_ESM.xlsx",
                         sheet_name="配列&波長", header=1)
rhomax = records.loc[records.dataset.eq("rhomax")].sort_values("source_row")
np.testing.assert_array_equal(rhomax.sequence, workbook.Sequence)
np.testing.assert_array_equal(rhomax.target, workbook.lmax)
np.testing.assert_array_equal(rhomax.background, workbook.Wildtype)
assert rhomax.groupby("background").effective_set.nunique().eq(1).all()
assert len(rhomax) == 884 and rhomax.variant_id.nunique() == 878

raw = pd.read_csv(ROOT / "data/extracted/flip_original_pdz/pdz3b1b2_norm_for_mochi.tsv",
                  sep="\t", usecols=["aa_seq", "fitness", "Nham_aa"], low_memory=False)
raw["expressed_sequence"] = raw.aa_seq.str.split("*", n=1, regex=False).str[0]
native = raw.loc[raw.Nham_aa.eq(0)]
assert len(native) == 1
wt = references.loc["pdz3", "sequence_a"] + references.loc["pdz3", "sequence_b"]
assert wt == native.aa_seq.iloc[0]
original = raw.groupby("expressed_sequence", sort=False).first().fitness
pdz = records.loc[records.dataset.eq("pdz3")]
source_rows = pdz.original_source_row.to_numpy(dtype=int)
np.testing.assert_array_equal(raw.iloc[source_rows].expressed_sequence, pdz.sequence.str.replace(":", "", regex=False))
np.testing.assert_allclose(raw.iloc[source_rows].fitness, pdz.target, atol=1e-7, rtol=0)
quartets = pd.read_parquet(source / "pdz3_quartets.parquet")
incomplete = pd.read_parquet(source / "pdz3_incomplete_quartets.parquet")
assert len(quartets) == manifest["pdz3_complete_quartets"]
assert len(incomplete) == manifest["pdz3_incomplete_quartets"]
assert not set(quartets.variant_id) & set(incomplete.variant_id)
double_ids = set(variants.loc[variants.dataset.eq("pdz3") & variants.n_mutations.eq(2) & variants.official_input].index)
assert double_ids == set(quartets.variant_id) | set(incomplete.variant_id)
for row in quartets.itertuples():
    identifiers = [row.wt_id, row.a_id, row.b_id, row.variant_id]
    components = variants.loc[identifiers]
    sequences = (components.sequence_a + components.sequence_b).tolist()
    assert sequences[0] == wt
    changed = [{position for position, (a, b) in enumerate(zip(wt, sequence)) if a != b} for sequence in sequences]
    assert [len(value) for value in changed] == [0, 1, 1, 2]
    assert not changed[1] & changed[2] and changed[1] | changed[2] == changed[3]
    for component in [1, 2]:
        position = next(iter(changed[component]))
        assert sequences[component][position] == sequences[3][position]
    values = original.loc[sequences].to_numpy()
    np.testing.assert_allclose(values, [row.phenotype_wt, row.phenotype_a, row.phenotype_b, row.phenotype_ab], atol=0, rtol=0)
    np.testing.assert_allclose(row.epsilon, values @ np.array([1., -1., -1., 1.]), atol=1e-14, rtol=0)
for row in incomplete.itertuples():
    assert len(row.missing_measured_components) > 0
    assert all(sequence not in original.index for sequence in row.missing_measured_components)
test_coverage = []
for split, group in pdz.loc[pdz.effective_set.eq("test")].groupby("official_split"):
    assert group.variant_id.nunique() == 579
    test_coverage.append({"split": split, "test_records": len(group),
                          "complete_quartets": int(group.variant_id.isin(quartets.variant_id).sum()),
                          "incomplete_quartets": int(group.variant_id.isin(incomplete.variant_id).sum())})
output = ROOT / "runs/flip_supplementary_data_verification"
output.mkdir(parents=True, exist_ok=True)
result = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "records_verified": len(records), "inputs_verified": len(variants), "official_splits": counts,
          "original_nucb_class_labels_verified": len(nucb), "rhomax_measurement_rows_verified": len(rhomax),
          "rhomax_cross_partition_sequence_or_background_leakage": False,
          "pdz3_original_record_rows_verified": len(pdz), "pdz3_complete_quartets_verified": len(quartets),
          "pdz3_incomplete_quartets_verified": len(incomplete), "pdz3_test_coverage": test_coverage,
          "official_sequences_labels_and_partitions_unchanged": True, "strict_tests_unchanged": True}
(output / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
