import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/extracted/proteingym/substitutions_raw_DMS"
CURATED = ROOT / "data/extracted/proteingym/DMS_ProteinGym_substitutions"
OUTPUT = ROOT / "data/processed/functional"
ASSAYS = [
    ("D7PM05_CLYGR_Somermeyer_2022", "replicates_mean_brightness", "linear fluorescence brightness"),
    ("GFP_AEQVI_Sarkisyan_2016", "mean_medianBrightness_per_aaseq", "log10 fluorescence"),
    ("HIS7_YEAST_Pokusaeva_2019", "selection", "selection fitness"),
    ("Q6WV12_9MAXI_Somermeyer_2022", "replicates_mean_brightness", "linear fluorescence brightness"),
    ("Q8WTC7_9CNID_Somermeyer_2022", "replicates_mean_brightness", "linear fluorescence brightness"),
    ("RASK_HUMAN_Weng_2022_abundance", "fitness", "abundance fitness"),
]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def canonical(sequence, mutation):
    if mutation == "WT":
        return "WT", sequence, []
    parts = []
    result = list(sequence)
    for token in mutation.split(":"):
        match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])", token)
        assert match is not None, mutation
        wt, position, mt = match.groups()
        index = int(position) - 1
        assert 0 <= index < len(sequence) and sequence[index] == wt, mutation
        assert wt != mt, mutation
        parts.append((index, token))
        result[index] = mt
    parts.sort()
    positions = [position for position, _ in parts]
    assert len(positions) == len(set(positions)), mutation
    return ":".join(token for _, token in parts), "".join(result), positions


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_table = pd.read_csv(ROOT / "external/ProteinGym/reference_files/DMS_substitutions.csv").set_index("DMS_id")
    observations, proteins, catalogs, counts, curated_checks = [], [], [], [], []
    for assay_index, (assay, phenotype, units) in enumerate(ASSAYS):
        path = RAW / (assay + ".csv")
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        curated = pd.read_csv(CURATED / (assay + ".csv"), low_memory=False)
        assert not curated.mutant.duplicated().any()
        sequence = reference_table.loc[assay, "target_seq"]
        assert set(sequence) <= AA
        mutations = {}
        for row in curated.itertuples():
            key, changed, positions = canonical(sequence, row.mutant)
            assert changed == row.mutated_sequence, (assay, row.mutant)
            mutations[row.mutant] = (key, changed, positions)
        if assay.startswith("GFP_"):
            wild = raw.numAAMutations.eq("0") & raw.mutant.eq("")
        else:
            wild = raw.mutant.str.lower().eq("wt")
        admitted = raw.mutant.isin(mutations) | wild
        counts.append({"assay_id": assay, "raw_rows": len(raw), "excluded_outside_curated_substitution_set": int((~admitted).sum()), "accepted_rows": int(admitted.sum()), "real_wild_type_rows": int(wild.sum())})
        current = raw.loc[admitted].copy()
        current["source_row"] = current.index
        current["canonical_input"] = current.mutant.where(~wild.loc[current.index], "WT")
        mutation_map = {name: canonical(sequence, name) for name in current.canonical_input.unique()}
        current["mutation_key"] = current.canonical_input.map({name: value[0] for name, value in mutation_map.items()})
        current["aa_seq"] = current.canonical_input.map({name: value[1] for name, value in mutation_map.items()})
        current["positions"] = current.canonical_input.map({name: value[2] for name, value in mutation_map.items()})
        current["n_mutations"] = current.positions.str.len()
        for sequence_column in ["fullseq", "mutated_sequence"]:
            if sequence_column in current:
                assert (current[sequence_column] == current.aa_seq).all(), (assay, sequence_column)
        current["point"] = pd.to_numeric(current[phenotype], errors="raise")
        assert np.isfinite(current.point).all()
        current["condition_id"] = current.block if assay.startswith("RASK_") else assay + "_original_assay"
        current["assay_id"] = assay
        current["source_file"] = str(path.relative_to(ROOT))
        current["raw_value"] = current[phenotype]
        current["observation_id"] = current.source_row.map(lambda row: f"func_obs_{assay_index:02d}_{row:07d}")
        condition_proteins = {}
        for condition in sorted(current.condition_id.unique()):
            protein_id = f"func_{len(proteins):04d}"
            condition_proteins[condition] = protein_id
            proteins.append({
                "protein_id": protein_id, "assay_id": assay, "condition_id": condition,
                "reference_sequence": sequence, "source_scaffold": assay,
                "origin": "natural", "units": units,
            })
        current["protein_id"] = current.condition_id.map(condition_proteins)
        current["measurement_status"] = "point"
        current["lower"], current["upper"] = current.point, current.point
        fields = ["observation_id", "source_file", "source_row", "assay_id", "condition_id", "protein_id", "mutation_key", "aa_seq", "positions", "n_mutations", "raw_value", "point", "lower", "upper", "measurement_status"]
        for field in ["sigma", "std_medianBrightness_per_aaseq", "n_replicates", "total_cell_count"]:
            if field in current:
                current[field] = pd.to_numeric(current[field].replace("", np.nan), errors="coerce")
                fields.append(field)
        observations.append(current[fields])
        observed = current.loc[current.n_mutations.gt(0)].groupby("mutant").point.mean()
        comparison = curated.set_index("mutant").DMS_score.to_frame().join(observed.rename("raw_mean"), how="left")
        assert comparison.raw_mean.notna().all(), assay
        difference = comparison.raw_mean - comparison.DMS_score
        curated_checks.append({
            "assay_id": assay, "curated_variants": len(curated), "raw_joined": int(comparison.raw_mean.notna().sum()),
            "max_absolute_score_difference": float(difference.abs().max()),
            "raw_vs_curated_pearson": float(comparison.raw_mean.corr(comparison.DMS_score)),
        })
        catalogs.append({
            "assay_id": assay, "source_file": str(path.relative_to(ROOT)), "curated_file": str((CURATED / (assay + ".csv")).relative_to(ROOT)),
            "data_version": "ProteinGym substitutions v1.3", "phenotype_column": phenotype, "units": units,
            "higher_is_better": True, "reference_sequence": sequence, "conditions": sorted(condition_proteins),
            "wild_type_observed": bool(wild.any()),
            "measurement_representation": "Original scalar measurements retained. No detection interval was substituted for a scalar value.",
            "raw_score_preservation": "Raw values retain their source assay scale; RASK blocks remain separate.",
        })
        print(json.dumps(counts[-1]), flush=True)
    obs = pd.concat(observations, ignore_index=True)
    identifiers = obs[["protein_id", "mutation_key"]].drop_duplicates().sort_values(["protein_id", "mutation_key"])
    identifiers["variant_id"] = [f"func_var_{i:07d}" for i in range(len(identifiers))]
    obs = obs.merge(identifiers, on=["protein_id", "mutation_key"], validate="many_to_one")
    grouped = obs.groupby("variant_id", sort=True)
    variants = grouped.agg(
        protein_id=("protein_id", "first"), assay_id=("assay_id", "first"), condition_id=("condition_id", "first"),
        mutation_key=("mutation_key", "first"), aa_seq=("aa_seq", "first"), positions=("positions", "first"),
        n_mutations=("n_mutations", "first"), point=("point", "mean"), n_observations=("observation_id", "size"),
        observations=("observation_id", list), source_rows=("source_row", list),
    ).reset_index()
    variants["lower"], variants["upper"] = variants.point, variants.point
    variants["all_point"] = True
    lookup = variants.set_index(["protein_id", "mutation_key"])
    quartets, incomplete = [], []
    for row in variants.loc[variants.n_mutations.eq(2)].itertuples():
        first, second = row.mutation_key.split(":")
        keys = [(row.protein_id, name) for name in ["WT", first, second]]
        missing = [key[1] for key in keys if key not in lookup.index]
        if missing:
            incomplete.append({"variant_id": row.variant_id, "protein_id": row.protein_id, "assay_id": row.assay_id, "missing_components": missing})
            continue
        wt, a, b = [lookup.loc[key] for key in keys]
        quartets.append({
            "quartet_id": f"func_q_{len(quartets):07d}", "assay_id": row.assay_id, "condition_id": row.condition_id,
            "protein_id": row.protein_id, "wt_id": wt.variant_id, "a_id": a.variant_id, "b_id": b.variant_id, "ab_id": row.variant_id,
            "mutation_key": row.mutation_key, "position_i": row.positions[0], "position_j": row.positions[1],
            "pair_id": f"{row.protein_id}:{row.positions[0]}:{row.positions[1]}",
            "epsilon": row.point - a.point - b.point + wt.point, "mu_ab": row.point - wt.point,
            "wt_point": wt.point, "all_point": True,
        })
    quartet = pd.DataFrame(quartets)
    component = variants.set_index("variant_id")
    independent = quartet.ab_id.map(component.point) - quartet.a_id.map(component.point) - quartet.b_id.map(component.point) + quartet.wt_id.map(component.point)
    np.testing.assert_allclose(independent, quartet.epsilon, rtol=0, atol=1e-10)
    for role in ["wt_id", "a_id", "b_id", "ab_id"]:
        assert (quartet[role].map(component.condition_id) == quartet.condition_id).all()
        assert (quartet[role].map(component.protein_id) == quartet.protein_id).all()
    pd.DataFrame(catalogs).to_parquet(OUTPUT / "catalog.parquet", index=False)
    pd.DataFrame(proteins).to_parquet(OUTPUT / "proteins.parquet", index=False)
    obs.to_parquet(OUTPUT / "observations.parquet", index=False)
    variants.to_parquet(OUTPUT / "variants.parquet", index=False)
    quartet.to_parquet(OUTPUT / "quartets.parquet", index=False)
    pd.DataFrame(incomplete).to_parquet(OUTPUT / "incomplete_quartets.parquet", index=False)
    pd.DataFrame(counts).to_csv(OUTPUT / "admission_counts.csv", index=False)
    pd.DataFrame(curated_checks).to_csv(OUTPUT / "raw_curated_comparison.csv", index=False)
    coverage = variants.loc[variants.n_mutations.eq(2)].groupby("assay_id").size().rename("double_variants").to_frame()
    coverage["complete_quartets"] = quartet.groupby("assay_id").size().reindex(coverage.index, fill_value=0)
    coverage["coverage"] = coverage.complete_quartets / coverage.double_variants
    coverage.to_csv(OUTPUT / "quartet_coverage.csv")
    result = {
        "complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "observations": len(obs), "variants": len(variants), "complete_quartets": len(quartet), "incomplete_quartets": len(incomplete),
        "coverage": coverage.reset_index().to_dict("records"), "raw_curated_checks": curated_checks,
        "raw_source": "ProteinGym v1.3 original DMS archive with independently identified WT records.",
        "conditions": "RASK measurements are kept within their original block. Shared biological sequences across blocks must stay together in downstream splits.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
