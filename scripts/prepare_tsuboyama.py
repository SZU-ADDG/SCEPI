import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/extracted/tsuboyama/Processed_K50_dG_datasets"
OUTPUT = ROOT / "data/processed/tsuboyama"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SUBSTITUTION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])")


def check_mutations(row, reference):
    sequence, label = row.aa_seq, row.mut_type
    if "ins" in label:
        return "insertion", None
    if "del" in label:
        return "deletion", None
    if any(aa not in ALPHABET for aa in sequence):
        return "nonstandard_sequence", None
    if len(sequence) != len(reference):
        return "length_mismatch", None
    declared = []
    if label != "wt":
        parts = label.split(":")
        matches = [SUBSTITUTION.fullmatch(part) for part in parts]
        if any(match is None for match in matches):
            return "unrecognized_mutation_label", None
        positions = [int(match[2]) - 1 for match in matches]
        if len(set(positions)) != len(positions):
            return "repeated_position", None
        for match, position in zip(matches, positions):
            if position >= len(reference) or reference[position] != match[1]:
                return "reference_mismatch", None
            if match[1] != match[3]:
                declared.append((position, match[1], match[3]))
    actual = [(i, wt, mt) for i, (wt, mt) in enumerate(zip(reference, sequence)) if wt != mt]
    if sorted(declared) != actual:
        return "sequence_label_mismatch", None
    return "accepted", actual


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv", low_memory=False)
    quality = pd.read_csv(SOURCE / "Single_DMS_list.csv", low_memory=False).set_index("name")
    assert raw.shape == (776298, 37)
    assert raw.WT_name.isin(quality.index).all()
    names = sorted(raw.WT_name.unique())
    proteins = quality.loc[names].reset_index().rename(columns={"name": "source_reference", "aa_seq": "reference_sequence"})
    proteins.insert(0, "protein_id", [f"tsub_{i:04d}" for i in range(len(proteins))])
    proteins["assay_id"] = "Tsuboyama2023_joint_dG"
    proteins["condition_id"] = "PBS_room_temperature_5min_joint_inference"
    proteins["source_scaffold"] = proteins.source_reference.str.split(".pdb", regex=False).str[0]
    proteins["origin"] = np.where(proteins.WT_cluster.astype(str).str.fullmatch(r"[0-9]+"), "natural", "designed")
    assert proteins.DMS_group.isin(["G0", "G1"]).all()
    reference_map = proteins.set_index("source_reference")
    rows = []
    for row in raw.itertuples():
        reference = reference_map.loc[row.WT_name]
        status, changes = check_mutations(row, reference.reference_sequence)
        rows.append({
            "protein_id": reference.protein_id, "sequence_status": status,
            "mutation_key": ":".join(f"{wt}{pos + 1}{mt}" for pos, wt, mt in changes) if changes else "",
            "n_mutations": len(changes) if changes is not None else -1,
            "positions": [pos for pos, _, _ in changes] if changes is not None else [],
            "wt_residues": "".join(wt for _, wt, _ in changes) if changes is not None else "",
            "mut_residues": "".join(mt for _, _, mt in changes) if changes is not None else "",
        })
    annotations = pd.DataFrame(rows)
    observations = pd.concat([raw, annotations], axis=1)
    observations.insert(0, "observation_id", [f"tsub_obs_{i:07d}" for i in range(len(raw))])
    observations.insert(1, "source_row", np.arange(len(raw)))
    text = raw.dG_ML.astype(str)
    numeric = pd.to_numeric(text, errors="coerce")
    assert (numeric.notna() | text.isin(["<-1", ">5", "-"])).all()
    observations["point"] = numeric
    observations["lower"] = numeric
    observations["upper"] = numeric
    observations["measurement_status"] = np.select(
        [text.eq("<-1"), text.eq(">5"), text.eq("-")],
        ["left_censored", "right_censored", "author_filtered"], default="point",
    )
    observations.loc[text.eq("<-1"), ["lower", "upper"]] = [-np.inf, -1.0]
    observations.loc[text.eq(">5"), ["lower", "upper"]] = [5.0, np.inf]
    observations["point_in_range"] = numeric.gt(-1) & numeric.lt(5)
    observations["accepted"] = observations.sequence_status.eq("accepted") & ~text.eq("-")
    observations["variant_id"] = pd.NA
    valid = observations.loc[observations.accepted].copy()
    valid["variant_id"] = valid.groupby(["protein_id", "mutation_key"], sort=True).ngroup().map(lambda value: f"tsub_var_{value:07d}")
    observations.loc[valid.index, "variant_id"] = valid.variant_id
    valid["all_point"] = valid.measurement_status.eq("point")
    valid["has_left_censoring"] = valid.measurement_status.eq("left_censored")
    valid["has_right_censoring"] = valid.measurement_status.eq("right_censored")
    valid["finite_lower"] = valid.lower.where(np.isfinite(valid.lower), 0)
    valid["finite_upper"] = valid.upper.where(np.isfinite(valid.upper), 0)
    grouped = valid.groupby(["protein_id", "mutation_key"], sort=True)
    variants = grouped.agg(
        variant_id=("variant_id", "first"), aa_seq=("aa_seq", "first"),
        n_mutations=("n_mutations", "first"), positions=("positions", "first"),
        wt_residues=("wt_residues", "first"), mut_residues=("mut_residues", "first"),
        point=("point", "mean"), lower=("finite_lower", "mean"), upper=("finite_upper", "mean"),
        all_point=("all_point", "all"), point_in_range=("point_in_range", "all"),
        left_censored=("has_left_censoring", "any"), right_censored=("has_right_censoring", "any"),
        n_observations=("observation_id", "size"), observations=("observation_id", list),
        source_rows=("source_row", list),
        ci_width_max=("deltaG_95CI", "max"), ci_width_mean=("deltaG_95CI", "mean"),
    ).reset_index()
    variants.loc[~variants.all_point, "point"] = np.nan
    variants.loc[variants.left_censored, "lower"] = -np.inf
    variants.loc[variants.right_censored, "upper"] = np.inf
    assert variants[["lower", "upper"]].notna().all().all()
    assert (variants.lower <= variants.upper).all()
    variants.drop(columns=["left_censored", "right_censored"], inplace=True)
    print(json.dumps({"stage": "biological_variants", "count": len(variants)}), flush=True)
    assert not variants.duplicated(["protein_id", "aa_seq"]).any()
    lookup = {(r.protein_id, r.mutation_key): r for r in variants.itertuples()}
    quartets = []
    missing = []
    for ab in variants.loc[variants.n_mutations.eq(2)].itertuples():
        a_key, b_key = ab.mutation_key.split(":")
        keys = [(ab.protein_id, key) for key in ("", a_key, b_key)]
        unavailable = [key[1] or "WT" for key in keys if key not in lookup]
        if unavailable:
            missing.append({"variant_id": ab.variant_id, "protein_id": ab.protein_id, "missing": unavailable})
            continue
        wt, a, b = [lookup[key] for key in keys]
        all_point = all(v.all_point for v in (wt, a, b, ab))
        main = all(v.point_in_range for v in (wt, a, b, ab))
        lower = ab.lower - a.upper - b.upper + wt.lower
        upper = ab.upper - a.lower - b.lower + wt.upper
        margin = min(min(v.point + 1, 5 - v.point) for v in (wt, a, b, ab)) if all_point else np.nan
        quartets.append({
            "quartet_id": f"tsub_q_{len(quartets):07d}", "protein_id": ab.protein_id,
            "wt_id": wt.variant_id, "a_id": a.variant_id, "b_id": b.variant_id, "ab_id": ab.variant_id,
            "mutation_key": ab.mutation_key, "position_i": ab.positions[0], "position_j": ab.positions[1],
            "pair_id": f"{ab.protein_id}:{ab.positions[0]}:{ab.positions[1]}",
            "epsilon": ab.point - a.point - b.point + wt.point if all_point else np.nan,
            "lower": lower, "upper": upper, "all_point": all_point, "main": main,
            "margin": margin, "mu_ab": ab.point - wt.point if ab.all_point and wt.all_point else np.nan,
            "mu_ab_lower": ab.lower - wt.upper, "mu_ab_upper": ab.upper - wt.lower,
            "wt_point": wt.point, "wt_lower": wt.lower, "wt_upper": wt.upper,
            "measurement_ci_width_max": max(v.ci_width_max for v in (wt, a, b, ab)),
            "covariance_available": False,
        })
    quartets = pd.DataFrame(quartets)
    assert quartets.quartet_id.is_unique and quartets.ab_id.is_unique
    summary = []
    summary.append({"stage": "source", "count": len(raw), "proteins": raw.WT_name.nunique()})
    for status, group in observations.groupby("sequence_status"):
        summary.append({"stage": "sequence_" + status, "count": len(group), "proteins": group.protein_id.nunique()})
    for status, group in observations.groupby("measurement_status"):
        summary.append({"stage": "measurement_" + status, "count": len(group), "proteins": group.protein_id.nunique()})
    for stage, frame in [("accepted_observations", valid), ("biological_variants", variants), ("complete_quartets", quartets), ("main_quartets", quartets.loc[quartets.main])]:
        summary.append({"stage": stage, "count": len(frame), "proteins": frame.protein_id.nunique()})
    for table, frame in [("observations", observations), ("variants", variants), ("quartets", quartets), ("proteins", proteins)]:
        frame.to_parquet(OUTPUT / f"{table}.parquet", index=False)
    pd.DataFrame(missing, columns=["variant_id", "protein_id", "missing"]).to_parquet(OUTPUT / "incomplete_quartets.parquet", index=False)
    pd.DataFrame(summary).to_csv(OUTPUT / "admission_counts.csv", index=False)
    SeqIO.write([SeqRecord(Seq(r.reference_sequence), id=r.protein_id, description=r.source_reference) for r in proteins.itertuples()], OUTPUT / "references.fasta", "fasta")
    catalog = pd.DataFrame([{
        "assay_id": "Tsuboyama2023_joint_dG", "condition_id": "PBS_room_temperature_5min_joint_inference",
        "doi": "10.1038/s41586-023-06328-6", "data_doi": "10.5281/zenodo.7992926",
        "version": "v2_230420", "license": "CC-BY-4.0", "unit": "kcal/mol",
        "orientation": "higher_unfolding_free_energy_is_more_stable", "source_column": "dG_ML",
        "source": str((SOURCE / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv").relative_to(ROOT)),
        "quality_source": str((SOURCE / "Single_DMS_list.csv").relative_to(ROOT)),
        "quality_groups": ["G0", "G1"], "protease_specific_fields": ["deltaG_t", "deltaG_c"],
    }])
    catalog.to_parquet(OUTPUT / "catalog.parquet", index=False)
    manifest = {
        "stage": "E0", "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_record": "7992926", "source_rows": len(raw), "counts": summary,
        "replicates": "Arithmetic mean within reference and amino-acid sequence. Every underlying observation is retained. Any censored replicate propagates an interval to the aggregate.",
        "main": "All four biological variants have only point observations strictly between -1 and 5 kcal/mol.",
        "mutation_validation": "Both declared reference residues and complete mutant sequence checked against the author's reference sequence.",
        "sequence_exclusions": observations.sequence_status.value_counts().to_dict(),
        "quartets_missing_components": len(missing), "main_proteins": int(quartets.loc[quartets.main].protein_id.nunique()),
        "main_pairs": int(quartets.loc[quartets.main].pair_id.nunique()),
        "main_margin025": int((quartets.main & quartets.margin.ge(0.25)).sum()),
        "main_margin050": int((quartets.main & quartets.margin.ge(0.50)).sum()),
        "protein_origin_counts": proteins.origin.value_counts().to_dict(),
        "outputs": [str(p.relative_to(ROOT)) for p in sorted(OUTPUT.glob("*")) if p.is_file() and p.name != "manifest.json"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
