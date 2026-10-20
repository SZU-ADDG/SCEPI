import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def reconstruct_primary():
    tables = {name: pd.read_parquet(ROOT / "evidence" / f"{name}_per_protein.parquet")
              for name in ["phenotype", "residual", "selection"]}
    keys = ["threshold", "arm", "protein_id"]
    phenotype = tables["phenotype"].loc[tables["phenotype"].phenotype_count.ge(30)]
    residual = tables["residual"].loc[tables["residual"].margin.eq(0)]
    selected = tables["selection"].loc[
        tables["selection"].task.eq("compensation") & tables["selection"].margin.eq(0)
        & tables["selection"].unit.eq("protein") & tables["selection"].budget.eq(10)
    ]
    native = selected.loc[selected.scoring.eq("native")]
    common = selected.loc[selected.scoring.eq("observed_main")]
    for table, expected in [(phenotype, 159), (residual, 155), (native, 120), (common, 120)]:
        assert table.groupby(keys).seed.nunique().eq(5).all()
        assert table.groupby(["threshold", "arm"]).protein_id.nunique().eq(expected).all()
    values = []
    for table, columns in [
        (phenotype, ["phenotype_rmse", "phenotype_spearman"]),
        (residual, ["epsilon_rmse"]),
        (native, ["compensation_hit_rate", "selected_mean_recovery"]),
        (common, ["compensation_hit_rate"]),
    ]:
        result = table.groupby(keys)[columns].mean().groupby(["threshold", "arm"]).mean()
        if table is common:
            result = result.rename(columns={"compensation_hit_rate": "common_main_hit_rate"})
        values.append(result)
    result = pd.concat(values, axis=1).sort_index()
    release = json.loads((ROOT / "reproducibility/release_manifest.json").read_text())
    assert len(result) == 2 * release["primary_complete_arms"]
    for threshold in [30, 25]:
        expected = pd.read_csv(ROOT / "evidence" / f"paper_table_family{threshold}.csv", index_col="arm")
        actual = result.loc[threshold].reindex(expected.index)
        np.testing.assert_allclose(actual[expected.columns], expected, rtol=0, atol=1e-12)
    return result, tables


def reconstruct_random(selection):
    random = pd.read_parquet(ROOT / "evidence/random_per_protein.parquet")
    np.testing.assert_allclose(random.random_hit_rate, random.pool_hits / random.pool_size, rtol=0, atol=1e-15)
    np.testing.assert_allclose(random.random_expected_hits, random.budget * random.random_hit_rate, rtol=0, atol=1e-15)
    selected = selection.loc[selection.task.eq("compensation") & selection.unit.eq("protein")].copy()
    keys = ["threshold", "fold", "margin", "protein_id", "family_group", "budget"]
    merged = selected.merge(random, on=keys, how="left", validate="many_to_one", suffixes=("", "_random"))
    assert merged.random_hit_rate.notna().all() and merged.pool_size.eq(merged.pool_size_random).all()
    assert merged.groupby([*keys, "arm", "scoring"]).seed.nunique().eq(5).all()
    per_protein = merged.groupby([*keys, "arm", "scoring"], as_index=False).agg(
        model_hit_rate=("compensation_hit_rate", "mean"), model_mean_recovery=("selected_mean_recovery", "mean"),
        random_hit_rate=("random_hit_rate", "first"), random_mean_recovery=("random_mean_recovery", "first"),
    )
    summary_keys = ["threshold", "margin", "scoring", "budget", "arm"]
    result = per_protein.groupby(summary_keys).mean(numeric_only=True)
    result["hit_rate_gain"] = result.model_hit_rate - result.random_hit_rate
    result["enrichment"] = result.model_hit_rate / result.random_hit_rate
    result["recovery_gain"] = result.model_mean_recovery - result.random_mean_recovery
    expected = pd.DataFrame(json.loads((ROOT / "evidence/random_summary.json").read_text())).set_index(summary_keys)
    fields = ["model_hit_rate", "random_hit_rate", "hit_rate_gain", "enrichment",
              "model_mean_recovery", "random_mean_recovery", "recovery_gain"]
    actual = result.reindex(expected.index)
    assert len(actual) == len(expected) == selection.arm.nunique() * 48
    np.testing.assert_allclose(actual[fields], expected[fields], rtol=0, atol=1e-12)
    return actual[fields], len(merged)


def analysis_panel(tables, row):
    metric = row["metric"]
    frame = tables["residual" if metric == "epsilon_rmse" else "selection"]
    mask = frame.threshold.eq(row["threshold"]) & frame.margin.eq(row["margin"])
    if metric != "epsilon_rmse":
        mask &= frame.task.eq("compensation") & frame.unit.eq("protein")
        mask &= frame.scoring.eq(row["scoring"]) & frame.budget.eq(row["budget"])
    return frame.loc[mask].pivot(index=["protein_id", "family_group", "seed"], columns="arm", values=metric)


def check_gain_summary(gain, row):
    grouped = gain.groupby("family_group")
    sums, counts = grouped.sum(), grouped.size()
    largest = counts.sort_values(ascending=False, kind="stable").index[0]
    leave_one = (gain.sum() - sums) / (len(gain) - counts)
    values = {"mean_gain": gain.mean(), "equal_group_mean_gain": grouped.mean().mean(),
              "without_largest_group_gain": leave_one.loc[largest],
              "leave_one_group_out_minimum": leave_one.min(), "leave_one_group_out_maximum": leave_one.max()}
    for key, value in values.items():
        np.testing.assert_allclose(value, row[key], rtol=0, atol=1e-12)
    assert len(gain) == row["proteins"] and len(counts) == row["family_groups"]
    assert int(counts.loc[largest]) == row["largest_group_proteins"]
    assert int(gain.gt(1e-12).sum()) == row["wins"]
    assert int(gain.lt(-1e-12).sum()) == row["losses"]
    assert int(gain.abs().le(1e-12).sum()) == row["ties"]
    return len(values)


def reconstruct_conditioning(tables):
    factorial = json.loads((ROOT / "evidence/factorial_summary.json").read_text())
    sensitivity = json.loads((ROOT / "evidence/group_sensitivity.json").read_text())
    assert len(factorial) == 120 and len(sensitivity) == 220
    checked = 0
    for row in factorial:
        cells = ["interaction", "values_only", "mask_only", "no_pair_measured"]
        repeated = analysis_panel(tables, row)[cells].dropna()
        assert repeated.groupby(["protein_id", "family_group"]).size().eq(5).all()
        means = repeated.groupby(["protein_id", "family_group"]).mean()
        direction = -1 if row["metric"] == "epsilon_rmse" else 1
        if row["contrast"] == "difference_of_conditional_gains":
            gain = direction * (means.interaction - means.mask_only - means.values_only + means.no_pair_measured)
        else:
            reference, control = row["reference"], row["control"]
            gain = direction * (means[reference] - means[control])
            np.testing.assert_allclose([means[reference].mean(), means[control].mean()],
                                       [row["reference_mean"], row["control_mean"]], rtol=0, atol=1e-12)
            repeats = (direction * (repeated[reference] - repeated[control])).groupby("seed").mean()
            assert set(map(str, repeats.index)) == set(row["repeat_gains"])
            for seed, value in repeats.items():
                np.testing.assert_allclose(value, row["repeat_gains"][str(seed)], rtol=0, atol=1e-12)
            checked += 7
        checked += check_gain_summary(gain, row)
    for row in sensitivity:
        repeated = analysis_panel(tables, row)[[row["reference"], row["control"]]].dropna()
        assert repeated.groupby(["protein_id", "family_group"]).size().eq(5).all()
        means = repeated.groupby(["protein_id", "family_group"]).mean()
        direction = -1 if row["metric"] == "epsilon_rmse" else 1
        checked += check_gain_summary(direction * (means[row["reference"]] - means[row["control"]]), row)
    return {"factorial_contrasts": len(factorial), "group_sensitivity_comparisons": len(sensitivity),
            "descriptive_values_checked": checked}


def check_model():
    import torch
    from epistasis.model import EpistasisModel, ModelConfig, PairEffect, PhenotypeMap, SingleEffect

    torch.set_num_threads(4)
    checkpoint = torch.load(ROOT / "examples/family30_fold0_seed11.pt", map_location="cpu", weights_only=True)
    model = EpistasisModel(
        SingleEffect(ModelConfig(**checkpoint["single_config"])), PairEffect(ModelConfig(**checkpoint["pair_config"])),
        PhenotypeMap(checkpoint["flexible"]), omit_pair=checkpoint["omit_pair"],
    ).eval()
    model.load_state_dict(checkpoint["model"])
    example = np.load(ROOT / "examples/held_out_quartets.npz", allow_pickle=False)
    features = {key.removeprefix("feature_"): torch.from_numpy(example[key]) for key in example.files if key.startswith("feature_")}
    with torch.inference_mode():
        prediction = model(features, torch.from_numpy(example["observed_singles"]),
                           torch.from_numpy(example["available"]), omega=float(example["omega"]))
    for field, expected in [("phenotype", "prediction_mu"), ("epsilon", "prediction_epsilon")]:
        np.testing.assert_allclose(prediction[field].numpy(), example[expected], rtol=1e-5, atol=1e-5)
    output = ROOT / "work/prediction_cli_check.csv"
    subprocess.run([sys.executable, "scripts/predict.py", "--checkpoint", "examples/family30_fold0_seed11.pt",
                    "--input", "examples/held_out_quartets.npz", "--output", str(output)], cwd=ROOT, check=True)
    actual = pd.read_csv(output).set_index("variant_id").loc[example["variant_id"]]
    for field in ["prediction_mu", "prediction_epsilon"]:
        np.testing.assert_allclose(actual[field], example[field], rtol=1e-5, atol=1e-5)
    assert actual.groupby("protein_id")["rank"].max().equals(actual.groupby("protein_id").size())
    return {"real_held_out_quartets": len(example["variant_id"]), "device": "cpu", "saved_prediction_tolerance": 1e-5,
            "public_prediction_cli_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "work/reproduced")
    parser.add_argument("--model-check", action="store_true")
    args = parser.parse_args()
    primary, tables = reconstruct_primary()
    random, comparisons = reconstruct_random(tables["selection"])
    conditioning = reconstruct_conditioning(tables)
    args.output.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.output / "primary_tables.csv")
    random.to_csv(args.output / "random_enrichment.csv")
    manifest = {"complete": True, "primary_model_rows": len(primary), "primary_values_checked": primary.size,
                "random_comparisons": len(random), "random_values_checked": random.size,
                "paired_selection_records": comparisons, "conditioning": conditioning, "mean_tolerance": 1e-12,
                "intervals": "Saved group-bootstrap intervals; this command independently reconstructs all reported means."}
    if args.model_check:
        manifest["model_check"] = check_model()
    (args.output / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
