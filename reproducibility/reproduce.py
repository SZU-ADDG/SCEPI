import argparse
import json
from pathlib import Path

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
    assert len(result) == 52
    for threshold in [30, 25]:
        expected = pd.read_csv(ROOT / "evidence" / f"paper_table_family{threshold}.csv", index_col="arm")
        actual = result.loc[threshold].reindex(expected.index)
        np.testing.assert_allclose(actual[expected.columns], expected, rtol=0, atol=1e-12)
    return result, tables["selection"]


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
    assert len(actual) == len(expected) == 1248
    np.testing.assert_allclose(actual[fields], expected[fields], rtol=0, atol=1e-12)
    return actual[fields], len(merged)


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
    return {"real_held_out_quartets": len(example["variant_id"]), "device": "cpu", "saved_prediction_tolerance": 1e-5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "work/reproduced")
    parser.add_argument("--model-check", action="store_true")
    args = parser.parse_args()
    primary, selection = reconstruct_primary()
    random, comparisons = reconstruct_random(selection)
    args.output.mkdir(parents=True, exist_ok=True)
    primary.to_csv(args.output / "primary_tables.csv")
    random.to_csv(args.output / "random_enrichment.csv")
    manifest = {"complete": True, "primary_model_rows": len(primary), "primary_values_checked": primary.size,
                "random_comparisons": len(random), "random_values_checked": random.size,
                "paired_selection_records": comparisons, "mean_tolerance": 1e-12,
                "intervals": "Saved group-bootstrap intervals; this command independently reconstructs all reported means."}
    if args.model_check:
        manifest["model_check"] = check_model()
    (args.output / "verification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
