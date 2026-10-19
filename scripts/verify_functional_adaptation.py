import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_family import load_model
from epistasis.functional import FunctionalData, FunctionalModel
from epistasis.model import EpistasisModel, ModelConfig, PairEffect, PhenotypeMap, SingleEffect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    root = (ROOT / args.run_root).resolve()
    assert root.is_relative_to(ROOT)
    records = []
    paths = sorted(root.glob("*/*/*/seed*/*/*/manifest.json"))
    assert paths
    for path in paths:
        information = json.loads(path.read_text())
        assert information["complete"]
        if not information["evaluated"]:
            continue
        train = pd.read_parquet(path.parent / "training_observations.parquet")
        validation = pd.read_parquet(path.parent / "validation_permissions.parquet")
        evaluation = pd.read_parquet(path.parent / "shared/predictions.parquet")
        testing = evaluation.loc[evaluation["set"].eq("test")]
        train_ids, validation_ids, test_ids = set(train.variant_id), set(validation.variant_id), set(testing.variant_id)
        assert not train_ids & (validation_ids | test_ids) and not validation_ids & test_ids
        proteins = sorted(set(train.protein_id) | set(validation.protein_id) | set(testing.protein_id))
        data = FunctionalData(ROOT, information["dataset"], proteins)
        training_rows = data.lookup.loc[sorted(train_ids)].to_numpy()
        values = data.metadata.iloc[training_rows].point.dropna().to_numpy()
        center, scale = information["normalization"]["training_mean"], information["normalization"]["training_std"]
        np.testing.assert_allclose([np.mean(values), np.std(values)], [center, scale], rtol=1e-10, atol=1e-10)
        features = data.features[information["protein_id"]]
        single_lookup = pd.Series(np.arange(len(features.single_metadata)), index=features.single_metadata.mutation_key)
        actual_singles = features.metadata.loc[features.metadata.n_mutations.eq(1)]
        wt_rows = features.metadata.index[features.metadata.n_mutations.eq(0)]
        for method in ["shared", "remove_interaction", "independent_no_r"]:
            output = path.parent / method
            arrays = np.load(output / "predictions.npz")
            assert arrays["variants"].shape == (len(features.metadata),)
            assert arrays["singles"].shape == (len(features.single_metadata),)
            counterfactual = arrays["singles"][actual_singles.mutation_key.map(single_lookup).to_numpy(dtype=int)]
            direct = arrays["variants"][actual_singles.index.to_numpy()]
            np.testing.assert_allclose((direct - center) / scale, (counterfactual - center) / scale, rtol=1e-6, atol=1e-5)
            if len(wt_rows):
                np.testing.assert_allclose((arrays["variants"][wt_rows] - center) / scale,
                                           (arrays["reference"] - center) / scale, rtol=1e-6, atol=1e-5)
            quartet_path = output / "quartet_predictions.parquet"
            if quartet_path.exists():
                quartets = pd.read_parquet(quartet_path)
                prediction = pd.Series(arrays["variants"], index=features.metadata.variant_id)
                computed = quartets.ab_id.map(prediction) - quartets.a_id.map(prediction) - quartets.b_id.map(prediction) + quartets.wt_id.map(prediction)
                np.testing.assert_allclose(computed, quartets.prediction_epsilon, rtol=0, atol=1e-10)
            checkpoint_path = path.parent / ("independent_no_r.pt" if method == "independent_no_r" else "best.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint["source_runs"]:
                sources = [load_model(ROOT / run).cpu() for run in checkpoint["source_runs"]]
            else:
                projection = checkpoint["projection"] if "projection" in checkpoint else checkpoint["selected"]["projection"]
                config = ModelConfig(projection=projection)
                sources = [EpistasisModel(SingleEffect(config), PairEffect(config), PhenotypeMap(False))]
            model = FunctionalModel(sources, data, training_rows, center, scale).cuda().eval()
            model.load_state_dict(checkpoint["model"])
            selected = features.metadata.groupby("n_mutations", group_keys=False).head(64)
            indices = data.lookup.loc[selected.variant_id].to_numpy()
            with torch.inference_mode():
                expected = model(data, indices, remove_interaction=method != "shared")["phenotype"].cpu().numpy()
            actual = (arrays["variants"][selected.index.to_numpy()] - center) / scale
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)
            records.append({"manifest": str(path.relative_to(ROOT)), "method": method, "single_counterfactuals": len(direct),
                            "reference_counterfactuals": len(wt_rows), "checkpoint_replay_variants": len(indices),
                            "maximum_normalized_replay_difference": float(np.max(np.abs(actual - expected))),
                            "label_disjointness": True, "training_normalization_verified": True})
            del model, sources, checkpoint
            gc.collect()
            torch.cuda.empty_cache()
    assert records
    output = root / "counterfactual_verification"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "checks.csv", index=False)
    result = {"complete": True, "source_commit": subprocess.check_output(
                  ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "verified_method_outputs": len(records), "verified_fits": len({row["manifest"] for row in records}),
              "single_counterfactuals": sum(row["single_counterfactuals"] for row in records),
              "reference_counterfactuals": sum(row["reference_counterfactuals"] for row in records),
              "checkpoint_replay_variants": sum(row["checkpoint_replay_variants"] for row in records),
              "maximum_normalized_replay_difference": max(row["maximum_normalized_replay_difference"] for row in records),
              "label_disjointness": True, "training_normalization_verified": True,
              "tolerance": {"absolute": 1e-5, "relative": 1e-6, "scale": "Training-standardized phenotype."}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
