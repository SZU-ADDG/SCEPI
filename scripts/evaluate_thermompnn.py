import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e1_thermompnn"


def metrics(frame):
    observed = frame["observed_stability"].to_numpy()
    prediction = frame["prediction_stability"].to_numpy()
    per_protein = []
    for name, group in frame.groupby("WT_name", sort=True):
        y = group.observed_stability.to_numpy()
        p = group.prediction_stability.to_numpy()
        per_protein.append({
            "protein": name, "n": len(group),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "spearman": float(spearmanr(y, p).statistic),
            "pearson": float(pearsonr(y, p).statistic),
        })
    return {
        "n": len(frame), "proteins": len(per_protein),
        "pooled_rmse": float(np.sqrt(np.mean((prediction - observed) ** 2))),
        "pooled_spearman": float(spearmanr(observed, prediction).statistic),
        "pooled_pearson": float(pearsonr(observed, prediction).statistic),
        "macro_protein_rmse": float(np.mean([x["rmse"] for x in per_protein])),
        "macro_protein_spearman": float(np.mean([x["spearman"] for x in per_protein])),
        "per_protein": per_protein,
    }


def main():
    reference = pd.read_csv(ROOT / "data/extracted/thermompnn/Megascale/csv/Megascale-D-test.csv")
    manifests = [json.loads((RUN / f"manifest_rank{rank}.json").read_text()) for rank in range(8)]
    assert all(x["complete"] for x in manifests)
    assert sum(x["rows"] for x in manifests) == len(reference) == 18574
    output = {}
    for model in ["ThermoMPNN-D", "ThermoMPNN"]:
        frames = []
        for ensemble in (1, 2, 3):
            frame = pd.concat([pd.read_parquet(RUN / f"{model}_ens{ensemble}_rank{rank}.parquet") for rank in range(8)])
            frame = frame.sort_values("source_row").reset_index(drop=True)
            assert len(frame) == len(reference) and frame.source_row.nunique() == len(reference)
            assert frame.WT_name.equals(reference.WT_name)
            assert frame.mut_type.equals(reference.mut_type)
            np.testing.assert_allclose(frame.ddG_ML, reference.ddG_ML, rtol=0, atol=1e-12)
            assert np.isfinite(frame.prediction_stability).all()
            frames.append(frame)
            output[f"{model}_ens{ensemble}"] = metrics(frame)
        mean = frames[0].copy()
        mean["prediction_stability"] = np.mean([x.prediction_stability.to_numpy() for x in frames], axis=0)
        mean["prediction_native"] = -mean.prediction_stability
        mean.to_parquet(RUN / f"{model}_ensemble_predictions.parquet", index=False)
        output[model + "_ensemble"] = metrics(mean)
    result = {
        "protocol": "published_Megascale-D_test",
        "models": output,
        "scope": "Public checkpoint inference on the original test split. This does not establish performance on newly isolated families.",
        "source_commits": sorted({x["source_commit"] for x in manifests}),
        "vendor_commits": sorted({x["vendor_commit"] for x in manifests}),
        "gpus": [x["gpu"] for x in manifests],
        "rank_manifests": [f"manifest_rank{rank}.json" for rank in range(8)],
    }
    (RUN / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({name: {k: v for k, v in value.items() if k != "per_protein"} for name, value in output.items()}, indent=2))


if __name__ == "__main__":
    main()
