import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e1_thermo_external"
ASSAYS = {
    "D7PM05_CLYGR": "D7PM05_CLYGR_Somermeyer_2022",
    "GFP_AEQVI": "GFP_AEQVI_Sarkisyan_2016",
    "HIS7_YEAST": "HIS7_YEAST_Pokusaeva_2019",
    "Q6WV12_9MAXI": "Q6WV12_9MAXI_Somermeyer_2022",
    "Q8WTC7_9CNID": "Q8WTC7_9CNID_Somermeyer_2022",
    "RASK_HUMAN": "RASK_HUMAN_Weng_2022_abundance",
}


def key(value):
    return ":".join(sorted(value.replace(";", ":").split(":"), key=lambda item: int(item[1:-1])))


def metrics(frame, physical):
    y = frame.observed_oriented.to_numpy()
    p = frame.prediction_stability.to_numpy()
    result = {"n": len(frame), "spearman": float(spearmanr(y, p).statistic), "pearson": float(pearsonr(y, p).statistic)}
    if physical:
        result["rmse_kcal_mol"] = float(np.sqrt(np.mean((y - p) ** 2)))
    return result


def main():
    manifests = [json.loads((RUN / f"manifest_rank{rank}.json").read_text()) for rank in range(8)]
    assert all(item["complete"] for item in manifests)
    jobs = [("PTMUL-D", ROOT / "data/extracted/thermompnn/PTMUL-D/csv/PTMUL-D.csv")]
    jobs += [(p.stem, p) for p in sorted((ROOT / "data/extracted/thermompnn/DMS/csv").glob("*.csv"))]
    output = {}
    orientations = {}
    for assay, source in jobs:
        reference = pd.read_csv(source)
        if assay != "PTMUL-D":
            official = pd.read_csv(ROOT / "data/extracted/proteingym/DMS_ProteinGym_substitutions" / f"{ASSAYS[assay]}.csv")
            official["key"] = official.mutant.map(key)
            assert official.key.is_unique
            matched = official.set_index("key").loc[reference.MUTS.map(key)]
            np.testing.assert_allclose(-reference.DDG.to_numpy(), matched.DMS_score.to_numpy(), rtol=1e-6, atol=1e-7)
            orientations[assay] = {"ProteinGym_assay": ASSAYS[assay], "matched_rows": len(matched), "equation": "DMS_score = -DDG", "higher_is_better": True}
        else:
            orientations[assay] = {"equation": "unfolding_ddG = -DDG", "unit": "kcal/mol", "source": "PTMUL-D/NOTE.txt"}
        output[assay] = {}
        for model in ["ThermoMPNN-D", "ThermoMPNN"]:
            frames = []
            for ensemble in (1, 2, 3):
                pieces = []
                for rank in range(8):
                    path = RUN / f"{assay}_{model}_ens{ensemble}_rank{rank}.parquet"
                    piece = pd.read_parquet(path)
                    piece["observed_oriented"] = -piece.DDG
                    piece.to_parquet(path, index=False)
                    pieces.append(piece)
                frame = pd.concat(pieces).sort_values("source_row").reset_index(drop=True)
                assert len(frame) == len(reference) and frame.source_row.nunique() == len(reference)
                assert frame.PDB.equals(reference.PDB) and frame.MUTS.equals(reference.MUTS)
                np.testing.assert_allclose(frame.DDG, reference.DDG, rtol=0, atol=1e-12)
                assert np.isfinite(frame.prediction_stability).all()
                output[assay][f"{model}_ens{ensemble}"] = metrics(frame, assay == "PTMUL-D")
                frames.append(frame)
            mean = frames[0].copy()
            mean["prediction_stability"] = np.mean([frame.prediction_stability.to_numpy() for frame in frames], axis=0)
            mean["prediction_native"] = -mean.prediction_stability
            mean.to_parquet(RUN / f"{assay}_{model}_ensemble_predictions.parquet", index=False)
            output[assay][model + "_ensemble"] = metrics(mean, assay == "PTMUL-D")
    result = {
        "stage": "E1", "protocol": "published_PTMUL-D_and_six_DMS", "models": output,
        "orientation_verification": orientations, "complete": True,
        "source_commits": sorted({item["source_commit"] for item in manifests}),
        "vendor_commits": sorted({item["vendor_commit"] for item in manifests}),
        "gpus": [item["gpu"] for item in manifests], "rank_manifests": [f"manifest_rank{rank}.json" for rank in range(8)],
        "scope": "Official public checkpoint inference on the complete original external test sets. Function predictions are ranked on each assay's own scale.",
    }
    (RUN / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({assay: {name: result for name, result in items.items() if name.endswith("_ensemble")} for assay, items in output.items()}, indent=2))


if __name__ == "__main__":
    main()
