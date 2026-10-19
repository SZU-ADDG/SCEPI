import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "runs/e1_kermut_workspace"
LARGE = {"POLG_CXB3N_Mattenberger_2021", "POLG_DEN26_Suphatrakul_2023"}


def validate_output(assay, split, output, dataset, directory):
    prediction = pd.read_csv(output)
    assert prediction.mutant.is_unique and np.isfinite(prediction[["y", "y_pred", "y_var"]]).all().all()
    assert prediction.y_var.ge(0).all()
    source = pd.read_csv(dataset)
    if split == "domain":
        single = source.loc[source.mutant.str.count(":").eq(0)]
        train = single.sample(frac=0.8, random_state=2024)
        expected_ids = set(source.mutant) - set(train.mutant)
        assert set(prediction.mutant) == expected_ids
        testing = source.set_index("mutant").reindex(prediction.mutant)
        values = torch.tensor(train.DMS_score.to_numpy(), dtype=torch.float32)
        targets = torch.tensor(testing.DMS_score.to_numpy(), dtype=torch.float32)
        np.testing.assert_allclose(((targets - values.mean()) / values.std()).numpy(), prediction.y, rtol=1e-5, atol=2e-5)
        prediction["prediction_raw"] = prediction.y_pred * float(values.std()) + float(values.mean())
        prediction["target_raw"] = testing.DMS_score.to_numpy()
        train[["mutant"]].to_csv(directory / "training_mutants.csv", index=False)
    else:
        assert set(prediction.mutant) == set(source.mutant) and len(prediction) == len(source)
        prediction["target_raw"], prediction["prediction_raw"] = np.nan, np.nan
        source_index = source.set_index("mutant")
        for fold, group in prediction.groupby("fold", sort=True):
            testing = source_index.reindex(group.mutant)
            assert testing[split].eq(fold).all()
            training = source.loc[source[split].ne(fold)]
            values = torch.tensor(training.DMS_score.to_numpy(), dtype=torch.float32)
            targets = torch.tensor(testing.DMS_score.to_numpy(), dtype=torch.float32)
            np.testing.assert_allclose(((targets - values.mean()) / values.std()).numpy(), group.y, rtol=1e-5, atol=2e-5)
            prediction.loc[group.index, "target_raw"] = testing.DMS_score.to_numpy()
            prediction.loc[group.index, "prediction_raw"] = group.y_pred * float(values.std()) + float(values.mean())
    prediction.to_parquet(directory / "verified_predictions.parquet", index=False)
    result = {"variants": len(prediction), "spearman": float(spearmanr(prediction.y, prediction.y_pred).statistic),
              "standardized_mse": float(np.mean((prediction.y_pred - prediction.y) ** 2)),
              "raw_rmse": float(np.sqrt(np.mean((prediction.prediction_raw - prediction.target_raw) ** 2)))}
    if split != "domain" and split != "fold_rand_multiples":
        published_path = ROOT / "data/extracted/kermut/results/predictions" / assay / f"kermut_{split}.csv"
        published = pd.read_csv(published_path).set_index("mutant").reindex(prediction.mutant)
        assert published["fold"].to_numpy().tolist() == prediction["fold"].tolist()
        np.testing.assert_allclose(published.y, prediction.y, rtol=1e-5, atol=2e-5)
        result["published_spearman"] = float(spearmanr(published.y, published.y_pred).statistic)
        result["published_standardized_mse"] = float(np.mean((published.y_pred - published.y) ** 2))
        result["prediction_difference_rmse"] = float(np.sqrt(np.mean((published.y_pred.to_numpy() - prediction.y_pred) ** 2)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--scope", choices=["singles", "multiples", "domain", "cpu"], required=True)
    parser.add_argument("--solver", choices=["author", "cholesky"], default="author")
    args = parser.parse_args()
    assert args.solver == "author" or args.scope in ["multiples", "domain"]
    preparation = json.loads((WORKSPACE / "manifest.json").read_text())
    assert preparation["complete"] and preparation["published_standardized_targets_verified"]
    reference = pd.read_csv(WORKSPACE / "data/DMS_substitutions.csv")
    if args.scope in ["multiples", "domain"]:
        reference = reference.loc[reference.includes_multiple_mutants & reference.DMS_total_number_mutants.lt(7500)
                                  & reference.DMS_id.ne("GCN4_YEAST_Staller_2018")]
        splits = ["domain"] if args.scope == "domain" else ["fold_rand_multiples"]
        kind = "multiples"
    else:
        reference = reference.loc[reference.DMS_id.isin(LARGE) if args.scope == "cpu" else ~reference.DMS_id.isin(LARGE)]
        splits = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
        kind = "singles"
    reference = reference.sort_values("DMS_id").iloc[args.rank::args.world_size]
    output_scope = args.scope + ("_exact" if args.solver == "cholesky" else "")
    output = ROOT / "runs/e1_kermut" / output_scope
    output.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    results = []
    for assay in reference.DMS_id:
        for split in splits:
            directory = output / assay / split
            directory.mkdir(parents=True, exist_ok=True)
            manifest_path = directory / "manifest.json"
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text())
                assert previous["complete"]
                results.append(str(manifest_path.relative_to(ROOT)))
                continue
            script = "proteingym_domains.py" if split == "domain" else "proteingym_benchmark.py"
            gp = "kermut_no_d" if assay == "BRCA2_HUMAN_Erwood_2022_HEK293T" else "kermut"
            command = [sys.executable, str(ROOT / "external/kermut/src/experiments" / script),
                       f"dataset={assay}", f"split_method={split}", f"gp={gp}",
                       "seed=2024", "overwrite=true", f"use_gpu={'false' if args.scope == 'cpu' else 'true'}"]
            prediction_name = "kermut_exact" if args.solver == "cholesky" else "kermut"
            if prediction_name != gp:
                command.append(f"++custom_name={prediction_name}")
            if args.solver == "cholesky":
                command = [sys.executable, str(ROOT / "scripts/kermut_exact_launcher.py"), command[1], *command[2:]]
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "external/kermut")}
            if args.scope == "cpu":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            started = time.monotonic()
            with (directory / "author.log").open("w") as handle:
                subprocess.run(command, cwd=WORKSPACE, env=environment, check=True, stdout=handle, stderr=subprocess.STDOUT)
            predictions = WORKSPACE / "results/predictions" / assay / f"{prediction_name}_{split}.csv"
            assert predictions.is_file(), f"Author program did not produce predictions: {assay} {split}"
            metrics = validate_output(assay, split, predictions, WORKSPACE / "data" / f"substitutions_{kind}" / f"{assay}.csv", directory)
            result = {
                "complete": True, "assay": assay, "split": split, "seed": 2024, "metrics": metrics,
                "source_commit": source_commit, "vendor_commit": preparation["vendor_commit"],
                "data_preparation": "runs/e1_kermut_workspace/manifest.json", "command": command,
                "model": gp, "device": "cpu" if args.scope == "cpu" else os.environ["CUDA_VISIBLE_DEVICES"],
                "elapsed_seconds": time.monotonic() - started,
                "unmodified_author_program": True, "output_variants_and_targets_verified": True,
                "linear_algebra": ("Exact Cholesky for covariance, marginal likelihood, and linear solves."
                                   if args.solver == "cholesky" else "Author default GPyTorch settings."),
                "torch": torch.__version__,
            }
            manifest_path.write_text(json.dumps(result, indent=2))
            results.append(str(manifest_path.relative_to(ROOT)))
            print(json.dumps({"assay": assay, "split": split, "complete": True, "metrics": metrics,
                              "seconds": result["elapsed_seconds"]}), flush=True)
    (output / f"manifest_rank{args.rank}.json").write_text(json.dumps({
        "complete": True, "config": vars(args), "assays": len(reference), "outputs": results,
        "source_commit": source_commit,
    }, indent=2))


if __name__ == "__main__":
    main()
