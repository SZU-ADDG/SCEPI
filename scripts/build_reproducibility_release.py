import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_pair import Labels
from train_single import FeatureArrays
from epistasis.model import ModelConfig


def copy(source, target):
    assert source.is_file(), source
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def training_stages(run):
    source = json.loads((ROOT / run / "manifest.json").read_text())
    assert source["complete"] and not source["smoke"]
    single_run = source["single_run"]
    single = json.loads((ROOT / single_run / "manifest.json").read_text())
    config = single["config"]
    single_args = ["scripts/train_single.py"]
    for field in ["threshold", "fold", "seed", "projection", "learning_rate", "representation", "batch_size"]:
        single_args += ["--" + field.replace("_", "-"), str(config[field])]
    stages = [{"name": "single_training", "arguments": single_args, "manifest": single_run + "/manifest.json"}]
    config = source["config"]
    pair_args = ["scripts/train_pair.py", "--single-run", single_run]
    for field in ["threshold", "fold", "seed", "projection", "learning_rate", "lambda_y", "representation", "objective", "batch_size"]:
        pair_args += ["--" + field.replace("_", "-"), str(config[field])]
    pair_config = ModelConfig(**source["pair_config"])
    pair_args += ["--pair-form", pair_config.pair_form, "--pair-single-input", pair_config.pair_single_input]
    for field in ["geometry", "flexible", "joint", "no_interaction"]:
        if config[field]:
            pair_args.append("--" + field.replace("_", "-"))
    stages.append({"name": "pair_training", "arguments": pair_args, "manifest": run + "/manifest.json"})
    for partition in ["validation", "test"]:
        stages.append({"name": "evaluation_" + partition,
                       "arguments": ["scripts/evaluate_family.py", "--run", run, "--partition", partition,
                                     "--batch-size", str(config["batch_size"])],
                       "manifest": run + f"/evaluation_{partition}/manifest.json"})
    return stages, source


def source_models(destination):
    protocol = json.loads((ROOT / "runs/key_ablations_20261015/protocol.json").read_text())
    selected = []
    for threshold in [30, 25]:
        selection = json.loads((ROOT / f"runs/e3/pair_repeats_full_g0_f1_j1_family{threshold}/selected_hyperparameters.json").read_text())
        for fold in range(5):
            run = Path(selection[f"family{threshold}/fold{fold}"]["run"])
            for seed in [11, 23, 47, 89, 131]:
                selected.append({"arm": "interaction", "threshold": threshold, "fold": fold, "seed": seed,
                                 "run": str(run.with_name(run.name.rsplit("_seed", 1)[0] + f"_seed{seed}"))})
    selected += [{key: row[key] for key in ["arm", "threshold", "fold", "seed", "run"]} for row in protocol["matched_runs"]]
    assert len(selected) == 150
    jobs, records = [], []
    for row in selected:
        stages, source = training_stages(row["run"])
        permissions = json.loads((ROOT / row["run"] / "permissions.json").read_text())
        assert not set(permissions["test"]) & (set(permissions["train"]) | set(permissions["validation"]))
        jobs.append({**row, "stages": stages})
        records.append({**row, "config": source["config"], "single_run": source["single_run"],
                        "single_config": source["single_config"], "pair_config": source["pair_config"],
                        "best_epoch": source["best_epoch"], "epochs": source["epochs"], "omega": source["omega"],
                        "validation_score": source["validation_score"], "permissions": permissions})
    dump(destination / "reproducibility/source_jobs.json", jobs)
    dump(destination / "reproducibility/source_models.json", records)
    for path in (ROOT / "runs/e3").glob("*/selected_hyperparameters.json"):
        copy(path, destination / path.relative_to(ROOT))
    return selected


def real_example(destination, source_run):
    source = ROOT / source_run
    manifest = json.loads((source / "manifest.json").read_text())
    prediction = pd.read_parquet(source / "evaluation_test/shared_Sall.parquet")
    prediction = prediction.loc[prediction.main & prediction.observed_a & prediction.observed_b].sort_values("variant_id").head(32)
    assert len(prediction) == 32
    permissions = json.loads((source / "permissions.json").read_text())
    assert set(prediction.protein_id) <= set(permissions["test"])
    single, double = FeatureArrays("singles"), FeatureArrays("doubles")
    labels = Labels(single, double, manifest["seed"])
    rows = double.metadata.set_index("variant_id").loc[prediction.variant_id, "matrix_row"].to_numpy()
    features = double.batch(rows, device="cpu")
    data = {"feature_" + key: value.numpy() for key, value in features.items()}
    data.update({"observed_singles": labels.pair_value[rows], "available": labels.pair_rank[rows] >= 0,
                 "variant_id": prediction.variant_id.to_numpy(dtype=str), "protein_id": prediction.protein_id.to_numpy(dtype=str),
                 "omega": np.array(manifest["omega"]), "prediction_mu": prediction.prediction_mu.to_numpy(),
                 "prediction_epsilon": prediction.prediction_epsilon.to_numpy()})
    (destination / "examples").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "examples/held_out_quartets.npz", **data)
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    retained = {key: checkpoint[key] for key in ["model", "single_config", "pair_config", "flexible", "omit_pair", "omega", "epoch"]}
    torch.save(retained, destination / "examples/family30_fold0_seed11.pt")
    dump(destination / "examples/provenance.json", {
        "source_run": source_run, "real_held_out_quartets": len(prediction),
        "source_threshold": 30, "source_fold": 0, "source_seed": 11,
        "target_combination_training_labels": 0, "permissions": permissions,
        "input": "Recorded ESM-2 features, measured constituent single effects, availability, and source-selected omega.",
        "reference": "Previously saved held-out S-all predictions; CPU reproduction checked at 1e-5 tolerance.",
    })


def environment(destination):
    command = ["/data1/doomx/anaconda3/bin/conda", "env", "export", "--prefix",
               "/data1/doomx/anaconda3/envs/recomb-3-epistasis", "--no-builds"]
    exported = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    specification = yaml.safe_load(exported)
    assert specification.pop("prefix") == "/data1/doomx/anaconda3/envs/recomb-3-epistasis"
    pip = next(value["pip"] for value in specification["dependencies"] if isinstance(value, dict) and "pip" in value)
    replacements = {
        "pymochi==0+untagged.1.gb71c738": "pymochi @ git+https://github.com/lehner-lab/MoCHI.git@b71c73806c2b7049bc05c499258c1cea6013189f",
        "esm-msr==0.3.0": "esm-msr @ git+https://github.com/SKTeamLab/esm-msr.git@6ecea374ecf2c661065449227a0e93cb825d5162",
    }
    assert set(replacements) <= set(pip)
    pip[:] = [replacements.get(value, value) for value in pip]
    output = destination / "configs/environment.yml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(specification, sort_keys=False))
    assert "prefix" not in yaml.safe_load(output.read_text())
    recorded = json.loads((ROOT / "runs/environment/manifest.json").read_text())
    dump(destination / "reproducibility/environment.json", {key: recorded[key] for key in ["python", "versions", "pip_check", "source_commit"]})
    copy(ROOT / "runs/environment/source_repositories.json", destination / "reproducibility/source_repositories.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "releases/scepi_submission_20261019")
    args = parser.parse_args()
    destination = args.output.resolve()
    assert destination.is_relative_to(ROOT / "releases")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["epistasis", "scripts"]:
        for path in (ROOT / name).iterdir():
            if path.is_file() and path.suffix in {".py", ".R", ".sh"}:
                copy(path, destination / path.relative_to(ROOT))
    for path in (ROOT / "reproducibility").iterdir():
        if path.is_file():
            copy(path, destination / "reproducibility" / path.name)
    copy(ROOT / "reproducibility/README.md", destination / "README.md")
    copy(ROOT / "reproducibility/LICENSE", destination / "LICENSE")
    copy(ROOT / "reproducibility/runtime.sh", destination / "scripts/runtime.sh")
    copy(ROOT / "reproducibility/reproduce.py", destination / "reproduce.py")
    for path in (ROOT / "paper").rglob("*"):
        if path.is_file() and path.suffix in {".tex", ".cls", ".tikz", ".svg", ".pdf"}:
            copy(path, destination / path.relative_to(ROOT))
    for path in (ROOT / "configs").glob("*.json"):
        copy(path, destination / path.relative_to(ROOT))
    for name in ["families_30.parquet", "families_25.parquet", "family_manifest.json", "combination_manifest.json"]:
        copy(ROOT / "data/splits" / name, destination / "data/splits" / name)
    for name in ["tsuboyama", "thermompnn", "esm_msr", "flip2", "proteingym"]:
        copy(ROOT / "work/sources" / (name + "_zenodo.json"), destination / "work/sources" / (name + "_zenodo.json"))
    primary = ROOT / "runs/key_ablations_20261015/paper_evidence"
    for name in ["phenotype", "residual", "selection"]:
        copy(primary / (name + "_per_protein.parquet"), destination / "evidence" / (name + "_per_protein.parquet"))
    for threshold in [30, 25]:
        copy(primary / f"verified_family{threshold}/paper_table.csv", destination / f"evidence/paper_table_family{threshold}.csv")
    random = ROOT / "runs/reviewer_revision_20261019/random_enrichment"
    copy(random / "random_per_protein.parquet", destination / "evidence/random_per_protein.parquet")
    copy(random / "summary.json", destination / "evidence/random_summary.json")
    copy(random / "manifest.json", destination / "evidence/random_manifest.json")
    copy(random / "manuscript_verification.json", destination / "evidence/random_manuscript_verification.json")
    copy(random / "figure_selection.parquet", destination / "evidence/figure_selection.parquet")
    revision = ROOT / "runs/reviewer_revision_20261019"
    for name in ["primary_verification.json", "completed_main_verification.json", "final_manifest.json"]:
        copy(revision / name, destination / "evidence" / name)
    copy(revision / "release_validation/verification.json", destination / "evidence/release_verification.json")
    for path in (ROOT / "runs/final_main_reporting_20261018").glob("*.csv"):
        copy(path, destination / "evidence/supplementary" / path.name)
    for name in ["rfa_constant_predictions.json", "manuscript_verification.json"]:
        copy(ROOT / "runs/final_main_reporting_20261018" / name, destination / "evidence/supplementary" / name)
    selected = source_models(destination)
    example = next(row for row in selected if row["arm"] == "interaction" and row["threshold"] == 30 and row["fold"] == 0 and row["seed"] == 11)
    real_example(destination, example["run"])
    environment(destination)
    snapshot = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    files = [str(path.relative_to(destination)) for path in sorted(destination.rglob("*")) if path.is_file()]
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    assert not any(part in {".git", ".venv", ".conda-env", "__pycache__"} for name in files for part in Path(name).parts)
    result = {"complete": True, "source_experiment_base_commit": snapshot, "source_snapshot": "Current saved project source files, including post-base method and reporting revisions.",
              "public_repository": "https://github.com/SZU-ADDG/SCEPI",
              "run_path": str(destination.relative_to(ROOT)), "source_models": len(selected),
              "primary_complete_arms": 26, "primary_evaluations": 1300,
              "esm_msr_scope": "One epoch-10 checkpoint, family30/fold0/seed11; remaining repeated training discontinued.",
              "files": files, "total_bytes": size, "environment_storage": "System Conda envs only; no environment is included."}
    dump(destination / "reproducibility/release_manifest.json", result)
    print(json.dumps({key: result[key] for key in ["complete", "run_path", "source_models", "total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
