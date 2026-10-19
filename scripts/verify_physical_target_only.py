import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from epistasis.model import EpistasisModel, ModelConfig, PairEffect, PhenotypeMap, SingleEffect
from train_pair import Labels
from train_single import FeatureArrays

parser = argparse.ArgumentParser()
parser.add_argument("--run-root", required=True)
parser.add_argument("--supervision", choices=["target_only", "transfer"], default="target_only")
args = parser.parse_args()
root = (ROOT / args.run_root).resolve()
assert root.is_relative_to(ROOT)
paths = sorted(root.glob("*/seed*/*/C*/manifest.json"))
assert paths
singles, doubles = FeatureArrays("singles"), FeatureArrays("doubles")
labels_by_seed = {}
records = []
for path in paths:
    information = json.loads(path.read_text())
    assert information["complete"]
    if not information["evaluated"]:
        continue
    assert information["supervision"] == args.supervision
    source = ROOT / information["source_run"]
    permissions = json.loads((source / "permissions.json").read_text())
    families = pd.read_parquet(ROOT / "data/splits/families_30.parquet").set_index("protein_id")
    relatives = set(families.index[families.family_group.eq(families.loc[information["protein_id"], "family_group"])])
    assert relatives <= set(permissions["test"])
    assert not relatives & (set(permissions["train"]) | set(permissions["validation"]))
    source_info = json.loads((source / "manifest.json").read_text())
    assert source_info["config"]["geometry"] == information.get("geometry", False)
    assert source_info["complete"] and not source_info["smoke"]
    seed = information["seed"]
    if seed not in labels_by_seed:
        labels_by_seed[seed] = Labels(singles, doubles, seed)
    labels = labels_by_seed[seed]
    training = pd.read_parquet(path.parent / "training_permissions.parquet")
    saved = pd.read_parquet(path.parent / "predictions.parquet")
    assert not set(training.variant_id) & set(saved.variant_id)
    assert set(training.protein_id) == {information["protein_id"]}
    if information["scheme"] != "random":
        allowed_doubles = doubles.metadata.loc[doubles.metadata.variant_id.isin(training.variant_id)]
        assert not set(allowed_doubles.pair_id) & set(saved.pair_id)
    checkpoint = torch.load(path.parent / "best.pt", map_location="cpu", weights_only=True)
    model = EpistasisModel(SingleEffect(ModelConfig(**checkpoint["single_config"])),
                            PairEffect(ModelConfig(**checkpoint["pair_config"])), PhenotypeMap(checkpoint["flexible"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    main_training_count = None
    if args.supervision == "transfer":
        source_checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        for name, value in model.single.state_dict().items():
            assert torch.equal(value, source_checkpoint["model"]["single." + name])
        assert information["supervised_source_weights_used"]
    if information.get("supervision") == "target_only":
        assert information["source_training_labels_used"] == 0 and not information["supervised_source_weights_used"]
        initial_run = ROOT / information["target_main_run"]
        main_info = json.loads((initial_run / "manifest.json").read_text())
        assert main_info["complete"] and main_info["source_training_labels_used"] == 0
        assert main_info["nonzero_gradient_batches"] > 0 and main_info["pair_weights_unchanged"]
        main = torch.load(initial_run / "best.pt", map_location="cpu", weights_only=True)
        for name, value in model.single.state_dict().items():
            assert torch.equal(value, main["model"]["single." + name])
        torch.manual_seed(seed)
        fresh = EpistasisModel(SingleEffect(ModelConfig(**checkpoint["single_config"])),
                               PairEffect(ModelConfig(**checkpoint["pair_config"])), PhenotypeMap(False))
        for name, value in fresh.pair.state_dict().items():
            assert torch.equal(value, main["model"]["pair." + name])
        main_training = pd.read_parquet(initial_run / "training_observations.parquet")
        assert main_training.n_mutations.le(1).all()
        assert set(main_training.protein_id) == {information["protein_id"]}
        assert set(main_training.observation_id) <= set(training.observation_id)
        assert not set(main_training.variant_id) & set(saved.variant_id)
        main_training_count = len(main_training)
    outputs, maximum_reference, maximum_single = [], 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(saved), 512):
            group = saved.iloc[start:start + 512]
            rows = group.matrix_row.to_numpy(dtype=int)
            data = doubles.batch(rows, device="cpu")
            values = torch.from_numpy(labels.pair_value[rows])
            allowed = torch.from_numpy(labels.pair_rank[rows] >= 0)
            predicted = model(data, values, allowed, omega=information["omega"])
            frame = pd.DataFrame({"prediction_mu": predicted["phenotype"].numpy(),
                                  "prediction_a": predicted["single"][:, 0].numpy(),
                                  "prediction_b": predicted["single"][:, 1].numpy(),
                                  "prediction_epsilon": predicted["epsilon"].numpy()})
            outputs.append(frame)
            for endpoint in [0, 1]:
                reference = {name: value.clone() for name, value in data.items()}
                reference["mut_aa"][:, endpoint] = reference["wt_aa"][:, endpoint]
                single = model(reference, values, allowed, omega=information["omega"])
                expected = predicted["single"][:, 1 - endpoint]
                torch.testing.assert_close(single["phenotype"], expected, rtol=1e-5, atol=1e-5)
                maximum_single = max(maximum_single, float((single["phenotype"] - expected).abs().max()))
            reference = {name: value.clone() for name, value in data.items()}
            reference["mut_aa"] = reference["wt_aa"].clone()
            wildtype = model(reference, values, allowed, omega=information["omega"])["phenotype"]
            torch.testing.assert_close(wildtype, torch.zeros_like(wildtype), rtol=0, atol=1e-5)
            maximum_reference = max(maximum_reference, float(wildtype.abs().max()))
    computed = pd.concat(outputs, ignore_index=True)
    np.testing.assert_allclose(computed, saved[computed.columns], rtol=1e-5, atol=1e-5)
    difference = float(np.max(np.abs(computed.to_numpy() - saved[computed.columns].to_numpy())))
    records.append({"manifest": str(path.relative_to(ROOT)), "replayed_variants": len(saved),
                    "maximum_prediction_difference": difference, "maximum_reference_prediction": maximum_reference,
                    "maximum_single_counterfactual_difference": maximum_single, "target_main_observations": main_training_count,
                    "label_disjointness": True, "frozen_main_verified": True})
    print(json.dumps(records[-1]), flush=True)
assert records
output = root / "verification"
output.mkdir(parents=True, exist_ok=True)
pd.DataFrame(records).to_csv(output / "checks.csv", index=False)
result = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
          "verified_fits": len(records), "replayed_variants": sum(row["replayed_variants"] for row in records),
          "maximum_prediction_difference": max(row["maximum_prediction_difference"] for row in records),
          "maximum_single_counterfactual_difference": max(row["maximum_single_counterfactual_difference"] for row in records),
          "maximum_reference_prediction": max(row["maximum_reference_prediction"] for row in records),
          "label_disjointness": True, "reference_and_single_constraints": True,
          "supervision": args.supervision, "source_family_exclusion": True,
          "initialization": "Fresh pair and target WT/single training." if args.supervision == "target_only" else "Verified supervised source checkpoint with frozen single head."}
(output / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
