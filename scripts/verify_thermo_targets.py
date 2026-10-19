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
from epistasis.thermo import CachedThermo, MeasuredTransitions, capture, new_model, tensor_batch, verify_cached_head
from epistasis.thermo_training import DoubleEvaluation, FEATURES, ThermoCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/thermo_target_smoke")
    args = parser.parse_args()
    root = (ROOT / args.run_root).resolve()
    assert root.is_relative_to(ROOT)
    paths = sorted(root.glob("family*/*/[SC]/*/seed*/*/*/manifest.json"))
    assert paths
    direct = pd.read_parquet(FEATURES / "direct_index.parquet")
    direct = direct.set_index(["protein_id", "mutation_key"], drop=False)
    columns = ["prediction_mu", "prediction_a", "prediction_b", "prediction_wt", "prediction_epsilon"]
    records, gradient_checks = [], []
    for head in ["additive", "epistatic"]:
        cache = ThermoCache(FEATURES / "direct", head)
        lookup = cache.table.set_index(["protein_id", "mutation_key"]).row
        for path in paths:
            information = json.loads(path.read_text())
            assert information["complete"]
            if information["method"] != "thermo_" + head or not information["evaluated"]:
                continue
            training = pd.read_parquet(path.parent / "training_variants.parquet")
            saved = pd.read_parquet(path.parent / "predictions.parquet").set_index("variant_id")
            assert training.variant_id.is_unique and saved.index.is_unique
            assert not set(training.variant_id) & set(saved.index)
            assert np.isfinite(training.native_ddg).all()
            if information["mode"] == "S":
                assert training.n_mutations.le(1).all()
            source = ROOT / information["source_manifest"]
            permissions = json.loads((source.parent / "permissions.json").read_text())
            family = pd.read_parquet(ROOT / "data/splits" / f"families_{information['threshold']}.parquet").set_index("protein_id")
            relatives = set(family.index[family.family_group.eq(family.loc[information["protein_id"], "family_group"])])
            assert relatives <= set(permissions["test"])
            assert not relatives & (set(permissions["train"]) | set(permissions["validation"]))

            original, _ = new_model(head, information["seed"])
            original = original.cuda().eval()
            model = CachedThermo(original, head).cuda().eval()
            source_state = torch.load(source.parent / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(source_state["model"])
            evaluator = DoubleEvaluation(cache, [information["protein_id"]])
            initial = evaluator.predict(model).set_index("variant_id")
            initial_saved = pd.read_parquet(path.parent / "initial_predictions.parquet").set_index("variant_id")
            np.testing.assert_allclose(initial.loc[initial_saved.index, columns], initial_saved[columns], rtol=1e-6, atol=1e-5)

            checkpoint = torch.load(path.parent / "best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            predicted = evaluator.predict(model).set_index("variant_id")
            replay_difference = float(np.max(np.abs(predicted.loc[saved.index, columns].to_numpy() - saved[columns].to_numpy())))
            np.testing.assert_allclose(predicted.loc[saved.index, columns], saved[columns], rtol=1e-6, atol=1e-5)
            identity = saved.prediction_mu - saved.prediction_a - saved.prediction_b + saved.prediction_wt
            np.testing.assert_allclose(identity, saved.prediction_epsilon, rtol=0, atol=1e-10)
            if head == "additive":
                np.testing.assert_allclose(saved.prediction_epsilon, 0, rtol=0, atol=1e-5)
            maximum_feature, maximum_native, checked = 0.0, 0.0, 0
            for role, table in [("train", training), ("evaluation", evaluator.table)]:
                orders = [1] if head == "additive" else [1, 2]
                for order in orders:
                    selected = table.loc[table.n_mutations.eq(order)].head(8)
                    if selected.empty:
                        continue
                    keys = list(zip(selected.protein_id, selected.mutation_key))
                    inputs = direct.loc[keys].reset_index(drop=True)
                    dataset = MeasuredTransitions(inputs)
                    batch = tensor_batch([dataset[index] for index in range(len(dataset))])
                    cache_rows = lookup.loc[keys].to_numpy(dtype=int)
                    original.eval()
                    model.eval()
                    with torch.no_grad():
                        fresh, author_prediction = capture(original, head, batch)
                        cached_batch = cache.batch(cache_rows)
                        cached_prediction = model(*cached_batch)[0]
                        torch.testing.assert_close(fresh, cached_batch[0], rtol=1e-6, atol=1e-5)
                        torch.testing.assert_close(author_prediction, cached_prediction, rtol=1e-6, atol=1e-5)
                    maximum_feature = max(maximum_feature, float((fresh - cached_batch[0]).abs().max()))
                    maximum_native = max(maximum_native, float((author_prediction - cached_prediction).abs().max()))
                    checked += len(inputs)
                    if role == "train":
                        targets = torch.tensor(inputs.native_ddg.to_numpy(dtype=np.float32), device="cuda")
                        check = verify_cached_head(original, head, batch, cached_batch[1], targets)
                        gradient_checks.append({"manifest": str(path.relative_to(ROOT)), "head": head, "mutation_order": order, **check})
            assert checked > 0
            records.append({"manifest": str(path.relative_to(ROOT)), "head": head, "mode": information["mode"],
                            "training_variants": len(training), "test_variants": int(saved["set"].eq("test").sum()),
                            "replayed_variants": len(saved), "author_forward_variants": checked,
                            "maximum_feature_difference": maximum_feature, "maximum_author_forward_difference": maximum_native,
                            "maximum_checkpoint_difference": replay_difference, "label_disjointness": True,
                            "family_exclusion": True, "source_initialization_verified": True})
            print(json.dumps(records[-1]), flush=True)
            del original, model, checkpoint, source_state, evaluator
            gc.collect()
            torch.cuda.empty_cache()
        del cache
        gc.collect()
    assert records and gradient_checks
    output = root / "verification"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "checks.csv", index=False)
    (output / "gradient_checks.json").write_text(json.dumps(gradient_checks, indent=2))
    result = {"complete": True, "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "verified_fits": len(records), "heads": sorted({row["head"] for row in records}),
              "modes": sorted({row["mode"] for row in records}),
              "replayed_variants": sum(row["replayed_variants"] for row in records),
              "author_forward_variants": sum(row["author_forward_variants"] for row in records),
              "gradient_comparisons": len(gradient_checks),
              "maximum_author_forward_difference": max(row["maximum_author_forward_difference"] for row in records),
              "maximum_checkpoint_difference": max(row["maximum_checkpoint_difference"] for row in records),
              "label_disjointness": True, "source_family_exclusion": True}
    (output / "manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
