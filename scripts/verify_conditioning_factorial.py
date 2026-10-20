from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis import model as current
from train_pair import Labels
from train_single import FeatureArrays

OUTPUT = ROOT / "runs/conditioning_factorial_20261021"


def build(module, checkpoint, single_input="calibrated", mask_input=None):
    single = module.SingleEffect(module.ModelConfig(**checkpoint["single_config"]))
    config = module.ModelConfig(**checkpoint["pair_config"])
    updates = {"pair_single_input": single_input}
    if mask_input is not None:
        updates["pair_mask_input"] = mask_input
    pair = module.PairEffect(replace(config, **updates))
    model = module.EpistasisModel(single, pair, module.PhenotypeMap(checkpoint["flexible"]))
    model.load_state_dict(checkpoint["model"])
    return model.cuda().eval()


def main():
    selection = json.loads((OUTPUT / "source_selection.json").read_text())
    spec = importlib.util.spec_from_file_location("conditioning_original_model", ROOT / "work/conditioning_factorial_20261021/before/model.py")
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    singles, doubles = FeatureArrays("singles"), FeatureArrays("doubles")
    checked, dimensions = [], set()
    for row in selection["selected"].values():
        checkpoint = torch.load(ROOT / row["run"] / "best.pt", map_location="cpu", weights_only=True)
        dimension = checkpoint["pair_config"]["projection"]
        if dimension in dimensions:
            continue
        permissions = json.loads((ROOT / row["run"] / "permissions.json").read_text())
        table = doubles.metadata.loc[doubles.metadata.protein_id.isin(permissions["train"]) & doubles.metadata.main]
        sample = table.iloc[:max(1, len(table) // 100)]
        labels = Labels(singles, doubles, 11)
        rows = sample.matrix_row.to_numpy()
        values, mask = labels.values("doubles", rows, np.full(len(rows), 3, dtype=np.int64))
        data = doubles.batch(rows)
        models = {
            "full": build(current, checkpoint),
            "removed": build(current, checkpoint, "predicted"),
            "values_only": build(current, checkpoint, "calibrated", "zero"),
            "mask_only": build(current, checkpoint, "predicted", "observed"),
        }
        with torch.inference_mode():
            for name, single_input in [("full", "calibrated"), ("removed", "predicted")]:
                legacy = build(original, checkpoint, single_input)
                previous = legacy(data, values, mask, omega=row["omega"])
                actual = models[name](data, values, mask, omega=row["omega"])
                for field in previous:
                    torch.testing.assert_close(actual[field], previous[field], rtol=0, atol=0)
                del legacy
            external = models["full"].calibrated_effects(data, values, mask, row["omega"])
            raw = models["full"].single(data)
            external_single = models["full"].mapping(external)
        parameter_counts = set()
        for name, model in models.items():
            captured, state_inputs = [], []
            pair_hook = model.pair.register_forward_pre_hook(lambda module, args: captured.append(args))
            state_hook = model.pair.network.register_forward_pre_hook(lambda module, args: state_inputs.append(args[0].detach()))
            prediction = model(data, values, mask, omega=row["omega"])
            pair_hook.remove()
            state_hook.remove()
            assert len(captured) == 1 and len(state_inputs) == 8
            expected_effects = raw if name in {"removed", "mask_only"} else external
            expected_mask = torch.zeros_like(mask) if name in {"removed", "values_only"} else mask
            torch.testing.assert_close(captured[0][1], expected_effects, rtol=0, atol=0)
            assert torch.equal(captured[0][2], expected_mask)
            torch.testing.assert_close(prediction["single"], external_single, rtol=0, atol=0)
            width = model.pair.encoder.width + 2
            active = data["mut_aa"] != data["wt_aa"]
            for state, flags in enumerate([(True, True), (True, False), (False, True), (False, False)]):
                ordered = state_inputs[2 * state]
                for endpoint in [0, 1]:
                    indicator = ordered[:, endpoint * width + model.pair.encoder.width + 1]
                    expected = expected_mask[:, endpoint] & active[:, endpoint] & flags[endpoint]
                    torch.testing.assert_close(indicator, expected.to(indicator.dtype), rtol=0, atol=0)
            truth = torch.from_numpy(sample.epsilon.to_numpy(dtype=np.float32)).cuda()
            loss = F.huber_loss(prediction["epsilon"], truth, delta=1.0)
            assert torch.isfinite(loss)
            loss.backward()
            gradients = [parameter.grad for parameter in model.pair.parameters() if parameter.grad is not None]
            assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
            assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
            parameter_counts.add(sum(parameter.numel() for parameter in model.parameters()))
        assert len(parameter_counts) == 1
        checked.append({"source_run": row["run"], "projection": dimension, "real_training_cycles": len(rows),
                        "training_cycle_fraction": len(rows) / len(table), "parameters": parameter_counts.pop(),
                        "legacy_full_and_removed_outputs_identical": True,
                        "external_single_predictions_identical": True,
                        "independent_value_and_mask_inputs_verified": True,
                        "all_four_WT_state_masks_verified": True, "real_label_backward_pass": True})
        dimensions.add(dimension)
        if dimensions == {64, 128}:
            break
    assert dimensions == {64, 128}
    result = {"complete": True, "performance_evaluation": False, "source_records": checked}
    (OUTPUT / "implementation_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
