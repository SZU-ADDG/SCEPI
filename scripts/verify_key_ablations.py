from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_pair import Labels
from scripts.train_single import FeatureArrays
from epistasis.model import EpistasisModel, ModelConfig, ObservationModel, PairEffect, PhenotypeMap, SingleEffect


def main():
    torch.set_float32_matmul_precision("highest")
    singles, doubles = FeatureArrays("singles"), FeatureArrays("doubles")
    labels = Labels(singles, doubles, 11)
    selected = json.loads((ROOT / "runs/e3/pair_repeats_full_g0_f1_j1_family30/selected_hyperparameters.json").read_text())
    records = []
    for key in ["family30/fold0", "family30/fold1"]:
        source = ROOT / selected[key]["run"]
        checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
        permissions = json.loads((source / "permissions.json").read_text())
        table = doubles.metadata.loc[doubles.metadata.main & doubles.metadata.protein_id.isin(permissions["validation"])].iloc[:128]
        rows = table.matrix_row.to_numpy()
        assert len(rows) == 128
        data = doubles.batch(rows)
        values, observed = labels.values("doubles", rows, np.full(len(rows), 3))
        assert observed.all()
        single_config = ModelConfig(**checkpoint["single_config"])
        config = ModelConfig(**checkpoint["pair_config"])
        configurations = {
            "interaction": config,
            "direct_residual": replace(config, pair_form="direct"),
            "no_pair_measured": replace(config, pair_single_input="predicted"),
        }
        outputs, counts, initial_states = {}, {}, {}
        for arm, current in configurations.items():
            torch.manual_seed(11)
            model = EpistasisModel(SingleEffect(single_config), PairEffect(current), PhenotypeMap(checkpoint["flexible"])).cuda()
            counts[arm] = sum(parameter.numel() for parameter in model.parameters())
            initial_states[arm] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            model.load_state_dict(checkpoint["model"])
            model.eval()
            output = model(data, values, observed, omega=1.0)
            outputs[arm] = {name: value.detach().clone() for name, value in output.items()}
            swapped = {name: value.flip(1) if value.ndim >= 2 else value for name, value in data.items()}
            reverse = model(swapped, values.flip(1), observed.flip(1), omega=1.0)
            torch.testing.assert_close(output["phenotype"], reverse["phenotype"], rtol=0, atol=1e-5)
            effects = model.calibrated_effects(data, values, observed, 1.0)
            active = data["mut_aa"] != data["wt_aa"]
            if arm == "direct_residual":
                expected = model.pair.symmetric_function(data, effects, observed, active)
                torch.testing.assert_close(output["interaction"], expected, rtol=0, atol=0)
            if arm == "interaction":
                first, second = active.clone(), active.clone()
                first[:, 1], second[:, 0] = False, False
                reference = torch.zeros_like(active)
                original = (model.pair.symmetric_function(data, effects, observed, active)
                            - model.pair.symmetric_function(data, effects, observed, first)
                            - model.pair.symmetric_function(data, effects, observed, second)
                            + model.pair.symmetric_function(data, effects, observed, reference))
                torch.testing.assert_close(output["interaction"], original, rtol=0, atol=0)
                torch.testing.assert_close(output["phenotype"], model.mapping(effects.sum(-1) + original), rtol=0, atol=0)
            if arm == "no_pair_measured":
                # 交换真实测量只用于检查信息通道，不用于训练或性能评价。
                exchanged = model(data, values.roll(1, 0), observed, omega=1.0)
                hidden = model(data, values, torch.zeros_like(observed), omega=1.0)
                torch.testing.assert_close(output["interaction"], exchanged["interaction"], rtol=0, atol=0)
                torch.testing.assert_close(output["interaction"], hidden["interaction"], rtol=0, atol=0)
                assert (output["single"] - exchanged["single"]).abs().max() > 0
                assert (output["single"] - hidden["single"]).abs().max() > 0
            target_y = torch.as_tensor(table.relative_point.to_numpy(dtype=np.float32), device="cuda")
            target_epsilon = torch.as_tensor(table.epsilon.to_numpy(dtype=np.float32), device="cuda")
            noise = ObservationModel().cuda()
            noise.load_state_dict(checkpoint["noise"])
            loss = F.huber_loss(output["epsilon"], target_epsilon, delta=1.0)
            loss += 0.1 * noise.nll(output["phenotype"], target_y, target_y, torch.ones_like(target_y, dtype=torch.bool)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            assert torch.isfinite(noise.sigma_raw.grad)
            del model, output, reverse, noise, loss
        assert len(set(counts.values())) == 1
        for arm in ["direct_residual", "no_pair_measured"]:
            assert initial_states[arm].keys() == initial_states["interaction"].keys()
            for name in initial_states[arm]:
                torch.testing.assert_close(initial_states[arm][name], initial_states["interaction"][name], rtol=0, atol=0)
            torch.testing.assert_close(outputs[arm]["single"], outputs["interaction"]["single"], rtol=0, atol=0)
        records.append({"source_run": str(source.relative_to(ROOT)), "real_validation_quartets": len(rows),
                        "projection": config.projection, "parameters": counts,
                        "same_initialization": True, "same_external_single_calibration": True,
                        "default_forward_unchanged": True, "direct_head_matches_symmetric_function": True,
                        "measured_values_and_availability_cannot_enter_removed_path": True,
                        "finite_real_data_forward_and_backward": True, "endpoint_symmetry": True})
    assert {record["projection"] for record in records} == {64, 128}
    output = ROOT / "runs/key_ablations_20261015"
    output.mkdir(parents=True, exist_ok=True)
    result = {"complete": True, "stage": "implementation_verification", "performance_evaluation": False, "records": records}
    (output / "implementation_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
