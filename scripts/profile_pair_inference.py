import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.model import EpistasisModel, ModelConfig, PairEffect, PhenotypeMap, SingleEffect

STUDY = ROOT / "runs/conditioning_factorial_20261021"


def load_model(run):
    checkpoint = torch.load(ROOT / run / "best.pt", map_location="cpu", weights_only=True)
    model = EpistasisModel(SingleEffect(ModelConfig(**checkpoint["single_config"])),
                           PairEffect(ModelConfig(**checkpoint["pair_config"])),
                           PhenotypeMap(checkpoint["flexible"]), omit_pair=checkpoint["omit_pair"])
    model.load_state_dict(checkpoint["model"])
    return model.eval(), float(checkpoint["omega"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=STUDY / "pair_inference_profile.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    selected = json.loads((args.study / "source_selection.json").read_text())["selected"]["family30/fold0"]
    full_run = next(row["run"] for row in selected["repeats"] if row["seed"] == 11)
    protocol = json.loads((args.study / "protocol.json").read_text())
    direct_run = next(row["run"] for row in protocol["matched_runs"] if row["arm"] == "final_direct_residual"
                      and row["threshold"] == 30 and row["fold"] == 0 and row["seed"] == 11)
    examples = args.input.resolve()
    data = np.load(examples, allow_pickle=False)
    features = {key.removeprefix("feature_"): torch.from_numpy(data[key]) for key in data.files if key.startswith("feature_")}
    observed, available = torch.from_numpy(data["observed_singles"]), torch.from_numpy(data["available"])
    assert len(observed) == 32
    models = {"contrast": load_model(full_run), "direct": load_model(direct_run)}
    assert models["contrast"][0].pair.config.pair_form == "contrast"
    assert models["direct"][0].pair.config.pair_form == "direct"
    counts = {name: sum(parameter.numel() for parameter in model.parameters()) for name, (model, _) in models.items()}
    assert len(set(counts.values())) == 1
    calls, timings = {}, {name: [] for name in models}
    with torch.inference_mode():
        for name, (model, omega) in models.items():
            observed_calls = []
            hook = model.pair.network.register_forward_hook(lambda *args: observed_calls.append(1))
            result = model(features, observed, available, omega=omega)
            assert torch.isfinite(result["phenotype"]).all()
            calls[name] = len(observed_calls)
            hook.remove()
            for _ in range(20):
                model(features, observed, available, omega=omega)
        assert calls == {"contrast": 8, "direct": 2}
        for iteration in range(100):
            order = list(models) if iteration % 2 == 0 else list(reversed(models))
            for name in order:
                model, omega = models[name]
                start = time.perf_counter_ns()
                model(features, observed, available, omega=omega)
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
    medians = {name: float(np.median(values)) for name, values in timings.items()}
    result = {"complete": True, "device": "cpu", "threads": torch.get_num_threads(), "batch_size": 32,
              "warmup_batches_per_model": 20, "timed_batches_per_model": 100,
              "full_run": full_run, "direct_run": direct_run, "real_feature_source": str(examples.relative_to(ROOT)),
              "parameters": counts, "ordered_pair_network_evaluations": calls,
              "symmetric_function_evaluations": {name: value // 2 for name, value in calls.items()},
              "median_ms": medians, "interquartile_ms": {name: np.quantile(values, [.25, .75]).tolist() for name, values in timings.items()},
              "contrast_to_direct_median_ratio": medians["contrast"] / medians["direct"],
              "scope": "Complete cached-feature predictor on 32 real held-out cycles; excludes frozen-encoder computation and feature-cache construction.",
              "torch_version": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
