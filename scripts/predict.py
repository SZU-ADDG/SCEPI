import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.model import EpistasisModel, ModelConfig, PairEffect, PhenotypeMap, SingleEffect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    assert args.batch_size > 0 and args.top_k > 0
    assert args.device != "cuda" or torch.cuda.is_available()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = EpistasisModel(SingleEffect(ModelConfig(**checkpoint["single_config"])),
                           PairEffect(ModelConfig(**checkpoint["pair_config"])),
                           PhenotypeMap(checkpoint["flexible"]), omit_pair=checkpoint["omit_pair"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(args.device).eval()
    data = np.load(args.input, allow_pickle=False)
    required = {"variant_id", "protein_id", "observed_singles", "available"}
    assert required <= set(data.files)
    size = len(data["variant_id"])
    assert size > 0 and len(data["protein_id"]) == size
    assert len(set(data["variant_id"])) == size
    assert data["observed_singles"].shape == data["available"].shape == (size, 2)
    assert np.isfinite(data["observed_singles"][data["available"]]).all()
    features = {key.removeprefix("feature_"): data[key] for key in data.files if key.startswith("feature_")}
    assert features and all(len(value) == size for value in features.values())
    predictions = []
    with torch.inference_mode():
        for offset in range(0, size, args.batch_size):
            batch = slice(offset, min(offset + args.batch_size, size))
            inputs = {key: torch.from_numpy(value[batch]).to(args.device) for key, value in features.items()}
            values = torch.from_numpy(data["observed_singles"][batch]).to(args.device)
            mask = torch.from_numpy(data["available"][batch]).to(args.device)
            result = model(inputs, values, mask, omega=checkpoint["omega"])
            assert all(torch.isfinite(result[key]).all() for key in ["phenotype", "epsilon", "single"])
            predictions.append(pd.DataFrame({"variant_id": data["variant_id"][batch], "protein_id": data["protein_id"][batch],
                                             "prediction_mu": result["phenotype"].cpu().numpy(),
                                             "prediction_epsilon": result["epsilon"].cpu().numpy(),
                                             "prediction_a": result["single"][:, 0].cpu().numpy(),
                                             "prediction_b": result["single"][:, 1].cpu().numpy(), "prediction_wt": 0.0}))
    frame = pd.concat(predictions, ignore_index=True)
    ranked = frame.sort_values(["protein_id", "prediction_mu", "variant_id"], ascending=[True, False, True], kind="stable")
    ranked["rank"] = ranked.groupby("protein_id").cumcount() + 1
    ranked["selected"] = ranked["rank"].le(args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(json.dumps({"complete": True, "candidates": len(frame), "proteins": frame.protein_id.nunique(),
                      "top_k": args.top_k, "selected_candidates": int(ranked.selected.sum()),
                      "source_selected_omega": checkpoint["omega"], "device": args.device,
                      "target_double_labels_used": 0, "output": str(args.output)}))


if __name__ == "__main__":
    main()
