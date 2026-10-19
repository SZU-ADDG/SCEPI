import argparse
from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "external" / "ThermoMPNN-D"
sys.path.insert(0, str(VENDOR))

from thermompnn.datasets.dataset_utils import Mutation
from thermompnn.datasets.v2_datasets import parse_PDB, tied_featurize_mut
from thermompnn.ssm_utils import get_config
from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese


class PublishedMegascale(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.structures = {}
        for name in self.frame.WT_name.unique():
            path = ROOT / "data/extracted/thermompnn/Megascale/pdb" / name.replace("|", ":")
            self.structures[name] = parse_PDB(str(path), input_chain_list=["A"])[0]
        for row in self.frame.itertuples():
            sequence = list(self.structures[row.WT_name]["seq"])
            for mutation in row.mut_type.split(":"):
                position = int(mutation[1:-1]) - 1
                assert sequence[position] == mutation[0], (row.WT_name, mutation)
                sequence[position] = mutation[-1]
            assert "".join(sequence) == row.aa_seq, (row.WT_name, row.mut_type)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        mutations = row.mut_type.split(":")
        value = deepcopy(self.structures[row.WT_name])
        value["mutation"] = Mutation(
            [int(m[1:-1]) - 1 for m in mutations],
            [m[0] for m in mutations],
            [m[-1] for m in mutations],
            float(row.ddG_ML),
            row.WT_name,
        )
        return value


def predict(model, loader, mode):
    values = []
    with torch.inference_mode():
        for batch in loader:
            X, S, mask, lengths, chain_M, chain_encoding, residue_idx, positions, wildtype, mutant, measured, atom_mask = batch
            tensors = [X, S, mask, chain_M, residue_idx, chain_encoding, positions, wildtype, mutant, measured, torch.as_tensor(atom_mask)]
            tensors = [x.cuda() for x in tensors]
            if mode == "epistatic":
                first, second = model(*tensors)
                prediction = (first + second).flatten() / 2
            else:
                prediction = None
                for index in range(tensors[6].shape[1]):
                    args = tensors.copy()
                    args[1] = args[1].clone()
                    for column in (6, 7, 8):
                        args[column] = args[column][:, index:index + 1]
                    part, _ = model(*args)
                    prediction = part.flatten() if prediction is None else prediction + part.flatten()
            assert torch.isfinite(prediction).all(), mode
            values.append(prediction.cpu().numpy())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.manual_seed(11)
    np.random.seed(11)
    root = ROOT / "runs" / ("e1_thermompnn_smoke" if args.smoke else "e1_thermompnn")
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data/extracted/thermompnn/Megascale/csv/Megascale-D-test.csv")
    assert len(frame) == 18574
    frame.insert(0, "source_row", np.arange(len(frame)))
    if args.smoke:
        frame = frame.iloc[:len(frame) // 100]
    frame = frame.iloc[args.rank::args.world_size].copy()
    dataset = PublishedMegascale(frame)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=partial(tied_featurize_mut, side_chains=False))
    started = time.monotonic()
    outputs = []
    for mode in ("epistatic", "additive"):
        cfg = get_config(mode)
        cfg.platform.thermompnn_dir = str(VENDOR)
        cls = TransferModelPLv2Siamese if mode == "epistatic" else TransferModelPLv2
        stem = "ThermoMPNN-D" if mode == "epistatic" else "ThermoMPNN"
        for ensemble in (1, 2, 3):
            checkpoint = VENDOR / "model_weights" / f"{stem}-ens{ensemble}.ckpt"
            model = cls.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location="cpu").model.cuda().eval()
            prediction = predict(model, loader, mode)
            result = frame[["source_row", "WT_name", "mut_type", "ddG_ML"]].copy()
            result["prediction_native"] = prediction
            result["prediction_stability"] = -prediction
            result["observed_stability"] = -result.ddG_ML
            path = root / f"{stem}_ens{ensemble}_rank{args.rank}.parquet"
            result.to_parquet(path, index=False)
            outputs.append(str(path.relative_to(ROOT)))
            print(json.dumps({"rank": args.rank, "model": stem, "ensemble": ensemble, "rows": len(result)}), flush=True)
            del model
            torch.cuda.empty_cache()
    manifest = {
        "stage": "E1", "protocol": "published_Megascale-D_test", "complete": True,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_commit": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
        "data_record": "13345274", "rank": args.rank, "world_size": args.world_size,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(),
        "torch": torch.__version__, "seed": 11, "rows": len(frame),
        "elapsed_seconds": time.monotonic() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "outputs": outputs,
    }
    (root / f"manifest_rank{args.rank}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
