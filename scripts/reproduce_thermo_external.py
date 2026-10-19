import argparse
from functools import partial
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from reproduce_thermompnn import ROOT, VENDOR, get_config, predict, TransferModelPLv2, TransferModelPLv2Siamese
from thermompnn.datasets.v2_datasets import ddgBenchDatasetv2, ProteinGymDataset, tied_featurize_mut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(11)
    np.random.seed(11)
    folder = "e1_thermo_external_smoke" if args.smoke else "e1_thermo_external"
    output = ROOT / "runs" / folder
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "data/extracted/thermompnn"
    jobs = [("PTMUL-D", base / "PTMUL-D/csv/PTMUL-D.csv", base / "PTMUL-D/pdb", ddgBenchDatasetv2)]
    jobs += [(p.stem, p, base / "DMS/pdb", ProteinGymDataset) for p in sorted((base / "DMS/csv").glob("*.csv"))]
    started = time.monotonic()
    products = []
    for name, csv_path, pdb_path, dataset_class in jobs:
        cfg = get_config("epistatic")
        cfg.platform.thermompnn_dir = str(VENDOR)
        cfg.data.dataset = "ptmul" if name == "PTMUL-D" else "proteingym"
        cfg.data.mut_types = ["double"]
        dataset = dataset_class(cfg, str(pdb_path), str(csv_path))
        frame = dataset.df.copy()
        assert frame.MUTS.str.count(";").eq(1).all(), name
        frame.insert(0, "source_row", np.arange(len(frame)))
        if args.smoke:
            frame = frame.iloc[:len(frame) // 100]
        frame = frame.iloc[args.rank::args.world_size].copy()
        if len(frame) == 0:
            continue
        indices = frame.source_row.tolist()
        loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=False, num_workers=0, collate_fn=partial(tied_featurize_mut, side_chains=False))
        for mode in ("epistatic", "additive"):
            cfg = get_config(mode)
            cfg.platform.thermompnn_dir = str(VENDOR)
            cls = TransferModelPLv2Siamese if mode == "epistatic" else TransferModelPLv2
            stem = "ThermoMPNN-D" if mode == "epistatic" else "ThermoMPNN"
            for ensemble in (1, 2, 3):
                checkpoint = VENDOR / "model_weights" / f"{stem}-ens{ensemble}.ckpt"
                model = cls.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location="cpu").model.cuda().eval()
                prediction = predict(model, loader, mode)
                result = frame[["source_row", "PDB", "MUTS", "DDG"]].copy()
                result["prediction_native"] = prediction
                result["prediction_stability"] = -prediction
                result["assay"] = name
                result["observed_oriented"] = -result.DDG
                result["unit"] = "kcal/mol" if name == "PTMUL-D" else "source_assay_scale"
                path = output / f"{name}_{stem}_ens{ensemble}_rank{args.rank}.parquet"
                result.to_parquet(path, index=False)
                products.append(str(path.relative_to(ROOT)))
                print(json.dumps({"rank": args.rank, "dataset": name, "model": stem, "ensemble": ensemble, "rows": len(result)}), flush=True)
                del model
                torch.cuda.empty_cache()
    manifest = {
        "stage": "E1", "protocol": "published_PTMUL-D_and_DMS", "complete": True,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "vendor_commit": subprocess.check_output(["git", "-C", str(VENDOR), "rev-parse", "HEAD"], text=True).strip(),
        "data_record": "13345274", "rank": args.rank, "world_size": args.world_size,
        "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(),
        "torch": torch.__version__, "seed": 11,
        "elapsed_seconds": time.monotonic() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "outputs": products,
    }
    (output / f"manifest_rank{args.rank}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
