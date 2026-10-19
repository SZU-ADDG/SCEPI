import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.thermo import CachedThermo, MeasuredTransitions, capture, new_model, tensor_batch, verify_cached_head


def cache(table, directory, mode, rank, world_size, smoke, batch_size, token_budget):
    output = directory / ("smoke" if smoke else "full") / mode
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"manifest_rank{rank}.json"
    if manifest_path.exists():
        information = json.loads(manifest_path.read_text())
        assert information["complete"]
        return information
    if smoke:
        table = table.iloc[::100].copy()
    if mode == "additive":
        table = table.loc[table.n_mutations.eq(1)].copy()
    table = table.iloc[rank::world_size].copy().reset_index(drop=True)
    table["cache_row"] = np.arange(len(table))
    table.to_parquet(output / f"metadata_rank{rank}.parquet", index=False)
    shape = (len(table), 2, 512) if mode == "epistatic" else (len(table), 384)
    path = output / f"features_rank{rank}.npy"
    if not len(table):
        np.save(path, np.empty(shape, dtype=np.float32))
        checks = []
    else:
        arrays = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        original, configuration = new_model(mode, 11)
        original = original.cuda().eval()
        wrapper = CachedThermo(original, mode).cuda().eval()
        checks, seen_orders, processed = [], set(), 0
        for (protein, order), group in table.groupby(["protein_id", "n_mutations"], sort=True):
            dataset = MeasuredTransitions(group)
            current_batch = min(batch_size, max(1, token_budget // int(group.background_sequence.str.len().max())))
            for start in range(0, len(group), current_batch):
                examples = [dataset[index] for index in range(start, min(start + current_batch, len(group)))]
                batch = tensor_batch(examples)
                counts = torch.full((len(examples),), int(order), device="cuda", dtype=torch.long)
                if order not in seen_orders:
                    observed = group.loc[group.native_ddg.notna()].iloc[:16]
                    if len(observed) >= 2:
                        real = MeasuredTransitions(observed)
                        check_batch = tensor_batch([real[index] for index in range(len(real))])
                        check_counts = torch.full((len(real),), int(order), device="cuda", dtype=torch.long)
                        target = torch.tensor(observed.native_ddg.to_numpy(dtype=np.float32), device="cuda")
                        checks.append(verify_cached_head(original, mode, check_batch, check_counts, target))
                        seen_orders.add(order)
                with torch.inference_mode():
                    values, author_prediction = capture(original, mode, batch)
                    predicted, _ = wrapper(values, counts, batch[7], batch[8])
                    torch.testing.assert_close(predicted, author_prediction, rtol=1e-6, atol=1e-5)
                    arrays[group.cache_row.iloc[start:start + len(examples)].to_numpy()] = values.cpu().numpy()
                processed += len(examples)
            print(json.dumps({"cache": str(output.relative_to(ROOT)), "protein": protein, "order": int(order),
                              "processed": processed, "rows": len(table)}), flush=True)
        assert processed == len(table) and checks
        arrays.flush()
        assert np.isfinite(arrays).all()
        del arrays, original, wrapper
        torch.cuda.empty_cache()
    information = {"complete": True, "mode": mode, "rank": rank, "world_size": world_size, "smoke": smoke,
                   "rows": len(table), "shape": shape, "dtype": "float32", "checks": checks,
                   "maximum_batch_size": batch_size, "maximum_sequence_tokens_per_batch": token_budget,
                   "frozen_encoder": "Original ProteinMPNN v_48_020; original feature construction captured before trainable attention.",
                   "trainable_layers": "Original light_attention and ddg_out, retained unchanged.",
                   "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                   "vendor_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/ThermoMPNN-D"), "rev-parse", "HEAD"], text=True).strip()}
    manifest_path.write_text(json.dumps(information, indent=2))
    return information


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["direct", "augmentation"], required=True)
    parser.add_argument("--mode", choices=["epistatic", "additive"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--thresholds", nargs="+", type=int, default=[30, 25])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47, 89, 131])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=2048)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    root = ROOT / "data/features/thermo/tsuboyama"
    started = time.monotonic()
    outputs = []
    if args.phase == "direct":
        table = pd.read_parquet(root / "direct_index.parquet")
        outputs.append(cache(table, root / "direct", args.mode, args.rank, args.world_size, args.smoke, args.batch_size, args.token_budget))
    else:
        assert args.mode == "epistatic"
        for threshold in args.thresholds:
            for fold in args.folds:
                for seed in args.seeds:
                    directory = root / f"family{threshold}" / f"fold{fold}" / f"seed{seed}"
                    assert json.loads((directory / "augmentation_manifest.json").read_text())["complete"]
                    table = pd.read_parquet(directory / "augmentation_index.parquet")
                    outputs.append(cache(table, directory / "augmentation", args.mode, args.rank, args.world_size, args.smoke, args.batch_size, args.token_budget))
    print(json.dumps({"complete": True, "config": vars(args), "rows": sum(item["rows"] for item in outputs),
                      "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
