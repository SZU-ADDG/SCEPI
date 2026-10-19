import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from esm_msr import inference, models

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    table = pd.read_parquet(ROOT / "data/processed/esm_msr_benchmark/variants.parquet")
    if args.smoke:
        table = table.iloc[::100].copy()
    groups = list(table.groupby(["partition", "code_wt", "mut_structure"], sort=True))
    groups = groups[args.rank::args.world_size]
    assert groups
    output = ROOT / "runs" / ("e1_esm_msr_smoke" if args.smoke else "e1_esm_msr")
    output.mkdir(parents=True, exist_ok=True)
    parsed = inference.parse_hparams_to_lora_config(str(ROOT / "models/esm-msr/hparams.yaml"))
    os.environ["INFRA_PROVIDER"] = "1"
    os.chdir(ROOT / "models/esm3-sm-open-v1")
    assert Path("data/weights/esm3_sm_open_v1.pth").is_file()
    started = time.monotonic()
    outputs = []
    for checkpoint_seed in [1, 2, 3]:
        torch.manual_seed(checkpoint_seed)
        model = models.MSRModel(
            lora_config={"wt_config": parsed["wt_config"], "mt_config": parsed["mt_config"]},
            shared_scale_init=1, shared_bias_init=0, adapter_mode=parsed["adapter_mode"],
            lora_mode=parsed["lora_mode"], model_dtype=torch.float32, inference_mode=True,
        ).cuda().eval()
        checkpoint = ROOT / "models/esm-msr" / f"lora_seed{checkpoint_seed}.safetensors"
        model.load_lora_weights(str(checkpoint))
        for (partition, code, background), group in groups:
            file = output / f"seed{checkpoint_seed}_{partition}_{group.code.iloc[0]}.parquet"
            if file.exists():
                previous = pd.read_parquet(file)
                assert len(previous) == len(group) and set(previous.sample_id) == set(group.sample_id)
                outputs.append(str(file.relative_to(ROOT)))
                continue
            backbone = None if background == "-" else background
            unique = group.drop_duplicates("mut_type").copy()
            input_data = inference.standardize_input_df(unique, backbone_mutation=backbone, quiet=True)
            current_started = time.monotonic()
            prediction = inference.infer_mutants(
                model=model, df=input_data, batch_size=args.batch_size, backbone_mutation=backbone,
                quiet=True, skip_additive=False, skip_reverse=False, mask_strategy=None,
                optimize_wt_pass=True, ignore_mismatch=False,
            )
            assert len(prediction) == len(unique)
            assert not prediction.mut_type_renumbered.duplicated().any()
            unique_mapping = input_data[["mut_type", "mut_type_renumbered"]]
            result = group.merge(unique_mapping, on="mut_type", validate="many_to_one")
            fields = ["mut_type_renumbered"] + [column for column in prediction if column.endswith("_pred") or column.endswith("_pred_additive")]
            result = result.merge(prediction[fields], on="mut_type_renumbered", validate="many_to_one")
            assert len(result) == len(group)
            assert np.isfinite(result[["wt_lora_pred", "mt_lora_pred", "combined_pred"]]).all().all()
            result["checkpoint_seed"] = checkpoint_seed
            result.to_parquet(file, index=False)
            outputs.append(str(file.relative_to(ROOT)))
            print(json.dumps({"seed": checkpoint_seed, "partition": partition, "protein": group.code.iloc[0], "rows": len(result), "elapsed_seconds": time.monotonic() - current_started}), flush=True)
        del model
        torch.cuda.empty_cache()
    manifest = {
        "complete": True, "smoke": args.smoke, "rank": args.rank, "world_size": args.world_size,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "data": "data/processed/esm_msr_benchmark/manifest.json", "checkpoint_seeds": [1, 2, 3],
        "skip_reverse": False, "skip_additive": False, "dtype": "float32", "outputs": outputs,
        "groups": len(groups), "variants_per_checkpoint": sum(len(group) for _, group in groups),
        "elapsed_seconds": time.monotonic() - started, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
    }
    (output / f"manifest_rank{args.rank}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
