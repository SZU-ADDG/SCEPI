import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from esm_msr.training import ESM3EpistasisLightningModule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.msr import FEATURES, evaluation_precision, make_loader, offline_model_directory

output = ROOT / "runs/msr_target_batch_diagnostic"
directory = output / "checkpoint_source"
configuration = json.loads((directory / "config.json").read_text())["author"]
previous = pd.read_parquet(output / "fp32_all_16.parquet")
metadata = pd.read_parquet(FEATURES / "variants.parquet")
inputs = metadata.loc[metadata.variant_id.isin(previous.index)]
logging.getLogger().setLevel(logging.WARNING)
offline_model_directory()
torch.manual_seed(987)
module = ESM3EpistasisLightningModule(**configuration).cuda().eval()
state = torch.load(directory / "best.ckpt", map_location="cpu", weights_only=True)
missing, unexpected = module.load_state_dict(state["state_dict"], strict=False)
assert not unexpected and not [name for name in missing if "lora" in name.lower() or "calibration" in name.lower()]
before = torch.get_float32_matmul_precision()
identifiers, predictions = [], []
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    with evaluation_precision():
        assert not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32
        for batch in make_loader(inputs, False, batch_size=8):
            identifiers.extend(batch["variant_id"])
            batch = {name: value.cuda() if torch.is_tensor(value) else value for name, value in batch.items()}
            values = module.model.forward_batch(batch, mask_strategy=configuration["mask_strategy"])["combined_pred"]
            predictions.extend(values.cpu().numpy().tolist())
    assert torch.is_autocast_enabled()
assert torch.get_float32_matmul_precision() == before
current = pd.Series(predictions, index=identifiers).loc[previous.index]
np.testing.assert_allclose(current, previous.combined_pred, rtol=1e-5, atol=1e-5)
diagnostic = json.loads((output / "manifest.json").read_text())
comparisons = [entry for entry in diagnostic["comparisons"] if entry["column"] == "combined_pred"]
assert max(entry["maximum_difference"] for entry in comparisons if entry["precision"] == "bf16") > 1e-3
assert max(entry["maximum_difference"] for entry in comparisons if entry["precision"] == "fp32") < 1e-5
result = {"complete": True, "replayed_genotypes": len(current), "independent_model_reload": True,
          "maximum_reload_difference": float(np.max(np.abs(current - previous.combined_pred))),
          "batch_sizes": [8, 16], "validation_and_test_composition_checked": True,
          "training_precision_restored": True,
          "checkpoint": str((directory / "best.ckpt").relative_to(ROOT))}
(output / "verification.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
