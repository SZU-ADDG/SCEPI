import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from esm_msr import inference, models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.msr import FEATURES, DirectDataset, collate_direct, offline_model_directory


def main():
    logging.getLogger().setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("highest")
    table = pd.read_parquet(FEATURES / "variants.parquet")
    doubles = table.loc[table.n_mutations.eq(2)]
    lengths = doubles.groupby("protein_id").background_sequence.first().str.len().sort_values()
    proteins = lengths.iloc[[0, len(lengths) // 2, -1]].index
    parsed = inference.parse_hparams_to_lora_config(str(ROOT / "models/esm-msr/hparams.yaml"))
    offline_model_directory()
    model = models.MSRModel(
        lora_config={"wt_config": parsed["wt_config"], "mt_config": parsed["mt_config"]},
        shared_scale_init=1, shared_bias_init=0, adapter_mode=parsed["adapter_mode"],
        lora_mode=parsed["lora_mode"], model_dtype=torch.float32, inference_mode=True).cuda().eval()
    model.load_lora_weights(str(ROOT / "models/esm-msr/lora_seed1.safetensors"))
    records = []
    for protein in proteins:
        selected = doubles.loc[doubles.protein_id.eq(protein)].iloc[:8]
        ids = set(selected.variant_id) | set(selected.single_a_id) | set(selected.single_b_id)
        inputs = table.loc[table.variant_id.isin(ids | {"msr_reference_" + protein})].copy()
        dataset = DirectDataset(inputs)
        predicted = []
        for start in range(0, len(dataset), 16):
            batch = collate_direct([dataset[index] for index in range(start, min(start + 16, len(dataset)))])
            batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.inference_mode():
                predicted.extend(model.forward_batch(batch, mask_strategy=None)["combined_pred"].cpu().numpy().tolist())
        current = inputs.assign(direct_prediction=predicted)
        author_input = current.loc[current.n_mutations.gt(0), ["mutation_key", "structure"]].copy()
        author_input["mut_type_renumbered"] = author_input.mutation_key
        author_input["mut_type_pdb"] = author_input.mutation_key
        author_input["pdb_file"] = author_input.structure.map(lambda value: str(ROOT / value))
        author_input["code"], author_input["chain"] = protein, "A"
        author = inference.infer_mutants(model, author_input, batch_size=16, quiet=True, skip_additive=False,
                                         skip_reverse=False, optimize_wt_pass=True, ignore_mismatch=False)
        compared = current.loc[current.n_mutations.gt(0)].merge(
            author[["mut_type_renumbered", "combined_pred"]], left_on="mutation_key",
            right_on="mut_type_renumbered", validate="one_to_one")
        np.testing.assert_allclose(compared.direct_prediction, compared.combined_pred, atol=2e-5, rtol=2e-5)
        reference_item = next(item for item in dataset.data if len(item["mutations"]) == 0)
        reference_batch = collate_direct([reference_item])
        reference_batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in reference_batch.items()}
        with torch.inference_mode():
            isolated_reference = model.forward_batch(reference_batch, mask_strategy=None)["combined_pred"].item()
        grouped_reference = current.loc[current.n_mutations.eq(0), "direct_prediction"].item()
        np.testing.assert_allclose(isolated_reference, grouped_reference, atol=2e-5, rtol=2e-5)
        assert np.isfinite(predicted).all()
        records.append({"protein": protein, "variants": len(compared),
                        "max_forward_difference": float(np.max(np.abs(compared.direct_prediction - compared.combined_pred))),
                        "reference_prediction": isolated_reference,
                        "reference_batch_difference": abs(isolated_reference - grouped_reference)})
    output = ROOT / "runs/verification/esm_msr"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"complete": True, "checks": records,
                "data": "data/features/esm_msr/tsuboyama/index_manifest.json",
                "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                "checkpoint": "models/esm-msr/lora_seed1.safetensors",
                "comparison": "Original infer_mutants on the same actual single and double genotypes; empty-genotype batch consistency."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
