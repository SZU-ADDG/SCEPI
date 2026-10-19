import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"


class Encoder:
    def __init__(self, batch_size):
        model_path = ROOT / "models/esm2_t33_650M_UR50D"
        metadata = json.loads((model_path / "download_manifest.json").read_text())
        assert metadata["revision"] == REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = EsmForMaskedLM.from_pretrained(model_path, local_files_only=True).cuda().eval()
        self.batch_size = batch_size
        self.aa_tokens = torch.tensor([self.tokenizer.convert_tokens_to_ids(aa) for aa in ALPHABET], device="cuda")

    @torch.inference_mode()
    def embeddings(self, sequences):
        length = len(sequences[0])
        assert all(len(sequence) == length for sequence in sequences)
        results = []
        for offset in range(0, len(sequences), self.batch_size):
            encoded = self.tokenizer(sequences[offset:offset + self.batch_size], padding=True, return_tensors="pt")
            assert encoded["input_ids"].shape[1] == length + 2
            encoded = {key: value.cuda() for key, value in encoded.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = self.model.esm(**encoded).last_hidden_state[:, 1:length + 1]
            assert torch.isfinite(hidden).all()
            results.append(hidden.float().cpu().numpy())
        return np.concatenate(results)

    @torch.inference_mode()
    def masked_log_probabilities(self, sequences, positions):
        assert len(sequences) == len(positions)
        results = []
        for offset in range(0, len(sequences), self.batch_size):
            chunk = sequences[offset:offset + self.batch_size]
            sites = torch.tensor(positions[offset:offset + self.batch_size], device="cuda") + 1
            encoded = self.tokenizer(chunk, padding=True, return_tensors="pt")
            encoded = {key: value.cuda() for key, value in encoded.items()}
            rows = torch.arange(len(chunk), device="cuda")
            encoded["input_ids"][rows, sites] = self.tokenizer.mask_token_id
            with torch.autocast("cuda", dtype=torch.float16):
                logits = self.model(**encoded).logits
            log_probs = logits[rows, sites].float().log_softmax(dim=-1)[:, self.aa_tokens]
            assert torch.isfinite(log_probs).all()
            results.append(log_probs.cpu().numpy())
        return np.concatenate(results)


def single_inputs(reference, variants):
    keys = sorted({key for mutation in variants.mutation_key for key in mutation.split(":") if key and key != "WT"}, key=lambda key: (int(key[1:-1]), key[-1]))
    sequences = []
    for key in keys:
        site = int(key[1:-1]) - 1
        assert reference[site] == key[0]
        sequences.append(reference[:site] + key[-1] + reference[site + 1:])
    return keys, sequences


def save_singles(encoder, reference, variants, destination, protein_id, smoke, smoke_fraction):
    keys, sequences = single_inputs(reference, variants)
    if smoke:
        count = max(1, int(len(keys) * smoke_fraction))
        keys, sequences = keys[:count], sequences[:count]
    partial = destination.with_suffix(".h5.part")
    started = time.monotonic()
    with h5py.File(partial, "w") as handle:
        handle.attrs["protein_id"] = protein_id
        handle.attrs["reference_sequence"] = reference
        handle.attrs["model_revision"] = REVISION
        handle.attrs["stage"] = "singles"
        handle.attrs["inference_precision"] = "float16_autocast"
        handle.create_dataset("wt", data=encoder.embeddings([reference])[0].astype(np.float16))
        handle.create_dataset("wt_masked_log_probs", data=encoder.masked_log_probabilities([reference] * len(reference), list(range(len(reference)))))
        handle.create_dataset("keys", data=np.asarray(keys, dtype=h5py.string_dtype()))
        representations = handle.create_dataset("single", shape=(len(keys), len(reference), 1280), dtype="float16", chunks=(1, len(reference), 1280), compression="lzf")
        for offset in range(0, len(sequences), encoder.batch_size):
            hidden = encoder.embeddings(sequences[offset:offset + encoder.batch_size])
            representations[offset:offset + len(hidden)] = hidden.astype(np.float16)
        handle.attrs["sequences"] = len(sequences)
        handle.attrs["elapsed_seconds"] = time.monotonic() - started
    os.replace(partial, destination)
    return len(sequences)


def save_variants(encoder, reference, variants, destination, protein_id, smoke, singles_path, smoke_fraction):
    if smoke:
        variants = variants.iloc[:max(1, int(len(variants) * smoke_fraction))]
    partial = destination.with_suffix(".h5.part")
    with h5py.File(singles_path, "r") as singles:
        assert singles.attrs["reference_sequence"] == reference
        wt_log_probs = singles["wt_masked_log_probs"][:]
    sequences = variants.aa_seq.tolist()
    means = np.empty((len(variants), 1280), dtype=np.float32)
    for offset in range(0, len(sequences), encoder.batch_size):
        hidden = encoder.embeddings(sequences[offset:offset + encoder.batch_size])
        means[offset:offset + len(hidden)] = hidden.mean(axis=1)
    scores = np.zeros(len(variants), dtype=np.float32)
    additive = np.zeros(len(variants), dtype=np.float32)
    masked_sequences, sites, result_indices, residues = [], [], [], []
    for index, row in enumerate(variants.itertuples()):
        for position, wt, mt in zip(row.positions, row.wt_residues, row.mut_residues):
            additive[index] += wt_log_probs[position, ALPHABET.index(mt)] - wt_log_probs[position, ALPHABET.index(wt)]
            if row.n_mutations == 1:
                scores[index] = additive[index]
            else:
                masked_sequences.append(row.aa_seq)
                sites.append(int(position))
                result_indices.append(index)
                residues.append((ALPHABET.index(wt), ALPHABET.index(mt)))
    for offset in range(0, len(sites), encoder.batch_size):
        log_probs = encoder.masked_log_probabilities(masked_sequences[offset:offset + encoder.batch_size], sites[offset:offset + encoder.batch_size])
        for local, values in enumerate(log_probs):
            index = offset + local
            wt, mt = residues[index]
            scores[result_indices[index]] += values[mt] - values[wt]
    with h5py.File(partial, "w") as handle:
        handle.attrs["protein_id"] = protein_id
        handle.attrs["reference_sequence"] = reference
        handle.attrs["model_revision"] = REVISION
        handle.attrs["stage"] = "variants"
        handle.create_dataset("variant_ids", data=np.asarray(variants.variant_id.tolist(), dtype=h5py.string_dtype()))
        handle.create_dataset("mean", data=means.astype(np.float16), compression="lzf")
        handle.create_dataset("context_score", data=scores)
        handle.create_dataset("additive_score", data=additive)
    os.replace(partial, destination)
    return len(variants)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--stage", choices=["singles", "variants"], default="singles")
    parser.add_argument("--dataset", default="tsuboyama")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-fraction", type=float, default=0.5)
    parser.add_argument("--shard-variants", action="store_true")
    args = parser.parse_args()
    assert 0 < args.smoke_fraction <= 1
    assert not args.shard_variants or args.stage == "variants"
    torch.manual_seed(11)
    data = ROOT / "data/processed" / args.dataset
    proteins = pd.read_parquet(data / "proteins.parquet").sort_values("protein_id")
    variants = pd.read_parquet(data / "variants.parquet")
    destination = ROOT / "data/features/esm2" / args.dataset / ("smoke" if args.smoke else "full")
    destination.mkdir(parents=True, exist_ok=True)
    selected = proteins if args.shard_variants else proteins.iloc[args.rank::args.world_size]
    if args.smoke and not args.shard_variants:
        selected = selected.iloc[:1]
    encoder = Encoder(args.batch_size)
    started = time.monotonic()
    outputs = []
    for protein in selected.itertuples():
        suffix = f"_rank{args.rank}" if args.shard_variants else ""
        path = destination / f"{protein.protein_id}_{args.stage}{suffix}.h5"
        group = variants.loc[variants.protein_id.eq(protein.protein_id)]
        if args.shard_variants:
            group = group.iloc[args.rank::args.world_size]
        assert len(group) > 0
        if path.exists():
            with h5py.File(path, "r") as handle:
                assert handle.attrs["model_revision"] == REVISION
                assert handle.attrs["reference_sequence"] == protein.reference_sequence
            outputs.append(str(path.relative_to(ROOT)))
            continue
        if args.stage == "singles":
            count = save_singles(encoder, protein.reference_sequence, group, path, protein.protein_id, args.smoke, args.smoke_fraction)
        else:
            singles_root = destination.parent / "full" if args.shard_variants else destination
            count = save_variants(encoder, protein.reference_sequence, group, path, protein.protein_id, args.smoke, singles_root / f"{protein.protein_id}_singles.h5", args.smoke_fraction)
        outputs.append(str(path.relative_to(ROOT)))
        print(json.dumps({"protein": protein.protein_id, "stage": args.stage, "sequences": count, "path": str(path.relative_to(ROOT))}), flush=True)
    manifest = {
        "stage": "frozen_ESM2_features", "feature_stage": args.stage, "complete": True,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "model": "facebook/esm2_t33_650M_UR50D", "revision": REVISION,
        "rank": args.rank, "world_size": args.world_size, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(), "batch_size": args.batch_size, "smoke": args.smoke,
        "smoke_fraction_per_selected_protein": args.smoke_fraction if args.smoke else None,
        "shard_variants": args.shard_variants,
        "elapsed_seconds": time.monotonic() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "outputs": outputs, "data_manifest": str((data / "manifest.json").relative_to(ROOT)),
    }
    (destination / f"{args.stage}_manifest_rank{args.rank}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
