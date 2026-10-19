import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from esm.pretrained import ESM3_structure_encoder_v0
from esm.utils.constants import esm3 as C
from esm.utils.structure.protein_chain import ProteinChain

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.msr import FEATURES, offline_model_directory


def prepare_index():
    table = pd.read_parquet(ROOT / "data/features/thermo/tsuboyama/direct_index.parquet")
    singles = table.loc[table.n_mutations.eq(1)].set_index(["protein_id", "mutation_key"])
    references, a_ids, b_ids, a_values, b_values = [], [], [], [], []
    for row in table.itertuples():
        keys = [f"{wild}{int(position) + 1}{mutant}"
                for position, wild, mutant in zip(row.positions, row.wt_residues, row.mut_residues)]
        if row.n_mutations == 1:
            a_ids.append(row.variant_id)
            b_ids.append("")
            a_values.append(row.native_ddg if row.training_point_eligible else np.nan)
            b_values.append(np.nan)
            continue
        first, second = singles.loc[(row.protein_id, keys[0])], singles.loc[(row.protein_id, keys[1])]
        a_ids.append(first.variant_id)
        b_ids.append(second.variant_id)
        a_values.append(first.native_ddg if first.training_point_eligible else np.nan)
        b_values.append(second.native_ddg if second.training_point_eligible else np.nan)
    table["single_a_id"], table["single_b_id"] = a_ids, b_ids
    table["native_single_a"], table["native_single_b"] = a_values, b_values
    for protein, group in table.groupby("protein_id", sort=True):
        row = group.iloc[0]
        references.append({"variant_id": "msr_reference_" + protein, "protein_id": protein,
                           "mutation_key": "", "positions": [], "wt_residues": "", "mut_residues": "",
                           "n_mutations": 0, "aa_seq": row.background_sequence,
                           "background_sequence": row.background_sequence, "structure": row.structure,
                           "native_ddg": np.nan, "training_point_eligible": False, "observed_variant": False,
                           "single_a_id": "", "single_b_id": "", "native_single_a": np.nan, "native_single_b": np.nan})
    table = pd.concat([pd.DataFrame(references), table], ignore_index=True)
    assert table.variant_id.is_unique
    table.to_parquet(FEATURES / "variants.parquet", index=False)
    record = {"complete": True, "variants": len(table), "proteins": table.protein_id.nunique(),
              "training_eligible": int(table.training_point_eligible.sum()),
              "reference_genotypes": len(references), "censored_labels_used_for_training": 0,
              "author_sign": "destabilizing native_ddg = negative relative unfolding free energy",
              "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    (FEATURES / "index_manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    FEATURES.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        prepare_index()
        return
    table = pd.read_parquet(FEATURES / "variants.parquet")
    directory = FEATURES / "structures"
    directory.mkdir(parents=True, exist_ok=True)
    offline_model_directory()
    encoder = ESM3_structure_encoder_v0("cuda").eval()
    outputs = []
    for index, (protein, group) in enumerate(table.groupby("protein_id", sort=True)):
        if index % args.world_size != args.rank:
            continue
        path = directory / (protein + ".pt")
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            assert cached["sequence"] == group.background_sequence.iloc[0]
            outputs.append(protein)
            continue
        row = group.iloc[0]
        chain = ProteinChain.from_pdb(str(ROOT / row.structure), "A", is_predicted=True)
        assert chain.sequence == row.background_sequence, protein
        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        with torch.inference_mode():
            _, tokens = encoder.encode(coords.cuda(), residue_index=residue_index.cuda())
        tokens = F.pad(tokens.squeeze(0).cpu(), (1, 1), value=0)
        tokens[0], tokens[-1] = C.STRUCTURE_BOS_TOKEN, C.STRUCTURE_EOS_TOKEN
        record = {"sequence": chain.sequence,
                  "coords_orig": F.pad(coords.cpu(), (0, 0, 0, 0, 1, 1), value=torch.inf),
                  "plddt": F.pad(plddt.cpu(), (1, 1), value=0),
                  "structure_tokens_orig": tokens, "residue_index": residue_index.cpu()}
        torch.save(record, path)
        outputs.append(protein)
        print(json.dumps({"protein": protein, "length": len(chain.sequence)}), flush=True)
    manifest = {"complete": True, "rank": args.rank, "world_size": args.world_size, "proteins": outputs,
                "frozen_structure_encoder": "ESM3_structure_encoder_v0",
                "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    (FEATURES / f"manifest_rank{args.rank}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
