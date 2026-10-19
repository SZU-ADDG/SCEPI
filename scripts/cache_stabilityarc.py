import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/stabilityarc"))
from scripts.stabilityarc_common import CACHE, DATA, load_features
from stabilityarc.contacts import extract_unsupervised_contact_map
from stabilityarc.esmc import FrozenESMC


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "manifest.json"
    assert not path.exists()
    proteins = pd.read_parquet(DATA / "proteins.parquet").sort_values("protein_id")
    assert proteins.protein_id.is_unique
    encoder = FrozenESMC(device="cuda")
    started = time.monotonic()
    records = []
    for protein in proteins.itertuples():
        output = CACHE / f"{protein.protein_id}.pt"
        if not output.exists():
            embedding = encoder.embed_sequence(protein.reference_sequence)
            contacts, contact_metadata = extract_unsupervised_contact_map(
                protein.reference_sequence, "cuda", encoder)
            payload = {"reference_sequence": protein.reference_sequence, "embedding": embedding,
                       "contacts": contacts, "contact_metadata": contact_metadata}
            partial = output.with_suffix(".partial")
            torch.save(payload, partial)
            partial.replace(output)
        load_features(protein.protein_id, protein.reference_sequence, "cpu")
        record = {"protein_id": protein.protein_id, "length": len(protein.reference_sequence),
                  "file": str(output.relative_to(ROOT))}
        records.append(record)
        print(json.dumps(record), flush=True)
    manifest = {"complete": True, "proteins": len(records), "source_labels_used": 0,
                "model": "ESMC-600M frozen wild-type embeddings and native entropy-gated attention contacts",
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "vendor_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "external/stabilityarc", text=True).strip(),
                "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "elapsed_seconds": time.monotonic() - started,
                "peak_vram_bytes": torch.cuda.max_memory_allocated(), "outputs": records}
    path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != "outputs"}), flush=True)


if __name__ == "__main__":
    main()
