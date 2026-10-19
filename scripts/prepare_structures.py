import json
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/tsuboyama"
OUTPUT = ROOT / "data/features/structures/tsuboyama"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parser = PDBParser(QUIET=True)
    proteins = pd.read_parquet(DATA / "proteins.parquet")
    candidates = {}
    for path in sorted((ROOT / "data/extracted/tsuboyama/AlphaFold_model_PDBs").glob("*.pdb")):
        model = parser.get_structure(path.stem, path)[0]
        for chain in model:
            residues = [residue for residue in chain if residue.id[0] == " "]
            sequence = "".join(seq1(residue.resname) for residue in residues)
            if not sequence or any("CA" not in residue for residue in residues):
                continue
            candidates.setdefault(sequence, []).append({
                "path": path, "chain": chain.id, "residues": residues,
                "coordinates": np.stack([residue["CA"].coord for residue in residues]),
                "confidence": np.array([residue["CA"].bfactor for residue in residues], dtype=np.float32) / 100,
            })
    mappings, availability = [], []
    for protein in proteins.itertuples():
        structures = candidates.get(protein.reference_sequence, [])
        if len(structures) == 0:
            availability.append({"protein_id": protein.protein_id, "available": False, "reason": "no_exact_reference_sequence_structure", "candidate_count": 0})
            continue
        selected = sorted(structures, key=lambda item: (item["path"].name != protein.source_reference.replace("|", "_"), str(item["path"])))[0]
        path = OUTPUT / f"{protein.protein_id}.npz"
        np.savez(path, coordinates=selected["coordinates"], confidence=selected["confidence"])
        for position, residue in enumerate(selected["residues"]):
            mappings.append({
                "protein_id": protein.protein_id, "sequence_position": position,
                "reference_residue": protein.reference_sequence[position], "pdb_chain": selected["chain"],
                "pdb_number": residue.id[1], "pdb_insertion_code": residue.id[2],
                "structure": str(selected["path"].relative_to(ROOT)), "confidence": float(selected["confidence"][position]),
            })
        availability.append({
            "protein_id": protein.protein_id, "available": True, "reason": "exact_reference_sequence",
            "candidate_count": len(structures), "structure": str(selected["path"].relative_to(ROOT)),
            "features": str(path.relative_to(ROOT)),
        })
    pd.DataFrame(mappings).to_parquet(OUTPUT / "residue_mapping.parquet", index=False)
    table = pd.DataFrame(availability)
    table.to_parquet(OUTPUT / "availability.parquet", index=False)
    manifest = {
        "source": "Zenodo7992926_AlphaFold_model_PDBs", "proteins": len(proteins),
        "exact_reference_structures": int(table.available.sum()), "missing_exact_structures": int((~table.available).sum()),
        "mapping": "Biopython PDBParser; full chain sequence must equal the reference. Missing exact structures are excluded from structural comparisons.",
        "outputs": ["availability.parquet", "residue_mapping.parquet", "protein_id.npz"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
