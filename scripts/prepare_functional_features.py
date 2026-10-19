import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/functional"
OUTPUT = ROOT / "data/processed/functional_doubles"
OUTPUT.mkdir(parents=True, exist_ok=True)
proteins = pd.read_parquet(SOURCE / "proteins.parquet")
references = proteins.set_index("protein_id").reference_sequence.to_dict()
variants = pd.read_parquet(SOURCE / "variants.parquet")
variants = variants.loc[variants.n_mutations.le(2)].copy()
variants["wt_residues"] = ["".join(references[row.protein_id][position] for position in row.positions) for row in variants.itertuples()]
variants["mut_residues"] = ["".join(row.aa_seq[position] for position in row.positions) for row in variants.itertuples()]
assert (variants.wt_residues.str.len() == variants.n_mutations).all()
assert (variants.mut_residues.str.len() == variants.n_mutations).all()
variants.to_parquet(OUTPUT / "variants.parquet", index=False)
proteins.to_parquet(OUTPUT / "proteins.parquet", index=False)
manifest = {
    "complete": True, "source_manifest": "data/processed/functional/manifest.json",
    "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "scope": "Six original ThermoMPNN-D/ProteinGym functional double-mutant assays with all measured WT and single-mutant components.",
    "variants": len(variants), "proteins": len(proteins), "mutation_counts": variants.n_mutations.value_counts().sort_index().to_dict(),
}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
