import json
from pathlib import Path
import subprocess

import pandas as pd
from esm_msr import preprocess_megascale, utils

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/extracted/esm_msr/data"
OUTPUT = ROOT / "data/processed/esm_msr_benchmark"
OUTPUT.mkdir(parents=True, exist_ok=True)
dataset = preprocess_megascale.MegaScaleDatasetPreprocessor(
    data_file=str(DATA / "tsuboyama/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"),
    af_model_folder=str(DATA / "tsuboyama/AlphaFold_model_PDBs"),
)
splits = dataset.create_training_splits(str(DATA / "hyperopt_splits.pkl"), -1)
frames = []
for partition in ["val", "test"]:
    table = utils.parse_multimutant_column(dataset.split_dfs[partition].copy(), "mut_type")
    table["id"] = table.code + "_" + table.mut_type
    table["mut_structure"] = table.mut_structure.fillna("-")
    for (code, background), group in table.groupby(["code_wt", "mut_structure"], sort=True):
        group = group.set_index("id")
        group = utils.sum_individual_mutation_scores(group, "ddG_ML", new_score_column="ddG_additive_ML")
        group["dddG_ML"] = group.ddG_ML - group.ddG_additive_ML
        group["partition"] = partition
        frames.append(group.reset_index())
table = pd.concat(frames, ignore_index=True)
table["sample_id"] = [f"msr_{i:07d}" for i in range(len(table))]
assert table.pdb_file.map(lambda path: Path(path).is_file()).all()
table.to_parquet(OUTPUT / "variants.parquet", index=False)
summary = table.groupby(["partition", "code"]).size().rename("variants").reset_index()
summary.to_csv(OUTPUT / "protein_counts.csv", index=False)
result = {
    "complete": True, "variants": len(table), "counts": table.groupby("partition").size().to_dict(),
    "proteins": table.groupby("partition").code.nunique().to_dict(),
    "author_repository_commit": subprocess.check_output(["git", "-C", str(ROOT / "external/esm-msr"), "rev-parse", "HEAD"], text=True).strip(),
    "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "split": str((DATA / "hyperopt_splits.pkl").relative_to(ROOT)),
    "preprocessing": "Unmodified author MegaScaleDatasetPreprocessor and utilities; validation and test partitions retained.",
}
(OUTPUT / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2), flush=True)
