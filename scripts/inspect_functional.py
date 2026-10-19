import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/functional_inspection"
OUTPUT.mkdir(parents=True, exist_ok=True)
results = []
for path in sorted((ROOT / "data/raw/flip2").glob("*/*.csv.gz")):
    frame = pd.read_csv(path, low_memory=False)
    result = {
        "source": str(path.relative_to(ROOT)), "rows": len(frame), "columns": frame.columns.tolist(),
        "sample": json.loads(frame.head(2).to_json(orient="records")),
        "sequence_lengths": frame.sequence.str.len().value_counts().to_dict(),
        "unique_sequences": frame.sequence.nunique(), "sets": frame["set"].value_counts(dropna=False).to_dict(),
        "validation": frame.validation.value_counts(dropna=False).to_dict(),
        "target": frame.target.describe().to_dict(),
    }
    results.append(result)
raw = ROOT / "data/extracted/proteingym/substitutions_raw_DMS"
names = ["D7PM05_CLYGR", "GFP_AEQVI", "HIS7_YEAST", "Q6WV12_9MAXI", "Q8WTC7_9CNID", "RASK_HUMAN"]
for name in names:
    for path in sorted(raw.glob(name + "*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        result = {
            "source": str(path.relative_to(ROOT)), "rows": len(frame), "columns": frame.columns.tolist(),
            "sample": json.loads(frame.head(4).to_json(orient="records")),
            "missing_by_column": frame.isna().sum().to_dict(),
        }
        for column in ["mutant", "mutation", "mutations", "aaMutations", "mutant_sequence"]:
            if column in frame:
                result[column + "_short_examples"] = frame.loc[frame[column].astype(str).str.len().lt(5), column].head(15).tolist()
        results.append(result)
(OUTPUT / "schemas.json").write_text(json.dumps(results, indent=2))
for entry in results:
    print(json.dumps({key: value for key, value in entry.items() if key not in {"sample", "sequence_lengths"}}, allow_nan=True), flush=True)
