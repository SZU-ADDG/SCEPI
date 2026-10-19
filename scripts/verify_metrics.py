import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.metrics import per_protein_metrics

paths = sorted((ROOT / "runs/e3/family30/fold0/pair").glob("*/manifest.json"))
assert paths
manifest = json.loads(paths[0].read_text())
run = paths[0].parent
prediction = pd.read_parquet(run / "validation_Z.parquet")
metadata = pd.read_parquet(ROOT / "data/features/training/tsuboyama/doubles/metadata.parquet")
extra = [column for column in metadata if column not in prediction or column == "variant_id"]
frame = prediction.merge(metadata[extra], on="variant_id", validate="one_to_one")
metrics = per_protein_metrics(frame).set_index("protein_id")
checks = 0
for protein, group in frame.groupby("protein_id", sort=True):
    expected = np.sqrt(np.mean((group.prediction_epsilon - group.epsilon) ** 2))
    assert np.isclose(metrics.loc[protein, "epsilon_rmse"], expected, atol=1e-12)
    pool = group.loc[group.wt_all_point & group.wt_point.le(4.5)]
    assert metrics.loc[protein, "selection_count"] == len(pool)
    if len(pool) >= 100:
        ranked = pool.sort_values(["prediction_mu", "mutation_key"], ascending=[False, True], kind="stable")
        for count in [10, 30]:
            selected = ranked.iloc[:count]
            expected_lower = float(selected.relative_lower.ge(0.5).mean())
            expected_upper = float(selected.relative_upper.ge(0.5).mean())
            assert np.isclose(metrics.loc[protein, f"precision{count}_lower"], expected_lower)
            assert np.isclose(metrics.loc[protein, f"precision{count}_upper"], expected_upper)
            assert expected_lower <= expected_upper
    checks += 1
output = ROOT / "runs/metrics_checks"
output.mkdir(parents=True, exist_ok=True)
result = {"complete": True, "real_prediction_run": str(run.relative_to(ROOT)), "proteins_checked": checks, "quartets_checked": len(frame)}
(output / "manifest.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
