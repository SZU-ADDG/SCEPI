from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_sall_full_target import paired_summary


def main():
    directory = ROOT / "runs/sall_full_target/paper_evidence/family30_snapshot"
    table = pd.read_parquet(directory / "selection_per_protein.parquet")
    table = table.loc[table.task.eq("compensation") & table.margin.eq(0) & table.budget.eq(10)
                      & table.scoring.eq("native") & table.arm.isin(["interaction", "global_nonlinear"])]
    coverage = Counter({(30, arm): 25 for arm in table.arm.unique()})
    summary = paired_summary(table, ["threshold", "task", "margin", "unit", "scoring", "budget"],
                             "compensation_hit_rate", True, coverage)
    assert len(summary) == 1 and summary[0]["complete_grid"]
    decoded = json.loads(json.dumps(summary))[0]
    protein = table.groupby(["protein_id", "arm"], observed=True).compensation_hit_rate.mean().unstack()
    assert protein.notna().all().all()
    np.testing.assert_allclose(decoded["mean_gain"], (protein.interaction - protein.global_nonlinear).mean(),
                               atol=1e-15, rtol=0)
    print(json.dumps({"checked": "real full-target compensation summary", "proteins": len(protein),
                      "mean_gain": decoded["mean_gain"], "json_roundtrip": True}), flush=True)


if __name__ == "__main__":
    main()
