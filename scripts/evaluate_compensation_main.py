import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from epistasis.compensation import SEEDS, definitions, destination, frozen_table, input_records, normalized_predictions, residual_metrics, selection_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(definitions()), required=True)
    parser.add_argument("--threshold", type=int, choices=[30, 25], required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    specification = definitions()[args.arm]
    assert args.threshold in specification["thresholds"] and args.condition in specification["conditions"]
    output = destination(args.arm, args.threshold, args.seed, args.fold, args.condition)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "manifest.json").exists()
    truth = frozen_table(args.threshold, args.seed, args.fold)
    if specification["requires_structure"]:
        assert truth.groupby("protein_id").structure_available.nunique().le(1).all()
        truth = truth.loc[truth.structure_available].copy()
    entries, missing = input_records(args.arm, args.threshold, args.seed, args.fold, args.condition, truth)
    assert not missing, missing
    frame, excluded, provenance = normalized_predictions(args.arm, args.threshold, args.seed, args.fold, args.condition, truth, entries)
    frame.to_parquet(output / "predictions.parquet", index=False)
    residual = residual_metrics(frame)
    selection = selection_metrics(frame)
    for table in [residual, selection]:
        for key, value in vars(args).items():
            table[key] = value
    residual.to_parquet(output / "residual_per_protein.parquet", index=False)
    selection.to_parquet(output / "selection_per_decision.parquet", index=False)
    primary = selection.loc[selection.task.eq("compensation") & selection.unit.eq("protein")
                            & selection.scoring.eq("observed_main") & selection.margin.eq(0) & selection.budget.eq(10)]
    manifest = {"complete": True, "smoke": False, "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                **vars(args), "protocol": "runs/compensation_main/protocol.json", "run_path": str(output),
                "input_predictions": provenance, "excluded_proteins": excluded,
                "normalized_variants": len(frame), "evaluated_proteins": frame.protein_id.nunique(),
                "primary_proteins": len(primary), "primary_families": primary.family_group.nunique(),
                "primary_selected_candidates": int(primary.budget.sum()),
                "primary_compensation_hits": int(primary.compensation_hits.sum()),
                "primary_compensation_hit_rate": float(primary.compensation_hit_rate.mean()) if len(primary) else None,
                "fixed_candidates": True, "four_model_predictions_verified": True, "source_family_permissions_verified": True,
                "outputs": ["predictions.parquet", "residual_per_protein.parquet", "selection_per_decision.parquet"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
