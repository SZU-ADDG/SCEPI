import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "runs/reviewer_revision_20261019/random_enrichment")
    args = parser.parse_args()
    records = json.loads((args.input / "summary.json").read_text())
    manifest = json.loads((args.input / "manifest.json").read_text())
    assert manifest["complete"] and len(records) == 1248
    indexed = {(row["threshold"], row["margin"], row["scoring"], row["budget"], row["arm"]): row for row in records}
    assert len(indexed) == len(records)
    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    checked = 0
    for budget in [1, 3, 10, 30]:
        row = indexed[(30, 0.0, "native", budget, "interaction")]
        lower, upper = row["enrichment_interval_95"]
        expected = (f"{budget} & {100 * row['model_hit_rate']:.2f} & {100 * row['hit_rate_gain']:.2f}"
                    f" & {row['enrichment']:.2f} [{lower:.2f}, {upper:.2f}] & {row['recovery_gain']:.3f}")
        assert expected in manuscript, expected
        checked += 6
    for threshold in [30, 25]:
        for margin in [0.0, 0.25, 0.5]:
            native = indexed[(threshold, margin, "native", 10, "interaction")]
            common = indexed[(threshold, margin, "observed_main", 10, "interaction")]
            assert native["random_hit_rate"] == common["random_hit_rate"]
            expected = (f"{threshold}\\% & {margin:.2f} & {native['proteins']}"
                        f" & {100 * native['random_hit_rate']:.2f} & {native['enrichment']:.2f} & {common['enrichment']:.2f}")
            assert expected in manuscript, expected
            checked += 4
    for arm in ["interaction", "thermo_epistatic", "kermut", "stabilityarc_kermut",
                "global_nonlinear", "additive", "mean_mlp", "mean_ridge"]:
        row = indexed[(30, 0.0, "native", 10, arm)]
        value = f"{row['enrichment']:.2f}"
        if arm == "mean_ridge":
            value = "\\textbf{" + value + "}"
        assert f" & {value} \\\\" in manuscript, (arm, value)
        checked += 1
    primary = indexed[(30, 0.0, "native", 10, "interaction")]
    assert primary["candidates"] == 79564 and primary["pool_hits"] == 1718
    assert "79,564" in manuscript and "1,718" in manuscript
    assert (f"Uniform random & --- & --- & {100 * primary['random_hit_rate']:.2f}"
            f" & ${primary['random_mean_recovery']:.3f}$ & 1.00") in manuscript
    for source in ["ThermoMPNN-D \\citep{dieckhaus2025}", "Mutate Everything \\citep{ouyang2023}",
                   "StabilityArc + Kermut \\citep{feller2026,groth2024}"]:
        assert source in manuscript
    figure = (ROOT / "paper/figures/compensation_discovery.tikz").read_text().strip()
    assert manuscript.count(figure) == 1
    assert figure.count("Uniform random") == 1
    report = {"complete": True, "random_comparisons": len(records), "manuscript_random_numeric_cells": checked + 4,
              "candidate_count": primary["candidates"], "pool_hits": primary["pool_hits"],
              "native_top10_enrichment": primary["enrichment"], "figure_fragment_matches": True,
              "information_table_citations": 3, "source": str(args.input.relative_to(ROOT))}
    (args.input / "manuscript_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
