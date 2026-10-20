import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/conditioning_factorial_20261021"
BASE = ROOT / "runs/key_ablations_20261015/paper_evidence"


def family_splits():
    records, rows = [], []
    for threshold in [30, 25]:
        family = pd.read_parquet(ROOT / "data/splits" / f"families_{threshold}.parquet")
        edges = pd.read_parquet(ROOT / "data/splits" / f"family_edges_{threshold}.parquet")
        assert family.protein_id.is_unique
        index = family.set_index("protein_id")
        assert (edges["query"].map(index.outer_group) == edges.target.map(index.outer_group)).all()
        assert (edges["query"].map(index.family_group) == edges.target.map(index.family_group)).all()
        graph = nx.from_pandas_edgelist(edges, "query", "target")
        graph.add_nodes_from(family.protein_id)
        components = list(nx.connected_components(graph))
        assert len(components) == family.family_group.nunique()
        for component in components:
            assert index.loc[list(component), "family_group"].nunique() == 1
        assert family.groupby("family_group").outer_group.nunique().eq(1).all()
        for fold in range(5):
            partitions = {"test": family.loc[family.outer_group.eq(fold)],
                          "validation": family.loc[family.outer_group.eq((fold + 1) % 5)],
                          "train": family.loc[~family.outer_group.isin([fold, (fold + 1) % 5])]}
            groups = [set(frame.family_group) for frame in partitions.values()]
            assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            rows.append({"threshold": threshold, "fold": fold,
                         **{name + "_proteins": len(frame) for name, frame in partitions.items()},
                         **{name + "_groups": frame.family_group.nunique() for name, frame in partitions.items()}})
        records.append({"threshold": threshold, "reference_backgrounds": len(family),
                        "sequence_homology_groups": family.homology_group.nunique(),
                        "merged_family_groups": family.family_group.nunique(),
                        "pfam_annotated_backgrounds": int(family.pfam_annotated.sum()),
                        "recorded_edges": len(edges), "cross_partition_edges": 0,
                        "edge_reasons": edges.reason.str.split(":").str[0].value_counts().to_dict()})
    pd.DataFrame(rows).to_csv(OUTPUT / "source_partition_counts.csv", index=False)
    return records


def combination_permissions():
    frame = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet")
    assert not frame.duplicated(["seed", "scheme", "variant_id"]).any()
    active = frame.loc[frame["set"].isin(["train", "validation", "test"])]
    assert active.train_rank.ge(0).eq(active["set"].eq("train")).all()
    grouped = active.loc[active.scheme.isin(["unseen_pair", "two_pair_secondary"])]
    assert grouped.groupby(["seed", "scheme", "protein_id", "pair_id"])["set"].nunique().eq(1).all()
    counts = grouped.groupby(["seed", "scheme", "protein_id", "set"]).agg(
        variants=("variant_id", "size"), site_pairs=("pair_id", "nunique")).reset_index()
    counts.to_parquet(OUTPUT / "combination_permission_counts.parquet", index=False)
    result = []
    for (seed, scheme), group in grouped.groupby(["seed", "scheme"]):
        result.append({"seed": int(seed), "scheme": scheme, "proteins": group.protein_id.nunique(),
                       "training_variants": int(group["set"].eq("train").sum()),
                       "validation_variants": int(group["set"].eq("validation").sum()),
                       "test_variants": int(group["set"].eq("test").sum()),
                       "train_validation_pair_overlap": 0, "train_test_pair_overlap": 0,
                       "validation_test_pair_overlap": 0})
    return result


def data_fields():
    path = ROOT / "data/processed/tsuboyama/observations.parquet"
    columns = pq.ParquetFile(path).schema_arrow.names
    frame = pd.read_parquet(path, columns=["accepted", "sequence_status", "dG_ML", "measurement_status", "point_in_range"])
    accepted = frame.sequence_status.eq("accepted") & frame.dG_ML.astype(str).ne("-")
    assert frame.accepted.equals(accepted)
    manifest = json.loads((ROOT / "data/processed/tsuboyama/manifest.json").read_text())
    assert len(frame) == manifest["source_rows"]
    assert int(accepted.sum()) == next(row["count"] for row in manifest["counts"] if row["stage"] == "accepted_observations")
    return {"source_rows": len(frame), "accepted_observations": int(accepted.sum()),
            "phenotype_field": "dG_ML", "unit": "kcal/mol", "observation_fields": columns,
            "measurement_status_counts": frame.measurement_status.value_counts().to_dict(),
            "admission": "Sequence-consistent substitution or wild type and author dG_ML not '-'. Dataset2/Dataset3 reference backgrounds belong to G0/G1 in Single_DMS_list.csv.",
            "censoring": "'<-1' becomes (-inf,-1]; '>5' becomes [5,inf). These records remain interval observations; numerical values are not clipped into point targets.",
            "main_evaluation": "Every member of a complete cycle has only point observations strictly between -1 and 5 kcal/mol.",
            "relative_effect": "mu_obs(M)=y(M)-y(WT), with the same reference background and measurement scale."}


def kermut_trace():
    table = pd.read_parquet(BASE / "phenotype_per_protein.parquet")
    table = table.loc[table.threshold.eq(30) & table.arm.eq("kermut"), ["protein_id", "seed"]].drop_duplicates()
    records = []
    for row in table.itertuples():
        directory = ROOT / "runs/kermut_target/S" / row.protein_id / f"seed{row.seed}/family/Sall"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["complete"] and manifest["evaluated"] and manifest["target_combination_labels"] == 0
        assert manifest["optimization"] == {"steps": 150, "optimizer": "AdamW", "lr": 0.1, "weight_decay": 0.01, "restarts": 1}
        history = pd.read_csv(directory / "optimization.csv")
        assert len(history) == 150 and np.array_equal(history.step, np.arange(150))
        values = history.negative_log_marginal_likelihood.to_numpy()
        assert np.isfinite(values).all()
        previous, terminal = values[-20:-10].mean(), values[-10:].mean()
        records.append({"protein_id": row.protein_id, "seed": int(row.seed), "steps": len(values),
                        "initial_loss": float(values[0]), "last_recorded_loss": float(values[-1]),
                        "minimum_recorded_loss": float(values.min()), "minimum_step": int(values.argmin()),
                        "terminal_window_change": float(abs(terminal - previous)),
                        "relative_terminal_window_change": float(abs(terminal - previous) / max(1.0, abs(terminal))),
                        "last_minus_minimum_loss": float(values[-1] - values.min())})
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT / "kermut_terminal_traces.csv", index=False)
    return {"fits": len(result), "proteins": result.protein_id.nunique(), "all_steps": 150,
            "median_relative_terminal_window_change": float(result.relative_terminal_window_change.median()),
            "p90_relative_terminal_window_change": float(result.relative_terminal_window_change.quantile(0.9)),
            "fraction_terminal_window_change_below_0_01": float(result.relative_terminal_window_change.lt(0.01).mean()),
            "median_last_minus_minimum_loss": float(result.last_minus_minimum_loss.median()),
            "protocol": "Fixed 150 AdamW steps, learning rate 0.1, weight decay 0.01, one initialization. Terminal-window changes are a descriptive diagnostic, not a new stopping criterion."}


def thermo_source():
    selected = {}
    for path in sorted((ROOT / "runs/thermo_family").glob("repeats_epistatic*/selected_hyperparameters.json")):
        for key, row in json.loads(path.read_text()).items():
            assert key not in selected
            selected[key] = row
    assert len(selected) == 10
    rows = []
    for key, chosen in selected.items():
        for seed in [11, 23, 47, 89, 131]:
            run = Path(chosen["run"])
            run = run.with_name(run.name.rsplit("_seed", 1)[0] + f"_seed{seed}")
            manifest = json.loads((ROOT / run / "manifest.json").read_text())
            assert manifest["complete"] and not manifest["smoke"] and manifest["source_label_permissions_verified"]
            rows.append({"key": key, "seed": seed, "run": str(run), "config": manifest["config"],
                         **{name: manifest[name] for name in ["best_epoch", "epochs", "optimizer", "selection_metric", "vendor_commit"]}})
    (OUTPUT / "thermompnn_source_protocols.json").write_text(json.dumps(rows, indent=2))
    return {"source_fits": len(rows), "maximum_epochs": 100, "patience": 10, "batch_size": 256,
            "selected_learning_rates": sorted({row["config"]["learning_rate"] for row in rows}),
            "selected_weight_decays": sorted({row["config"]["weight_decay"] for row in rows}),
            "minimum_realized_epochs": min(row["epochs"] for row in rows),
            "maximum_realized_epochs": max(row["epochs"] for row in rows)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {"complete": True, "family_partitions": family_splits(), "combination_permissions": combination_permissions(),
              "data_fields": data_fields(), "kermut_optimization": kermut_trace(), "thermompnn_source": thermo_source(),
              "family_versions": json.loads((ROOT / "data/splits/family_manifest.json").read_text()),
              "new_model_training": False}
    (OUTPUT / "protocol_disclosure.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ["complete", "family_partitions", "kermut_optimization", "thermompnn_source"]}, indent=2))


if __name__ == "__main__":
    main()
