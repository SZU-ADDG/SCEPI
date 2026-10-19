import numpy as np
import pandas as pd


def correlation(observed, predicted, method):
    table = pd.DataFrame({"observed": observed, "predicted": predicted}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(table) < 2 or table.observed.nunique() < 2 or table.predicted.nunique() < 2:
        return float("nan")
    return float(table.observed.corr(table.predicted, method=method))


def per_protein_metrics(frame):
    assert not frame.variant_id.duplicated().any()
    assert np.isfinite(frame[["prediction_mu", "prediction_epsilon", "prediction_a", "prediction_b", "prediction_wt"]]).all().all()
    np.testing.assert_allclose(
        frame.prediction_epsilon,
        frame.prediction_mu - frame.prediction_a - frame.prediction_b + frame.prediction_wt,
        atol=1e-5, rtol=1e-5,
    )
    results = []
    for protein, group in frame.groupby("protein_id", sort=True):
        main = group.loc[group.main.fillna(False)]
        point = group.loc[np.isfinite(group.relative_point)]
        entry = {"protein_id": protein, "variants": len(group), "main_quartets": len(main), "main_eligible": len(main) >= 30}
        for name, subset in [
            ("epsilon", main),
            ("epsilon_margin025", main.loc[main.margin.ge(0.25)]),
            ("epsilon_margin050", main.loc[main.margin.ge(0.5)]),
            ("epsilon_all_point", group.loc[np.isfinite(group.epsilon)]),
        ]:
            entry[name + "_count"] = len(subset)
            entry[name + "_rmse"] = float(np.sqrt(np.mean((subset.prediction_epsilon - subset.epsilon) ** 2))) if len(subset) else float("nan")
        if len(main):
            squared = (main.prediction_epsilon - main.epsilon) ** 2
            entry["pair_macro_epsilon_rmse"] = float(squared.groupby(main.pair_id).mean().pow(0.5).mean())
            entry["epsilon_spearman"] = correlation(main.epsilon, main.prediction_epsilon, "spearman")
        else:
            entry["pair_macro_epsilon_rmse"] = entry["epsilon_spearman"] = float("nan")
        entry["phenotype_count"] = len(point)
        entry["phenotype_rmse"] = float(np.sqrt(np.mean((point.prediction_mu - point.relative_point) ** 2))) if len(point) else float("nan")
        entry["phenotype_spearman"] = correlation(point.relative_point, point.prediction_mu, "spearman")
        entry["phenotype_pearson"] = correlation(point.relative_point, point.prediction_mu, "pearson")
        interval = group.loc[group.relative_point.isna() & group.relative_lower.notna() & group.relative_upper.notna()]
        distances = np.maximum(np.maximum(interval.relative_lower - interval.prediction_mu, interval.prediction_mu - interval.relative_upper), 0.0)
        entry["interval_count"] = len(interval)
        entry["interval_distance_rmse"] = float(np.sqrt(np.mean(distances ** 2))) if len(interval) else float("nan")
        pool = group.loc[group.wt_all_point & group.wt_point_in_range & group.wt_point.le(4.5)].copy()
        entry["selection_count"] = len(pool)
        entry["selection_eligible"] = len(pool) >= 100
        entry["certain_beneficial_count"] = int(pool.relative_lower.ge(0.5).sum())
        entry["possible_beneficial_count"] = int(pool.relative_upper.ge(0.5).sum())
        ranked = pool.sort_values(["prediction_mu", "mutation_key"], ascending=[False, True], kind="stable")
        for count in [10, 30]:
            selected = ranked.iloc[:count]
            entry[f"precision{count}_lower"] = float(selected.relative_lower.ge(0.5).mean()) if len(pool) >= 100 else float("nan")
            entry[f"precision{count}_upper"] = float(selected.relative_upper.ge(0.5).mean()) if len(pool) >= 100 else float("nan")
            entry[f"selected{count}_point_count"] = int(selected.relative_point.notna().sum()) if len(pool) >= 100 else 0
        results.append(entry)
    return pd.DataFrame(results)
