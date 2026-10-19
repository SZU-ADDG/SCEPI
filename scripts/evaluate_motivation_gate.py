import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_single import ROOT
from epistasis.metrics import per_protein_metrics

ARMS = {
    "additive": "pair_grid_full_g0_f0_j1_n1",
    "global_nonlinear": "pair_grid_full_g0_f1_j1_n1",
    "interaction": "pair_grid_full_g0_f1_j1",
}
CONTROLS = ("additive", "global_nonlinear")
CONDITIONS = ("Z", "S128")
COMPARISON_METRICS = ("epsilon_rmse", "epsilon_margin050_rmse", "precision10_lower", "precision10_upper")


def select_models():
    selected = {}
    for arm, grid_name in ARMS.items():
        jobs = json.loads((ROOT / "runs/e3" / grid_name / "jobs.json").read_text())
        arm_runs = {}
        for fold in range(5):
            candidates = [job for job in jobs if job["threshold"] == 30 and job["fold"] == fold]
            assert len(candidates) == 12
            complete = []
            for job in candidates:
                run = ROOT / job["run"]
                manifest = json.loads((run / "manifest.json").read_text())
                assert manifest["complete"] and not manifest["smoke"]
                config = manifest["config"]
                assert config["threshold"] == 30 and config["fold"] == fold
                assert config["joint"] and config["flexible"] == (arm != "additive")
                assert config["no_interaction"] == (arm != "interaction")
                assert config["seed"] == 11 and config["objective"] == "combined"
                assert config["representation"] == "full" and not config["geometry"]
                complete.append((float(manifest["validation_score"]), job["run"]))
            score, run = min(complete, key=lambda item: (item[0], item[1]))
            arm_runs[str(fold)] = {"run": run, "validation_score": score}
        selected[arm] = arm_runs
    return selected


def audit_dataset():
    quartets = pd.read_parquet(
        ROOT / "data/processed/tsuboyama/quartets.parquet",
        columns=["quartet_id", "protein_id", "pair_id", "wt_id", "a_id", "b_id", "ab_id", "main", "all_point"],
    )
    variants = pd.read_parquet(ROOT / "data/processed/tsuboyama/variants.parquet", columns=["variant_id"])
    families = pd.read_parquet(
        ROOT / "data/splits/families_30.parquet",
        columns=["protein_id", "family_group", "outer_group", "pfam_annotated"],
    )
    assert quartets.quartet_id.is_unique
    component_columns = ["wt_id", "a_id", "b_id", "ab_id"]
    missing_components = int((~quartets[component_columns].isin(set(variants.variant_id))).to_numpy().sum())
    assert missing_components == 0
    main = quartets.loc[quartets.main & quartets.all_point]
    main_proteins = set(main.protein_id)
    main_families = families.loc[families.protein_id.isin(main_proteins)]
    return {
        "unique_quartets": int(len(quartets)),
        "missing_components": missing_components,
        "main_point_quartets": int(len(main)),
        "main_proteins": int(main.protein_id.nunique()),
        "main_site_pairs": int(main.pair_id.nunique()),
        "family_groups_total": int(families.family_group.nunique()),
        "family_groups_in_main": int(main_families.family_group.nunique()),
        "pfam_annotated_family_groups_in_main": int(
            main_families.loc[main_families.pfam_annotated].family_group.nunique()
        ),
        "max_outer_groups_per_family": int(families.groupby("family_group").outer_group.nunique().max()),
    }
def hierarchical_interval(frame, draws, seed):
    families = np.array(sorted(frame.family_group.unique()))
    members = {
        family: frame.loc[frame.family_group.eq(family), "benefit"].to_numpy(dtype=float)
        for family in families
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        sampled_values = []
        for family in sampled_families:
            values = members[family]
            sampled_values.extend(rng.choice(values, size=len(values), replace=True))
        samples[draw] = np.mean(sampled_values)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def build_metrics(selected):
    split = pd.read_parquet(ROOT / "data/splits/combination_splits.parquet")
    pair_test = split.loc[
        split.seed.eq(11) & split.scheme.eq("unseen_pair") & split["set"].eq("test"),
        ["variant_id", "protein_id"],
    ]
    assert not pair_test.variant_id.duplicated().any()
    records = []
    coverage = []
    for arm, folds in selected.items():
        for fold_text, choice in folds.items():
            fold = int(fold_text)
            run = ROOT / choice["run"]
            source = json.loads((run / "manifest.json").read_text())
            config = source["config"]
            assert config["threshold"] == 30 and config["fold"] == fold and source["seed"] == 11
            assert source["complete"] and not source["smoke"]
            evaluation = run / "evaluation_test"
            target = pd.read_parquet(evaluation / "target_permissions.parquet")
            assert target.index.name == "protein_id"
            target = target.reset_index()
            allowed = pair_test.loc[pair_test.protein_id.isin(target.protein_id)]
            allowed_ids = set(allowed.variant_id)
            assert allowed_ids
            for condition in CONDITIONS:
                predictions = pd.read_parquet(evaluation / f"shared_{condition}.parquet")
                assert not predictions.variant_id.duplicated().any()
                frame = predictions.loc[predictions.variant_id.isin(allowed_ids)].copy()
                assert set(frame.variant_id) == allowed_ids, (arm, fold, condition, len(frame), len(allowed_ids))
                protein_metrics = per_protein_metrics(frame)
                metadata = target[
                    ["protein_id", "family_group", "outer_group", "pfam_annotated", "available_single_labels"]
                ]
                protein_metrics = protein_metrics.merge(metadata, on="protein_id", validate="one_to_one")
                protein_metrics["budget_eligible"] = (
                    True if condition == "Z" else protein_metrics.available_single_labels.ge(128)
                )
                measurement_metrics = ["epsilon_rmse", "epsilon_margin050_rmse", "precision10_lower", "precision10_upper"]
                protein_metrics.loc[~protein_metrics.budget_eligible, measurement_metrics] = np.nan
                protein_metrics.loc[~protein_metrics.main_eligible, ["epsilon_rmse", "epsilon_margin050_rmse"]] = np.nan
                protein_metrics["arm"] = arm
                protein_metrics["fold"] = fold
                protein_metrics["condition"] = condition
                protein_metrics["test_variants"] = len(frame)
                records.append(protein_metrics)
                coverage.append({
                    "arm": arm,
                    "fold": fold,
                    "condition": condition,
                    "test_variants": len(frame),
                    "test_proteins": int(frame.protein_id.nunique()),
                    "main_eligible_proteins": int((protein_metrics.main_eligible & protein_metrics.budget_eligible).sum()),
                    "selection_eligible_proteins": int((protein_metrics.selection_eligible & protein_metrics.budget_eligible).sum()),
                    "main_quartets": int(protein_metrics.main_quartets.sum()),
                    "families": int(protein_metrics.family_group.nunique()),
                })
    return pd.concat(records, ignore_index=True), coverage


def compare_models(metrics, draws, seed):
    comparisons = []
    index = 0
    for condition in CONDITIONS:
        subset = metrics.loc[metrics.condition.eq(condition)]
        for control in CONTROLS:
            for metric in COMPARISON_METRICS:
                wide = subset.pivot(
                    index=["fold", "protein_id", "family_group"],
                    columns="arm",
                    values=metric,
                ).reset_index()
                wide = wide.replace([np.inf, -np.inf], np.nan)
                wide = wide.dropna(subset=["interaction", control]).copy()
                assert len(wide)
                if metric.startswith("epsilon_"):
                    wide["benefit"] = wide[control] - wide["interaction"]
                else:
                    wide["benefit"] = wide["interaction"] - wide[control]
                lower, upper = hierarchical_interval(wide, draws, seed + index)
                fold_benefits = {
                    str(int(fold)): float(group.benefit.mean())
                    for fold, group in wide.groupby("fold", sort=True)
                }
                target_wins = int(wide.benefit.gt(0).sum())
                target_losses = int(wide.benefit.lt(0).sum())
                target_ties = int(wide.benefit.eq(0).sum())
                comparisons.append({
                    "condition": condition,
                    "control": control,
                    "metric": metric,
                    "gain": float(wide.benefit.mean()),
                    "target_wins": target_wins,
                    "target_losses": target_losses,
                    "target_ties": target_ties,
                    "target_win_rate": target_wins / len(wide),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "proteins": int(wide.protein_id.nunique()),
                    "families": int(wide.family_group.nunique()),
                    "positive_folds": int(sum(value > 0 for value in fold_benefits.values())),
                    "folds_evaluated": int(len(fold_benefits)),
                    "fold_benefits": fold_benefits,
                })
                index += 1
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261009)
    args = parser.parse_args()
    assert args.bootstrap_draws >= 1000
    output = ROOT / "runs/motivation_gate"
    output.mkdir(parents=True, exist_ok=True)
    selected = select_models()
    selection_record = {
        "selection_scope": "Minimum source-validation score among 12 configurations for each arm and family30 fold.",
        "models": selected,
    }
    (output / "selected_runs.json").write_text(json.dumps(selection_record, indent=2))
    if args.select_only:
        print(json.dumps(selection_record, indent=2))
        return

    dataset_audit = audit_dataset()
    metrics, coverage = build_metrics(selected)
    s128 = metrics.loc[metrics.condition.eq("S128")]
    candidate_counts = s128.pivot(
        index=["fold", "protein_id"], columns="arm", values="selection_count"
    )
    candidate_eligibility = s128.pivot(
        index=["fold", "protein_id"], columns="arm", values="selection_eligible"
    )
    assert candidate_counts.notna().all().all()
    assert candidate_counts.nunique(axis=1).eq(1).all()
    assert candidate_eligibility.nunique(axis=1).eq(1).all()

    comparisons = compare_models(metrics, args.bootstrap_draws, args.bootstrap_seed)
    residual_rows = [
        row for row in comparisons
        if row["metric"] == "epsilon_rmse" and row["condition"] in CONDITIONS
    ]
    candidate_rows = [
        row for row in comparisons
        if row["metric"] == "precision10_lower" and row["condition"] == "S128"
    ]
    assert len(residual_rows) == 4 and len(candidate_rows) == 2
    residual_checks = [
        row["gain"] > 0 and row["folds_evaluated"] == 5 and row["positive_folds"] >= 4
        for row in residual_rows
    ]
    candidate_checks = [row["gain"] > 0 for row in candidate_rows]
    boundary_rows = [
        row for row in comparisons
        if row["metric"] == "epsilon_margin050_rmse"
        or (row["condition"] == "S128" and row["metric"] == "precision10_upper")
    ]
    boundary_checks = [row["gain"] > 0 for row in boundary_rows]
    gate = {
        "decision_criterion": {
            "residual": "Positive mean RMSE gain in Z and S128 against both controls, with at least 4/5 positive fold means; bootstrap intervals are descriptive and do not determine retention.",
            "candidate": "Positive mean S128 P@10 lower-bound gain against both controls.",
            "measurement_boundary": "Positive mean margin050 residual RMSE gain and positive mean S128 P@10 upper-bound gain against both controls.",
        },
        "residual_improvement_pass": all(residual_checks),
        "candidate_lower_bound_gain_pass": all(candidate_checks),
        "measurement_boundary_mean_gain_pass": all(boundary_checks),
        "retained": all(residual_checks) and all(candidate_checks) and all(boundary_checks),
        "residual_checks": residual_checks,
        "candidate_checks": candidate_checks,
        "boundary_checks": boundary_checks,
    }

    base = metrics.loc[metrics.arm.eq("interaction") & metrics.condition.eq("Z")]
    eligible = base.loc[base.main_eligible & base.epsilon_rmse.notna()]
    candidate = s128.loc[s128.arm.eq("interaction") & s128.selection_eligible & s128.budget_eligible]
    support = {
        "strict_test_variants": int(base.groupby("fold").test_variants.first().sum()),
        "strict_test_proteins": int(base.protein_id.nunique()),
        "budget128_eligible_proteins": int(base.available_single_labels.ge(128).sum()),
        "residual_eligible_proteins": int(eligible.protein_id.nunique()),
        "residual_families": int(eligible.family_group.nunique()),
        "candidate_eligible_proteins": int(candidate.protein_id.nunique()),
        "candidate_families": int(candidate.family_group.nunique()),
        "main_quartets": int(base.main_quartets.sum()),
        "margin050_quartets": int(base.epsilon_margin050_count.sum()),
    }
    summary = {
        "selection_scope": selection_record["selection_scope"],
        "evaluation_scope": "Seed11 unseen_pair test variants intersected with family30 held-out proteins; source test labels were not used for model selection.",
        "dataset_audit": dataset_audit,
        "bootstrap": {
            "method": "Resample families with replacement, then proteins within each sampled family with replacement.",
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
        },
        "support": support,
        "coverage_by_arm_fold_condition": coverage,
        "comparisons": comparisons,
        "gate": gate,
        "models": selected,
    }
    metrics.to_parquet(output / "per_protein_metrics.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# RECOMB-3 动机实验门槛",
        "",
        f"判定：{'通过' if gate['retained'] else '未通过'}",
        "",
        "评估集为 seed11 的 unseen_pair 测试位点对与 family30 留出蛋白的交集。模型配置只按五折源验证集分数选择。",
        "",
        "去留依据为严格测试集上的平均增益；95% 分层家族自助区间用于描述不确定性，不作为去留条件。残差改善还要求五折中至少四折平均方向为正。",
        "",
        f"主数据审计：{dataset_audit['unique_quartets']} 个唯一四元组，四组件缺失 {dataset_audit['missing_components']} 个；主要点值分析覆盖 {dataset_audit['main_point_quartets']} 个四元组、{dataset_audit['main_proteins']} 个蛋白背景和 {dataset_audit['main_site_pairs']} 个位点对。",
        f"家族覆盖：family30 共 {dataset_audit['family_groups_total']} 组；主分析涉及 {dataset_audit['family_groups_in_main']} 组，其中 {dataset_audit['pfam_annotated_family_groups_in_main']} 组有 Pfam 标注；每个家族组最多对应 {dataset_audit['max_outer_groups_per_family']} 个外层留出组。",
        f"测试覆盖：{support['strict_test_variants']} 个双突变体，{support['strict_test_proteins']} 个蛋白，S128 预算合格 {support['budget128_eligible_proteins']} 个；"
        f"残差可评蛋白 {support['residual_eligible_proteins']} 个、{support['residual_families']} 个家族；"
        f"候选可评蛋白 {support['candidate_eligible_proteins']} 个、{support['candidate_families']} 个家族。",
        "",
        "下表的增益为正表示相互作用模型更好。残差增益定义为对照 RMSE 减去相互作用 RMSE；"
        "P@10 增益定义为相互作用值减去对照值。区间使用分层家族自助法。",
        "逐蛋白比较按相同外层测试折配对。RMSE 较低、P@10 较高均计为相互作用模型获胜；指标相等计为平局。",
        "",
        "| 条件 | 对照模型 | 指标 | 平均增益 | 95% 区间 | target 胜/负/平 | 胜率 | 蛋白数 | 家族数 | 正向折数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['condition']} | {row['control']} | {row['metric']} | {row['gain']:.4f} | "
            f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] | "
            f"{row['target_wins']}/{row['target_losses']}/{row['target_ties']} | "
            f"{row['target_win_rate']:.1%} | {row['proteins']} | {row['families']} | "
            f"{row['positive_folds']}/{row['folds_evaluated']} |"
        )
    lines.extend([
        "",
        f"主残差门槛：{'通过' if gate['residual_improvement_pass'] else '未通过'}。"
        "要求 Z 与 S128 条件下相对两个对照模型的平均 RMSE 增益均为正，且至少四折方向一致。",
        f"保守候选收益门槛：{'通过' if gate['candidate_lower_bound_gain_pass'] else '未通过'}。"
        "要求 S128 的 P@10 下界平均增益相对两个对照模型均为正。",
        f"测量边界平均收益门槛：{'通过' if gate['measurement_boundary_mean_gain_pass'] else '未通过'}。"
        "要求 margin≥0.5 kcal/mol 的残差 RMSE 平均增益及 S128 P@10 上界平均增益相对两个对照模型均为正。",
        f"测量边界补充分析包括 margin≥0.5 kcal/mol 的残差 RMSE，以及 P@10 的上下界；"
        f"主分析含 {support['margin050_quartets']} 个高边界距离四元组。",
        "",
        "五折模型路径、逐蛋白指标、家族自助区间和覆盖计数保存在同目录的 JSON 与 parquet 文件中。",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"support": support, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
