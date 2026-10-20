import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/key_ablations_20261015/paper_evidence"
OUTPUT = ROOT / "paper/figures"
NOTE = ROOT / "work/conditioning_factorial_20261021/figure4/component_analysis_note.md"

SEEDS = [11, 23, 47, 89, 131]
PRIMARY = [
    "interaction",
    "additive",
    "global_nonlinear",
    "thermo_epistatic",
    "kermut",
    "stabilityarc_kermut",
    "mean_mlp",
    "mean_ridge",
]
PRIMARY_LABELS = {
    "interaction": "SCEPI",
    "additive": "Additive",
    "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D",
    "kermut": "Kermut",
    "stabilityarc_kermut": "StabilityArc + Kermut",
    "mean_mlp": "Mean MLP",
    "mean_ridge": "Mean ridge",
}
PRIMARY_SHORT = {
    "interaction": "SCEPI",
    "additive": "Additive",
    "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D",
    "kermut": "Kermut",
    "stabilityarc_kermut": "StabilityArc + Kermut",
    "mean_mlp": "Mean MLP",
    "mean_ridge": "Mean ridge",
}
PRIMARY_COLORS = {
    "interaction": "#2878B5",
    "thermo_epistatic": "#E68632",
    "kermut": "#32936F",
    "stabilityarc_kermut": "#3F9BA8",
    "global_nonlinear": "#B89B3C",
    "additive": "#6F7785",
    "mean_mlp": "#9368A6",
    "mean_ridge": "#AC6D79",
}

ABLATIONS = [
    "interaction",
    "remove_interaction",
    "direct_residual",
    "no_pair_measured",
    "no_cross",
    "reference_context",
    "identity_features",
    "phenotype_only",
    "residual_only",
]
ABLATION_LABELS = {
    "interaction": "SCEPI",
    "remove_interaction": "Pair removed",
    "direct_residual": "Direct head",
    "no_pair_measured": "Predicted singles",
    "no_cross": "Local context",
    "reference_context": "WT context",
    "identity_features": "Identity features",
    "phenotype_only": "Phenotype loss",
    "residual_only": "Residual loss",
}
ABLATION_DISPLAY_COLORS = {
    arm: ("#2878B5" if arm == "interaction" else "#7B858E")
    for arm in ABLATIONS
}

# factorial 实验在各项指标中使用同一组四状态，固定顺序以便并列读取残差和选择面板。
FACTORIAL_CELLS = ["interaction", "values_only", "mask_only", "no_pair_measured"]
FACTORIAL_LABELS = {
    "interaction": "SCEPI",
    "values_only": "Values only",
    "mask_only": "Indicators only",
    "no_pair_measured": "Predicted singles",
}
FACTORIAL_COLORS = {
    "interaction": "#2878B5",
    "values_only": "#9368A6",
    "mask_only": "#3F9BA8",
    "no_pair_measured": "#7B858E",
}
CONTRAST_ORDER = [
    "joint_input",
    "values_with_mask",
    "values_without_mask",
    "mask_with_values",
    "mask_without_values",
]
CONTRAST_LABELS = {
    "joint_input": "Joint input",
    "values_with_mask": "Values given indicators",
    "values_without_mask": "Values without indicators",
    "mask_with_values": "Indicators given values",
    "mask_without_values": "Indicators without values",
}
CONTRAST_COLORS = {
    "joint_input": "#2878B5",
    "values_with_mask": "#9368A6",
    "values_without_mask": "#B89B3C",
    "mask_with_values": "#3F9BA8",
    "mask_without_values": "#6F7785",
}
CONTRAST_ARMS = {
    "joint_input": ("interaction", "no_pair_measured"),
    "values_with_mask": ("interaction", "mask_only"),
    "values_without_mask": ("values_only", "no_pair_measured"),
    "mask_with_values": ("interaction", "values_only"),
    "mask_without_values": ("mask_only", "no_pair_measured"),
}
AUXILIARY_ARMS = ["interaction", "direct_residual", "remove_interaction"]
AUXILIARY_LABELS = {
    "interaction": "SCEPI",
    "direct_residual": "Direct head",
    "remove_interaction": "Pair removed",
}


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#D9DEE3", linewidth=0.45, alpha=0.76)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#8D969E")
        spine.set_linewidth(0.65)
    ax.tick_params(axis="both", colors="#27323A", labelsize=10, length=3.0, width=0.65, pad=2)


def factorial_run_from_input(path):
    """检测已完成 factorial 分析并返回运行根目录。"""
    path = Path(path)
    if path.name == "paper_evidence":
        candidate = path.parent
    else:
        candidate = path
    if (candidate / "factorial_summary.json").is_file() and (candidate / "paper_evidence").is_dir():
        return candidate
    return None


def load_factorial_data(run_root):
    """读取完成的 factorial 证据，拒绝部分完成的运行。"""
    evidence = run_root / "paper_evidence"
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert manifest["complete"]
    assert int(manifest["completed_evaluations"]) == int(manifest["expected_evaluations"])
    assert int(manifest["waiting_evaluations"]) == 0
    verification = json.loads((evidence / "integration_verification.json").read_text())
    assert verification["complete"]
    analysis = json.loads((run_root / "analysis_manifest.json").read_text())
    assert analysis["complete"]
    selection = pd.read_parquet(evidence / "selection_per_protein.parquet")
    residual = pd.read_parquet(evidence / "residual_per_protein.parquet")
    summary = json.loads((run_root / "factorial_summary.json").read_text())
    assert isinstance(summary, list) and summary
    assert set(FACTORIAL_CELLS).issubset(set(selection.arm.unique()))
    assert set(FACTORIAL_CELLS).issubset(set(residual.arm.unique()))
    required = {
        "threshold", "margin", "metric", "contrast", "mean_gain",
        "gain_interval_95", "proteins", "family_groups",
    }
    contrast_required = {"reference", "control", "reference_mean", "control_mean"}
    for row in summary:
        assert required.issubset(row)
        assert row["contrast"] in {"difference_of_conditional_gains", *CONTRAST_ORDER}
        assert len(row["gain_interval_95"]) == 2
        plotted_residual = row["contrast"] in CONTRAST_ARMS and row["metric"] == "epsilon_rmse"
        plotted_selection = (
            row["contrast"] in CONTRAST_ARMS
            and row["metric"] == "compensation_hit_rate"
            and row.get("scoring") == "native"
        )
        if plotted_residual or plotted_selection:
            assert contrast_required.issubset(row)
            assert (row["reference"], row["control"]) == CONTRAST_ARMS[row["contrast"]]
            assert len(row["repeat_gains"]) == 5
    return {
        "run_root": run_root,
        "evidence": evidence,
        "selection": selection,
        "residual": residual,
        "summary": summary,
        "manifest": manifest,
    }


def factorial_repeat_summary(frame, metric, threshold, margin, arms):
    """先在 protein 内平均，再返回五个匹配重复值。"""
    folded = fold_seed_protein_values(frame, metric, ["threshold", "margin", "arm"])
    selected = folded.loc[
        folded.threshold.eq(threshold)
        & folded.margin.eq(margin)
        & folded.arm.isin(arms)
    ].copy()
    assert selected.groupby(["arm", "protein_id"]).seed.nunique().eq(5).all()
    protein_sets = [set(selected.loc[selected.arm.eq(arm), "protein_id"]) for arm in arms]
    assert protein_sets and all(current == protein_sets[0] for current in protein_sets[1:])
    result = selected.groupby(["arm", "seed"], as_index=False)[metric].mean()
    assert result.groupby("arm").size().eq(5).all()
    return result


def factorial_metric_frame(data, threshold, margin, metric, scoring=None):
    if metric == "epsilon_rmse":
        frame = data["residual"].loc[
            data["residual"].threshold.eq(threshold)
            & data["residual"].margin.eq(margin)
            & data["residual"].arm.isin(FACTORIAL_CELLS)
        ].copy()
    elif metric == "compensation_hit_rate":
        frame = data["selection"].loc[
            data["selection"].threshold.eq(threshold)
            & data["selection"].margin.eq(margin)
            & data["selection"].task.eq("compensation")
            & data["selection"].unit.eq("protein")
            & data["selection"].budget.eq(10)
            & data["selection"].scoring.eq(scoring)
            & data["selection"].arm.isin(FACTORIAL_CELLS)
        ].copy()
    else:
        raise ValueError(metric)
    assert not frame.empty
    return frame


def factorial_summary_rows(data, threshold, margin, metric, scoring=None):
    rows = []
    for row in data["summary"]:
        if int(row["threshold"]) != int(threshold) or float(row["margin"]) != float(margin):
            continue
        if row["metric"] != metric:
            continue
        if scoring is None:
            assert "scoring" not in row or row["scoring"] is None
        elif row.get("scoring") != scoring:
            continue
        if row["contrast"] in CONTRAST_ORDER:
            rows.append(row)
    rows.sort(key=lambda row: CONTRAST_ORDER.index(row["contrast"]))
    assert [row["contrast"] for row in rows] == CONTRAST_ORDER
    return rows


def draw_factorial_metric(ax, values, metric, order, labels, colors, panel_label, unit_label, xlabel, xticks=None, xticklabels=None):
    """绘制四状态指标面板，每个状态显示五个重复点。"""
    positions = np.arange(len(order))
    extents = []
    for position, arm in enumerate(order):
        current = values.loc[values.arm.eq(arm), metric].sort_index().to_numpy(dtype=float)
        assert len(current) == 5 and np.isfinite(current).all()
        jitter = np.linspace(-0.13, 0.13, len(current))
        color = colors[arm]
        ax.scatter(
            current,
            np.full(len(current), positions[position]) + jitter,
            s=22.0,
            facecolors="white",
            edgecolors=color,
            linewidths=0.85,
            zorder=3,
        )
        mean = float(current.mean())
        sd = float(current.std(ddof=1))
        ax.errorbar(
            mean,
            positions[position],
            xerr=sd,
            fmt="none",
            ecolor=color,
            elinewidth=1.3,
            capsize=2.1,
            capthick=0.7,
            zorder=4,
        )
        ax.scatter(
            [mean],
            [positions[position]],
            marker="D",
            s=25,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=5,
        )
        extents.extend([float(current.min()), float(current.max()), mean - sd, mean + sd])
    minimum, maximum = min(extents), max(extents)
    span = max(maximum - minimum, 0.05 if metric == "epsilon_rmse" else 0.005)
    padding = max(0.10 * span, 0.004 if metric == "epsilon_rmse" else 0.0006)
    ax.set_xlim(minimum - padding, maximum + padding)
    ax.set_yticks(positions)
    ax.set_yticklabels([labels[arm] for arm in order])
    ax.set_ylim(-0.55, len(order) - 0.45)
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel, fontsize=9.5, labelpad=4)
    style_axis(ax)
    ax.tick_params(axis="y", labelsize=9.0)
    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.99, 1.08, unit_label, transform=ax.transAxes, fontsize=9.5, ha="right", va="top", color="#4A555D")


def draw_contrast_panel(ax, rows, panel_label, xlabel, unit_label, scale=1.0, show_y_labels=True, xticks=None, xticklabels=None):
    """绘制五个匹配重复增益及其 mean ± SD。"""
    positions = np.arange(len(rows))
    ax.axvline(0.0, color="#46515A", linewidth=0.85, linestyle=(0, (3, 2)), zorder=1)
    extents = [0.0]
    for position, row in enumerate(rows):
        contrast = row["contrast"]
        color = CONTRAST_COLORS[contrast]
        repeats = np.asarray(list(row["repeat_gains"].values()), dtype=float) * scale
        mean = float(repeats.mean())
        sd = float(repeats.std(ddof=1))
        assert len(repeats) == 5 and np.isfinite(repeats).all()
        assert np.isfinite(mean) and np.isfinite(sd)
        y_jitter = np.linspace(-0.12, 0.12, len(repeats))
        ax.scatter(
            repeats,
            np.full(len(repeats), position) + y_jitter,
            s=20.0,
            facecolors="white",
            edgecolors=color,
            linewidths=0.8,
            zorder=3,
        )
        ax.errorbar(
            mean,
            position,
            xerr=sd,
            fmt="none",
            ecolor=color,
            elinewidth=1.4,
            capsize=2.1,
            capthick=0.7,
            zorder=4,
        )
        ax.scatter(
            [mean],
            [position],
            marker="D",
            s=25,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=5,
        )
        extents.extend([float(repeats.min()), float(repeats.max()), mean - sd, mean + sd])
    minimum, maximum = min(extents), max(extents)
    span = max(maximum - minimum, 0.05 * scale)
    padding = max(0.12 * span, 0.006 * scale)
    ax.set_xlim(minimum - padding, maximum + padding)
    ax.set_yticks(positions)
    if show_y_labels:
        ax.set_yticklabels([CONTRAST_LABELS[row["contrast"]] for row in rows])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    if xticks is not None:
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel, fontsize=9.5, labelpad=4)
    style_axis(ax)
    ax.tick_params(axis="y", labelsize=8.4)
    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.99, 1.08, unit_label, transform=ax.transAxes, fontsize=9.5, ha="right", va="top", color="#4A555D")


def write_factorial_note(data, threshold, margin, output_note):
    evidence = data["evidence"]
    residual = factorial_metric_frame(data, threshold, margin, "epsilon_rmse")
    selection = factorial_metric_frame(data, threshold, margin, "compensation_hit_rate", "native")
    residual_values = factorial_repeat_summary(residual, "epsilon_rmse", threshold, margin, FACTORIAL_CELLS)
    selection_values = factorial_repeat_summary(selection, "compensation_hit_rate", threshold, margin, FACTORIAL_CELLS)
    residual_rows = []
    for arm in FACTORIAL_CELLS:
        current = residual_values.loc[residual_values.arm.eq(arm), "epsilon_rmse"]
        proteins = int(residual.loc[residual.arm.eq(arm), "protein_id"].nunique())
        residual_rows.append(f"| {FACTORIAL_LABELS[arm]} (`{arm}`) | {current.mean():.6f} | {current.std(ddof=1):.6f} | 5 | {proteins} |")
    selection_rows = []
    for arm in FACTORIAL_CELLS:
        current = selection_values.loc[selection_values.arm.eq(arm), "compensation_hit_rate"]
        proteins = int(selection.loc[selection.arm.eq(arm), "protein_id"].nunique())
        selection_rows.append(f"| {FACTORIAL_LABELS[arm]} (`{arm}`) | {100 * current.mean():.4f} | {100 * current.std(ddof=1):.4f} | 5 | {proteins} |")
    lines = [
        "# Figure 4 factorial evidence",
        "",
        f"- Source run: `{data['run_root']}`.",
        f"- Evidence tables: `{evidence}/residual_per_protein.parquet` and `{evidence}/selection_per_protein.parquet`.",
        f"- Filter: family{threshold}, margin={margin:g}; residual metric `epsilon_rmse`; native compensation selection with `budget=10`.",
        "- Each repeat value averages the assigned held-out fold within protein and then averages proteins equally. Points show the five matched training repeats; diamonds and whiskers show the repeat mean and standard deviation.",
        "- Panels (c) and (d) show the five repeat gains as open points, with the repeat mean and standard deviation as the diamond and whisker. The declared homology-group bootstrap intervals remain in `factorial_summary.json` and are retained as evidence without being drawn in the main figure.",
        "- A positive residual gain is `control RMSE - state RMSE`. A positive selection gain is `state hit rate - control hit rate`.",
        "",
        "## Panel a: four input states, residual RMSE",
        "",
        "| State | Mean RMSE | Repeat SD | Repeats | Proteins |",
        "|---|---:|---:|---:|---:|",
        *residual_rows,
        "",
        "## Panel b: four input states, native top-10 hit rate",
        "",
        "| State | Mean hit rate (%) | Repeat SD (percentage points) | Repeats | Proteins |",
        "|---|---:|---:|---:|---:|",
        *selection_rows,
        "",
        "## Conditional contrast fields",
        "",
        "- Required rows: `threshold`, `margin`, `metric`, `scoring`, `contrast`, `reference`, `control`, `reference_mean`, `control_mean`, `mean_gain`, `gain_interval_95`, `repeat_gains`, `proteins`, and `family_groups`.",
        "- Plotted contrasts: `joint_input`, `values_with_mask`, `values_without_mask`, `mask_with_values`, and `mask_without_values`.",
        "",
        f"Generated artifacts: `{output_note.parent}/component_analysis.pdf`, `.svg`, `.png`, and `.tikz`.",
    ]
    output_note.parent.mkdir(parents=True, exist_ok=True)
    output_note.write_text("\n".join(lines) + "\n")


def load_data():
    selection = pd.read_parquet(INPUT / "selection_per_protein.parquet")
    residual = pd.read_parquet(INPUT / "residual_per_protein.parquet")
    expected_arms = {
        "interaction", "additive", "global_nonlinear", "remove_interaction",
        "frozen_main", "identity_mapping", "reference_context", "identity_features",
        "no_cross", "residual_only", "phenotype_only", "geometry",
        "geometry_permuted_retrained", "geometry_permuted_inference",
        "geometry_rewired_inference", "geometry_rewired_retrained", "esm2_additive",
        "esm2_context", "mean_ridge", "mean_mlp", "kermut", "thermo_additive",
        "thermo_epistatic", "stabilityarc_kermut", "direct_residual", "no_pair_measured",
    }
    assert set(selection.arm.unique()) == expected_arms
    assert set(residual.arm.unique()) == expected_arms
    selection = selection.loc[selection.threshold.eq(30)].copy()
    residual = residual.loc[residual.threshold.eq(30)].copy()
    assert set(selection.threshold.unique()) == {30}
    assert set(residual.threshold.unique()) == {30}
    assert set(selection.seed.unique()) == set(SEEDS)
    assert set(residual.seed.unique()) == set(SEEDS)
    assert set(selection.condition.dropna().unique()) == {"Sall"}
    assert set(residual.condition.dropna().unique()) == {"Sall"}
    assert selection.loc[selection.condition.isna(), "arm"].isin({"stabilityarc_kermut"}).all()
    assert residual.loc[residual.condition.isna(), "arm"].isin({"stabilityarc_kermut"}).all()
    return selection, residual


def fold_seed_protein_values(frame, value, keys):
    expected = [*keys, "protein_id", "seed", "fold"]
    folded = frame.groupby(expected, as_index=False)[value].mean()
    assert folded.groupby([*keys, "protein_id", "seed"]).size().eq(1).all()
    return folded.groupby([*keys, "protein_id", "seed"], as_index=False)[value].mean()


def fold_seed_protein_mean(frame, value, keys):
    seed_values = fold_seed_protein_values(frame, value, keys)
    assert seed_values.groupby([*keys, "protein_id"]).seed.nunique().eq(5).all()
    return seed_values.groupby([*keys, "protein_id"], as_index=False)[value].mean()


def primary_selection_summary(selection):
    frame = selection.loc[
        selection.task.eq("compensation")
        & selection.scoring.eq("observed_main")
        & selection.budget.eq(10)
        & selection.arm.isin(PRIMARY)
    ].copy()
    assert set(frame.margin.unique()) == {0.0, 0.25, 0.5}
    result = fold_seed_protein_mean(frame, "compensation_hit_rate", ["arm", "margin"])
    assert result.groupby(["arm", "margin"]).size().to_dict() == {
        (arm, margin): (115 if margin == 0.5 else 120)
        for arm in PRIMARY
        for margin in [0.0, 0.25, 0.5]
    }
    return result


def ablation_residual_summary(residual):
    frame = residual.loc[residual.margin.eq(0.0) & residual.arm.isin(ABLATIONS)].copy()
    result = fold_seed_protein_mean(frame, "epsilon_rmse", ["arm", "margin"])
    assert result.groupby("arm").size().eq(155).all()
    return result


def ablation_selection_summary(selection):
    frame = selection.loc[
        selection.task.eq("compensation")
        & selection.scoring.eq("native")
        & selection.margin.eq(0.0)
        & selection.budget.eq(10)
        & selection.arm.isin(ABLATIONS)
    ].copy()
    result = fold_seed_protein_mean(frame, "compensation_hit_rate", ["arm", "margin"])
    assert result.groupby("arm").size().eq(120).all()
    return result


def repeat_macro(values, metric, group_keys):
    result = values.groupby([*group_keys, "seed"], as_index=False)[metric].mean()
    assert result.groupby([*group_keys, "seed"]).size().eq(1).all()
    assert result.groupby(group_keys).seed.nunique().eq(5).all()
    return result


def draw_repeat_summary(ax, values, order, labels, colors, metric, xlabel, panel_label, unit_label, show_y_labels=True):
    positions = np.arange(len(order))
    extents = []
    for position, arm in enumerate(order):
        current = values.loc[values.arm.eq(arm), metric].to_numpy(dtype=float)
        assert len(current) == 5
        jitter = np.linspace(-0.12, 0.12, len(current))
        ax.scatter(
            current,
            np.full(len(current), positions[position]) + jitter,
            s=24.0,
            facecolors="none",
            edgecolors=colors[arm],
            linewidths=0.82,
            alpha=0.95,
            zorder=2,
        )
        mean = float(np.mean(current))
        sd = float(np.std(current, ddof=1))
        ax.errorbar(
            mean,
            positions[position],
            xerr=sd,
            fmt="none",
            ecolor=colors[arm],
            elinewidth=1.3,
            capsize=2.0,
            capthick=0.65,
            zorder=3,
        )
        ax.scatter(
            [mean],
            [positions[position]],
            marker="D",
            s=24,
            facecolor="white",
            edgecolor=colors[arm],
            linewidth=0.95,
            zorder=4,
        )
        extents.extend([float(np.min(current)), float(np.max(current)), mean - sd, mean + sd])
    minimum = min(extents)
    maximum = max(extents)
    span = max(maximum - minimum, 0.05)
    padding = max(0.08 * span, 0.006)
    ax.set_xlim(minimum - padding, maximum + padding)
    ax.set_yticks(positions)
    if show_y_labels:
        ax.set_yticklabels([labels[arm] for arm in order])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=4)
    style_axis(ax)
    ax.text(-0.14, 1.08, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.99, 1.08, unit_label, transform=ax.transAxes, fontsize=10, ha="right", va="top", color="#4A555D")


def draw_boundary(ax, values):
    margins = [0.0, 0.25, 0.5]
    x = np.arange(len(margins))
    for arm in PRIMARY:
        current = values.loc[values.arm.eq(arm)].groupby("margin", as_index=False).compensation_hit_rate.mean()
        current = current.set_index("margin").reindex(margins)
        assert current.compensation_hit_rate.notna().all()
        ax.plot(
            x,
            current.compensation_hit_rate.to_numpy(dtype=float),
            color=PRIMARY_COLORS[arm],
            linewidth=1.45 if arm == "interaction" else 0.9,
            marker="o",
            markersize=4.2 if arm == "interaction" else 3.3,
            markerfacecolor="white",
            markeredgecolor=PRIMARY_COLORS[arm],
            markeredgewidth=0.72,
            label=PRIMARY_LABELS[arm],
            zorder=4 if arm == "interaction" else 3,
        )
    ax.set_xticks(x, ["0.00\n$n=120$", "0.25\n$n=120$", "0.50\n$n=115$"])
    ax.set_xlabel("Boundary margin (kcal/mol)", fontsize=10, labelpad=10)
    ax.set_ylabel("Hit rate", fontsize=10, labelpad=4)
    ax.set_ylim(0.0, 0.14)
    ax.set_xlim(-0.20, 2.20)
    style_axis(ax)
    ax.text(-0.14, 1.08, "b.", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.99, 1.08, "compensation, $K=10$", transform=ax.transAxes, fontsize=10, ha="right", va="top", color="#4A555D")


def repeat_stats(summary, value):
    rows = []
    for arm in summary.arm.unique():
        current = summary.loc[summary.arm == arm, value]
        rows.append({
            "arm": arm,
            "mean": float(current.mean()),
            "sd": float(current.std(ddof=1)),
            "repeats": int(current.size),
        })
    return pd.DataFrame(rows)


def boundary_stats(summary):
    rows = []
    for margin in [0.0, 0.25, 0.5]:
        for arm in PRIMARY:
            current = summary.loc[(summary.arm == arm) & (summary.margin == margin), "compensation_hit_rate"]
            rows.append({
                "arm": arm,
                "margin": margin,
                "mean": float(current.mean()),
                "sd": float(current.std(ddof=1)),
                "repeats": int(current.size),
            })
    return pd.DataFrame(rows)


def write_note(primary_protein, primary_repeat, residual_protein, residual_repeat, native_protein, native_repeat, selection, residual_raw):
    primary_rows = repeat_stats(primary_repeat.loc[primary_repeat.margin.eq(0.0)], "compensation_hit_rate")
    boundary_rows = boundary_stats(primary_repeat)
    residual_rows = repeat_stats(residual_repeat, "epsilon_rmse")
    native_rows = repeat_stats(native_repeat, "compensation_hit_rate")
    groups = selection.loc[
        selection.task.eq("compensation")
        & selection.scoring.eq("observed_main")
        & selection.budget.eq(10)
        & selection.arm.eq("interaction")
    ].groupby("margin").family_group.nunique()
    residual_groups = residual_raw.loc[residual_raw.margin.eq(0.0) & residual_raw.arm.eq("interaction")].family_group.nunique()
    native_groups = selection.loc[
        selection.task.eq("compensation")
        & selection.scoring.eq("native")
        & selection.margin.eq(0.0)
        & selection.budget.eq(10)
        & selection.arm.eq("interaction")
    ].family_group.nunique()
    main_group_counts = [int(groups.get(margin, 0)) for margin in [0.0, 0.25, 0.5]]
    main_protein_counts = [
        int(primary_protein.loc[primary_protein.margin.eq(margin)].protein_id.nunique())
        for margin in [0.0, 0.25, 0.5]
    ]
    input_label = INPUT.relative_to(ROOT).as_posix()
    manifest_path = INPUT / "manifest.json"
    assert manifest_path.is_file()
    input_manifest = json.loads(manifest_path.read_text())
    expected_evaluations = int(input_manifest["expected_evaluations"])
    completed_evaluations = int(input_manifest["completed_evaluations"])
    waiting_evaluations = int(input_manifest["waiting_evaluations"])
    assert completed_evaluations + waiting_evaluations == expected_evaluations
    completed_grids = input_manifest.get("completed_grids", [])
    primary_figure_input = input_label == "runs/key_ablations_20261015/paper_primary_figures"
    if primary_figure_input:
        assert input_manifest["complete"]
        assert (expected_evaluations, completed_evaluations, waiting_evaluations) == (650, 650, 0)
        assert len(completed_grids) == 26
        assert all(int(row["threshold"]) == 30 and int(row["evaluations"]) == 25 for row in completed_grids)
        figure_scope = "family30 primary grouping only; this figure does not certify the two-threshold key-ablation campaign."
    else:
        figure_scope = "The plotted rows are filtered to threshold 30; the input manifest may also record other thresholds or waiting jobs."
    grid_scope = ", ".join(
        f"{row['arm']}@threshold{row['threshold']}={row['evaluations']}"
        for row in completed_grids
    )

    lines = [
        "# Components figure evidence",
        "",
        "The figure reports common-main-effect and measurement-boundary comparisons together with pair-related ablations under one equal-protein macro-average protocol. The ablation panels retain the measured values of every complete control, including controls that exceed the shared model on a displayed metric.",
        "",
        "## Artifact and input",
        "",
        f"- Source tables: `{input_label}/selection_per_protein.parquet` and `{input_label}/residual_per_protein.parquet`.",
        f"- Scope: {figure_scope}",
        f"- Aggregation manifest: `{input_label}/manifest.json`; complete={input_manifest['complete']}, completed evaluations={completed_evaluations}/{expected_evaluations}, waiting evaluations={waiting_evaluations}, completed grids={len(completed_grids)}.",
        f"- Completed-grid coverage recorded by the manifest: {grid_scope}.",
        f"- Manifest source run: `{input_manifest.get('run_path', 'unspecified')}`; external sources: {', '.join(input_manifest.get('external_sources', [])) or 'none'}.",
        "- Plotted scope: threshold 30, condition `Sall`, complete target double-mutant evaluation, five folds and five training repetitions per arm.",
        "- No site-pair test mask is applied. Selection uses `task=compensation`, `budget=10`, and `unit=protein`.",
        "- Each protein belongs to one held-out outer fold. For each seed, the metric from that protein's assigned test fold contributes one protein-level value; the five-seed repeat value is the equal-protein macro mean. Panels (a), (c), and (d) show the five repeat values as open points, with a diamond and whisker for the mean and standard deviation across repeats. Panel (b) shows the equal-protein means at each boundary margin; the seed and fold identifiers are omitted from the graphic.",
        "- Panel (b) is the only panel with a method-color legend. Panels (c) and (d) use neutral gray for all controls and blue for SCEPI; row labels identify each ablation.",
        "- The vector and raster outputs are `paper/figures/component_analysis.pdf`, `paper/figures/component_analysis.svg`, and `paper/figures/component_analysis.png`.",
        "",
        "## Panel cohorts",
        "",
        f"- Panels (a) and (b): fixed `observed_main`, `K=10` compensation selection. Margins 0.00, 0.25, and 0.50 contain {main_protein_counts[0]}, {main_protein_counts[1]}, and {main_protein_counts[2]} proteins, with {main_group_counts[0]}, {main_group_counts[1]}, and {main_group_counts[2]} homology groups, respectively. The panel (a) values use the margin 0.00 cohort.",
        f"- Panel (c): residual RMSE at margin 0.00, 155 proteins and {residual_groups} homology groups for every displayed arm.",
        f"- Panel (d): native compensation hit rate at margin 0.00 and `K=10`, {int(native_protein.loc[native_protein.arm.eq('interaction')].protein_id.nunique())} proteins and {native_groups} homology groups for every displayed arm.",
        "",
        "## Panel (a): fixed observed-main comparison",
        "",
        "| Arm | Mean hit rate | Across-repeat SD | Repeats | Proteins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary_rows.itertuples(index=False):
        proteins = int(primary_protein.loc[(primary_protein.arm == row.arm) & (primary_protein.margin == 0.0)].protein_id.nunique())
        lines.append(f"| {PRIMARY_LABELS[row.arm]} (`{row.arm}`) | {row.mean:.6f} | {row.sd:.6f} | {row.repeats} | {proteins} |")
    lines += [
        "",
        "## Panel (b): measurement-boundary sensitivity",
        "",
        "| Arm | Margin 0.00 | Margin 0.25 | Margin 0.50 |",
        "|---|---:|---:|---:|",
    ]
    for arm in PRIMARY:
        cells = []
        for margin in [0.0, 0.25, 0.5]:
            row = boundary_rows.loc[(boundary_rows.arm == arm) & (boundary_rows.margin == margin)].iloc[0]
            protein_n = int(primary_protein.loc[(primary_protein.arm == arm) & (primary_protein.margin == margin)].protein_id.nunique())
            cells.append(f"{row['mean']:.6f} (n={protein_n}, SD={row['sd']:.6f})")
        lines.append(f"| {PRIMARY_LABELS[arm]} (`{arm}`) | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Panel (c): residual-error ablations",
        "",
        "| Arm | Residual RMSE mean | Across-repeat SD | Repeats | Proteins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in residual_rows.itertuples(index=False):
        proteins = int(residual_protein.loc[residual_protein.arm == row.arm].protein_id.nunique())
        lines.append(f"| {ABLATION_LABELS[row.arm]} (`{row.arm}`) | {row.mean:.6f} | {row.sd:.6f} | {row.repeats} | {proteins} |")
    lines += [
        "",
        "## Panel (d): native compensation ablations",
        "",
        "| Arm | Native `K=10` hit rate mean | Across-repeat SD | Repeats | Proteins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in native_rows.itertuples(index=False):
        proteins = int(native_protein.loc[native_protein.arm == row.arm].protein_id.nunique())
        lines.append(f"| {ABLATION_LABELS[row.arm]} (`{row.arm}`) | {row.mean:.6f} | {row.sd:.6f} | {row.repeats} | {proteins} |")
    lines += [
        "",
        "## Arm identity and interpretation",
        "",
        "- `interaction` is SCEPI, the shared flexible mapping and jointly optimized interaction model.",
        "- `remove_interaction` is the inference control that zeros the learned pair interaction while keeping the trained model and mapping. It is the pair-removed control.",
        "- `no_cross` removes cross-endpoint contextual features from the pair representation.",
        "- `reference_context` is the WT-context representation (`representation=wt`). It retains WT context and sequence identity while removing mutation delta features.",
        "- `identity_features` uses identity-only pair features (`representation=identity`).",
        "- `phenotype_only` trains with the phenotype likelihood objective only; `residual_only` trains with the residual objective only. These are training-loss ablations.",
        "- `direct_residual` uses a same-capacity symmetric direct residual function $r=F(a,b)$ with the shared training recipe.",
        "- `no_pair_measured`: $F$ receives model-predicted single effects and zero observation indicators. The external calibration mechanism is retained, and training weights are independently fitted.",
        "- The source table also contains `identity_mapping`, `frozen_main`, geometry controls, ESM2 controls, `thermo_additive`, and the target baselines. They remain in the evidence table and are excluded from this compact component figure because their roles belong to mapping, frozen-single, structural-intervention, representation-pretraining, or external-baseline analyses. `stabilityarc_kermut` is included in panels (a) and (b) as the external StabilityArc + Kermut comparison with Kermut target calibration under the same selection protocol.",
        "",
        "## Caption draft",
        "",
        "**Component and measurement-boundary analysis on the full S-all target.** (a) Five-repeat compensation hit rates at a fixed observed-main-effect score and candidate budget of 10. (b) The same comparison after requiring each complete quartet to remain at least 0, 0.25, or 0.50 kcal/mol from the assay boundary; cohort sizes are shown under the x-axis. (c) Five-repeat residual RMSE for pair-removal, direct-residual, pair-availability, representation, and training-loss controls. (d) Five-repeat native top-10 compensation hit rates for the same controls. Open points in (a), (c), and (d) are repeat-level equal-protein means; diamonds and horizontal whiskers show the mean and standard deviation across the five repeats. Panel (b) uses the method legend; rows in (c) and (d) identify their corresponding controls. Each protein contributes once per seed from its assigned held-out outer fold.",
    ]
    NOTE.write_text("\n".join(lines) + "\n")


def tex_number(value):
    return f"{float(value):.12f}".rstrip("0").rstrip(".") or "0"


def tikz_axis_start(x, y, xmin, xmax, ymin, ymax, xticks, yticklabels, xlabel, ylabel="", xticklabels=None, ytick_values=None, extra_options=None):
    if ytick_values is None:
        ytick_values = list(range(len(yticklabels)))
    if extra_options is None:
        extra_options = []
    return [
        "\\begin{axis}[",
        f"at={{({x}mm,{y}mm)}}, anchor=south west,",
        "width=49mm, height=42mm,",
        "scale only axis,",
        f"xmin={tex_number(xmin)}, xmax={tex_number(xmax)}, ymin={tex_number(ymin)}, ymax={tex_number(ymax)},",
        "axis line style={draw=gray!65},",
        "grid=major, grid style={draw=gray!20, line width=0.35pt},",
        "tick label style={font=\\fontsize{10}{12}\\selectfont, text=black!68},",
        "label style={font=\\fontsize{10}{12}\\selectfont, text=black!78},",
        "xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},",
        f"xtick={{{','.join(tex_number(value) for value in xticks)}}},",
        f"ytick={{{','.join(tex_number(value) for value in ytick_values)}}},",
        f"yticklabels={{{','.join(yticklabels)}}},",
        "yticklabel style={font=\\fontsize{10}{12}\\selectfont, text=black!78},",
        f"xlabel={{{xlabel}}}, ylabel={{{ylabel}}},",
        "clip=false,",
        *([f"xticklabels={{{','.join(xticklabels)}}},"] if xticklabels is not None else []),
        *extra_options,
        "]",
    ]


def add_repeat_panel(lines, values, order, labels, colors, metric, axis, panel_label, title, xlabel, xmin, xmax, xticks, show_y_labels=True, panel_label_x=-0.18):
    panel_values = values.loc[values.arm.isin(order), metric].to_numpy(dtype=float)
    assert len(panel_values) == len(order) * 5
    assert np.isfinite(panel_values).all()
    assert panel_values.min() >= xmin and panel_values.max() <= xmax
    yticklabels = [labels[arm] for arm in order] if show_y_labels else [""] * len(order)
    lines.extend(tikz_axis_start(*axis, xmin, xmax, -0.55, len(order) - 0.55, xticks, yticklabels, xlabel))
    lines.append(f"\\node[anchor=south west, font=\\bfseries\\fontsize{{10}}{{12}}\\selectfont] at (rel axis cs:{panel_label_x},1.04) {{{panel_label}}};")
    lines.append(f"\\node[anchor=south east, font=\\fontsize{{10}}{{12}}\\selectfont, text=black!60] at (rel axis cs:1,1.04) {{{title}}};")
    for position, arm in enumerate(order):
        current = values.loc[values.arm.eq(arm)].sort_values("seed")[metric].to_numpy(dtype=float)
        assert len(current) == 5
        mean = float(current.mean())
        sd = float(current.std(ddof=1))
        color = colors[arm]
        points = " ".join(f"({tex_number(value)},{position})" for value in current)
        lo = tex_number(mean - sd)
        hi = tex_number(mean + sd)
        mean_text = tex_number(mean)
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{lo},{position}) -- (axis cs:{hi},{position});")
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{lo},{position - 0.10}) -- (axis cs:{lo},{position + 0.10});")
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{hi},{position - 0.10}) -- (axis cs:{hi},{position + 0.10});")
        lines.append(f"\\addplot[only marks, mark=o, mark size=2.1pt, color={color}, mark options={{draw={color}, fill=white}}] coordinates {{{points}}};")
        lines.append(f"\\addplot[only marks, mark=diamond*, mark size=2.7pt, color={color}, mark options={{draw={color}, fill=white}}] coordinates {{({mean_text},{position})}};")
    lines.append("\\end{axis}")


def add_boundary_panel(lines, values):
    primary_colors = {
        "interaction": "SharedBlue",
        "additive": "AdditiveGray",
        "global_nonlinear": "GlobalGold",
        "thermo_epistatic": "ThermoOrange",
        "kermut": "KermutGreen",
        "stabilityarc_kermut": "StabilityArcTeal",
        "mean_mlp": "MeanPurple",
        "mean_ridge": "RidgeRose",
    }
    lines.extend(tikz_axis_start(
        108, 58, -0.20, 2.20, 0, 0.14, [0, 1, 2], ["0.00", "0.05", "0.10"],
        "Boundary margin (kcal/mol)", "Hit rate",
        xticklabels=["0.00", "0.25", "0.50"],
        ytick_values=[0, 0.05, 0.10],
        extra_options=[
            "legend to name=componentanalysislegend,",
            "legend style={draw=none, font=\\fontsize{10}{12}\\selectfont, legend columns=4, column sep=3pt, row sep=0pt},",
            "legend cell align=left,",
            "xlabel style={yshift=-6pt},",
        ],
    ))
    lines.append("\\node[anchor=south west, font=\\bfseries\\fontsize{10}{12}\\selectfont] at (rel axis cs:0.02,1.04) {b.};")
    lines.append("\\node[anchor=south east, font=\\fontsize{10}{12}\\selectfont, text=black!60] at (rel axis cs:1,1.04) {compensation, $K=10$};")
    lines.append("\\addplot[forget plot, draw=none] coordinates {(0,0)};")
    for arm in PRIMARY:
        current = values.loc[values.arm.eq(arm)].groupby("margin", as_index=False).compensation_hit_rate.mean().set_index("margin").reindex([0.0, 0.25, 0.5])
        points = " ".join(f"({index},{tex_number(value)})" for index, value in enumerate(current.compensation_hit_rate.to_numpy(dtype=float)))
        color = primary_colors[arm]
        width = "1.25pt" if arm == "interaction" else "0.75pt"
        size = "2.5pt" if arm == "interaction" else "2.0pt"
        lines.append(f"\\addplot[color={color}, line width={width}, mark=o, mark size={size}, mark options={{draw={color}, fill=white}}] coordinates {{{points}}};")
        legend_label = PRIMARY_LABELS[arm]
        lines.append(f"\\addlegendentry{{{legend_label}}}")
    for index, count in enumerate([120, 120, 115]):
        lines.append(f"\\node[anchor=north, font=\\fontsize{{10}}{{12}}\\selectfont] at (axis cs:{index},-0.015) {{$n={count}$}};")
    lines.append("\\end{axis}")


def write_tikz(primary_protein, primary_repeat, residual_protein, residual_repeat, native_protein, native_repeat):
    primary_colors = {
        "interaction": "SharedBlue",
        "additive": "AdditiveGray",
        "global_nonlinear": "GlobalGold",
        "thermo_epistatic": "ThermoOrange",
        "kermut": "KermutGreen",
        "stabilityarc_kermut": "StabilityArcTeal",
        "mean_mlp": "MeanPurple",
        "mean_ridge": "RidgeRose",
    }
    ablation_colors = {arm: ("SharedBlue" if arm == "interaction" else "NeutralGray") for arm in ABLATIONS}
    lines = [
        "\\begin{tikzpicture}[font=\\sffamily\\fontsize{10}{12}\\selectfont]",
        "\\definecolor{SharedBlue}{HTML}{2878B5}",
        "\\definecolor{ThermoOrange}{HTML}{E68632}",
        "\\definecolor{KermutGreen}{HTML}{32936F}",
        "\\definecolor{StabilityArcTeal}{HTML}{3F9BA8}",
        "\\definecolor{GlobalGold}{HTML}{B89B3C}",
        "\\definecolor{AdditiveGray}{HTML}{6F7785}",
        "\\definecolor{MeanPurple}{HTML}{9368A6}",
        "\\definecolor{RidgeRose}{HTML}{AC6D79}",
        "\\definecolor{NeutralGray}{HTML}{7B858E}",
    ]
    add_repeat_panel(
        lines,
        primary_repeat.loc[primary_repeat.margin.eq(0.0)],
        PRIMARY,
        PRIMARY_SHORT,
        primary_colors,
        "compensation_hit_rate",
        (47, 58),
        "a.",
        "$K=10$, $n=120$",
        "Hit rate",
        0.038,
        0.122,
        [0.04, 0.06, 0.08, 0.10, 0.12],
    )
    add_boundary_panel(lines, primary_protein)
    add_repeat_panel(
        lines,
        residual_repeat,
        ABLATIONS,
        ABLATION_LABELS,
        ablation_colors,
        "epsilon_rmse",
        (47, -5),
        "c.",
        "margin=0, $n=155$",
        "Residual RMSE (kcal/mol)",
        0.54,
        1.10,
        [0.6, 0.8, 1.0],
    )
    add_repeat_panel(
        lines,
        native_repeat,
        ABLATIONS,
        ABLATION_LABELS,
        ablation_colors,
        "compensation_hit_rate",
        (108, -5),
        "d.",
        "$K=10$, $n=120$",
        "Native hit rate",
        0.035,
        0.14,
        [0.04, 0.08, 0.12],
        show_y_labels=False,
        panel_label_x=0.02,
    )
    lines.append("\\node[anchor=north, yshift=-3mm, font=\\fontsize{10}{12}\\selectfont] at (current bounding box.south) {\\pgfplotslegendfromname{componentanalysislegend}};")
    lines.append("\\end{tikzpicture}")
    (OUTPUT / "component_analysis.tikz").write_text("\n".join(lines) + "\n")


def factorial_tikz_repeat_panel(lines, values, metric, order, labels, colors, axis, panel_label, title, xlabel, show_y_labels=True, xticks=None, xticklabels=None):
    current_values = values.loc[values.arm.isin(order), metric].to_numpy(dtype=float)
    assert len(current_values) == len(order) * 5
    minimum, maximum = float(current_values.min()), float(current_values.max())
    span = max(maximum - minimum, 0.05 if metric == "epsilon_rmse" else 0.5)
    padding = max(0.10 * span, 0.004 if metric == "epsilon_rmse" else 0.06)
    xmin, xmax = minimum - padding, maximum + padding
    ticks = np.linspace(xmin, xmax, 4) if xticks is None else np.asarray(xticks, dtype=float)
    yticklabels = [labels[arm] for arm in order] if show_y_labels else [""] * len(order)
    lines.extend(tikz_axis_start(
        *axis,
        xmin,
        xmax,
        -0.55,
        len(order) - 0.45,
        ticks,
        yticklabels,
        xlabel,
        xticklabels=xticklabels,
    ))
    lines.append(f"\\node[anchor=south west, font=\\bfseries\\fontsize{{10}}{{12}}\\selectfont] at (rel axis cs:-0.18,1.04) {{{panel_label}}};")
    lines.append(f"\\node[anchor=south east, font=\\fontsize{{10}}{{12}}\\selectfont, text=black!60] at (rel axis cs:1,1.04) {{{title}}};")
    for position, arm in enumerate(order):
        values_for_arm = values.loc[values.arm.eq(arm)].sort_values("seed")[metric].to_numpy(dtype=float)
        assert len(values_for_arm) == 5
        mean = float(values_for_arm.mean())
        sd = float(values_for_arm.std(ddof=1))
        color = colors[arm]
        lo, hi = tex_number(mean - sd), tex_number(mean + sd)
        points = " ".join(f"({tex_number(value)},{position})" for value in values_for_arm)
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{lo},{position}) -- (axis cs:{hi},{position});")
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{lo},{position - 0.10}) -- (axis cs:{lo},{position + 0.10});")
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{hi},{position - 0.10}) -- (axis cs:{hi},{position + 0.10});")
        lines.append(f"\\addplot[only marks, mark=o, mark size=2.0pt, color={color}, mark options={{draw={color}, fill=white}}] coordinates {{{points}}};")
        lines.append(f"\\addplot[only marks, mark=diamond*, mark size=2.6pt, color={color}, mark options={{draw={color}, fill=white}}] coordinates {{({tex_number(mean)},{position})}};")
    lines.append("\\end{axis}")


def factorial_tikz_contrast_panel(lines, rows, axis, panel_label, title, xlabel, scale=1.0, show_y_labels=True, xticks=None, xticklabels=None):
    tikz_colors = {
        "joint_input": "FactorBlue",
        "values_with_mask": "ValuesPurple",
        "values_without_mask": "ContrastGold",
        "mask_with_values": "MaskTeal",
        "mask_without_values": "ContrastGray",
    }
    repeats = np.asarray([value for row in rows for value in list(row["repeat_gains"].values())], dtype=float) * scale
    repeat_means = np.asarray([np.mean(list(row["repeat_gains"].values())) for row in rows], dtype=float) * scale
    repeat_sds = np.asarray([np.std(list(row["repeat_gains"].values()), ddof=1) for row in rows], dtype=float) * scale
    extents = np.concatenate([repeats, repeat_means - repeat_sds, repeat_means + repeat_sds, np.asarray([0.0])])
    minimum, maximum = float(np.min(extents)), float(np.max(extents))
    span = max(maximum - minimum, 0.05 * scale)
    padding = max(0.12 * span, 0.006 * scale)
    xmin, xmax = minimum - padding, maximum + padding
    ticks = np.linspace(xmin, xmax, 4) if xticks is None else np.asarray(xticks, dtype=float)
    yticklabels = [CONTRAST_LABELS[row["contrast"]] if show_y_labels else "" for row in rows]
    lines.extend(tikz_axis_start(
        *axis,
        xmin,
        xmax,
        -0.55,
        len(rows) - 0.45,
        ticks,
        yticklabels,
        xlabel,
        xticklabels=xticklabels,
    ))
    lines.append(f"\\node[anchor=south west, font=\\bfseries\\fontsize{{10}}{{12}}\\selectfont] at (rel axis cs:-0.18,1.04) {{{panel_label}}};")
    lines.append(f"\\node[anchor=south east, font=\\fontsize{{10}}{{12}}\\selectfont, text=black!60] at (rel axis cs:1,1.04) {{{title}}};")
    lines.append("\\draw[dashed, gray!75, line width=0.6pt] (axis cs:0,-0.45) -- (axis cs:0,4.45);")
    for position, row in enumerate(rows):
        contrast = row["contrast"]
        color = tikz_colors[contrast]
        values_for_arm = np.asarray(list(row["repeat_gains"].values()), dtype=float) * scale
        mean = float(values_for_arm.mean())
        sd = float(values_for_arm.std(ddof=1))
        lo, hi = tex_number(mean - sd), tex_number(mean + sd)
        points = " ".join(f"({tex_number(value)},{position})" for value in values_for_arm)
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{lo},{position}) -- (axis cs:{hi},{position});")
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{lo},{position - 0.10}) -- (axis cs:{lo},{position + 0.10});")
        lines.append(f"\\draw[{color}, line width=0.65pt] (axis cs:{hi},{position - 0.10}) -- (axis cs:{hi},{position + 0.10});")
        lines.append(f"\\addplot[only marks, mark=o, mark size=2.0pt, color={color}, mark options={{draw={color}, fill=white}}] coordinates {{{points}}};")
        lines.append(f"\\addplot[only marks, mark=diamond*, mark size=2.6pt, color={color}, mark options={{draw={color}, fill=white}}] coordinates {{({tex_number(mean)},{position})}};")
    lines.append("\\end{axis}")


def write_factorial_tikz(data, threshold, margin, output):
    residual = factorial_metric_frame(data, threshold, margin, "epsilon_rmse")
    selection = factorial_metric_frame(data, threshold, margin, "compensation_hit_rate", "native")
    residual_n = int(residual.loc[residual.arm.eq("interaction"), "protein_id"].nunique())
    selection_n = int(selection.loc[selection.arm.eq("interaction"), "protein_id"].nunique())
    residual_values = factorial_repeat_summary(residual, "epsilon_rmse", threshold, margin, FACTORIAL_CELLS)
    selection_values = factorial_repeat_summary(selection, "compensation_hit_rate", threshold, margin, FACTORIAL_CELLS).copy()
    selection_values["compensation_hit_rate"] *= 100.0
    residual_rows = factorial_summary_rows(data, threshold, margin, "epsilon_rmse")
    selection_rows = factorial_summary_rows(data, threshold, margin, "compensation_hit_rate", "native")
    colors = {
        "interaction": "FactorBlue",
        "values_only": "ValuesPurple",
        "mask_only": "MaskTeal",
        "no_pair_measured": "NeutralGray",
    }
    lines = [
        "\\begin{tikzpicture}[font=\\sffamily\\fontsize{10}{12}\\selectfont]",
        "\\definecolor{FactorBlue}{HTML}{2878B5}",
        "\\definecolor{ValuesPurple}{HTML}{9368A6}",
        "\\definecolor{MaskTeal}{HTML}{3F9BA8}",
        "\\definecolor{NeutralGray}{HTML}{7B858E}",
        "\\definecolor{ContrastGold}{HTML}{B89B3C}",
        "\\definecolor{ContrastGray}{HTML}{6F7785}",
    ]
    factorial_tikz_repeat_panel(
        lines,
        residual_values,
        "epsilon_rmse",
        FACTORIAL_CELLS,
        FACTORIAL_LABELS,
        colors,
        (47, 58),
        "a.",
        f"family{threshold}, $n={residual_n}$",
        "Residual RMSE (kcal/mol)",
        xticks=[0.60, 0.65, 0.70, 0.75, 0.80],
        xticklabels=["0.60", "0.65", "0.70", "0.75", "0.80"],
    )
    factorial_tikz_repeat_panel(
        lines,
        selection_values,
        "compensation_hit_rate",
        FACTORIAL_CELLS,
        FACTORIAL_LABELS,
        colors,
        (108, 58),
        "b.",
        f"native, $K=10$, $n={selection_n}$",
        "Native top-10 hit rate (\\%)",
        show_y_labels=False,
        xticks=[7, 9, 11, 13],
        xticklabels=["7", "9", "11", "13"],
    )
    factorial_tikz_contrast_panel(
        lines,
        residual_rows,
        (47, -5),
        "c.",
        "repeat mean $\\pm$ SD",
        "Residual gain (kcal/mol)",
        xticks=[0.0, 0.05, 0.10, 0.15, 0.20],
        xticklabels=["0", "0.05", "0.10", "0.15", "0.20"],
    )
    factorial_tikz_contrast_panel(
        lines,
        selection_rows,
        (108, -5),
        "d.",
        "repeat mean $\\pm$ SD",
        "Hit-rate gain (p.p.)",
        scale=100.0,
        show_y_labels=False,
        xticks=[-2, 0, 2, 4],
        xticklabels=["$-2$", "0", "2", "4"],
    )
    lines.append("\\end{tikzpicture}")
    output.write_text("\n".join(lines) + "\n")


def render_factorial(data, output, note, threshold, margin):
    residual = factorial_metric_frame(data, threshold, margin, "epsilon_rmse")
    selection = factorial_metric_frame(data, threshold, margin, "compensation_hit_rate", "native")
    residual_n = int(residual.loc[residual.arm.eq("interaction"), "protein_id"].nunique())
    selection_n = int(selection.loc[selection.arm.eq("interaction"), "protein_id"].nunique())
    residual_values = factorial_repeat_summary(residual, "epsilon_rmse", threshold, margin, FACTORIAL_CELLS)
    selection_values = factorial_repeat_summary(selection, "compensation_hit_rate", threshold, margin, FACTORIAL_CELLS)
    selection_values["compensation_hit_rate"] *= 100.0
    residual_rows = factorial_summary_rows(data, threshold, margin, "epsilon_rmse")
    selection_rows = factorial_summary_rows(data, threshold, margin, "compensation_hit_rate", "native")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(6.45, 4.65), facecolor="white")
    layout = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.78,
        hspace=0.88,
        left=0.31,
        right=0.99,
        top=0.92,
        bottom=0.19,
    )
    ax_a = fig.add_subplot(layout[0, 0])
    ax_b = fig.add_subplot(layout[0, 1])
    ax_c = fig.add_subplot(layout[1, 0])
    ax_d = fig.add_subplot(layout[1, 1])
    draw_factorial_metric(
        ax_a,
        residual_values,
        "epsilon_rmse",
        FACTORIAL_CELLS,
        FACTORIAL_LABELS,
        FACTORIAL_COLORS,
        "a.",
        f"family{threshold}, n={residual_n}",
        "Residual RMSE (kcal/mol)",
        xticks=[0.60, 0.65, 0.70, 0.75, 0.80],
        xticklabels=["0.60", "0.65", "0.70", "0.75", "0.80"],
    )
    draw_factorial_metric(
        ax_b,
        selection_values,
        "compensation_hit_rate",
        FACTORIAL_CELLS,
        FACTORIAL_LABELS,
        FACTORIAL_COLORS,
        "b.",
        f"native, $K=10$, n={selection_n}",
        "Native top-10 hit rate (%)",
        xticks=[7, 9, 11, 13],
        xticklabels=["7", "9", "11", "13"],
    )
    draw_contrast_panel(
        ax_c,
        residual_rows,
        "c.",
        "Residual gain (kcal/mol)",
        "repeat mean ± SD",
        xticks=[0.0, 0.05, 0.10, 0.15, 0.20],
        xticklabels=["0", "0.05", "0.10", "0.15", "0.20"],
    )
    draw_contrast_panel(
        ax_d,
        selection_rows,
        "d.",
        "Hit-rate gain (p.p.)",
        "repeat mean ± SD",
        scale=100.0,
        show_y_labels=False,
        xticks=[-2, 0, 2, 4],
        xticklabels=["−2", "0", "2", "4"],
    )
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "component_analysis.pdf", facecolor="white")
    fig.savefig(output / "component_analysis.svg", facecolor="white")
    fig.savefig(output / "component_analysis.png", dpi=320, facecolor="white")
    plt.close(fig)
    write_factorial_note(data, threshold, margin, note)
    write_factorial_tikz(data, threshold, margin, output / "component_analysis.tikz")
    print("factorial residual repeat means")
    print(residual_values.groupby("arm")["epsilon_rmse"].mean().to_string())
    print("factorial native hit-rate repeat means")
    print(selection_values.groupby("arm")["compensation_hit_rate"].mean().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="evidence directory or completed factorial run root")
    parser.add_argument("--output", type=Path, help="directory for component figure artifacts")
    parser.add_argument("--note", type=Path, help="path for the evidence note")
    parser.add_argument("--threshold", type=int, default=30, choices=[25, 30])
    parser.add_argument("--margin", type=float, default=0.0, choices=[0.0, 0.5])
    args = parser.parse_args()
    global INPUT, OUTPUT, NOTE
    if args.input is not None:
        INPUT = args.input if args.input.is_absolute() else ROOT / args.input
    if args.output is not None:
        OUTPUT = args.output if args.output.is_absolute() else ROOT / args.output
    if args.note is not None:
        NOTE = args.note if args.note.is_absolute() else ROOT / args.note
    factorial_root = factorial_run_from_input(INPUT)
    if factorial_root is not None:
        data = load_factorial_data(factorial_root)
        render_factorial(data, OUTPUT, NOTE, args.threshold, args.margin)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    selection, residual = load_data()
    primary_protein = primary_selection_summary(selection)
    primary_frame = selection.loc[
        selection.task.eq("compensation")
        & selection.scoring.eq("observed_main")
        & selection.budget.eq(10)
        & selection.arm.isin(PRIMARY)
    ].copy()
    primary_seed = fold_seed_protein_values(primary_frame, "compensation_hit_rate", ["arm", "margin"])
    primary_repeat = repeat_macro(primary_seed, "compensation_hit_rate", ["arm", "margin"])
    residual_protein = ablation_residual_summary(residual)
    residual_frame = residual.loc[residual.margin.eq(0.0) & residual.arm.isin(ABLATIONS)].copy()
    residual_seed = fold_seed_protein_values(residual_frame, "epsilon_rmse", ["arm", "margin"])
    residual_repeat = repeat_macro(residual_seed, "epsilon_rmse", ["arm", "margin"])
    native_protein = ablation_selection_summary(selection)
    native_frame = selection.loc[
        selection.task.eq("compensation")
        & selection.scoring.eq("native")
        & selection.margin.eq(0.0)
        & selection.budget.eq(10)
        & selection.arm.isin(ABLATIONS)
    ].copy()
    native_seed = fold_seed_protein_values(native_frame, "compensation_hit_rate", ["arm", "margin"])
    native_repeat = repeat_macro(native_seed, "compensation_hit_rate", ["arm", "margin"])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(6.45, 4.65), facecolor="white")
    layout = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.00, 1.00],
        height_ratios=[1.00, 1.00],
        wspace=0.55,
        hspace=0.82,
        left=0.28,
        right=0.99,
        top=0.92,
        bottom=0.23,
    )
    ax_a = fig.add_subplot(layout[0, 0])
    ax_b = fig.add_subplot(layout[0, 1])
    ax_c = fig.add_subplot(layout[1, 0])
    ax_d = fig.add_subplot(layout[1, 1])

    primary_a = primary_repeat.loc[primary_repeat.margin.eq(0.0)].copy()
    draw_repeat_summary(
        ax_a,
        primary_a,
        PRIMARY,
        PRIMARY_SHORT,
        PRIMARY_COLORS,
        "compensation_hit_rate",
        "Hit rate",
        "a.",
        "$K=10$, n=120",
    )
    draw_boundary(ax_b, primary_protein)

    draw_repeat_summary(
        ax_c,
        residual_repeat,
        ABLATIONS,
        ABLATION_LABELS,
        ABLATION_DISPLAY_COLORS,
        "epsilon_rmse",
        "Residual RMSE (kcal/mol)",
        "c.",
        "margin=0, n=155",
    )

    draw_repeat_summary(
        ax_d,
        native_repeat,
        ABLATIONS,
        ABLATION_LABELS,
        ABLATION_DISPLAY_COLORS,
        "compensation_hit_rate",
        "Native hit rate",
        "d.",
        "$K=10$, n=120",
        show_y_labels=False,
    )

    handles = [
        Line2D(
            [0], [0], color=PRIMARY_COLORS[arm], linewidth=1.5 if arm == "interaction" else 0.95,
            marker="o", markerfacecolor="white", markeredgecolor=PRIMARY_COLORS[arm],
            markeredgewidth=0.7, markersize=4.2, label=PRIMARY_LABELS[arm],
        )
        for arm in PRIMARY
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.035),
        ncol=4,
        frameon=False,
        fontsize=10,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "component_analysis.pdf", facecolor="white")
    fig.savefig(OUTPUT / "component_analysis.svg", facecolor="white")
    fig.savefig(OUTPUT / "component_analysis.png", dpi=320, facecolor="white")
    plt.close(fig)

    write_note(primary_protein, primary_repeat, residual_protein, residual_repeat, native_protein, native_repeat, selection, residual)
    write_tikz(primary_protein, primary_repeat, residual_protein, residual_repeat, native_protein, native_repeat)
    print(repeat_stats(primary_repeat.loc[primary_repeat.margin.eq(0.0)], "compensation_hit_rate").to_string(index=False))
    print(boundary_stats(primary_repeat).to_string(index=False))
    print(repeat_stats(residual_repeat, "epsilon_rmse").to_string(index=False))
    print(repeat_stats(native_repeat, "compensation_hit_rate").to_string(index=False))


if __name__ == "__main__":
    main()
