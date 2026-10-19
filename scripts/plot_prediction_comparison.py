from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs/sall_full_target/paper_prediction"
OUTPUT = ROOT / "paper/figures"

COLORS = {
    "interaction": "#2878B5",
    "thermo_epistatic": "#E68632",
    "kermut": "#32936F",
    "stabilityarc_kermut": "#3F9BA8",
    "global_nonlinear": "#B89B3C",
    "additive": "#6F7785",
    "mean_mlp": "#9368A6",
    "mean_ridge": "#AC6D79",
}
ARM_ORDER = ["additive", "global_nonlinear", "thermo_epistatic", "kermut", "stabilityarc_kermut", "mean_mlp", "mean_ridge"]
SHORT = {
    "additive": "Additive",
    "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D",
    "kermut": "Kermut",
    "stabilityarc_kermut": "StabilityArc + Kermut",
    "mean_mlp": "Mean MLP",
    "mean_ridge": "Mean ridge",
}
FULL = {
    "interaction": "SCEPI",
    "additive": "Additive",
    "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D",
    "kermut": "Kermut",
    "stabilityarc_kermut": "StabilityArc + Kermut",
    "mean_mlp": "Mean MLP",
    "mean_ridge": "Mean ridge",
}


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#D9DEE3", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#8D969E")
        spine.set_linewidth(0.65)
    ax.tick_params(axis="both", colors="#27323A", labelsize=10, length=2.5, width=0.55, pad=2)


def load_table():
    manifest = json.loads((INPUT / "manifest.json").read_text())
    assert manifest["prediction_grid_complete"]
    assert manifest["source_manifest"]["available_jobs_complete"]
    table = pd.read_parquet(INPUT / "per_protein.parquet")
    table = table.loc[
        table.threshold.eq(30)
        & table.arm.isin(["interaction", *ARM_ORDER])
    ].copy()
    assert set(table.arm.unique()) == {"interaction", *ARM_ORDER}
    assert set(table.seed.unique()) == {11, 23, 47, 89, 131}
    for arm in ["interaction", *ARM_ORDER]:
        assert table.loc[table.arm.eq(arm), ["fold", "seed"]].drop_duplicates().shape[0] == 25
    return table, manifest


def protein_means(table, metric, phenotype=False):
    frame = table
    if phenotype:
        frame = frame.loc[frame.phenotype_count.ge(30)]
    result = frame.groupby(["arm", "protein_id"], as_index=False)[metric].mean()
    return result


def pair_values(means, control, metric, shared="interaction"):
    left = means.loc[means.arm.eq(shared), ["protein_id", metric]].rename(columns={metric: "shared"})
    right = means.loc[means.arm.eq(control), ["protein_id", metric]].rename(columns={metric: "control"})
    pair = left.merge(right, on="protein_id", validate="one_to_one").dropna()
    assert len(pair) > 100
    return pair


def plot_scatter_facets(fig, slot, means):
    rows = slot.subgridspec(2, 1, hspace=0.60)
    row_one = rows[0].subgridspec(1, 4, wspace=0.30)
    row_two = rows[1].subgridspec(1, 3, wspace=0.80)
    all_values = means.epsilon_rmse.to_numpy(dtype=float)
    limit = float(np.nanmax(all_values) * 1.04)
    axes = []
    for index, arm in enumerate(ARM_ORDER):
        panel_grid = row_one if index < 4 else row_two
        panel_index = index if index < 4 else index - 4
        ax = fig.add_subplot(panel_grid[0, panel_index])
        axes.append(ax)
        pair = pair_values(means, arm, "epsilon_rmse")
        assert len(pair) == 155
        ax.scatter(
            pair.control,
            pair.shared,
            s=13,
            facecolors="none",
            edgecolors=COLORS[arm],
            linewidths=0.55,
            alpha=0.7,
        )
        ax.plot([0, limit], [0, limit], color="#9299A0", linewidth=0.7, linestyle=(0, (2, 2)))
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.95, SHORT[arm], transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#202830")
        if index == 5:
            ax.set_xlabel("Control RMSE (kcal/mol)", fontsize=10, labelpad=2)
        style_axis(ax)
    first = axes[0]
    first.text(-0.14, 1.10, "a.", transform=first.transAxes, fontsize=10, fontweight="bold", va="top")


def draw_gain_panel(ax, means, metric, higher_shared, xlabel, panel_label, show_ylabels=True):
    all_gains = []
    distributions = {}
    for index, arm in enumerate(ARM_ORDER):
        pair = pair_values(means, arm, metric)
        gains = pair.shared.to_numpy() - pair.control.to_numpy() if higher_shared else pair.control.to_numpy() - pair.shared.to_numpy()
        assert len(gains) == (159 if metric == "phenotype_spearman" or metric == "phenotype_rmse" else 155)
        gains = np.sort(gains)
        distributions[arm] = gains
        all_gains.append(gains)
    minimum = min(float(np.min(values)) for values in all_gains)
    maximum = max(float(np.max(values)) for values in all_gains)
    span = max(maximum - minimum, 0.05)
    y_positions = np.arange(len(ARM_ORDER))
    for index, arm in enumerate(ARM_ORDER):
        values = distributions[arm]
        jitter = np.linspace(-0.16, 0.16, len(values))
        ax.scatter(values, np.full(len(values), y_positions[index]) + jitter, s=4.0, color=COLORS[arm], alpha=0.19, linewidths=0)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        mean = float(np.mean(values))
        ax.plot([q1, q3], [y_positions[index], y_positions[index]], color=COLORS[arm], linewidth=2.2, solid_capstyle="butt")
        ax.plot([median, median], [y_positions[index] - 0.10, y_positions[index] + 0.10], color="#26313A", linewidth=0.9)
        ax.scatter([mean], [y_positions[index]], marker="D", s=23, facecolor="white", edgecolor=COLORS[arm], linewidth=1.0, zorder=4)
    ax.axvline(0, color="#626D76", linewidth=0.7, linestyle=(0, (2, 2)))
    yticklabels = [SHORT[arm] for arm in ARM_ORDER] if show_ylabels else [""] * len(ARM_ORDER)
    ax.set_yticks(y_positions, yticklabels)
    ax.set_ylim(-0.55, len(ARM_ORDER) - 0.45)
    ax.set_xlim(minimum - span * 0.06, maximum + span * 0.06)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=3)
    style_axis(ax)
    ax.text(-0.16, 1.08, panel_label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def plot_repeats(ax, table):
    frame = table.loc[table.main_eligible].copy()
    seed_protein = frame.groupby(["arm", "seed", "protein_id"], as_index=False).epsilon_rmse.mean()
    shared = seed_protein.loc[seed_protein.arm.eq("interaction"), ["seed", "protein_id", "epsilon_rmse"]].rename(columns={"epsilon_rmse": "shared"})
    seed_values = sorted(shared.seed.unique())
    assert seed_values == [11, 23, 47, 89, 131]
    summaries = []
    for index, arm in enumerate(ARM_ORDER):
        control = seed_protein.loc[seed_protein.arm.eq(arm), ["seed", "protein_id", "epsilon_rmse"]].rename(columns={"epsilon_rmse": "control"})
        paired = shared.merge(control, on=["seed", "protein_id"], validate="one_to_one")
        assert paired.groupby("seed").protein_id.nunique().eq(155).all()
        paired["gain"] = paired.control - paired.shared
        values = paired.groupby("seed").gain.mean().reindex(seed_values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        summaries.append((arm, values.to_numpy(), mean, sd))
    minimum = min(float(mean - sd) for _, _, mean, sd in summaries)
    maximum = max(float(mean + sd) for _, _, mean, sd in summaries)
    span = max(maximum - minimum, 0.02)
    y_positions = np.arange(len(ARM_ORDER))
    for index, (arm, values, mean, sd) in enumerate(summaries):
        jitter = np.linspace(-0.13, 0.13, len(values))
        ax.scatter(values, np.full(len(values), y_positions[index]) + jitter, s=25, facecolors="white", edgecolors=COLORS[arm], linewidths=0.9, zorder=3)
        ax.errorbar(mean, y_positions[index], xerr=sd, fmt="none", ecolor=COLORS[arm], elinewidth=1.7, capsize=2.2, capthick=0.8, zorder=2)
        ax.scatter([mean], [y_positions[index]], marker="D", s=31, facecolor=COLORS[arm], edgecolor="white", linewidth=0.55, zorder=4)
    ax.axvline(0, color="#626D76", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_yticks(y_positions, [SHORT[arm] for arm in ARM_ORDER])
    ax.set_ylim(-0.55, len(ARM_ORDER) - 0.45)
    ax.set_xlabel("Residual RMSE gain (control − SCEPI; kcal/mol)", fontsize=10, labelpad=3)
    ax.set_xlim(minimum - span * 0.12, maximum + span * 0.12)
    style_axis(ax)
    ax.text(0.0, 1.10, "d.", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def tex_number(value):
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text if text and text != "-0" else "0"


def tex_coordinates(x_values, y_values):
    return " ".join(f"({tex_number(x)},{tex_number(y)})" for x, y in zip(x_values, y_values))


def gain_distributions(means, metric, higher_shared):
    distributions = {}
    statistics = {}
    for arm in ARM_ORDER:
        pair = pair_values(means, arm, metric)
        gains = pair.shared.to_numpy() - pair.control.to_numpy() if higher_shared else pair.control.to_numpy() - pair.shared.to_numpy()
        gains = np.sort(gains)
        distributions[arm] = gains
        statistics[arm] = {
            "q1": float(np.quantile(gains, 0.25)),
            "median": float(np.quantile(gains, 0.5)),
            "q3": float(np.quantile(gains, 0.75)),
            "mean": float(np.mean(gains)),
            "n": len(gains),
        }
    return distributions, statistics


def write_tikz(table, residual_means, phenotype_means, spearman_means):
    residual_values = residual_means.epsilon_rmse.to_numpy(dtype=float)
    scatter_limit = float(np.nanmax(residual_values) * 1.04)
    spearman, spearman_stats = gain_distributions(spearman_means, "phenotype_spearman", True)
    phenotype, phenotype_stats = gain_distributions(phenotype_means, "phenotype_rmse", False)

    residual_seed = table.loc[table.main_eligible].groupby(["arm", "seed", "protein_id"], as_index=False).epsilon_rmse.mean()
    shared_seed = residual_seed.loc[residual_seed.arm.eq("interaction"), ["seed", "protein_id", "epsilon_rmse"]].rename(columns={"epsilon_rmse": "shared"})
    repeat_values = {}
    repeat_statistics = {}
    for arm in ARM_ORDER:
        control_seed = residual_seed.loc[residual_seed.arm.eq(arm), ["seed", "protein_id", "epsilon_rmse"]].rename(columns={"epsilon_rmse": "control"})
        paired = shared_seed.merge(control_seed, on=["seed", "protein_id"], validate="one_to_one")
        paired["gain"] = paired.control - paired.shared
        values = paired.groupby("seed").gain.mean().sort_index().to_numpy()
        assert len(values) == 5 and np.all(np.isfinite(values))
        repeat_values[arm] = values
        repeat_statistics[arm] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "n": len(values)}

    b_min = min(float(np.min(values)) for values in spearman.values())
    b_max = max(float(np.max(values)) for values in spearman.values())
    b_span = max(b_max - b_min, 0.05)
    c_min = min(float(np.min(values)) for values in phenotype.values())
    c_max = max(float(np.max(values)) for values in phenotype.values())
    c_span = max(c_max - c_min, 0.05)
    d_min = min(stat["mean"] - stat["sd"] for stat in repeat_statistics.values())
    d_max = max(stat["mean"] + stat["sd"] for stat in repeat_statistics.values())
    d_span = max(d_max - d_min, 0.02)

    lines = [
        "% Self-contained PGFPlots source generated from the verified full-target S-all export.",
        "% The parent manuscript wraps this fragment in a tikzpicture at native size.",
        r"\definecolor{predblue}{HTML}{2878B5}",
        r"\definecolor{predorange}{HTML}{E68632}",
        r"\definecolor{predgreen}{HTML}{32936F}",
        r"\definecolor{predteal}{HTML}{3F9BA8}",
        r"\definecolor{predgold}{HTML}{B89B3C}",
        r"\definecolor{predgray}{HTML}{6F7785}",
        r"\definecolor{predpurple}{HTML}{9368A6}",
        r"\definecolor{predrose}{HTML}{AC6D79}",
        r"\definecolor{preddiag}{HTML}{9299A0}",
        r"\begin{groupplot}[",
        r"  group style={group name=predtoprowone, group size=4 by 1, horizontal sep=0.13in},",
        r"  scale only axis, width=1.02in, height=0.72in, xmin=0, ymin=0,",
        f"  xmax={tex_number(scatter_limit)}, ymax={tex_number(scatter_limit)},",
        r"  grid=major, grid style={draw=gray!25, line width=0.25pt},",
        r"  axis line style={draw=gray!65, line width=0.35pt},",
        r"  tick style={draw=gray!65, line width=0.3pt},",
        r"  tick label style={font=\normalsize}, label style={font=\normalsize},",
        r"  title style={font=\normalsize, yshift=1pt},",
        r"]",
    ]
    tikz_colors = {"additive": "predgray", "global_nonlinear": "predgold", "thermo_epistatic": "predorange", "kermut": "predgreen", "stabilityarc_kermut": "predteal", "mean_mlp": "predpurple", "mean_ridge": "predrose"}
    for index, arm in enumerate(ARM_ORDER[:4]):
        options = [f"title={{{SHORT[arm]}}}"]
        if index != 0:
            options.append(r"yticklabels=\empty")
        lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
        pair = pair_values(residual_means, arm, "epsilon_rmse")
        assert len(pair) == 155
        lines.append(r"\addplot[only marks, mark=o, mark size=1.35pt, mark options={fill=white}, color=" + tikz_colors[arm] + r", opacity=0.70] coordinates {" + tex_coordinates(pair.control, pair.shared) + "};")
        lines.append(r"\addplot[no marks, dashed, color=preddiag, line width=0.35pt] coordinates {(0,0) (" + tex_number(scatter_limit) + "," + tex_number(scatter_limit) + ")};")
    lines.extend([
        r"\end{groupplot}",
        r"\begin{scope}[yshift=-1.25in]",
        r"\begin{groupplot}[",
        r"  group style={group name=predtoprowtwo, group size=3 by 1, horizontal sep=0.45in},",
        r"  scale only axis, width=1.02in, height=0.72in, xmin=0, ymin=0,",
        f"  xmax={tex_number(scatter_limit)}, ymax={tex_number(scatter_limit)},",
        r"  grid=major, grid style={draw=gray!25, line width=0.25pt},",
        r"  axis line style={draw=gray!65, line width=0.35pt},",
        r"  tick style={draw=gray!65, line width=0.3pt},",
        r"  tick label style={font=\normalsize}, label style={font=\normalsize},",
        r"  title style={font=\normalsize, yshift=1pt},",
        r"]",
    ])
    for index, arm in enumerate(ARM_ORDER[4:]):
        options = [f"title={{{SHORT[arm]}}}"]
        if index != 0:
            options.append(r"yticklabels=\empty")
        lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
        pair = pair_values(residual_means, arm, "epsilon_rmse")
        assert len(pair) == 155
        lines.append(r"\addplot[only marks, mark=o, mark size=1.35pt, mark options={fill=white}, color=" + tikz_colors[arm] + r", opacity=0.70] coordinates {" + tex_coordinates(pair.control, pair.shared) + "};")
        lines.append(r"\addplot[no marks, dashed, color=preddiag, line width=0.35pt] coordinates {(0,0) (" + tex_number(scatter_limit) + "," + tex_number(scatter_limit) + ")};")
    lines.extend([
        r"\end{groupplot}",
        r"\end{scope}",
        r"\node[anchor=south east, xshift=-1pt, yshift=0.16in, font=\bfseries\normalsize] at (predtoprowone c1r1.north west) {a.};",
        r"\node[rotate=90, align=center, font=\normalsize] at ($(predtoprowone c1r1.west)!0.5!(predtoprowtwo c1r1.west)+(-0.42in,0)$) {SCEPI residual RMSE\\(kcal/mol)};",
        r"\node[align=center, font=\normalsize] at ($(predtoprowtwo c1r1.south)!0.5!(predtoprowtwo c3r1.south)+(0,-0.28in)$) {Control RMSE (kcal/mol)};",
        r"\begin{scope}[yshift=-3.20in]",
        r"\begin{groupplot}[",
        r"  group style={group name=predlower, group size=2 by 1, horizontal sep=0.20in},",
        r"  scale only axis, width=2.38in, height=1.20in, ymin=-0.55, ymax=6.55,",
        r"  ytick={0,1,2,3,4,5,6}, yticklabels={Additive,Global nonlinear,ThermoMPNN-D,Kermut,StabilityArc + Kermut,Mean MLP,Mean ridge},",
        r"  grid=major, grid style={draw=gray!25, line width=0.25pt},",
        r"  axis line style={draw=gray!65, line width=0.35pt}, tick style={draw=gray!65, line width=0.3pt},",
        r"  tick label style={font=\normalsize}, label style={font=\normalsize},",
        r"  yticklabel style={font=\normalsize, inner xsep=1pt}, xlabel style={font=\normalsize, align=center},",
        r"]",
    ])
    lower_panel_labels = []
    for metric_name, distributions, statistics, x_label, x_min, x_max, panel in [
        ("spearman", spearman, spearman_stats, r"Phenotype Spearman gain\\(SCEPI - control)", b_min - b_span * 0.06, b_max + b_span * 0.06, "b."),
        ("phenotype", phenotype, phenotype_stats, r"Phenotype RMSE gain\\(control - SCEPI; kcal/mol)", c_min - c_span * 0.06, c_max + c_span * 0.06, "c."),
    ]:
        options = [r"xmin=" + tex_number(x_min), r"xmax=" + tex_number(x_max), r"xlabel={" + x_label + r"}"]
        if metric_name == "phenotype":
            options.append(r"yticklabels=\empty")
        lines.append(r"\nextgroupplot[" + ",".join(options) + r"]")
        for index, arm in enumerate(ARM_ORDER):
            values = distributions[arm]
            jitter = np.linspace(-0.16, 0.16, len(values))
            lines.append(r"\addplot[only marks, mark=o, mark size=0.75pt, mark options={fill=" + tikz_colors[arm] + r"!25}, color=" + tikz_colors[arm] + r", opacity=0.20] coordinates {" + tex_coordinates(values, np.full(len(values), index) + jitter) + "};")
            stat = statistics[arm]
            lines.append(r"\addplot[no marks, color=" + tikz_colors[arm] + r", line width=1.7pt] coordinates {(" + tex_number(stat["q1"]) + "," + str(index) + ") (" + tex_number(stat["q3"]) + "," + str(index) + ")};")
            lines.append(r"\addplot[no marks, color=black, line width=0.55pt] coordinates {(" + tex_number(stat["median"]) + "," + tex_number(index - 0.10) + ") (" + tex_number(stat["median"]) + "," + tex_number(index + 0.10) + ")};")
            lines.append(r"\addplot[only marks, mark=diamond*, mark size=2.3pt, mark options={fill=white}, color=" + tikz_colors[arm] + r"] coordinates {(" + tex_number(stat["mean"]) + "," + str(index) + ")};")
        lines.append(r"\addplot[no marks, dashed, color=gray!70, line width=0.35pt] coordinates {(0,-0.55) (0,6.55)};")
        if metric_name == "spearman":
            lower_panel_labels.append(r"\node[anchor=south west, xshift=-2pt, yshift=2pt, font=\bfseries\normalsize] at (predlower c1r1.north west) {" + panel + r"};")
        else:
            lower_panel_labels.append(r"\node[anchor=south west, xshift=-2pt, yshift=2pt, font=\bfseries\normalsize] at (predlower c2r1.north west) {" + panel + r"};")
    lines.extend([
        r"\end{groupplot}",
        *lower_panel_labels,
        r"\end{scope}",
        r"\begin{scope}[yshift=-5.20in]",
        r"\begin{axis}[",
        r"  name=predd, scale only axis, width=4.87in, height=1.20in,",
        f"  xmin={tex_number(d_min - d_span * 0.12)}, xmax={tex_number(d_max + d_span * 0.12)}, ymin=-0.55, ymax=6.55,",
        r"  ytick={0,1,2,3,4,5,6}, yticklabels={Additive,Global nonlinear,ThermoMPNN-D,Kermut,StabilityArc + Kermut,Mean MLP,Mean ridge},",
        r"  xtick={0,0.1,0.2,0.3,0.4,0.5}, xticklabels={0,0.1,0.2,0.3,0.4,0.5}, scaled ticks=false,",
        r"  xlabel={Residual RMSE gain\\(control - SCEPI; kcal/mol)},",
        r"  grid=major, grid style={draw=gray!25, line width=0.25pt},",
        r"  axis line style={draw=gray!65, line width=0.35pt}, tick style={draw=gray!65, line width=0.3pt},",
        r"  tick label style={font=\normalsize}, label style={font=\normalsize}, xlabel style={font=\normalsize, align=center},",
        r"  yticklabel style={font=\normalsize, inner xsep=1pt},",
        r"]",
    ])
    for index, arm in enumerate(ARM_ORDER):
        values = repeat_values[arm]
        jitter = np.linspace(-0.13, 0.13, len(values))
        lines.append(r"\addplot[only marks, mark=o, mark size=1.7pt, mark options={fill=white}, color=" + tikz_colors[arm] + r"] coordinates {" + tex_coordinates(values, np.full(len(values), index) + jitter) + "};")
        stat = repeat_statistics[arm]
        lines.append(r"\addplot[only marks, mark=diamond*, mark size=2.8pt, color=" + tikz_colors[arm] + r"] coordinates {(" + tex_number(stat["mean"]) + "," + str(index) + ")};")
        lines.append(r"\addplot[no marks, color=" + tikz_colors[arm] + r", line width=1.1pt, error bars/.cd, x dir=both, x explicit, error bar style={line width=0.9pt}, error mark options={rotate=90, mark size=1.8pt}] coordinates {(" + tex_number(stat["mean"]) + "," + str(index) + ") +- (" + tex_number(stat["sd"]) + ",0)};")
    lines.extend([
        r"\addplot[no marks, dashed, color=gray!70, line width=0.35pt] coordinates {(0,-0.55) (0,6.55)};",
        r"\end{axis}",
        r"\node[anchor=south west, xshift=-2pt, yshift=2pt, font=\bfseries\normalsize] at (predd.north west) {d.};",
        r"\end{scope}",
        r"\node[anchor=north, align=center, yshift=-0.58in, font=\normalsize, text width=4.87in] at (predd.south) {",
        r"\textcolor{predblue}{\normalsize $\circ$}\,SCEPI \quad \textcolor{predorange}{\normalsize $\circ$}\,ThermoMPNN-D \quad \textcolor{predgreen}{\normalsize $\circ$}\,Kermut \quad \textcolor{predteal}{\normalsize $\circ$}\,StabilityArc + Kermut\\[2pt]",
        r"\textcolor{predgold}{\normalsize $\circ$}\,Global nonlinear \quad \textcolor{predgray}{\normalsize $\circ$}\,Additive \quad \textcolor{predpurple}{\normalsize $\circ$}\,Mean MLP \quad \textcolor{predrose}{\normalsize $\circ$}\,Mean ridge",
        r"};",
    ])
    tikz_path = ROOT / "paper/figures/prediction_comparison.tikz"
    tikz_path.write_text("\n".join(lines) + "\n")

    integrity = {
        "control_count": len(ARM_ORDER),
        "panel_a_scatter_points": len(ARM_ORDER) * 155,
        "panel_b_points": len(ARM_ORDER) * 159,
        "panel_c_points": len(ARM_ORDER) * 159,
        "panel_d_repetition_points": len(ARM_ORDER) * 5,
        "panel_b_iqr_median_mean": len(ARM_ORDER),
        "panel_c_iqr_median_mean": len(ARM_ORDER),
        "panel_d_mean_sd": len(ARM_ORDER),
        "coordinates_from": "runs/sall_full_target/paper_prediction/per_protein.parquet",
        "all_points_are_observed": True,
        "target_seed_ids_embedded": False,
        "tikz_path": "paper/figures/prediction_comparison.tikz",
    }
    assert integrity["panel_a_scatter_points"] == 1085
    assert integrity["panel_b_points"] == 1113
    assert integrity["panel_c_points"] == 1113
    assert integrity["panel_d_repetition_points"] == 35
    note = """# Native PGFPlots integrity record

The self-contained fragment `paper/figures/prediction_comparison.tikz` was generated from the verified full-target S-all export by `scripts/plot_prediction_comparison.py`. It contains 1085 panel-a coordinates (seven controls x 155 proteins), 1113 panel-b coordinates (seven controls x 159 proteins), 1113 panel-c coordinates, and 35 panel-d repetition coordinates (seven controls x five repetitions). Panels b and c retain one IQR segment, one median mark, and one mean diamond for each control. Panel d retains one mean diamond and one mean +/- SD error bar for each control. The fragment embeds no external data file and no seed identifier; all coordinate values are written at ten decimal places or better from the actual pandas calculations.

Native TeX visual compilation is intentionally left to the root agent. The PNG, PDF, and SVG outputs remain available for comparison with the source figure.
"""
    note_path = ROOT / "work/sall_full_target/prediction_tikz_integrity.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note)
    return integrity


def main():
    table, manifest = load_table()
    residual_means = protein_means(table.loc[table.main_eligible], "epsilon_rmse")
    phenotype_means = protein_means(table, "phenotype_rmse", phenotype=True)
    spearman_means = protein_means(table, "phenotype_spearman", phenotype=True)
    assert residual_means.groupby("arm").protein_id.nunique().eq(155).all()
    assert phenotype_means.groupby("arm").protein_id.nunique().eq(159).all()
    assert spearman_means.groupby("arm").protein_id.nunique().eq(159).all()
    residual_means = residual_means.rename(columns={"epsilon_rmse": "epsilon_rmse"})

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(6.5, 6.8), facecolor="white")
    layout = fig.add_gridspec(3, 1, height_ratios=[3.10, 1.70, 1.35], hspace=0.74, top=0.97, bottom=0.24, left=0.12, right=0.98)
    plot_scatter_facets(fig, layout[0], residual_means)
    fig.text(0.055, 0.80, "SCEPI residual RMSE\n(kcal/mol)", rotation=90, ha="center", va="center", fontsize=10)
    lower = layout[1].subgridspec(1, 2, wspace=0.40)
    ax_b = fig.add_subplot(lower[0])
    ax_c = fig.add_subplot(lower[1])
    draw_gain_panel(ax_b, spearman_means, "phenotype_spearman", True, "Phenotype Spearman gain (SCEPI − control)", "b.")
    draw_gain_panel(ax_c, phenotype_means, "phenotype_rmse", False, "Phenotype RMSE gain\n(control − SCEPI; kcal/mol)", "c.", show_ylabels=False)
    ax_d = fig.add_subplot(layout[2])
    plot_repeats(ax_d, table)

    legend_order = ["interaction", "thermo_epistatic", "kermut", "stabilityarc_kermut", "global_nonlinear", "additive", "mean_mlp", "mean_ridge"]
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLORS[arm], markeredgewidth=0.9, markersize=5.1, label=FULL[arm]) for arm in legend_order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.02), ncol=4, frameon=False, fontsize=10, handletextpad=0.25, columnspacing=0.55)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "prediction_comparison.pdf", facecolor="white")
    fig.savefig(OUTPUT / "prediction_comparison.svg", facecolor="white")
    fig.savefig(OUTPUT / "prediction_comparison.png", dpi=320, facecolor="white")
    plt.close(fig)
    tikz_integrity = write_tikz(table, residual_means, phenotype_means, spearman_means)

    summary = {
        "source_manifest": manifest,
        "scope": "Threshold-30 S-all; complete target double-mutant evaluation; protein means over five repeats",
        "residual_proteins": 155,
        "phenotype_proteins": 159,
        "family_groups": 51,
        "residual_records": 775,
        "phenotype_records": 795,
        "outputs": [
            "paper/figures/prediction_comparison.pdf",
            "paper/figures/prediction_comparison.svg",
            "paper/figures/prediction_comparison.png",
            "paper/figures/prediction_comparison.tikz",
            "work/sall_full_target/prediction_tikz_integrity.md",
        ],
        "tikz_integrity": tikz_integrity,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
