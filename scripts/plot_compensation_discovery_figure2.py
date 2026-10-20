from pathlib import Path
import argparse
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_INPUT = ROOT / "work/reviewer_revision_20261019/random_enrichment/figure_selection.parquet"
DEFAULT_RANDOM_INPUT = ROOT / "work/reviewer_revision_20261019/random_enrichment/random_per_protein.parquet"
DEFAULT_OUTPUT = ROOT / "paper/figures"

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
CONTROL_ORDER = [
    "additive",
    "global_nonlinear",
    "thermo_epistatic",
    "kermut",
    "stabilityarc_kermut",
    "mean_mlp",
    "mean_ridge",
]
ARM_ORDER = ["interaction", *CONTROL_ORDER]
LABELS = {
    "interaction": "SCEPI",
    "additive": "Additive",
    "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D",
    "kermut": "Kermut",
    "stabilityarc_kermut": "StabilityArc + Kermut",
    "mean_mlp": "Mean MLP",
    "mean_ridge": "Mean ridge",
}
SHORT = {
    "interaction": "SCEPI",
    "additive": "Additive",
    "global_nonlinear": "Global nonlinear",
    "thermo_epistatic": "ThermoMPNN-D",
    "kermut": "Kermut",
    "stabilityarc_kermut": "StabilityArc + Kermut",
    "mean_mlp": "Mean MLP",
    "mean_ridge": "Mean ridge",
}
BUDGETS = [1, 3, 10, 30]
SEEDS = [11, 23, 47, 89, 131]


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#D9DEE3", linewidth=0.45, alpha=0.78)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#8D969E")
        spine.set_linewidth(0.65)
    ax.tick_params(axis="both", colors="#27323A", labelsize=10, length=2.5, width=0.55, pad=2)


def load_data(selection_input):
    table = pd.read_parquet(selection_input)
    table = table.loc[
        table.task.eq("compensation")
        & table.margin.eq(0.0)
        & table.unit.eq("protein")
        & table.scoring.eq("native")
        & table.threshold.eq(30)
        & (table.condition.eq("Sall") | (table.arm.eq("stabilityarc_kermut") & table.condition.isna()))
        & table.arm.isin(ARM_ORDER)
        & table.budget.isin(BUDGETS)
    ].copy()
    assert set(table.arm.unique()) == set(ARM_ORDER)
    assert set(table.budget.unique()) == set(BUDGETS)
    assert set(table.seed.unique()) == set(SEEDS)
    assert table.fold.nunique() == 5
    assert table.groupby(["arm", "protein_id", "seed", "fold", "budget"]).size().eq(1).all()
    assert table.compensation_hit_rate.notna().all()
    assert table.selected_mean_recovery.notna().all()
    counts = table.groupby("arm").protein_id.nunique()
    assert counts.eq(120).all()
    return table


def load_random_expectation(random_input):
    random_table = pd.read_parquet(random_input)
    random_table = random_table.loc[
        random_table.threshold.eq(30)
        & random_table.margin.eq(0.0)
        & random_table.budget.isin(BUDGETS)
    ].copy()
    assert set(random_table.budget.unique()) == set(BUDGETS)
    assert random_table.groupby("budget").protein_id.nunique().eq(120).all()
    assert random_table.random_hit_rate.notna().all()
    assert random_table.random_mean_recovery.notna().all()
    expectation = (
        random_table.groupby("budget", as_index=False)[
            ["random_hit_rate", "random_mean_recovery"]
        ]
        .mean()
    )
    assert len(expectation) == len(BUDGETS)
    return expectation


def repeat_protein_means(table, value):
    """Retain one protein-level value per seed and candidate budget."""
    seed_protein = (
        table.groupby(["arm", "protein_id", "seed", "budget"], as_index=False)[value]
        .mean()
    )
    assert seed_protein.groupby(["arm", "budget", "seed"]).protein_id.nunique().eq(120).all()
    return seed_protein


def curve_summary(seed_protein, value):
    per_seed = (
        seed_protein.groupby(["arm", "seed", "budget"], as_index=False)[value]
        .mean()
    )
    summary = (
        per_seed.groupby(["arm", "budget"], as_index=False)[value]
        .agg(mean="mean", sd=lambda x: x.std(ddof=1))
    )
    assert summary.groupby("arm").budget.nunique().eq(4).all()
    assert per_seed.groupby(["arm", "budget"]).size().eq(5).all()
    return per_seed, summary


def protein_average(seed_protein, value):
    result = seed_protein.groupby(["arm", "protein_id", "budget"], as_index=False)[value].mean()
    assert result.groupby(["arm", "budget"]).protein_id.nunique().eq(120).all()
    return result


def paired_repeat_gains(seed_protein, value, budget):
    frame = seed_protein.loc[seed_protein.budget.eq(budget)].copy()
    shared = frame.loc[frame.arm.eq("interaction"), ["protein_id", "seed", value]].rename(columns={value: "shared"})
    assert shared.groupby("seed").protein_id.nunique().eq(120).all()
    rows = []
    for arm in CONTROL_ORDER:
        control = frame.loc[frame.arm.eq(arm), ["protein_id", "seed", value]].rename(columns={value: "control"})
        paired = shared.merge(control, on=["protein_id", "seed"], validate="one_to_one")
        assert paired.groupby("seed").protein_id.nunique().eq(120).all()
        paired["gain"] = paired.shared - paired.control
        repeat = paired.groupby("seed", as_index=False).gain.mean()
        repeat["arm"] = arm
        rows.append(repeat)
    result = pd.concat(rows, ignore_index=True)
    assert result.groupby("arm").seed.nunique().eq(5).all()
    return result


def plot_budget_curves(
    ax,
    summary,
    random_expectation,
    value,
    ylabel,
    panel_label,
    ylim=None,
    label_x=-0.12,
    label_y=1.08,
):
    x = np.arange(len(BUDGETS))
    random_value = (
        "random_hit_rate" if value == "compensation_hit_rate" else "random_mean_recovery"
    )
    ax.plot(
        x,
        random_expectation.set_index("budget").reindex(BUDGETS)[random_value],
        color="#313A40",
        linewidth=1.0,
        linestyle=(0, (3.2, 2.2)),
        label="Uniform random",
        zorder=1,
    )
    for arm in ARM_ORDER:
        current = summary.loc[summary.arm.eq(arm)].set_index("budget").reindex(BUDGETS)
        line_width = 1.55 if arm == "interaction" else 0.95
        marker_size = 4.5 if arm == "interaction" else 3.4
        ax.plot(
            x,
            current["mean"],
            color=COLORS[arm],
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgewidth=0.75,
            markeredgecolor=COLORS[arm],
            label=LABELS[arm],
            zorder=4 if arm == "interaction" else 3,
        )
    ax.set_xticks(x, [str(value) for value in BUDGETS])
    ax.set_xlabel("Candidate budget $K$", fontsize=10, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)
    ax.text(label_x, label_y, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def plot_gain_distribution(ax, protein_values, panel_label):
    positions = np.arange(len(CONTROL_ORDER))
    all_values = []
    for pos, arm in enumerate(CONTROL_ORDER):
        values = protein_values.loc[protein_values.arm.eq(arm), "gain"].to_numpy(dtype=float)
        assert len(values) == 120
        all_values.append(values)
        jitter = np.linspace(-0.15, 0.15, len(values))
        ax.scatter(
            values,
            np.full(len(values), positions[pos]) + jitter,
            s=4.2,
            facecolors="none",
            edgecolors=COLORS[arm],
            linewidths=0.45,
            alpha=0.40,
            zorder=2,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        mean = float(np.mean(values))
        ax.plot([q1, q3], [positions[pos], positions[pos]], color=COLORS[arm], linewidth=2.0, zorder=3)
        ax.plot([median, median], [positions[pos] - 0.10, positions[pos] + 0.10], color="#26313A", linewidth=0.85, zorder=4)
        ax.scatter([mean], [positions[pos]], marker="D", s=22, facecolor="white", edgecolor=COLORS[arm], linewidth=0.9, zorder=5)
    minimum = min(float(np.min(values)) for values in all_values)
    maximum = max(float(np.max(values)) for values in all_values)
    span = max(maximum - minimum, 0.05)
    ax.axvline(0.0, color="#626D76", linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
    ax.set_yticks(positions, [SHORT[arm] for arm in CONTROL_ORDER])
    ax.set_ylim(-0.55, len(CONTROL_ORDER) - 0.45)
    ax.set_xlim(minimum - span * 0.08, maximum + span * 0.08)
    ax.set_xlabel("Native $K=10$ hit-rate gain\n(SCEPI $-$ control)", fontsize=10, labelpad=3)
    style_axis(ax)
    ax.text(-0.12, 1.08, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.99, 1.08, "n=120 proteins", transform=ax.transAxes, fontsize=10, ha="right", va="top", color="#5D6871")


def plot_repeat_gains(ax, repeat_values, panel_label):
    positions = np.arange(len(CONTROL_ORDER))
    summaries = []
    for arm in CONTROL_ORDER:
        values = repeat_values.loc[repeat_values.arm.eq(arm), "gain"].to_numpy(dtype=float)
        assert len(values) == 5
        summaries.append((arm, values, float(values.mean()), float(values.std(ddof=1))))
    minimum = min(mean - sd for _, _, mean, sd in summaries)
    maximum = max(mean + sd for _, _, mean, sd in summaries)
    span = max(maximum - minimum, 0.04)
    for pos, (arm, values, mean, sd) in enumerate(summaries):
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            values,
            np.full(len(values), positions[pos]) + jitter,
            s=25,
            facecolors="white",
            edgecolors=COLORS[arm],
            linewidths=0.9,
            zorder=3,
        )
        ax.errorbar(
            mean,
            positions[pos],
            xerr=sd,
            fmt="none",
            ecolor=COLORS[arm],
            elinewidth=1.55,
            capsize=2.2,
            capthick=0.75,
            zorder=2,
        )
        ax.scatter([mean], [positions[pos]], marker="D", s=29, facecolor=COLORS[arm], edgecolor="white", linewidth=0.55, zorder=4)
    ax.axvline(0.0, color="#626D76", linewidth=0.7, linestyle=(0, (2, 2)))
    ax.set_yticks(positions)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(-0.55, len(CONTROL_ORDER) - 0.45)
    ax.set_xlim(minimum - span * 0.12, maximum + span * 0.12)
    ax.set_xticks([0.00, 0.02, 0.04, 0.06, 0.08], ["0.00", "0.02", "0.04", "0.06", "0.08"])
    ax.set_xlabel("Native $K=10$ hit-rate gain\n(SCEPI $-$ control)", fontsize=10, labelpad=3)
    style_axis(ax)
    ax.text(-0.12, 1.08, panel_label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.99, 1.08, "5 repeats", transform=ax.transAxes, fontsize=10, ha="right", va="top", color="#5D6871")


def tikz_number(value):
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def tikz_coordinates(values_x, values_y):
    return " ".join(f"({tikz_number(x)},{tikz_number(y)})" for x, y in zip(values_x, values_y))


def write_tikz(hit_summary, recovery_summary, random_expectation, protein_gains, repeat_gains, output_dir):
    color_names = {
        "interaction": "SharedBlue",
        "additive": "AdditiveGray",
        "global_nonlinear": "GlobalGold",
        "thermo_epistatic": "ThermoOrange",
        "kermut": "KermutGreen",
        "stabilityarc_kermut": "StabilityArcTeal",
        "mean_mlp": "MeanMlpPurple",
        "mean_ridge": "MeanRidgeRose",
    }
    tikz = []
    append = tikz.append
    append("% Self-contained PGFPlots fragment generated from the verified family30 S-all selection tables.")
    append("% The 120 protein-level coordinates for each control in panels c and d are embedded explicitly.")
    append(r"\begin{tikzpicture}")
    append(r"\definecolor{SharedBlue}{HTML}{2878B5}")
    append(r"\definecolor{AdditiveGray}{HTML}{6F7785}")
    append(r"\definecolor{GlobalGold}{HTML}{B89B3C}")
    append(r"\definecolor{ThermoOrange}{HTML}{E68632}")
    append(r"\definecolor{KermutGreen}{HTML}{32936F}")
    append(r"\definecolor{StabilityArcTeal}{HTML}{3F9BA8}")
    append(r"\definecolor{MeanMlpPurple}{HTML}{9368A6}")
    append(r"\definecolor{MeanRidgeRose}{HTML}{AC6D79}")
    append(r"\definecolor{CompensationDark}{HTML}{26313A}")
    append(r"\pgfplotsset{compat=1.18}")
    append(r"\begin{groupplot}[")
    append(r"  group style={group size=2 by 2, horizontal sep=14mm, vertical sep=16mm},")
    append(r"  width=54mm, height=31mm, scale only axis,")
    append(r"  tick label style={font=\normalsize}, label style={font=\normalsize},")
    append(r"  axis line style={draw={gray!60}, line width=0.35pt},")
    append(r"  grid=major, grid style={draw={gray!25}, line width=0.25pt},")
    append(r"  tick style={draw={gray!60}, line width=0.35pt},")
    append(r"  every axis plot/.append style={line join=round},")
    append(r"]")

    for panel, value, ylabel, title, ylim in [
        ("a.", "compensation_hit_rate", "Compensation hit rate", "a.", (0.0, 0.20)),
        ("b.", "selected_mean_recovery", r"Mean recovery (kcal/mol)", "b.", (-1.0, -0.05)),
    ]:
        summary = hit_summary if value == "compensation_hit_rate" else recovery_summary
        random_value = (
            "random_hit_rate" if value == "compensation_hit_rate" else "random_mean_recovery"
        )
        append(r"\nextgroupplot[")
        append("  title={\\textbf{" + title + "}}, title style={at={(axis description cs:-0.14,1.10)}, anchor=south west, font=\\normalsize},")
        append(f"  ylabel={{{ylabel}}}, xmin=-0.15, xmax=3.15, ymin={tikz_number(ylim[0])}, ymax={tikz_number(ylim[1])},")
        append(r"  xtick={0,1,2,3}, xticklabels={1,3,10,30}, xlabel={Candidate budget $K$},")
        if value == "compensation_hit_rate":
            append(r"  legend to name=compensationdiscoverylegend, legend style={draw=none, legend columns=3, font=\normalsize, column sep=4pt, row sep=1pt},")
        append(r"]")
        random_current = random_expectation.set_index("budget").reindex(BUDGETS)
        append(
            f"  \\addplot+[color=CompensationDark, dashed, line width=0.7pt, mark=none] coordinates "
            f"{{{tikz_coordinates(range(len(BUDGETS)), random_current[random_value].to_numpy(dtype=float))}}};"
        )
        if value == "compensation_hit_rate":
            append(r"  \addlegendentry{Uniform random}")
        for arm in ARM_ORDER:
            current = summary.loc[summary.arm.eq(arm)].set_index("budget").reindex(BUDGETS)
            xs = range(len(BUDGETS))
            ys = current["mean"].to_numpy(dtype=float)
            width = "1.1pt" if arm == "interaction" else "0.7pt"
            mark_size = "1.9pt" if arm == "interaction" else "1.5pt"
            append(
                f"  \\addplot+[color={color_names[arm]}, line width={width}, mark=o, mark size={mark_size}, "
                f"mark options={{fill=white}}] coordinates {{{tikz_coordinates(xs, ys)}}};"
            )
            if value == "compensation_hit_rate":
                append(f"  \\addlegendentry{{{LABELS[arm]}}}")

    gain_values = []
    for arm in CONTROL_ORDER:
        values = protein_gains.loc[protein_gains.arm.eq(arm), "gain"].to_numpy(dtype=float)
        assert len(values) == 120
        gain_values.append(values)
    gain_minimum = min(float(np.min(values)) for values in gain_values)
    gain_maximum = max(float(np.max(values)) for values in gain_values)
    gain_span = max(gain_maximum - gain_minimum, 0.05)
    gain_xmin = gain_minimum - gain_span * 0.08
    gain_xmax = gain_maximum + gain_span * 0.08
    append(r"\nextgroupplot[")
    append(r"  xlabel={Native $K=10$ hit-rate gain\\(SCEPI $-$ control)},")
    append(r"  xlabel style={align=center},")
    max_position = len(CONTROL_ORDER) - 1
    tick_positions = ",".join(str(position) for position in range(len(CONTROL_ORDER)))
    tick_labels = ",".join(SHORT[arm] for arm in CONTROL_ORDER)
    append(f"  ymin=-0.55, ymax={tikz_number(max_position + 0.45)}, xmin={tikz_number(gain_xmin)}, xmax={tikz_number(gain_xmax)},")
    append(f"  ytick={{{tick_positions}}}, yticklabels={{{tick_labels}}},")
    append(r"  title={\textbf{c.}}, title style={at={(axis description cs:-0.14,1.10)}, anchor=south west, font=\normalsize},")
    append(r"]")
    for position, arm in enumerate(CONTROL_ORDER):
        values = protein_gains.loc[protein_gains.arm.eq(arm), "gain"].to_numpy(dtype=float)
        assert len(values) == 120
        jitter = np.linspace(-0.15, 0.15, len(values))
        point_y = np.full(len(values), position, dtype=float) + jitter
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        mean = float(np.mean(values))
        color = color_names[arm]
        append(
            f"  \\addplot+[only marks, mark=o, mark size=0.65pt, mark options={{draw={color}, fill=white, opacity=0.42}}] "
            f"coordinates {{{tikz_coordinates(values, point_y)}}};"
        )
        append(f"  \\addplot[color={color}, line width=1.25pt] coordinates {{({tikz_number(q1)},{position}) ({tikz_number(q3)},{position})}};")
        append(f"  \\addplot[color=CompensationDark, line width=0.5pt] coordinates {{({tikz_number(median)},{tikz_number(position - 0.10)}) ({tikz_number(median)},{tikz_number(position + 0.10)})}};")
        append(
            f"  \\addplot+[only marks, mark=diamond*, mark size=2.0pt, mark options={{draw={color}, fill=white}}] "
            f"coordinates {{({tikz_number(mean)},{position})}};"
        )
    append(f"  \\addplot[color=gray!65, dashed, line width=0.35pt] coordinates {{(0,-0.55) (0,{tikz_number(max_position + 0.45)})}};")
    append(r"  \node[anchor=south east, font=\normalsize, text=gray!70] at (axis description cs:1,1) {n=120 proteins};")

    repeat_minimum = min(float(repeat_gains.loc[repeat_gains.arm.eq(arm), "gain"].mean() - repeat_gains.loc[repeat_gains.arm.eq(arm), "gain"].std(ddof=1)) for arm in CONTROL_ORDER)
    repeat_maximum = max(float(repeat_gains.loc[repeat_gains.arm.eq(arm), "gain"].mean() + repeat_gains.loc[repeat_gains.arm.eq(arm), "gain"].std(ddof=1)) for arm in CONTROL_ORDER)
    repeat_span = max(repeat_maximum - repeat_minimum, 0.04)
    repeat_xmin = repeat_minimum - repeat_span * 0.12
    repeat_xmax = repeat_maximum + repeat_span * 0.12
    append(r"\nextgroupplot[")
    append(r"  xlabel={Native $K=10$ hit-rate gain\\(SCEPI $-$ control)},")
    append(r"  xlabel style={align=center},")
    append(f"  xmin={tikz_number(repeat_xmin)}, xmax={tikz_number(repeat_xmax)}, ymin=-0.55, ymax={tikz_number(max_position + 0.45)}, scaled x ticks=false,")
    append(r"  xtick={0,0.02,0.04,0.06,0.08}, xticklabels={0.00,0.02,0.04,0.06,0.08},")
    append(f"  ytick={{{tick_positions}}}, yticklabels={{}},")
    append(r"  title={\textbf{d.}}, title style={at={(axis description cs:-0.14,1.10)}, anchor=south west, font=\normalsize},")
    append(r"]")
    for position, arm in enumerate(CONTROL_ORDER):
        values = repeat_gains.loc[repeat_gains.arm.eq(arm), "gain"].to_numpy(dtype=float)
        assert len(values) == 5
        jitter = np.linspace(-0.12, 0.12, len(values))
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        color = color_names[arm]
        append(
            f"  \\addplot+[only marks, mark=o, mark size=1.55pt, mark options={{draw={color}, fill=white}}] "
            f"coordinates {{{tikz_coordinates(values, np.full(len(values), position, dtype=float) + jitter)}}};"
        )
        append(f"  \\addplot[color={color}, line width=1.15pt] coordinates {{({tikz_number(mean - sd)},{position}) ({tikz_number(mean + sd)},{position})}};")
        append(f"  \\addplot[color={color}, line width=0.6pt] coordinates {{({tikz_number(mean - sd)},{tikz_number(position - 0.07)}) ({tikz_number(mean - sd)},{tikz_number(position + 0.07)})}};")
        append(f"  \\addplot[color={color}, line width=0.6pt] coordinates {{({tikz_number(mean + sd)},{tikz_number(position - 0.07)}) ({tikz_number(mean + sd)},{tikz_number(position + 0.07)})}};")
        append(
            f"  \\addplot+[only marks, mark=diamond*, mark size=2.0pt, mark options={{draw=white, fill={color}}}] "
            f"coordinates {{({tikz_number(mean)},{position})}};"
        )
    append(f"  \\addplot[color=gray!65, dashed, line width=0.35pt] coordinates {{(0,-0.55) (0,{tikz_number(max_position + 0.45)})}};")
    append(r"  \node[anchor=south east, font=\normalsize, text=gray!70] at (axis description cs:1,1) {5 repeats};")
    append(r"\end{groupplot}")
    append(r"\node[anchor=north, yshift=-3mm] at (current bounding box.south) {\pgfplotslegendfromname{compensationdiscoverylegend}};")
    append(r"\end{tikzpicture}")
    (output_dir / "compensation_discovery.tikz").write_text("\n".join(tikz) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot Figure 2 from a paper_evidence selection table and random_enrichment analysis."
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="Directory containing selection_per_protein.parquet.",
    )
    parser.add_argument(
        "--random-dir",
        type=Path,
        help="Directory containing random_per_protein.parquet.",
    )
    parser.add_argument(
        "--selection-input",
        type=Path,
        help="Explicit selection table; overrides --evidence-dir and MAIN_INPUT.",
    )
    parser.add_argument(
        "--random-input",
        type=Path,
        help="Explicit random expectation table; overrides --random-dir and RANDOM_INPUT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for compensation_discovery.{pdf,png,svg,tikz}.",
    )
    return parser.parse_args()


def resolve_input(explicit, directory, environment_name, directory_filename, default):
    if explicit is not None:
        return explicit
    if directory is not None:
        return directory / directory_filename
    environment_value = os.environ.get(environment_name)
    return Path(environment_value) if environment_value else default


def main():
    args = parse_args()
    selection_input = resolve_input(
        args.selection_input,
        args.evidence_dir,
        "MAIN_INPUT",
        "selection_per_protein.parquet",
        DEFAULT_SELECTION_INPUT,
    )
    random_input = resolve_input(
        args.random_input,
        args.random_dir,
        "RANDOM_INPUT",
        "random_per_protein.parquet",
        DEFAULT_RANDOM_INPUT,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    table = load_data(selection_input)
    random_expectation = load_random_expectation(random_input)
    hit_seed = repeat_protein_means(table, "compensation_hit_rate")
    recovery_seed = repeat_protein_means(table, "selected_mean_recovery")
    _, hit_summary = curve_summary(hit_seed, "compensation_hit_rate")
    _, recovery_summary = curve_summary(recovery_seed, "selected_mean_recovery")
    hit_protein = protein_average(hit_seed, "compensation_hit_rate")
    hit_protein_k10 = hit_protein.loc[hit_protein.budget.eq(10)].copy()
    shared_k10 = hit_protein_k10.loc[hit_protein_k10.arm.eq("interaction"), ["protein_id", "compensation_hit_rate"]].rename(columns={"compensation_hit_rate": "shared"})
    gains = []
    for arm in CONTROL_ORDER:
        control = hit_protein_k10.loc[hit_protein_k10.arm.eq(arm), ["protein_id", "compensation_hit_rate"]].rename(columns={"compensation_hit_rate": "control"})
        paired = shared_k10.merge(control, on="protein_id", validate="one_to_one")
        assert len(paired) == 120
        paired["gain"] = paired.shared - paired.control
        paired["arm"] = arm
        gains.append(paired[["protein_id", "arm", "gain"]])
    protein_gains = pd.concat(gains, ignore_index=True)
    repeat_gains = paired_repeat_gains(hit_seed, "compensation_hit_rate", budget=10)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(6.5, 4.6), facecolor="white")
    layout = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.06, 0.94],
        height_ratios=[0.93, 1.07],
        wspace=0.42,
        hspace=0.82,
        left=0.255,
        right=0.985,
        top=0.935,
        bottom=0.36,
    )
    ax_a = fig.add_subplot(layout[0, 0])
    ax_b = fig.add_subplot(layout[0, 1])
    ax_c = fig.add_subplot(layout[1, 0])
    ax_d = fig.add_subplot(layout[1, 1])
    plot_budget_curves(ax_a, hit_summary, random_expectation, "compensation_hit_rate", "Compensation hit rate", "a.", ylim=(0.0, 0.20), label_x=-0.18, label_y=1.15)
    plot_budget_curves(ax_b, recovery_summary, random_expectation, "selected_mean_recovery", "Mean recovery\n(kcal/mol)", "b.", ylim=(-1.0, -0.05))
    plot_gain_distribution(ax_c, protein_gains, "c.")
    plot_repeat_gains(ax_d, repeat_gains, "d.")

    legend_order = ARM_ORDER
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[arm],
            linewidth=1.55 if arm == "interaction" else 1.0,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COLORS[arm],
            markeredgewidth=0.65,
            markersize=4.4,
            label=LABELS[arm],
        )
        for arm in legend_order
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#313A40",
            linewidth=1.0,
            linestyle=(0, (3.2, 2.2)),
            label="Uniform random",
        )
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.025), ncol=3, frameon=False, fontsize=10, handletextpad=0.35, columnspacing=1.05)
    for suffix in ["pdf", "svg"]:
        fig.savefig(output_dir / ("compensation_discovery." + suffix), facecolor="white")
    fig.savefig(output_dir / "compensation_discovery.png", dpi=320, facecolor="white")
    plt.close(fig)
    write_tikz(hit_summary, recovery_summary, random_expectation, protein_gains, repeat_gains, output_dir)

    print("cohort proteins=120, family_groups=%d, seeds=5, folds=5" % table.family_group.nunique())
    print("selection input=%s" % selection_input)
    print("random expectation input=%s" % random_input)
    print("selection records=%d" % len(table))
    print("hit curves")
    print(hit_summary.to_string(index=False))
    print("recovery curves")
    print(recovery_summary.to_string(index=False))
    print("uniform random expectation")
    print(random_expectation.to_string(index=False))
    print("protein gain summary")
    print(protein_gains.groupby("arm").gain.agg(["mean", "std", "min", "max", lambda x: (x > 0).sum(), lambda x: (x == 0).sum()]).rename(columns={"<lambda_0>": "wins", "<lambda_1>": "ties"}).to_string())
    print("repeat gain summary")
    print(repeat_gains.groupby("arm").gain.agg(["mean", "std", "min", "max"]).to_string())
    print("figure output=%s" % output_dir)
    print("tikz output=%s" % (output_dir / "compensation_discovery.tikz"))


if __name__ == "__main__":
    main()
