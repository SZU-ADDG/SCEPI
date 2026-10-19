from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/figures"
BLUE = "#2878B5"
GREEN = "#32936F"
GRAY = "#6F7785"


def box(ax, xy, width, height, text, color=BLUE, fill="#F4F8FC", fontsize=10):
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.012,rounding_size=0.025",
                              edgecolor=color, facecolor=fill, linewidth=0.8))
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, first, second, color=GRAY, style="-", connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(first, second, arrowstyle="-|>", mutation_scale=8,
                                color=color, linewidth=0.8, linestyle=style, connectionstyle=connection))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.96, bottom=0.04)
    ax.set(xlim=(0, 10.0), ylim=(0, 4.0))
    ax.axis("off")
    for letter, x, width in [("a", 0.05, 2.9), ("b", 3.12, 3.48), ("c", 6.77, 3.18)]:
        ax.add_patch(FancyBboxPatch((x, 0.92), width, 2.8, boxstyle="round,pad=0.025,rounding_size=0.045",
                                  facecolor="white", edgecolor="#CDD3D9", linewidth=0.7))
        ax.text(x + 0.04, 3.82, letter, fontsize=11, fontweight="bold", ha="left")
    box(ax, (0.23, 3.02), 0.66, 0.42, "$s_p$", color=GRAY, fill="#F6F7F8")
    box(ax, (1.07, 3.02), 0.66, 0.42, "$x_p(A)$", color=GRAY, fill="#F6F7F8")
    box(ax, (1.91, 3.02), 0.66, 0.42, "$x_p(B)$", color=GRAY, fill="#F6F7F8")
    for x in [0.56, 1.40, 2.24]:
        arrow(ax, (x, 3.0), (x, 2.77))
    box(ax, (0.23, 2.17), 2.35, 0.58, "Frozen protein\nencoder")
    ax.text(1.41, 1.77, "Reference and mutant\nsequence features\nAmino-acid identities", ha="center", va="center", fontsize=10)
    arrow(ax, (2.97, 2.43), (3.3, 2.43))
    ax.text(4.84, 3.34, "SCEPI interaction function", ha="center", fontsize=10)
    for x, y, label in [(3.45, 2.68, "$F(a,b)$"), (5.10, 2.68, "$F(a,w_j)$"),
                         (3.45, 1.95, "$F(w_i,b)$"), (5.10, 1.95, "$F(w_i,w_j)$")]:
        box(ax, (x, y), 1.24, 0.49, label)
    ax.text(4.85, 2.93, "$-$", ha="center", va="center", fontsize=12)
    ax.text(4.85, 2.20, "$+$", ha="center", va="center", fontsize=12)
    ax.text(3.28, 2.20, "$-$", ha="center", va="center", fontsize=12)
    ax.text(4.84, 1.56, "$r_{ij}(a,b)$", ha="center", va="center", color=BLUE, fontsize=11)
    ax.text(4.84, 1.15, "$r_{ij}=r_{ji}$\n$r_{ij}(a,w_j)=r_{ij}(w_i,b)=0$", ha="center", va="center", fontsize=10)
    arrow(ax, (6.62, 2.43), (6.94, 2.43))
    box(ax, (6.85, 2.93), 3.0, 0.5, "Single effects + availability\n$\\widetilde a_i,\\ \\widetilde a_j,\\ m_i,\\ m_j$", color=GREEN, fill="#F3F9F6", fontsize=10)
    ax.plot([8.35, 8.35, 4.84], [3.45, 3.65, 3.65], color=GREEN, linewidth=0.8)
    arrow(ax, (4.84, 3.65), (4.84, 3.52), color=GREEN)
    arrow(ax, (8.35, 2.91), (8.35, 2.73), color=GREEN)
    box(ax, (7.03, 2.14), 2.64, 0.57, "$\\widehat\\mu(AB)=\\tau\\,g(\\widetilde a_i+\\widetilde a_j+r_{ij})$", fontsize=10)
    arrow(ax, (8.35, 2.11), (8.35, 1.91))
    box(ax, (7.03, 1.30), 2.64, 0.59, "Rank double mutants\nSelect the top $K$", fontsize=10)
    box(ax, (0.27, 0.04), 3.02, 0.58, "Source quartets:\nWT, A, B, AB", color=GREEN, fill="#F3F9F6", fontsize=10)
    arrow(ax, (3.33, 0.34), (3.69, 0.34), color=GREEN)
    ax.text(5.57, 0.34, "Train shared parameters", ha="center", va="center", color=GREEN, fontsize=10)
    arrow(ax, (4.83, 0.60), (4.83, 0.89), color=GREEN, style="--")
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUTPUT / ("method_overview." + suffix), dpi=320, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
