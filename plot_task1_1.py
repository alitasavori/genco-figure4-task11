"""Task 1.1 panel of GENCO Figure 4, drawn from this run's test scores.

Colors, category order, and axis text follow
gridfm-graphkit scripts/pfdelta/make_pbl_barplots.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from common import CATEGORIES, MODELS, RESULTS, read_csv

PALETTE = {
    "CANOS-PF": "#D9A6D9",
    "GNS-S": "#F27AA4",
    "PFNet": "#8FD694",
    "GENCO Base": "#69D8FF",
}
PAPER_YLIM = (0.0, 1.8)
PUBLISHED_YTICKS = [0.0, 0.45, 0.9, 1.35, 1.8]


def _font(size: float) -> font_manager.FontProperties:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "serif"
    return font_manager.FontProperties(family=family, size=size)


def _rows_for(summary: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    grouped: dict[str, dict[str, tuple[float, float]]] = {}
    for row in summary:
        model = row["model"]
        category = row["test_category"]
        if model not in MODELS or category not in CATEGORIES:
            continue
        if str(row.get("n_seeds", "")) != "3":
            continue
        grouped.setdefault(model, {})[category] = (
            float(row["mean_across_seeds"]),
            float(row["std_across_seeds"]),
        )
    complete = {}
    for model, by_category in grouped.items():
        if all(category in by_category for category in CATEGORIES):
            if all(np.isfinite(by_category[category]).all() for category in CATEGORIES):
                complete[model] = by_category
    return complete


def _draw(grouped: dict, output: Path, ylim: tuple[float, float], yticks: list[float]) -> None:
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    models = [model for model in MODELS if model in grouped]
    fig, ax = plt.subplots(figsize=(8.5, 8))
    x = np.arange(len(CATEGORIES))
    width = 0.18
    fontsize = 18
    for index, model in enumerate(models):
        means = [grouped[model][category][0] for category in CATEGORIES]
        stds = [grouped[model][category][1] for category in CATEGORIES]
        offset = (index - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=model,
            color=PALETTE[model],
            edgecolor="white",
            capsize=3,
            error_kw={"linewidth": 1.2},
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(CATEGORIES))
    ax.set_ylabel("Power Balance Loss (Mean) [p.u.]")
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    tick_font = _font(fontsize)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(tick_font)
    ax.yaxis.label.set_fontproperties(_font(fontsize))
    ax.text(
        0.97,
        0.95,
        "Task 1.1",
        transform=ax.transAxes,
        fontproperties=_font(fontsize),
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    legend = ax.legend(frameon=False, prop=_font(14), loc="upper left", title="Model")
    legend.get_title().set_fontproperties(_font(14))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    png = output.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {output}")
    print(f"wrote {png}")


def _ylim_for(grouped: dict) -> tuple[tuple[float, float], list[float]]:
    ymax = max(mean + std for model in grouped.values() for mean, std in model.values())
    if ymax <= PAPER_YLIM[1]:
        return PAPER_YLIM, list(PUBLISHED_YTICKS)
    ymax = max(ymax * 1.1, 1e-6)
    magnitude = 10 ** np.floor(np.log10(ymax))
    ymax = float(np.ceil(ymax / magnitude * 4) / 4 * magnitude)
    return (0.0, ymax), [i * ymax / 4 for i in range(5)]


def plot_task11(results_dir: Path | None = None) -> list[Path]:
    results_dir = results_dir or RESULTS
    summary = read_csv(results_dir / "task1_1_summary.csv")
    measured = _rows_for(summary)
    pdf = results_dir / "figure4_task1.1.pdf"
    note = results_dir / "figure4_task1.1_NOT_READY.txt"
    if len(measured) != len(MODELS):
        if pdf.exists():
            pdf.unlink()
        png = pdf.with_suffix(".png")
        if png.exists():
            png.unlink()
        missing = [model for model in MODELS if model not in measured]
        note.write_text(
            "Figure 4 Task 1.1 was not written.\n"
            f"Models without three finite test seeds: {missing}.\n",
            encoding="utf-8",
        )
        print(f"figure not written; see {note}")
        return []
    if note.exists():
        note.unlink()
    ylim, yticks = _ylim_for(measured)
    _draw(measured, pdf, ylim, yticks)
    return [pdf]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Task 1.1 panel of GENCO Figure 4.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    plot_task11(args.results)


if __name__ == "__main__":
    main()
