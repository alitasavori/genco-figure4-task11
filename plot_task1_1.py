"""Task 1.1 panel of GENCO Figure 4.

Colors, category order, and axis text follow
gridfm-graphkit scripts/pfdelta/make_pbl_barplots.py. The published figure
uses the authors' rounded table. The retrained figure uses only rows whose
measured_source is an independent training run with three finite seeds.
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
from published import PUBLISHED_SOURCE

PALETTE = {
    "CANOS-PF": "#D9A6D9",
    "GNS-S": "#F27AA4",
    "PFNet": "#8FD694",
    "GENCO Base": "#69D8FF",
}
RETRAINED_SOURCES = {"independent_training"}
PUBLISHED_YLIM = (0.0, 1.8)
PUBLISHED_YTICKS = [0.0, 0.45, 0.9, 1.35, 1.8]


def _font(size: float) -> font_manager.FontProperties:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Times New Roman" if "Times New Roman" in available else "serif"
    return font_manager.FontProperties(family=family, size=size)


def _rows_for(summary: list[dict], source: str) -> dict[str, dict[str, tuple[float, float]]]:
    selected = [row for row in summary if row.get("measured_source") == source]
    grouped: dict[str, dict[str, tuple[float, float]]] = {}
    for row in selected:
        model = row["model"]
        category = row["test_category"]
        if model not in MODELS or category not in CATEGORIES:
            continue
        n_seeds = row.get("n_seeds", "")
        if source in RETRAINED_SOURCES and str(n_seeds) != "3":
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


def _ylim_for(grouped: dict, published_scale: bool) -> tuple[tuple[float, float], list[float]]:
    ymax = max(mean + std for model in grouped.values() for mean, std in model.values())
    if published_scale and ymax <= PUBLISHED_YLIM[1]:
        return PUBLISHED_YLIM, list(PUBLISHED_YTICKS)
    ymax = max(ymax * 1.1, 1e-6)
    magnitude = 10 ** np.floor(np.log10(ymax))
    ymax = float(np.ceil(ymax / magnitude * 4) / 4 * magnitude)
    return (0.0, ymax), [i * ymax / 4 for i in range(5)]


def plot_task11(results_dir: Path | None = None) -> list[Path]:
    results_dir = results_dir or RESULTS
    summary = read_csv(results_dir / "task1_1_summary.csv")
    written: list[Path] = []
    published = _rows_for(summary, PUBLISHED_SOURCE)
    if len(published) != len(MODELS):
        raise FileNotFoundError(
            "Published Task 1.1 rows are incomplete. Run reproduce_genco.py --mode published first."
        )
    published_pdf = results_dir / "figure4_task1.1_published.pdf"
    ylim, yticks = _ylim_for(published, published_scale=True)
    _draw(published, published_pdf, ylim, yticks)
    written.append(published_pdf)

    retrained = {}
    for source in RETRAINED_SOURCES:
        retrained.update(_rows_for(summary, source))
    retrained_pdf = results_dir / "figure4_task1.1_retrained.pdf"
    note = results_dir / "figure4_task1.1_retrained_NOT_RUN.txt"
    if len(retrained) != len(MODELS):
        if retrained_pdf.exists():
            retrained_pdf.unlink()
        png = retrained_pdf.with_suffix(".png")
        if png.exists():
            png.unlink()
        missing = [model for model in MODELS if model not in retrained]
        note.write_text(
            "No retrained Figure 4 panel was written.\n"
            f"Models without three finite independent seeds: {missing}.\n"
            "Published table values were not copied into this figure.\n",
            encoding="utf-8",
        )
        print(f"retrained figure not written; see {note}")
    else:
        if note.exists():
            note.unlink()
        ylim, yticks = _ylim_for(retrained, published_scale=False)
        _draw(retrained, retrained_pdf, ylim, yticks)
        written.append(retrained_pdf)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Task 1.1 panel of GENCO Figure 4.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    plot_task11(args.results)


if __name__ == "__main__":
    main()
