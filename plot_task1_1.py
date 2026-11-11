"""Task 1.1 panel of GENCO Figure 4, and TPGT-vs-published bar plots for every PFΔ IEEE 118 task.

Colors, category order, and axis text follow
gridfm-graphkit scripts/pfdelta/make_pbl_barplots.py.
The published table is table_a_8_genco_exponent_fixed.tex (``PUBLISHED``; sources in ``PUBLISHED_SOURCES``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CATEGORIES, MODELS, RESULTS, read_csv
from plot_style import FONT_LABEL, FONT_LEGEND, apply_plot_style, font_properties, style_axis

PALETTE = {
    "CANOS-PF": "#D9A6D9",
    "GNS-S": "#F27AA4",
    "PFNet": "#8FD694",
    "GENCO Base": "#69D8FF",
}
PAPER_YLIM = (0.0, 1.8)
PUBLISHED_YTICKS = [0.0, 0.45, 0.9, 1.35, 1.8]

TPGT_COLOR = "#1F4B99"

# Mean ± std (p.u., 100 MVA base, three seeds) of mean power-balance loss from the table the authors
# plot as GENCO Figure 4, keyed by task id ("1.3" is the "1.3, 2.1" row). GNS is labeled GNS-S.
PUBLISHED = {
    "1.1": {
        "GENCO Base": {"N": (1.4e-3, 0.4e-3), "N-1": (4.2e-2, 2.2e-2), "N-2": (5.0e-2, 2.6e-2), "C2I": (3.0e-1, 0.8e-1)},
        "CANOS-PF": {"N": (2.7e-2, 0.03e-2), "N-1": (7.0e-1, 0.8e-1), "N-2": (5.6e-1, 0.7e-1), "C2I": (1.1, 0.1)},
        "GNS-S": {"N": (5.3e-1, 2.7e-1), "N-1": (6.8e-1, 2.6e-1), "N-2": (7.1e-1, 2.5e-1), "C2I": (1.0, 0.2)},
        "PFNet": {"N": (3.2e-1, 0.2e-1), "N-1": (5.2e-1, 0.2e-1), "N-2": (5.3e-1, 0.3e-1), "C2I": (1.4, 0.1)},
    },
    "1.2": {
        "GENCO Base": {"N": (1.7e-3, 1.3e-3), "N-1": (3.7e-3, 2.5e-3), "N-2": (5.5e-3, 3.7e-3), "C2I": (1.8e-1, 0.4e-1)},
        "CANOS-PF": {"N": (3.8e-2, 0.2e-2), "N-1": (5.9e-2, 0.2e-2), "N-2": (7.7e-2, 0.3e-2), "C2I": (7.5e-1, 0.5e-1)},
        "GNS-S": {"N": (3.6e-1, 0.3e-1), "N-1": (4.0e-1, 0.3e-1), "N-2": (4.1e-1, 0.3e-1), "C2I": (8.2e-1, 1.1e-1)},
        "PFNet": {"N": (3.2e-1, 0.2e-1), "N-1": (4.0e-1, 0.2e-1), "N-2": (4.4e-1, 0.2e-1), "C2I": (1.2, 0.05)},
    },
    "1.3": {
        "GENCO Base": {"N": (2.7e-3, 2.2e-3), "N-1": (4.6e-3, 3.4e-3), "N-2": (5.9e-3, 4.3e-3), "C2I": (2.2e-1, 0.8e-1)},
        "CANOS-PF": {"N": (4.0e-2, 0.4e-2), "N-1": (5.6e-2, 0.5e-2), "N-2": (6.3e-2, 0.5e-2), "C2I": (7.2e-1, 0.9e-1)},
        "GNS-S": {"N": (3.5e-1, 0.4e-1), "N-1": (3.8e-1, 0.4e-1), "N-2": (3.9e-1, 0.4e-1), "C2I": (7.7e-1, 0.5e-1)},
        "PFNet": {"N": (3.4e-1, 0.08e-1), "N-1": (4.0e-1, 0.05e-1), "N-2": (4.2e-1, 0.09e-1), "C2I": (1.1, 0.03)},
    },
    "2.3": {
        "GENCO Base": {"N": (5.0e-3, 2.5e-3), "N-1": (8.1e-3, 3.7e-3), "N-2": (9.9e-3, 4.2e-3), "C2I": (2.5e-1, 0.4e-1)},
        "CANOS-PF": {"N": (9.0e-2, 1.7e-2), "N-1": (1.2e-1, 0.2e-1), "N-2": (1.3e-1, 0.2e-1), "C2I": (8.3e-1, 0.4e-1)},
        "GNS-S": {"N": (8.3e-1, 2.6e-1), "N-1": (8.8e-1, 2.7e-1), "N-2": (8.9e-1, 2.7e-1), "C2I": (1.3, 0.3)},
        "PFNet": {"N": (4.1e-1, 0.6e-1), "N-1": (4.9e-1, 0.7e-1), "N-2": (5.1e-1, 0.7e-1), "C2I": (1.1, 0.08)},
    },
    "4.1": {
        "GENCO Base": {"N": (2.7e-2, 1.8e-2), "N-1": (3.6e-2, 2.4e-2), "N-2": (4.2e-2, 2.7e-2), "C2I": (2.1e-1, 1.2e-1)},
        "CANOS-PF": {"N": (1.6e-1, 0.1e-1), "N-1": (2.0e-1, 0.2e-1), "N-2": (2.1e-1, 0.2e-1), "C2I": (8.0e-1, 0.4e-1)},
        "GNS-S": {"N": (3.5e-1, 0.1e-1), "N-1": (3.7e-1, 0.2e-1), "N-2": (3.8e-1, 0.2e-1), "C2I": (6.6e-1, 0.3e-1)},
        "PFNet": {"N": (4.5e-1, 0.4e-1), "N-1": (5.2e-1, 0.09e-1), "N-2": (5.3e-1, 0.07e-1), "C2I": (1.3, 0.04)},
    },
    "4.2": {
        "GENCO Base": {"N": (4.4e-2, 3.8e-2), "N-1": (6.0e-2, 5.0e-2), "N-2": (7.0e-2, 5.7e-2), "C2I": (1.8e-1, 1.2e-1)},
        "CANOS-PF": {"N": (1.2e-1, 0.02e-1), "N-1": (1.5e-1, 0.02e-1), "N-2": (1.6e-1, 0.03e-1), "C2I": (4.3e-1, 0.1e-1)},
        "GNS-S": {"N": (4.0e-1, 0.9e-1), "N-1": (4.6e-1, 1.0e-1), "N-2": (4.6e-1, 1.0e-1), "C2I": (6.4e-1, 1.4e-1)},
        "PFNet": {"N": (5.6e-1, 0.7e-1), "N-1": (6.6e-1, 0.7e-1), "N-2": (6.7e-1, 0.7e-1), "C2I": (1.2, 0.1)},
    },
    "4.3": {
        "GENCO Base": {"N": (1.5e-1, 0.5e-1), "N-1": (2.0e-1, 0.6e-1), "N-2": (2.2e-1, 0.6e-1), "C2I": (1.3e-1, 0.5e-1)},
        "CANOS-PF": {"N": (4.5e-1, 0.2e-1), "N-1": (5.7e-1, 0.2e-1), "N-2": (6.0e-1, 0.3e-1), "C2I": (3.7e-1, 0.2e-1)},
        "GNS-S": {"N": (6.3e-1, 2.2e-1), "N-1": (7.3e-1, 2.4e-1), "N-2": (7.3e-1, 2.4e-1), "C2I": (7.0e-1, 2.8e-1)},
        "PFNet": {"N": (1.2, 0.04), "N-1": (1.4, 0.04), "N-2": (1.5, 0.02), "C2I": (1.1, 0.07)},
    },
}
PUBLISHED_TASK11 = PUBLISHED["1.1"]

_GENCO_TABLE = (
    "gridfm-graphkit 310e2f989b46058f3e6c5b4a56b7d03de06a3340 scripts/pfdelta/table_a_8_genco_exponent_fixed.tex"
)
_PFDELTA_TABLE = "PFΔ arXiv 2510.22048 Table A.8 (Power Balance Loss (Mean))"
PUBLISHED_SOURCES = {
    task: (
        f"GENCO Base: {_GENCO_TABLE}, row {row} (= scripts/pfdelta/results/{task}/metrics_summary.csv rounded); "
        f"CANOS-PF, GNS-S (GNS), PFNet: same table, identical to {_PFDELTA_TABLE}, row {row}"
    )
    for task, row in (
        ("1.1", "1.1"),
        ("1.2", "1.2"),
        ("1.3", "1.3, 2.1"),
        ("2.3", "2.3"),
        ("4.1", "4.1"),
        ("4.2", "4.2"),
        ("4.3", "4.3"),
    )
}


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
    apply_plot_style()
    models = [model for model in MODELS if model in grouped]
    fig, ax = plt.subplots(figsize=(8.5, 8))
    x = np.arange(len(CATEGORIES))
    width = 0.18
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
    style_axis(ax)
    ax.text(
        0.97,
        0.95,
        "Task 1.1",
        transform=ax.transAxes,
        fontproperties=font_properties(FONT_LABEL),
        va="top",
        ha="right",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    legend = ax.legend(frameon=False, prop=font_properties(FONT_LEGEND), loc="upper left", title="Model")
    legend.get_title().set_fontproperties(font_properties(FONT_LEGEND))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        metadata={"Title": "GENCO Figure 4, Task 1.1"},
    )
    plt.close(fig)
    for stale in (output.with_suffix(".pdf"), output.with_suffix(".png")):
        if stale.exists():
            stale.unlink()
    print(f"wrote {output}")
    try:
        from IPython.display import SVG, display

        display(SVG(filename=str(output)))
    except Exception:
        pass


def _ylim_for(grouped: dict) -> tuple[tuple[float, float], list[float]]:
    ymax = max(mean + std for model in grouped.values() for mean, std in model.values())
    if ymax <= PAPER_YLIM[1]:
        return PAPER_YLIM, list(PUBLISHED_YTICKS)
    ymax = max(ymax * 1.1, 1e-6)
    magnitude = 10 ** np.floor(np.log10(ymax))
    ymax = float(np.ceil(ymax / magnitude * 4) / 4 * magnitude)
    return (0.0, ymax), [i * ymax / 4 for i in range(5)]


def plot_tpgt_comparison(
    task: str,
    label: str,
    tpgt: dict[str, tuple[float, float]],
    output: Path,
    tpgt_label: str = "TPGT (this run)",
) -> list[Path]:
    """Mean PBL per test category: published models of ``task`` next to TPGT; writes ``output`` (SVG) and PNG."""
    published = PUBLISHED.get(task, {})
    models = [model for model in MODELS if model in published]
    labels = [f"{model} (published)" for model in models] + [tpgt_label]
    colors = [PALETTE[model] for model in models] + [TPGT_COLOR]
    means = [[published[model][category][0] for category in CATEGORIES] for model in models]
    stds = [[published[model][category][1] for category in CATEGORIES] for model in models]
    means.append([tpgt[category][0] for category in CATEGORIES])
    stds.append([tpgt[category][1] for category in CATEGORIES])
    top = max(mean + std for row_mean, row_std in zip(means, stds) for mean, std in zip(row_mean, row_std))
    apply_plot_style()
    figure, axis = plt.subplots(figsize=(10, 8))
    positions = np.arange(len(CATEGORIES))
    width = 0.75 / len(labels)
    for index, name in enumerate(labels):
        offset = (index - (len(labels) - 1) / 2) * width
        axis.bar(positions + offset, means[index], width, yerr=stds[index], label=name, color=colors[index], capsize=2)
    axis.set_xticks(positions)
    axis.set_xticklabels(list(CATEGORIES))
    axis.set_ylabel("Power Balance Loss (Mean) [p.u.]")
    axis.set_ylim(0, max(PAPER_YLIM[1], 1.1 * top))
    axis.yaxis.grid(True)
    style_axis(axis)
    axis.set_title(f"Task {label}", fontproperties=font_properties(FONT_LABEL))
    axis.legend(frameon=False, prop=font_properties(FONT_LEGEND))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    figure.savefig(output, format="svg", bbox_inches="tight")
    figure.savefig(png, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output} and {png.name}", flush=True)
    return [output, png]


def plot_task11(results_dir: Path | None = None) -> list[Path]:
    results_dir = results_dir or RESULTS
    summary = read_csv(results_dir / "task1_1_summary.csv")
    measured = _rows_for(summary)
    figure = results_dir / "figure4_task1.1.svg"
    note = results_dir / "figure4_task1.1_NOT_READY.txt"
    if len(measured) != len(MODELS):
        for path in (figure, figure.with_suffix(".pdf"), figure.with_suffix(".png")):
            if path.exists():
                path.unlink()
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
    _draw(measured, figure, ylim, yticks)
    return [figure]


def plot_published(results_dir: Path | None = None) -> list[Path]:
    """Draw Task 1.1 from the published table. This does not train or test."""
    results_dir = results_dir or RESULTS
    note = results_dir / "figure4_task1.1_NOT_READY.txt"
    if note.exists():
        note.unlink()
    print("Task 1.1 published power-balance loss (mean ± std).")
    for model in MODELS:
        parts = [
            f"{category} {mean:.6g} ± {std:.6g}"
            for category, (mean, std) in PUBLISHED_TASK11[model].items()
        ]
        print(f"{model}: " + "; ".join(parts))
    figure = results_dir / "figure4_task1.1.svg"
    ylim, yticks = _ylim_for(PUBLISHED_TASK11)
    _draw(PUBLISHED_TASK11, figure, ylim, yticks)
    return [figure]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the Task 1.1 panel of GENCO Figure 4.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--published",
        action="store_true",
        help="Draw Task 1.1 from the published table, without training.",
    )
    args = parser.parse_args()
    if args.published:
        plot_published(args.results)
    else:
        plot_task11(args.results)


if __name__ == "__main__":
    main()
