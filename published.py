"""Task 1.1 numbers transcribed from the authors' Figure 4 table.

The bar chart uses the rounded LaTeX values parsed by
gridfm-graphkit scripts/pfdelta/make_pbl_barplots.py. Those values are the
published comparison. They are not measurements from this workspace.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from common import CATEGORIES, RESULTS, SUMMARY_COLUMNS, TASK, upsert_csv

# table_a_8_genco_exponent_fixed.tex, Task 1.1 only.
# GNS is the table name; the figure labels it GNS-S.
TABLE_TASK_11 = {
    "GENCO Base": [
        r"1.4e-3\std{0.4e-3}",
        r"4.2e-2\std{2.2e-2}",
        r"5.0e-2\std{2.6e-2}",
        r"3.0e-1\std{0.8e-1}",
    ],
    "PFNet": [
        r"3.2e-1\std{0.2e-1}",
        r"5.2e-1\std{0.2e-1}",
        r"5.3e-1\std{0.3e-1}",
        r"1.4\std{0.1}",
    ],
    "CANOS-PF": [
        r"2.7e-2\std{0.03e-2}",
        r"7.0e-1\std{0.8e-1}",
        r"5.6e-1\std{0.7e-1}",
        r"1.1\std{0.1}",
    ],
    "GNS-S": [
        r"5.3e-1\std{2.7e-1}",
        r"6.8e-1\std{2.6e-1}",
        r"7.1e-1\std{2.5e-1}",
        r"1.0\std{0.2}",
    ],
}

# scripts/pfdelta/results/1.1/metrics_summary.csv, unrounded GENCO means.
# The figure does not plot these directly; make_pbl_barplots.py parses the
# rounded LaTeX column instead.
GENCO_UNROUNDED = {
    "N": (0.0014365429679552711, 0.00038566372924967053),
    "N-1": (0.042099264462788895, 0.02183983789776223),
    "N-2": (0.05048130909601847, 0.026133216315799984),
    "C2I": (0.3007470300462511, 0.082013997457341),
}

PUBLISHED_SOURCE = "published_figure4_table"
PLOT_NAMES = {"GNS": "GNS-S"}


def _parse_sci(token: str) -> float:
    match = re.fullmatch(r"([\d.]+)(?:e([+-]?\d+))?", token)
    if not match:
        raise ValueError(f"Cannot parse scientific token: {token!r}")
    value = float(match.group(1))
    exponent = int(match.group(2)) if match.group(2) is not None else 0
    return value * (10**exponent)


def parse_metric(cell: str) -> tuple[float, float]:
    """Parse one mean/std cell the same way as make_pbl_barplots.py."""
    match = re.search(
        r"([\d.]+(?:e[+-]?\d+)?)\\std\{([\d.]+(?:e[+-]?\d+)?)\}",
        cell,
    )
    if not match:
        raise ValueError(f"Cannot parse metric cell: {cell!r}")
    return _parse_sci(match.group(1)), _parse_sci(match.group(2))


def published_stats() -> dict[str, dict[str, tuple[float, float]]]:
    parsed: dict[str, dict[str, tuple[float, float]]] = {}
    for model, cells in TABLE_TASK_11.items():
        parsed[model] = {}
        for category, cell in zip(CATEGORIES, cells):
            parsed[model][category] = parse_metric(cell)
    return parsed


def write_published_tables(results_dir: Path | None = None) -> Path:
    results_dir = results_dir or RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    stats = published_stats()
    seed_rows = []
    summary_rows = []
    for model, by_category in stats.items():
        for category, (mean, std) in by_category.items():
            seed_rows.append(
                {
                    "model": model,
                    "task": TASK,
                    "test_category": category,
                    "seed": "aggregate",
                    "mean_pbl_pu": f"{mean:.12g}",
                    "checkpoint_path": "",
                    "source": PUBLISHED_SOURCE,
                }
            )
            summary_rows.append(
                {
                    "model": model,
                    "test_category": category,
                    "mean_across_seeds": f"{mean:.12g}",
                    "std_across_seeds": f"{std:.12g}",
                    "published_mean": f"{mean:.12g}",
                    "published_std": f"{std:.12g}",
                    "abs_diff": "0",
                    "rel_diff": "0",
                    "measured_source": PUBLISHED_SOURCE,
                    "n_seeds": "",
                }
            )
    seed_path = results_dir / "task1_1_seed_metrics.csv"
    summary_path = results_dir / "task1_1_summary.csv"
    upsert_csv(
        seed_path,
        ["model", "task", "test_category", "seed", "mean_pbl_pu", "checkpoint_path", "source"],
        seed_rows,
        ["model", "task", "test_category", "seed", "source"],
    )
    upsert_csv(summary_path, SUMMARY_COLUMNS, summary_rows, ["model", "test_category", "measured_source"])
    unrounded = results_dir / "genco_task1.1_metrics_summary_unrounded.csv"
    with unrounded.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "test_category", "mean_pbl_pu", "std_pbl_pu", "source"],
        )
        writer.writeheader()
        for category, (mean, std) in GENCO_UNROUNDED.items():
            writer.writerow(
                {
                    "model": "GENCO Base",
                    "test_category": category,
                    "mean_pbl_pu": mean,
                    "std_pbl_pu": std,
                    "source": "genco_metrics_summary_csv",
                }
            )
    return summary_path
