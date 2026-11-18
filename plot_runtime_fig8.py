"""GENCO paper §5.4 Figure 8 + Table 6 style runtime comparison (TPGT vs published).

Protocol follows gridfm-graphkit ``genco-paper-repro`` in-memory PF benchmarks:
preload scenarios, exclude warmup, amortized ``outer_elapsed_ms / num_samples`` at the
best batch size (IEEE from 64, GOC from 16). Published numbers ship in
``data/pf_runtime_published.json`` (regenerate with ``scripts/build_pf_runtime_published.py``).

The paper cell requires NVIDIA H100 (name contains ``H100``). An L4 cell uses the
**same protocol** with ``gpu_needle=\"L4\"`` for exploratory timing (not paper-comparable).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_plot_style, style_axis

REPO_ROOT = Path(__file__).resolve().parent
PUBLISHED_JSON = REPO_ROOT / "data" / "pf_runtime_published.json"
PAPER_GPU_NEEDLE = "H100"
MIN_BENCH_SECONDS = 20.0
WARMUP_BATCHES = 10

BUS_ORDER = (14, 30, 57, 118, 500, 2000, 10000)
NETWORK_LABELS = {
    14: "IEEE 14",
    30: "IEEE 30",
    57: "IEEE 57",
    118: "IEEE 118",
    500: "GOC 500",
    2000: "GOC 2000",
    10000: "GOC 10000",
}

UTILITY_BY_GRID: dict[int, str] = {
    14: "None. Slower than AC-PF; AC-PF is preferred for accuracy.",
    30: "None. Slower than AC-PF; AC-PF is preferred for accuracy.",
    57: "None. Slower than AC-PF; AC-PF is preferred for accuracy.",
    118: "None. Slower than AC-PF; AC-PF is preferred for accuracy.",
    500: "None. Slower than AC-PF; AC-PF is preferred for accuracy.",
    2000: (
        "Much faster than AC-PF; slower than DC-PF with similar residuals, "
        "but complete solutions (reactive power and voltage magnitude)."
    ),
    10000: (
        "Much faster than AC-PF; slower than DC-PF with similar residuals, "
        "but complete solutions (reactive power and voltage magnitude)."
    ),
}

# Task 3.1 PFΔ cross-grid eval (N topology, condition N).
TPGT_BENCH_GRIDS = ("57", "118", "500")


def load_published(path: Path = PUBLISHED_JSON) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_pf_runtime_published.py when online."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def paper_gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the §5.4 runtime protocol.")
    return torch.cuda.get_device_name(0)


def require_gpu(needle: str = PAPER_GPU_NEEDLE, *, purpose: str | None = None) -> str:
    """Require CUDA device whose name contains ``needle`` (case-insensitive)."""
    name = paper_gpu_name()
    needle_u = str(needle).strip().upper()
    if not needle_u:
        raise ValueError("gpu needle must be non-empty (e.g. 'H100' or 'L4')")
    if needle_u not in name.upper():
        why = purpose or (
            "GENCO paper §5.4 timings were measured on an exclusive NVIDIA H100 SXM5 80GB"
            if needle_u == PAPER_GPU_NEEDLE
            else f"This cell requires a GPU whose name contains {needle_u!r}"
        )
        raise SystemExit(
            f"{why}. This runtime has {name!r}. Stopping before TPGT benchmark "
            "(published curves/tables can still be plotted offline)."
        )
    return name


def require_paper_gpu() -> str:
    return require_gpu(PAPER_GPU_NEEDLE)


def batch_sweep(bus_count: int) -> list[int]:
    start = 64 if bus_count < 500 else 16
    sizes: list[int] = []
    batch = start
    while batch <= 16_384:
        sizes.append(batch)
        batch *= 2
    return sizes


def _fmt_speedup(ratio: float) -> str:
    return f"{ratio:.1f}×"


def _fmt_ms(ms: float) -> str:
    if ms < 0.01:
        return f"{ms:.4f}"
    if ms < 1:
        return f"{ms:.3f}"
    return f"{ms:.2f}"


def table6_rows_for_grid(
    grid: int,
    published: dict,
    tpgt_ms: float | None,
    *,
    tpgt_label: str = "TPGT (measured)",
) -> list[dict[str, str]]:
    g = published["grids"][str(grid)]
    ac_ms = g["powermodels"]["pf"]["ms_per_instance"]
    dc_ms = g["powermodels"]["dcpf"]["ms_per_instance"]
    rows: list[dict[str, str]] = []

    def add_row(method: str, ms: float, dc_over_genco: float | None = None) -> None:
        rows.append(
            {
                "method": method,
                "ms": _fmt_ms(ms),
                "vs_ac": _fmt_speedup(ac_ms / ms),
                "dc_over_genco": "—" if dc_over_genco is None else _fmt_speedup(dc_over_genco),
                "vs_dc": _fmt_speedup(dc_ms / ms),
            }
        )

    add_row("AC-PF (PowerModels)", ac_ms)
    add_row("DC-PF (PowerModels)", dc_ms)
    for model in ("base", "small", "tiny"):
        gm = g["genco"][model]
        add_row(f"GENCO {model.capitalize()}", gm["ms_per_instance"], gm.get("dc_over_genco_active_res"))
    if tpgt_ms is not None:
        add_row(tpgt_label, tpgt_ms)
    return rows


def write_table6_html(
    grid: int,
    rows: list[dict[str, str]],
    out_dir: Path,
    *,
    name_suffix: str = "",
) -> Path:
    utility = UTILITY_BY_GRID.get(grid, "")
    lines = [
        "<table>",
        "<thead><tr>",
        "<th>Method</th><th>ms / instance</th>",
        "<th>Speedup vs AC-PF</th><th>DC-PF / GENCO active res.</th><th>Speedup vs DC-PF</th>",
        "</tr></thead><tbody>",
    ]
    for row in rows:
        lines.append(
            "<tr>"
            f"<td>{row['method']}</td>"
            f"<td>{row['ms']}</td>"
            f"<td>{row['vs_ac']}</td>"
            f"<td>{row['dc_over_genco']}</td>"
            f"<td>{row['vs_dc']}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    if utility:
        lines.append(f"<p><b>Utility</b> (GENCO Tiny regime, paper Table 6): {utility}</p>")
    path = out_dir / f"table6_runtime_grid{grid}{name_suffix}.html"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def plot_figure8(
    published: dict,
    tpgt_by_bus: dict[int, float],
    out_path: Path,
    *,
    tpgt_legend: str = "TPGT (measured)",
) -> None:
    apply_plot_style()
    bus_counts = list(BUS_ORDER)
    genco_best: dict[str, list[float]] = {m: [] for m in ("base", "small", "tiny")}
    ac_pf: list[float] = []
    dc_pf: list[float] = []
    tpgt_line: list[float] = []

    for buses in bus_counts:
        g = published["grids"][str(buses)]
        for model in ("base", "small", "tiny"):
            genco_best[model].append(g["genco"][model]["ms_per_instance"])
        ac_pf.append(g["powermodels"]["pf"]["ms_per_instance"])
        dc_pf.append(g["powermodels"]["dcpf"]["ms_per_instance"])
        tpgt_line.append(tpgt_by_bus.get(buses, float("nan")))

    labels = [NETWORK_LABELS[b] for b in bus_counts]
    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    split = bus_counts.index(500) + 1
    axis.plot(bus_counts[:split], ac_pf[:split], color="#2ca02c", linestyle="-", marker="D", label="AC-PF (NLsolve)")
    if split < len(bus_counts):
        axis.plot(
            bus_counts[split:],
            ac_pf[split:],
            color="#7fbc41",
            linestyle="-",
            marker="D",
            label="AC-PF (JuMP)",
        )
    axis.plot(bus_counts, dc_pf, color="#2ca02c", linestyle=":", marker="D", label="DC-PF")
    for model, linestyle in (("base", "-"), ("small", "--"), ("tiny", ":")):
        axis.plot(
            bus_counts,
            genco_best[model],
            color="#1f6fff",
            linestyle=linestyle,
            marker="o",
            label=f"GENCO {model.capitalize()}",
        )
    tpgt_arr = np.array(tpgt_line, dtype=float)
    if np.any(np.isfinite(tpgt_arr)):
        axis.plot(
            bus_counts,
            tpgt_arr,
            color="#1F4B99",
            linestyle="-",
            marker="s",
            linewidth=2.2,
            label=tpgt_legend,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(11.5, 12_000)
    axis.set_ylim(0.0045, 100)
    axis.set_xlabel("Bus count")
    axis.set_ylabel("Amortized per-instance\nruntime [ms]")
    axis.grid(True, which="both", alpha=0.2)
    style_axis(axis)
    axis.legend(loc="upper left", frameon=False, ncol=2)
    top = axis.twiny()
    top.set_xscale("log")
    top.set_xlim(axis.get_xlim())
    top.set_xticks(bus_counts)
    top.set_xticklabels(labels, rotation=30, ha="left", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def benchmark_tpgt_grid(
    *,
    model,
    split,
    network,
    device,
    bus_count: int,
) -> tuple[float, int]:
    """Return best (ms/instance, batch_size) for one grid on the paper in-memory protocol."""
    import torch

    from train_tpgt_task11 import clear_pair_embed_cache, pair_features, run_batch

    clear_pair_embed_cache([network])
    pair = pair_features(model, network)
    n_cases = int(split.count)
    best_ms = float("inf")
    best_batch = 0

    for batch_size in batch_sweep(bus_count):
        if batch_size > n_cases and n_cases < batch_size:
            continue
        indices = np.arange(min(batch_size, n_cases))
        for _ in range(WARMUP_BATCHES):
            run_batch(model, split, network, indices, device, pair=pair)
        if device.type == "cuda":
            torch.cuda.synchronize()

        total_samples = 0
        wall_start = time.perf_counter()
        while time.perf_counter() - wall_start < MIN_BENCH_SECONDS:
            run_batch(model, split, network, indices, device, pair=pair)
            total_samples += len(indices)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
        ms_per = elapsed_ms / max(total_samples, 1)
        print(
            f"    batch {batch_size}: {ms_per:.4f} ms/instance ({total_samples} cases)",
            flush=True,
        )
        if ms_per < best_ms:
            best_ms = ms_per
            best_batch = batch_size
    return best_ms, best_batch


def run_tpgt_benchmarks(
    *,
    data_root: Path,
    ckpt_root: Path,
    active_cache: Path,
    persistent_cache: Path,
    arch: str = "v9",
    gpu_needle: str = PAPER_GPU_NEEDLE,
) -> dict[int, dict[str, float]]:
    """Benchmark TPGT on Task 3.1 N splits for grids we have in PFΔ task 3.1."""
    import torch

    from train_tpgt_task11 import (
        TPGT,
        TASKS,
        apply_architecture_config,
        cached_split,
        ensure_runtime_packages,
        eval_checkpoint_payload,
        raw_dir,
        warm_split_caches,
    )

    gpu = require_gpu(gpu_needle)
    print(f"Timing GPU: {gpu} (needle={gpu_needle!r})", flush=True)
    ensure_runtime_packages()
    apply_architecture_config()
    spec = TASKS["3.1"]
    device = torch.device("cuda")
    warm_split_caches(
        data_root,
        active_cache,
        persistent_cache,
        spec,
        weight_run=ckpt_root,
        load_valid=False,
    )
    seed_dir = ckpt_root / "seed_0"
    picked = eval_checkpoint_payload(seed_dir, device)
    if picked is None:
        raise FileNotFoundError(f"No compatible checkpoint in {seed_dir}")
    state, epoch, ckpt_path = picked
    model = TPGT().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded TPGT {arch} epoch {epoch} from {ckpt_path}", flush=True)

    out: dict[int, dict[str, float]] = {}
    for grid in TPGT_BENCH_GRIDS:
        split = cached_split(
            active_cache,
            persistent_cache,
            raw_dir(data_root, "test", "N", "N", grid),
            f"test_{grid}_N_N",
        )
        network = split.networks[0]
        buses = int(network.ybus.shape[0])
        print(f"Benchmark grid {grid} ({buses} buses, {split.count} scenarios)", flush=True)
        ms, batch = benchmark_tpgt_grid(
            model=model,
            split=split,
            network=network,
            device=device,
            bus_count=buses,
        )
        out[buses] = {"ms_per_instance": ms, "batch_size": float(batch), "grid": grid}
        print(f"  best: {ms:.4f} ms @ batch {batch}", flush=True)
    return out


def write_outputs(
    results_dir: Path,
    published: dict,
    tpgt_by_bus: dict[int, float],
    *,
    figure_name: str = "figure8_runtime_pf.png",
    summary_name: str = "runtime_fig8_summary.json",
    tpgt_label: str = "TPGT (measured)",
    table_suffix: str = "",
    extra_summary: dict[str, Any] | None = None,
) -> tuple[Path, list[Path]]:
    fig_path = results_dir / figure_name
    plot_figure8(published, tpgt_by_bus, fig_path, tpgt_legend=tpgt_label)
    table_paths: list[Path] = []
    for grid in BUS_ORDER:
        tpgt_ms = tpgt_by_bus.get(grid)
        rows = table6_rows_for_grid(grid, published, tpgt_ms, tpgt_label=tpgt_label)
        table_paths.append(
            write_table6_html(grid, rows, results_dir, name_suffix=table_suffix)
        )
    summary: dict[str, Any] = {
        "tpgt_ms_by_bus": tpgt_by_bus,
        "published_source": published.get("source"),
        "tpgt_label": tpgt_label,
    }
    if extra_summary:
        summary.update(extra_summary)
    (results_dir / summary_name).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return fig_path, table_paths


def display_runtime_fig8(
    results_dir: Path,
    *,
    figure_name: str = "figure8_runtime_pf.png",
    table_suffix: str = "",
) -> None:
    """Notebook helper: show figure + per-grid Table 6 HTML."""
    from IPython.display import HTML, Image, display

    fig = results_dir / figure_name
    if fig.is_file():
        display(Image(filename=str(fig)))
    else:
        print(f"Missing figure: {fig}", flush=True)
    for grid in BUS_ORDER:
        html_path = results_dir / f"table6_runtime_grid{grid}{table_suffix}.html"
        if html_path.is_file():
            print(f"Table 6 style — {NETWORK_LABELS[grid]}", flush=True)
            display(HTML(html_path.read_text(encoding="utf-8")))


def ensure_published_json(path: Path = PUBLISHED_JSON) -> Path:
    if path.is_file():
        return path
    import subprocess
    import sys

    script = REPO_ROOT / "scripts" / "build_pf_runtime_published.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing {path} and no {script} to rebuild it.")
    subprocess.check_call([sys.executable, str(script), "--output", str(path)])
    return path


def run_notebook_section54(
    *,
    data_root: Path,
    weight_run: Path,
    task_run: Path,
    arch: str = "v9",
    rebench: bool = True,
    published_json: Path = PUBLISHED_JSON,
    gpu_needle: str = PAPER_GPU_NEEDLE,
    bench_cache_name: str | None = None,
    figure_name: str | None = None,
    summary_name: str | None = None,
    tpgt_label: str | None = None,
) -> Path:
    """Colab entry: GPU gate + TPGT timing vs published GENCO curves/tables.

    Default ``gpu_needle=\"H100\"`` is the paper cell. Pass ``gpu_needle=\"L4\"`` for the
    same in-memory protocol on L4 (exploratory; not paper-comparable absolute times).
    ``weight_run`` is the Task 1.1 checkpoint folder containing ``seed_0/``.
    """
    from train_tpgt_task11 import TASKS, split_cache_dirs

    results = task_run / "results"
    results.mkdir(parents=True, exist_ok=True)
    ensure_published_json(published_json)
    published = load_published(published_json)
    tpgt_by_bus: dict[int, float] = {}
    needle = str(gpu_needle).strip().upper() or PAPER_GPU_NEEDLE
    suffix = "" if needle == PAPER_GPU_NEEDLE else f"_{needle.lower()}"
    cache_path = results / (bench_cache_name or f"runtime_tpgt_bench{suffix}.json")
    fig_name = figure_name or f"figure8_runtime_pf{suffix}.png"
    sum_name = summary_name or f"runtime_fig8_summary{suffix}.json"
    label = tpgt_label or f"TPGT ({arch}, {needle})"

    gpu = require_gpu(
        needle,
        purpose=(
            "GENCO paper §5.4 timings were measured on an exclusive NVIDIA H100 SXM5 80GB"
            if needle == PAPER_GPU_NEEDLE
            else f"This exploratory §5.4 cell requires NVIDIA {needle}"
        ),
    )
    print(f"Timing GPU: {gpu} (needle={needle})", flush=True)
    if needle != PAPER_GPU_NEEDLE:
        print(
            "Note: GENCO Base/Small/Tiny and PowerModels curves are still the "
            "paper's H100-published numbers; only the TPGT points are measured here.",
            flush=True,
        )

    if rebench or not cache_path.is_file():
        active, persistent = split_cache_dirs(task_run, TASKS["3.1"])
        raw = run_tpgt_benchmarks(
            data_root=data_root,
            ckpt_root=weight_run,
            active_cache=active,
            persistent_cache=persistent,
            arch=arch,
            gpu_needle=needle,
        )
        cache_path.write_text(
            json.dumps({str(k): v for k, v in raw.items()}, indent=2),
            encoding="utf-8",
        )
    else:
        raw = {int(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
        print(f"Reusing TPGT timings from {cache_path}", flush=True)
    for buses, payload in raw.items():
        tpgt_by_bus[int(buses)] = float(payload["ms_per_instance"])

    write_outputs(
        results,
        published,
        tpgt_by_bus,
        figure_name=fig_name,
        summary_name=sum_name,
        tpgt_label=label,
        table_suffix=suffix,
        extra_summary={
            "gpu": gpu,
            "gpu_needle": needle,
            "arch": arch,
            "weight_run": str(weight_run),
            "bench_cache": str(cache_path),
            "figure": fig_name,
            "table_suffix": suffix,
            "paper_comparable": needle == PAPER_GPU_NEEDLE,
        },
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true", help="Skip TPGT timing (published only).")
    parser.add_argument("--published-json", type=Path, default=PUBLISHED_JSON)
    args = parser.parse_args()
    ensure_published_json(args.published_json)
    published = load_published(args.published_json)
    tpgt_by_bus: dict[int, float] = {}
    if not args.plot_only:
        raise SystemExit("Use the Task1_1 §5.4 cell or run_notebook_section54() with cache paths.")
    write_outputs(args.results_dir, published, tpgt_by_bus)
    print(f"Wrote {args.results_dir / 'figure8_runtime_pf.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
