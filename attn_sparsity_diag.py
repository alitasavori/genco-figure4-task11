"""Step-2 GRIT attention sparsity diagnostics (no retrain).

Captures dense attention from a trained grit checkpoint (v16/v9), writes bus×bus
heatmaps, reports how many senders retain 90/95/99% of attention mass (and of
weighted-message magnitude), and scores optimistic top-k renormalized PBL.
"""

from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_style import apply_plot_style, style_axis

MASS_THRESHOLDS = (0.90, 0.95, 0.99)
DEFAULT_TOPKS = (4, 8, 16, 24, 32)


@contextlib.contextmanager
def grit_attention_capture(
    model,
    *,
    contrib: bool = True,
    topk: int | None = None,
) -> Iterator[None]:
    """Enable per-stage attention dumps (and optional top-k renormalization) for grit."""
    stages = list(model.stages)
    prev = []
    for stage in stages:
        prev.append(
            (
                getattr(stage, "_capture_attn", False),
                getattr(stage, "_capture_attn_contrib", False),
                getattr(stage, "_attn_topk", None),
            )
        )
        stage._capture_attn = True
        stage._capture_attn_contrib = bool(contrib)
        stage._attn_topk = topk
        stage._last_attn = None
        stage._last_attn_contrib = None
    try:
        yield
    finally:
        for stage, (cap, con, tk) in zip(stages, prev):
            stage._capture_attn = cap
            stage._capture_attn_contrib = con
            stage._attn_topk = tk


def read_stage_attention(model) -> list[dict[str, torch.Tensor | None]]:
    """Return ``[{attn, contrib}, ...]`` from the last captured forward (CPU float)."""
    out: list[dict[str, torch.Tensor | None]] = []
    for stage in model.stages:
        attn = getattr(stage, "_last_attn", None)
        contrib = getattr(stage, "_last_attn_contrib", None)
        out.append(
            {
                "attn": None if attn is None else attn.detach().float().cpu(),
                "contrib": None if contrib is None else contrib.detach().float().cpu(),
            }
        )
    return out


def _k_to_mass(row: np.ndarray, thresholds: tuple[float, ...] = MASS_THRESHOLDS) -> dict[str, float]:
    """Smallest k such that the top-k entries cover each mass threshold (row sums to ~1)."""
    total = float(row.sum())
    if total <= 0 or not np.isfinite(total):
        return {f"k@{int(t * 100)}": float("nan") for t in thresholds}
    order = np.sort(row)[::-1]
    csum = np.cumsum(order) / total
    out: dict[str, float] = {}
    for t in thresholds:
        hit = np.nonzero(csum >= t)[0]
        out[f"k@{int(t * 100)}"] = float(hit[0] + 1) if hit.size else float(len(row))
    out["entropy"] = float(-np.sum(np.where(row > 0, row * np.log(row + 1e-12), 0.0)))
    out["max"] = float(order[0] / total) if order.size else 0.0
    return out


def summarize_mass(
    tensor: torch.Tensor,
    types: np.ndarray,
    *,
    type_names: tuple[str, ...] = ("PQ", "PV", "Slack"),
) -> dict[str, Any]:
    """``tensor``: (B, H, N, N) attention or contribution magnitudes."""
    arr = tensor.numpy()
    batch, heads, buses, _ = arr.shape
    # Zero diagonal (no self-attention) already; still guard.
    eye = np.eye(buses, dtype=bool)
    per_recv: list[dict[str, float]] = []
    for b in range(batch):
        for h in range(heads):
            for i in range(buses):
                row = arr[b, h, i].copy()
                row[eye[i]] = 0.0
                stats = _k_to_mass(row)
                stats["batch"] = b
                stats["head"] = h
                stats["recv"] = i
                tid = int(types[i]) if i < len(types) else -1
                stats["recv_type"] = type_names[tid] if 0 <= tid < len(type_names) else str(tid)
                per_recv.append(stats)

    def _mean_keys(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
        return {
            key: float(np.nanmean([r[key] for r in rows])) if rows else float("nan")
            for key in keys
        }

    keys = [f"k@{int(t * 100)}" for t in MASS_THRESHOLDS] + ["entropy", "max"]
    overall = _mean_keys(per_recv, keys)
    by_head = {
        str(h): _mean_keys([r for r in per_recv if int(r["head"]) == h], keys)
        for h in range(heads)
    }
    by_type = {
        name: _mean_keys([r for r in per_recv if r["recv_type"] == name], keys)
        for name in type_names
    }
    return {
        "overall": overall,
        "by_head": by_head,
        "by_recv_type": by_type,
        "n_rows": len(per_recv),
        "shape": [batch, heads, buses, buses],
    }


def mean_attention_grid(
    attn: torch.Tensor,
    *,
    head: int | None = None,
) -> np.ndarray:
    """Average (B[, H], N, N) → (N, N) heatmap grid."""
    x = attn.numpy()
    if head is not None:
        x = x[:, int(head)]
    else:
        x = x.mean(axis=1)
    return x.mean(axis=0)


def plot_attention_heatmap(
    grid: np.ndarray,
    out_path: Path,
    *,
    title: str,
    types: np.ndarray | None = None,
    type_names: tuple[str, ...] = ("PQ", "PV", "Slack"),
    log1p: bool = False,
) -> None:
    apply_plot_style()
    data = np.log1p(np.maximum(grid, 0.0) * 1e3) if log1p else grid
    buses = int(grid.shape[0])
    fig, axis = plt.subplots(figsize=(8.5, 7.2))
    im = axis.imshow(data, aspect="equal", interpolation="nearest", cmap="magma")
    axis.set_xlabel("Sender bus")
    axis.set_ylabel("Receiver bus")
    axis.set_title(title)
    cbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    cbar.set_label("log1p(1000·α)" if log1p else "Attention weight α")
    if types is not None and buses <= 80:
        # Light type ticks so PQ/PV/Slack structure is readable on case57.
        ticks = list(range(0, buses, max(1, buses // 12)))
        axis.set_xticks(ticks)
        axis.set_yticks(ticks)
    style_axis(axis)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_topk_mass_curve(
    attn: torch.Tensor,
    out_path: Path,
    *,
    title: str,
) -> dict[str, float]:
    """Mean cumulative attention mass vs #senders kept (sorted); marks 90/95/99%."""
    apply_plot_style()
    arr = attn.numpy().mean(axis=1)  # mean heads → (B, N, N)
    batch, buses, _ = arr.shape
    curves = []
    for b in range(batch):
        for i in range(buses):
            row = arr[b, i].copy()
            row[i] = 0.0
            total = row.sum()
            if total <= 0:
                continue
            order = np.sort(row)[::-1]
            curves.append(np.cumsum(order) / total)
    if not curves:
        return {}
    stacked = np.stack(curves, axis=0)
    mean_c = stacked.mean(axis=0)
    xs = np.arange(1, buses + 1)
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(xs, mean_c, color="#1F4B99", linewidth=2.0, label="Mean cumulative mass")
    marks = {}
    for t in MASS_THRESHOLDS:
        hit = np.nonzero(mean_c >= t)[0]
        k = int(hit[0] + 1) if hit.size else buses
        marks[f"k@{int(t * 100)}"] = float(k)
        axis.axhline(t, color="#888888", linestyle=":", linewidth=1.0)
        axis.axvline(k, color="#C44E52", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.text(k + 0.3, t - 0.04, f"{int(t*100)}% @ k={k}", fontsize=11)
    axis.set_xlabel("Senders kept (sorted by α)")
    axis.set_ylabel("Cumulative attention mass")
    axis.set_xlim(1, buses)
    axis.set_ylim(0.0, 1.02)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    style_axis(axis)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return marks


def _sample_indices(split, network_id: int, count: int) -> np.ndarray:
    return _tpgt().debug_sample_index(split, network_id=network_id, count=count)


def _tpgt():
    """Import ``train_tpgt_task11`` and sync arch/case from env (needed under ``%run``)."""
    import train_tpgt_task11 as tpgt

    # ``%run train_tpgt_task11.py`` configures ``__main__``, not this import. Re-apply from env.
    tpgt.apply_architecture_config()
    return tpgt


def _forward_capture(model, split, network, index, device, *, contrib: bool, topk: int | None):
    tpgt = _tpgt()
    case_pbl = tpgt.case_pbl
    forward_cases = tpgt.forward_cases
    move_to = tpgt.move_to
    pair_features = tpgt.pair_features

    device_network = move_to(network, device)
    pair = pair_features(model, device_network)
    with torch.inference_mode():
        with grit_attention_capture(model, contrib=contrib, topk=topk):
            _prediction, _target, residuals = forward_cases(
                model, split, device_network, index, device, pair
            )
            captured = read_stage_attention(model)
    pbl = float(case_pbl(residuals[-1]).mean().item())
    types = device_network.types.detach().cpu().numpy()
    return captured, types, pbl


def run_attention_sparsity_diagnostic(
    model,
    data: Path,
    device: torch.device,
    active_cache: Path,
    persistent_cache: Path,
    *,
    out_dir: Path,
    cases: int = 16,
    labels: tuple[str, ...] = ("N", "N-1", "N-2"),
    topks: tuple[int, ...] = DEFAULT_TOPKS,
    eval_batch_size: int | None = None,
) -> dict[str, Any]:
    """Full Step-2 diagnostic: heatmaps, mass tables, optimistic top-k PBL."""
    tpgt = _tpgt()
    ATTENTION_KIND = tpgt.ATTENTION_KIND
    DEPTH = tpgt.DEPTH
    HEADS = tpgt.HEADS
    TEST_PARTS = tpgt.TEST_PARTS
    TYPE_NAMES = tpgt.TYPE_NAMES
    cached_split = tpgt.cached_split
    raw_dir = tpgt.raw_dir
    CASE_NAME = tpgt.CASE_NAME

    if ATTENTION_KIND != "grit":
        raise RuntimeError(f"--attn-sparsity requires grit attention, got {ATTENTION_KIND}")
    print(f"Using CASE_NAME={CASE_NAME} for split paths under {data}", flush=True)
    probe = raw_dir(data, "test", "n", "feasible")
    if not (probe / "bus_data.parquet").is_file():
        raise FileNotFoundError(
            f"Missing test N feasible parquet at {probe / 'bus_data.parquet'} "
            f"(CASE_NAME={CASE_NAME}). Check GENCO_TPGT_CASE and the vendor pack."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "case": CASE_NAME,
        "depth": int(DEPTH),
        "heads": int(HEADS),
        "cases": int(cases),
        "labels": list(labels),
        "mass_thresholds": list(MASS_THRESHOLDS),
        "topks": list(topks),
        "by_label": {},
    }

    mass_csv = out_dir / "attention_mass_by_stage.csv"
    with mass_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "label",
                "stage",
                "metric",
                "group",
                "k@90",
                "k@95",
                "k@99",
                "entropy",
                "max",
            ]
        )

        for label, topology, condition in TEST_PARTS:
            if label not in labels or condition != "feasible":
                continue
            print(f"=== Attention sparsity: {label} ===", flush=True)
            split = cached_split(
                active_cache,
                persistent_cache,
                raw_dir(data, "test", topology, condition),
                f"test_{topology}_{condition}",
            )
            # Intact N: network 0. Contingencies: largest group (most cases).
            if label == "N":
                net_id = 0
            else:
                counts = torch.bincount(split.network_index, minlength=len(split.networks))
                net_id = int(torch.argmax(counts))
            network = split.networks[net_id]
            index = _sample_indices(split, net_id, cases)
            print(
                f"  network_id={net_id} buses={int(network.ybus.shape[0])} "
                f"cases={len(index)}",
                flush=True,
            )

            captured, types, _dense_pbl = _forward_capture(
                model, split, network, index, device, contrib=True, topk=None
            )
            label_summary: dict[str, Any] = {
                "network_id": net_id,
                "n_buses": int(network.ybus.shape[0]),
                "n_cases": int(len(index)),
                "stages": [],
            }

            # Aggregate mean attention over stages for the main 2D grid.
            stage_grids = []
            for stage_i, payload in enumerate(captured):
                attn = payload["attn"]
                contrib = payload["contrib"]
                if attn is None:
                    continue
                attn_stats = summarize_mass(attn, types, type_names=TYPE_NAMES)
                contrib_stats = (
                    summarize_mass(contrib, types, type_names=TYPE_NAMES)
                    if contrib is not None
                    else None
                )
                grid = mean_attention_grid(attn)
                stage_grids.append(grid)
                stem = f"{label}_stage{stage_i + 1:02d}"
                plot_attention_heatmap(
                    grid,
                    out_dir / f"heatmap_{stem}_mean_heads.png",
                    title=f"{label} stage {stage_i + 1}: mean α (batch×heads)",
                    types=types,
                )
                plot_attention_heatmap(
                    grid,
                    out_dir / f"heatmap_{stem}_mean_heads_log.png",
                    title=f"{label} stage {stage_i + 1}: log1p mean α",
                    types=types,
                    log1p=True,
                )
                np.save(out_dir / f"attn_grid_{stem}.npy", grid)
                curve_marks = plot_topk_mass_curve(
                    attn,
                    out_dir / f"mass_curve_{stem}.png",
                    title=f"{label} stage {stage_i + 1}: cumulative attention mass",
                )
                # Per-head grids for first/last stage only (avoid PNG spam).
                if stage_i in (0, len(captured) - 1):
                    for h in range(int(attn.shape[1])):
                        hgrid = mean_attention_grid(attn, head=h)
                        plot_attention_heatmap(
                            hgrid,
                            out_dir / f"heatmap_{stem}_head{h}.png",
                            title=f"{label} stage {stage_i + 1} head {h}",
                            types=types,
                        )

                stage_entry = {
                    "stage": stage_i + 1,
                    "attention_mass": attn_stats,
                    "message_contrib_mass": contrib_stats,
                    "curve_k": curve_marks,
                }
                label_summary["stages"].append(stage_entry)

                for metric_name, stats in (
                    ("attention", attn_stats),
                    ("message_contrib", contrib_stats),
                ):
                    if stats is None:
                        continue
                    for group, payload_stats in (
                        ("overall", stats["overall"]),
                        *[("head_" + h, v) for h, v in stats["by_head"].items()],
                        *[("type_" + t, v) for t, v in stats["by_recv_type"].items()],
                    ):
                        writer.writerow(
                            [
                                label,
                                stage_i + 1,
                                metric_name,
                                group,
                                payload_stats.get("k@90", ""),
                                payload_stats.get("k@95", ""),
                                payload_stats.get("k@99", ""),
                                payload_stats.get("entropy", ""),
                                payload_stats.get("max", ""),
                            ]
                        )

                print(
                    f"  stage {stage_i + 1}: attn k@90/95/99 = "
                    f"{attn_stats['overall'].get('k@90', float('nan')):.1f} / "
                    f"{attn_stats['overall'].get('k@95', float('nan')):.1f} / "
                    f"{attn_stats['overall'].get('k@99', float('nan')):.1f}",
                    flush=True,
                )

            if stage_grids:
                mean_all = np.mean(np.stack(stage_grids, axis=0), axis=0)
                plot_attention_heatmap(
                    mean_all,
                    out_dir / f"heatmap_{label}_all_stages_mean.png",
                    title=f"{label}: mean α over stages×batch×heads",
                    types=types,
                )
                plot_attention_heatmap(
                    mean_all,
                    out_dir / f"heatmap_{label}_all_stages_mean_log.png",
                    title=f"{label}: log1p mean α over stages",
                    types=types,
                    log1p=True,
                )
                np.save(out_dir / f"attn_grid_{label}_all_stages.npy", mean_all)
                label_summary["primary_heatmap"] = f"heatmap_{label}_all_stages_mean.png"

            # Optimistic top-k: dense scores → keep k → renormalize (still needs dense scoring).
            topk_rows: list[dict[str, Any]] = []
            _, _, dense_pbl = _forward_capture(
                model, split, network, index, device, contrib=False, topk=None
            )
            topk_rows.append({"k": "dense", "pbl": float(dense_pbl)})
            print(f"  dense sample PBL={dense_pbl:.6g}", flush=True)

            for k in topks:
                _, _, pbl_k = _forward_capture(
                    model, split, network, index, device, contrib=False, topk=int(k)
                )
                topk_rows.append(
                    {
                        "k": int(k),
                        "pbl": float(pbl_k),
                        "delta": float(pbl_k - dense_pbl),
                    }
                )
                print(
                    f"  top-k={k} sample PBL={pbl_k:.6g} (Δ {pbl_k - dense_pbl:+.4g})",
                    flush=True,
                )

            label_summary["topk_pbl_sample"] = topk_rows
            summary["by_label"][label] = label_summary

            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary_path = out_dir / "attn_sparsity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {mass_csv}", flush=True)
    print(f"Heatmaps under {out_dir}", flush=True)
    return summary


def display_attn_sparsity(out_dir: Path, labels: tuple[str, ...] = ("N", "N-1", "N-2")) -> None:
    """Notebook helper: show primary heatmaps + mass CSV."""
    from IPython.display import Image, display
    import pandas as pd

    for label in labels:
        for name in (
            f"heatmap_{label}_all_stages_mean.png",
            f"heatmap_{label}_all_stages_mean_log.png",
        ):
            path = out_dir / name
            if path.is_file():
                print(path.name, flush=True)
                display(Image(filename=str(path)))
        # Last-stage mean heads if all-stages missing
        stages = sorted(out_dir.glob(f"heatmap_{label}_stage*_mean_heads.png"))
        if stages:
            last = stages[-1]
            print(last.name, flush=True)
            display(Image(filename=str(last)))
        curve = sorted(out_dir.glob(f"mass_curve_{label}_stage*.png"))
        if curve:
            print(curve[-1].name, flush=True)
            display(Image(filename=str(curve[-1])))
    csv_path = out_dir / "attention_mass_by_stage.csv"
    if csv_path.is_file():
        display(pd.read_csv(csv_path))
