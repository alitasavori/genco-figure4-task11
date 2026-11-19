"""Step-2 GRIT attention sparsity diagnostics (no retrain).

Captures dense attention from a trained grit checkpoint (v16/v9), writes bus×bus
attention (α) and message-magnitude (α‖v‖) heatmaps, reports how many senders
retain 90/95/99% of attention mass (and of weighted-message magnitude), and
scores optimistic top-k renormalized PBL.
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import shutil
import tempfile
import zipfile
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


def _fsync_file(path: Path) -> None:
    """Best-effort durable write so Colab Drive sync sees the file before session death."""
    try:
        with path.open("rb", buffering=0) as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _fsync_file(path)


def _write_text(path: Path, text: str) -> None:
    _write_bytes(path, text.encode("utf-8"))


def _local_staging_dir(drive_out: Path) -> Path:
    """Local SSD staging — Colab Drive FUSE often drops many small PNGs written directly."""
    if Path("/content").is_dir():
        stage = Path("/content") / f"{drive_out.name}_staging"
    else:
        stage = Path(tempfile.mkdtemp(prefix=f"{drive_out.name}_"))
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    return stage


def publish_artifacts_to_drive(stage_dir: Path, drive_out: Path) -> list[Path]:
    """Copy staged files to Drive, verify sizes, and write a single zip for reliable download."""
    drive_out.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    for src in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
        rel = src.relative_to(stage_dir)
        dst = drive_out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _fsync_file(dst)
        if not dst.is_file() or dst.stat().st_size != src.stat().st_size:
            raise RuntimeError(
                f"Drive publish failed for {rel}: "
                f"src={src.stat().st_size if src.is_file() else 'missing'} "
                f"dst={dst.stat().st_size if dst.is_file() else 'missing'}"
            )
        published.append(dst)

    zip_path = drive_out / "attn_sparsity_artifacts.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(p for p in stage_dir.rglob("*") if p.is_file()):
            zf.write(src, arcname=str(src.relative_to(stage_dir)).replace("\\", "/"))
    _fsync_file(zip_path)
    published.append(zip_path)

    pngs = [p for p in published if p.suffix.lower() == ".png"]
    print(
        f"Published {len(published)} files to Drive ({len(pngs)} PNGs) under {drive_out}",
        flush=True,
    )
    print(f"  zip: {zip_path} ({zip_path.stat().st_size} bytes)", flush=True)
    for path in pngs[:12]:
        print(f"  png: {path.name} ({path.stat().st_size} bytes)", flush=True)
    if len(pngs) > 12:
        print(f"  ... and {len(pngs) - 12} more PNGs", flush=True)
    return published


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
    cbar_label: str = "Attention weight α",
) -> None:
    apply_plot_style()
    buses = int(grid.shape[0])
    fig, axis = plt.subplots(figsize=(8.5, 7.2))
    im = axis.imshow(grid, aspect="equal", interpolation="nearest", cmap="magma")
    axis.set_xlabel("Sender bus")
    axis.set_ylabel("Receiver bus")
    axis.set_title(title)
    cbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
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
    _fsync_file(out_path)


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
    _fsync_file(out_path)
    return marks


def _sample_indices(split, network_id: int, count: int) -> np.ndarray:
    return _tpgt().debug_sample_index(split, network_id=network_id, count=count)


def _tpgt():
    """Resolve the trainer module with the active ``CASE_NAME`` / arch.

    Under ``%run train_tpgt_task11.py``, config lives on ``__main__``. A plain
    ``import train_tpgt_task11`` is a second module that can stay at default
    ``case118`` (and Colab may keep a stale ``attn_sparsity_diag`` in
    ``sys.modules`` across pulls). Prefer ``__main__`` when it looks like the
    trainer; otherwise import + ``apply_architecture_config()``.
    """
    import sys

    main = sys.modules.get("__main__")
    if (
        main is not None
        and getattr(main, "__file__", None)
        and str(getattr(main, "__file__", "")).endswith("train_tpgt_task11.py")
        and hasattr(main, "raw_dir")
        and hasattr(main, "CASE_NAME")
        and hasattr(main, "cached_split")
    ):
        return main

    import train_tpgt_task11 as tpgt

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

    drive_out = Path(out_dir)
    drive_out.mkdir(parents=True, exist_ok=True)
    # Write PNGs/CSV/JSON on local disk first; publish to Drive at the end.
    stage_dir = _local_staging_dir(drive_out)
    work_dir = stage_dir
    print(f"Staging artifacts locally at {work_dir} (then copy to {drive_out})", flush=True)
    summary: dict[str, Any] = {
        "case": CASE_NAME,
        "depth": int(DEPTH),
        "heads": int(HEADS),
        "cases": int(cases),
        "labels": list(labels),
        "mass_thresholds": list(MASS_THRESHOLDS),
        "topks": list(topks),
        "by_label": {},
        "drive_out": str(drive_out),
        "stage_dir": str(stage_dir),
    }

    mass_csv = work_dir / "attention_mass_by_stage.csv"
    mass_rows: list[list[Any]] = []

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

        # Aggregate mean attention / message over stages for the main 2D grids.
        stage_grids = []
        stage_msg_grids = []
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
                work_dir / f"heatmap_{stem}_mean_heads.png",
                title=f"{label} stage {stage_i + 1}: mean α (batch×heads)",
                types=types,
                cbar_label="Attention weight α",
            )
            if contrib is not None:
                msg_grid = mean_attention_grid(contrib)
                stage_msg_grids.append(msg_grid)
                plot_attention_heatmap(
                    msg_grid,
                    work_dir / f"heatmap_{stem}_message_mean_heads.png",
                    title=f"{label} stage {stage_i + 1}: mean α‖v‖ (batch×heads)",
                    types=types,
                    cbar_label="Message magnitude α‖v‖",
                )
                msg_npy = work_dir / f"msg_grid_{stem}.npy"
                np.save(msg_npy, msg_grid)
                _fsync_file(msg_npy)
            grid_path = work_dir / f"attn_grid_{stem}.npy"
            np.save(grid_path, grid)
            _fsync_file(grid_path)
            curve_marks = plot_topk_mass_curve(
                attn,
                work_dir / f"mass_curve_{stem}.png",
                title=f"{label} stage {stage_i + 1}: cumulative attention mass",
            )
            # Per-head attention grids for first/last stage only (avoid PNG spam).
            if stage_i in (0, len(captured) - 1):
                for h in range(int(attn.shape[1])):
                    hgrid = mean_attention_grid(attn, head=h)
                    plot_attention_heatmap(
                        hgrid,
                        work_dir / f"heatmap_{stem}_head{h}.png",
                        title=f"{label} stage {stage_i + 1} head {h}",
                        types=types,
                        cbar_label="Attention weight α",
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
                    mass_rows.append(
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
                work_dir / f"heatmap_{label}_all_stages_mean.png",
                title=f"{label}: mean α over stages×batch×heads",
                types=types,
                cbar_label="Attention weight α",
            )
            np_path = work_dir / f"attn_grid_{label}_all_stages.npy"
            np.save(np_path, mean_all)
            _fsync_file(np_path)
            label_summary["primary_heatmap"] = f"heatmap_{label}_all_stages_mean.png"
        if stage_msg_grids:
            mean_msg = np.mean(np.stack(stage_msg_grids, axis=0), axis=0)
            plot_attention_heatmap(
                mean_msg,
                work_dir / f"heatmap_{label}_all_stages_message_mean.png",
                title=f"{label}: mean α‖v‖ over stages×batch×heads",
                types=types,
                cbar_label="Message magnitude α‖v‖",
            )
            msg_all_path = work_dir / f"msg_grid_{label}_all_stages.npy"
            np.save(msg_all_path, mean_msg)
            _fsync_file(msg_all_path)
            label_summary["primary_message_heatmap"] = (
                f"heatmap_{label}_all_stages_message_mean.png"
            )

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
        writer.writerows(mass_rows)
        fh.flush()
        os.fsync(fh.fileno())

    summary_path = work_dir / "attn_sparsity_summary.json"
    _write_text(summary_path, json.dumps(summary, indent=2))
    files = sorted(p for p in work_dir.rglob("*") if p.is_file())
    manifest_lines = [f"{p.relative_to(work_dir)}\t{p.stat().st_size}" for p in files]
    _write_text(work_dir / "MANIFEST.txt", "\n".join(manifest_lines) + "\n")
    print(f"Wrote staged {summary_path}", flush=True)
    print(f"Wrote staged {mass_csv}", flush=True)
    print(f"Staged {len(files)} files under {work_dir}", flush=True)
    published = publish_artifacts_to_drive(work_dir, drive_out)
    summary["published_files"] = [str(path) for path in published]
    summary["n_png"] = sum(1 for path in published if path.suffix.lower() == ".png")
    _write_text(drive_out / "attn_sparsity_summary.json", json.dumps(summary, indent=2))
    return summary



def display_attn_sparsity(out_dir: Path, labels: tuple[str, ...] = ("N", "N-1", "N-2")) -> None:
    """Notebook helper: show primary attention + message heatmaps + mass CSV."""
    from IPython.display import Image, display
    import pandas as pd

    for label in labels:
        for name in (
            f"heatmap_{label}_all_stages_mean.png",
            f"heatmap_{label}_all_stages_message_mean.png",
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
        msg_stages = sorted(
            out_dir.glob(f"heatmap_{label}_stage*_message_mean_heads.png")
        )
        if msg_stages:
            last_msg = msg_stages[-1]
            print(last_msg.name, flush=True)
            display(Image(filename=str(last_msg)))
        curve = sorted(out_dir.glob(f"mass_curve_{label}_stage*.png"))
        if curve:
            print(curve[-1].name, flush=True)
            display(Image(filename=str(curve[-1])))
    csv_path = out_dir / "attention_mass_by_stage.csv"
    if csv_path.is_file():
        display(pd.read_csv(csv_path))
