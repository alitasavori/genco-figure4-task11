"""Experiment B — numerical rank of learned GRIT attention (diagnostic, no retrain).

For each stage/head, SVD of the attention matrix A^{(ℓ,h)} and of pair-conditioned
message magnitudes α‖v‖; report cumulative Frobenius energy E(r) at r=4,8,16,32 and
optimistic low-rank row-renormalized PBL on N / N-1 / N-2.

SVD at inference would be expensive; this only diagnoses whether a structured
low-rank GRIT approximation is worth building separately.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import attn_sparsity_diag
from attn_sparsity_diag import (
    _forward_capture,
    _fsync_file,
    _local_staging_dir,
    _sample_indices,
    _tpgt,
    _write_text,
    publish_artifacts_to_drive,
    read_stage_attention,
)
from plot_style import apply_plot_style, style_axis

DEFAULT_RANKS = (4, 8, 16, 32)
ENERGY_THRESHOLDS = (0.90, 0.95, 0.99)


def singular_energy_stats(
    matrix: np.ndarray,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
) -> dict[str, float]:
    """SVD energy for one (N, N) matrix (Frobenius): E(r)=sum_{k≤r} σ² / sum σ²."""
    flat = np.asarray(matrix, dtype=np.float64)
    if flat.ndim != 2 or flat.shape[0] == 0:
        return {}
    # Nonnegative matrices can still have negative SVD entries after truncation;
    # energy uses singular values of the raw matrix.
    try:
        singular = np.linalg.svd(flat, compute_uv=False)
    except np.linalg.LinAlgError:
        return {}
    power = np.square(singular)
    total = float(power.sum())
    if total <= 0 or not np.isfinite(total):
        return {f"E@{r}": float("nan") for r in ranks} | {
            f"rank@{int(t * 100)}": float("nan") for t in ENERGY_THRESHOLDS
        }
    csum = np.cumsum(power) / total
    out: dict[str, float] = {
        "n": float(flat.shape[0]),
        "nuclear": float(singular.sum()),
        "frobenius": float(np.sqrt(total)),
        "sigma1_frac": float(power[0] / total),
    }
    for r in ranks:
        rr = min(int(r), len(csum))
        out[f"E@{r}"] = float(csum[rr - 1]) if rr > 0 else 0.0
    for t in ENERGY_THRESHOLDS:
        hit = np.nonzero(csum >= t)[0]
        out[f"rank@{int(t * 100)}"] = float(hit[0] + 1) if hit.size else float(len(csum))
    return out


def summarize_tensor_rank(
    tensor: torch.Tensor,
    *,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
) -> dict[str, Any]:
    """``tensor``: (B, H, N, N) — mean energy stats over batch×heads, plus per-head."""
    arr = tensor.numpy()
    batch, heads, buses, _ = arr.shape
    per: list[dict[str, float]] = []
    for b in range(batch):
        for h in range(heads):
            stats = singular_energy_stats(arr[b, h], ranks=ranks)
            if not stats:
                continue
            stats["batch"] = float(b)
            stats["head"] = float(h)
            per.append(stats)

    keys = (
        [f"E@{r}" for r in ranks]
        + [f"rank@{int(t * 100)}" for t in ENERGY_THRESHOLDS]
        + ["sigma1_frac", "nuclear", "frobenius"]
    )

    def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
        return {
            key: float(np.nanmean([r[key] for r in rows])) if rows else float("nan")
            for key in keys
        }

    return {
        "overall": _mean(per),
        "by_head": {
            str(h): _mean([r for r in per if int(r["head"]) == h]) for h in range(heads)
        },
        "n": buses,
        "n_matrices": len(per),
        "shape": [batch, heads, buses, buses],
    }


def plot_energy_curve(
    tensor: torch.Tensor,
    out_path: Path,
    *,
    title: str,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
) -> dict[str, float]:
    """Mean cumulative singular-energy curve over batch×heads; mark r in ``ranks``."""
    apply_plot_style()
    arr = tensor.numpy()
    batch, heads, buses, _ = arr.shape
    curves: list[np.ndarray] = []
    for b in range(batch):
        for h in range(heads):
            try:
                singular = np.linalg.svd(arr[b, h].astype(np.float64), compute_uv=False)
            except np.linalg.LinAlgError:
                continue
            power = np.square(singular)
            total = float(power.sum())
            if total <= 0:
                continue
            curves.append(np.cumsum(power) / total)
    if not curves:
        return {}
    stacked = np.stack(curves, axis=0)
    mean_c = stacked.mean(axis=0)
    xs = np.arange(1, len(mean_c) + 1)
    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(xs, mean_c, color="#1F4B99", linewidth=2.0, label="Mean E(r)")
    marks: dict[str, float] = {}
    for r in ranks:
        rr = min(int(r), len(mean_c))
        if rr <= 0:
            continue
        e = float(mean_c[rr - 1])
        marks[f"E@{r}"] = e
        axis.axvline(rr, color="#C44E52", linestyle="--", linewidth=1.0, alpha=0.75)
        axis.text(rr + 0.3, min(0.98, e + 0.02), f"r={rr}\nE={e:.2f}", fontsize=10)
    for t in ENERGY_THRESHOLDS:
        axis.axhline(t, color="#888888", linestyle=":", linewidth=1.0)
    axis.set_xlabel("Rank r")
    axis.set_ylabel("Cumulative energy E(r)")
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


def run_attention_rank_diagnostic(
    model,
    data: Path,
    device: torch.device,
    active_cache: Path,
    persistent_cache: Path,
    *,
    out_dir: Path,
    cases: int = 16,
    labels: tuple[str, ...] = ("N", "N-1", "N-2"),
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    eval_batch_size: int | None = None,
) -> dict[str, Any]:
    """Full Experiment B: SVD energy tables/curves + optimistic low-rank PBL."""
    del eval_batch_size  # kept for CLI symmetry with sparsity diag
    tpgt = _tpgt()
    ATTENTION_KIND = tpgt.ATTENTION_KIND
    DEPTH = tpgt.DEPTH
    HEADS = tpgt.HEADS
    TEST_PARTS = tpgt.TEST_PARTS
    cached_split = tpgt.cached_split
    raw_dir = tpgt.raw_dir
    CASE_NAME = tpgt.CASE_NAME

    if ATTENTION_KIND != "grit":
        raise RuntimeError(f"--attn-rank requires grit attention, got {ATTENTION_KIND}")
    print(f"Using CASE_NAME={CASE_NAME} for split paths under {data}", flush=True)
    probe = raw_dir(data, "test", "n", "feasible")
    if not (probe / "bus_data.parquet").is_file():
        raise FileNotFoundError(
            f"Missing test N feasible parquet at {probe / 'bus_data.parquet'} "
            f"(CASE_NAME={CASE_NAME})."
        )

    drive_out = Path(out_dir)
    drive_out.mkdir(parents=True, exist_ok=True)
    stage_dir = _local_staging_dir(drive_out)
    work_dir = stage_dir
    print(f"Staging artifacts locally at {work_dir} (then copy to {drive_out})", flush=True)

    summary: dict[str, Any] = {
        "case": CASE_NAME,
        "depth": int(DEPTH),
        "heads": int(HEADS),
        "cases": int(cases),
        "labels": list(labels),
        "ranks": list(ranks),
        "energy_thresholds": list(ENERGY_THRESHOLDS),
        "by_label": {},
        "drive_out": str(drive_out),
        "stage_dir": str(stage_dir),
        "note": (
            "Diagnostic only: E(r) from SVD of captured dense α / α‖v‖. "
            "Low-rank PBL compares three post-softmax SVD modes in-process: "
            "raw (UΣVᵀ only), clamp_renorm (clamp≥0 + row-norm; first attn-rank "
            "behavior), repaired (+ restore original >0 support). Also reports "
            "||repaired−raw||_F and ||repaired−clamp_renorm||_F. Identical "
            "clamp_renorm vs repaired PBL means support masking was a no-op, not "
            "that the hook failed. Production low-rank GRIT needs a separate impl."
        ),
    }

    energy_csv = work_dir / "attention_rank_energy_by_stage.csv"
    energy_rows: list[list[Any]] = []
    pbl_csv = work_dir / "attention_rank_pbl.csv"
    pbl_rows: list[list[Any]] = []

    for label, topology, condition in TEST_PARTS:
        if label not in labels or condition != "feasible":
            continue
        print(f"=== Attention rank: {label} ===", flush=True)
        split = cached_split(
            active_cache,
            persistent_cache,
            raw_dir(data, "test", topology, condition),
            f"test_{topology}_{condition}",
        )
        if label == "N":
            net_id = 0
        else:
            counts = torch.bincount(split.network_index, minlength=len(split.networks))
            net_id = int(torch.argmax(counts))
        network = split.networks[net_id]
        index = _sample_indices(split, net_id, cases)
        n_buses = int(network.ybus.shape[0])
        print(
            f"  network_id={net_id} buses={n_buses} cases={len(index)}",
            flush=True,
        )

        captured, _types, _dense_pbl = _forward_capture(
            model, split, network, index, device, contrib=True, topk=None
        )
        label_summary: dict[str, Any] = {
            "network_id": net_id,
            "n_buses": n_buses,
            "n_cases": int(len(index)),
            "stages": [],
        }

        for stage_i, payload in enumerate(captured):
            attn = payload["attn"]
            contrib = payload["contrib"]
            if attn is None:
                continue
            attn_rank = summarize_tensor_rank(attn, ranks=ranks)
            contrib_rank = (
                summarize_tensor_rank(contrib, ranks=ranks) if contrib is not None else None
            )
            stem = f"{label}_stage{stage_i + 1:02d}"
            curve_marks = plot_energy_curve(
                attn,
                work_dir / f"energy_curve_{stem}_attention.png",
                title=f"{label} stage {stage_i + 1}: attention singular energy",
                ranks=ranks,
            )
            if contrib is not None:
                plot_energy_curve(
                    contrib,
                    work_dir / f"energy_curve_{stem}_message.png",
                    title=f"{label} stage {stage_i + 1}: message α‖v‖ singular energy",
                    ranks=ranks,
                )

            ov = attn_rank["overall"]
            print(
                f"  stage {stage_i + 1}: attn E@4/8/16/32 = "
                f"{ov.get('E@4', float('nan')):.3f} / "
                f"{ov.get('E@8', float('nan')):.3f} / "
                f"{ov.get('E@16', float('nan')):.3f} / "
                f"{ov.get('E@32', float('nan')):.3f}  "
                f"rank@90/95/99 = "
                f"{ov.get('rank@90', float('nan')):.1f} / "
                f"{ov.get('rank@95', float('nan')):.1f} / "
                f"{ov.get('rank@99', float('nan')):.1f}",
                flush=True,
            )
            if contrib_rank is not None:
                cov = contrib_rank["overall"]
                print(
                    f"           msg  E@4/8/16/32 = "
                    f"{cov.get('E@4', float('nan')):.3f} / "
                    f"{cov.get('E@8', float('nan')):.3f} / "
                    f"{cov.get('E@16', float('nan')):.3f} / "
                    f"{cov.get('E@32', float('nan')):.3f}  "
                    f"rank@90/95/99 = "
                    f"{cov.get('rank@90', float('nan')):.1f} / "
                    f"{cov.get('rank@95', float('nan')):.1f} / "
                    f"{cov.get('rank@99', float('nan')):.1f}",
                    flush=True,
                )

            label_summary["stages"].append(
                {
                    "stage": stage_i + 1,
                    "attention_rank": attn_rank,
                    "message_contrib_rank": contrib_rank,
                    "curve_E": curve_marks,
                }
            )

            for metric_name, stats in (
                ("attention", attn_rank),
                ("message_contrib", contrib_rank),
            ):
                if stats is None:
                    continue
                for group, payload_stats in (
                    ("overall", stats["overall"]),
                    *[("head_" + h, v) for h, v in stats["by_head"].items()],
                ):
                    energy_rows.append(
                        [
                            label,
                            stage_i + 1,
                            metric_name,
                            group,
                            n_buses,
                            *[payload_stats.get(f"E@{r}", "") for r in ranks],
                            payload_stats.get("rank@90", ""),
                            payload_stats.get("rank@95", ""),
                            payload_stats.get("rank@99", ""),
                            payload_stats.get("sigma1_frac", ""),
                        ]
                    )

        # Offline repair delta on captured dense α (same process; no saved PBL reuse).
        # Δ_repair = ||α_repaired − α_raw||_F  (and vs clamp_renorm = pre-b6db7c0 path).
        repair_rows: list[dict[str, Any]] = []
        lowrank_variants = _tpgt()._lowrank_attention_variants
        print("  repair check (captured dense α → SVD variants):", flush=True)
        for r in ranks:
            deltas_raw: list[float] = []
            deltas_clamp: list[float] = []
            for payload in captured:
                attn = payload["attn"]
                if attn is None:
                    continue
                variants = lowrank_variants(attn.to(device), int(r))
                raw = variants["raw"]
                clamp_renorm = variants["clamp_renorm"]
                repaired = variants["repaired"]
                deltas_raw.append(float(torch.linalg.vector_norm(repaired - raw).item()))
                deltas_clamp.append(
                    float(torch.linalg.vector_norm(repaired - clamp_renorm).item())
                )
            d_raw = float(np.mean(deltas_raw)) if deltas_raw else float("nan")
            d_clamp = float(np.mean(deltas_clamp)) if deltas_clamp else float("nan")
            repair_rows.append(
                {
                    "rank": int(r),
                    "delta_repair_vs_raw_fro": d_raw,
                    "delta_repair_vs_clamp_renorm_fro": d_clamp,
                }
            )
            print(
                f"    r={r}: ||repaired−raw||_F={d_raw:.6g}  "
                f"||repaired−clamp_renorm||_F={d_clamp:.6g}",
                flush=True,
            )
        label_summary["repair_delta"] = repair_rows

        # Optimistic low-rank PBL: raw vs clamp_renorm vs repaired (same process).
        _, _, dense_pbl = _forward_capture(
            model, split, network, index, device, contrib=False, topk=None
        )
        topk_rows: list[dict[str, Any]] = [{"rank": "dense", "pbl": float(dense_pbl)}]
        pbl_rows.append([label, "dense", "", float(dense_pbl), 0.0])
        print(f"  dense sample PBL={dense_pbl:.6g}", flush=True)

        for r in ranks:
            mode_pbls: dict[str, float] = {}
            for mode in ("raw", "clamp_renorm", "repaired"):
                _, _, pbl_r = _forward_capture_lowrank(
                    model,
                    split,
                    network,
                    index,
                    device,
                    rank=int(r),
                    mode=mode,
                )
                mode_pbls[mode] = float(pbl_r)
                delta = float(pbl_r - dense_pbl)
                pbl_rows.append([label, mode, int(r), float(pbl_r), delta])
            topk_rows.append({"rank": int(r), **mode_pbls})
            print(
                f"  low-rank r={r} PBL raw/clamp_renorm/repaired = "
                f"{mode_pbls['raw']:.6g} / {mode_pbls['clamp_renorm']:.6g} / "
                f"{mode_pbls['repaired']:.6g}",
                flush=True,
            )

        label_summary["lowrank_pbl_sample"] = topk_rows
        summary["by_label"][label] = label_summary
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with energy_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "label",
                "stage",
                "metric",
                "group",
                "n",
                *[f"E@{r}" for r in ranks],
                "rank@90",
                "rank@95",
                "rank@99",
                "sigma1_frac",
            ]
        )
        writer.writerows(energy_rows)
        fh.flush()
        os.fsync(fh.fileno())

    with pbl_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "kind", "rank", "pbl", "delta_vs_dense"])
        writer.writerows(pbl_rows)
        fh.flush()
        os.fsync(fh.fileno())

    summary_path = work_dir / "attn_rank_summary.json"
    _write_text(summary_path, json.dumps(summary, indent=2))
    files = sorted(p for p in work_dir.rglob("*") if p.is_file())
    _write_text(
        work_dir / "MANIFEST.txt",
        "\n".join(f"{p.relative_to(work_dir)}\t{p.stat().st_size}" for p in files) + "\n",
    )
    print(f"Wrote staged {summary_path}", flush=True)
    print(f"Wrote staged {energy_csv}", flush=True)
    print(f"Wrote staged {pbl_csv}", flush=True)
    published = publish_artifacts_to_drive(
        work_dir, drive_out, zip_name="attn_rank_artifacts.zip"
    )
    summary["published_files"] = [str(path) for path in published]
    summary["n_png"] = sum(1 for path in published if path.suffix.lower() == ".png")
    _write_text(drive_out / "attn_rank_summary.json", json.dumps(summary, indent=2))
    return summary


def _forward_capture_lowrank(
    model,
    split,
    network,
    index,
    device,
    *,
    rank: int,
    mode: str = "repaired",
):
    """Forward with SVD-truncated attention; ``mode`` in raw / clamp_renorm / repaired."""
    # Always resolve capture from the live module (Colab may keep a stale import).
    capture_cm = attn_sparsity_diag.grit_attention_capture
    tpgt = _tpgt()
    case_pbl = tpgt.case_pbl
    forward_cases = tpgt.forward_cases
    move_to = tpgt.move_to
    pair_features = tpgt.pair_features

    device_network = move_to(network, device)
    pair = pair_features(model, device_network)
    with torch.inference_mode():
        with capture_cm(
            model,
            contrib=False,
            topk=None,
            lowrank=int(rank),
            lowrank_mode=str(mode),
        ):
            _prediction, _target, residuals = forward_cases(
                model, split, device_network, index, device, pair
            )
            captured = read_stage_attention(model)
    pbl = float(case_pbl(residuals[-1]).mean().item())
    types = device_network.types.detach().cpu().numpy()
    return captured, types, pbl


def display_attn_rank(out_dir: Path, labels: tuple[str, ...] = ("N", "N-1", "N-2")) -> None:
    """Notebook helper: show energy curves + PBL / energy CSVs."""
    from IPython.display import Image, display
    import pandas as pd

    for label in labels:
        curves = sorted(out_dir.glob(f"energy_curve_{label}_stage*_attention.png"))
        if curves:
            # First + last stage attention curves
            for path in (curves[0], curves[-1]):
                print(path.name, flush=True)
                display(Image(filename=str(path)))
        msg = sorted(out_dir.glob(f"energy_curve_{label}_stage*_message.png"))
        if msg:
            print(msg[-1].name, flush=True)
            display(Image(filename=str(msg[-1])))
    pbl_path = out_dir / "attention_rank_pbl.csv"
    if pbl_path.is_file():
        print("attention_rank_pbl.csv", flush=True)
        display(pd.read_csv(pbl_path))
    energy_path = out_dir / "attention_rank_energy_by_stage.csv"
    if energy_path.is_file():
        print("attention_rank_energy_by_stage.csv (overall rows)", flush=True)
        df = pd.read_csv(energy_path)
        display(df[df["group"] == "overall"])
