#!/usr/bin/env python3
"""Phase 3 §3 — PFΔ Newton traces with interpretable CSV reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.data.csv_reports import manifest_to_triples_csv, write_csv  # noqa: E402
from jmat.data.manifest import assign_pfdelta_bucket, write_manifest  # noqa: E402
from jmat.data.shard_io import save_triple_npz  # noqa: E402
from jmat.data.splits import HOLDOUT_GRIDS, TRAIN_GRIDS  # noqa: E402
from jmat.pf.newton_logger import newton_trace  # noqa: E402
from jmat.pf.pfdelta_case import case_from_pfdelta_json  # noqa: E402
from jmat.pf.pfdelta_paths import (  # noqa: E402
    diagnose_raw_dirs,
    resolve_pfdelta_raw_dir,
    resolve_vendor_for_grid,
)
from pfdelta_case_parquet import HF_REPO, _load_json, list_raw_sample_paths  # noqa: E402


def process_grid(
    raw_dir: Path,
    grid_id: str,
    source_split: str,
    topology: str,
    num_oc: int,
    out: Path,
    *,
    lin_tol: float = 1e-8,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    manifest: list[dict] = []
    oc_rows: list[dict] = []
    iter_rows: list[dict] = []
    stats = {
        "requested_oc": num_oc,
        "processed_oc": 0,
        "failed_oc": 0,
        "saved_shards": 0,
        "newton_iters_total": 0,
        "converged_oc": 0,
    }
    paths = list_raw_sample_paths(raw_dir)[:num_oc]
    print(f"  {grid_id}: {len(paths)} OCs (requested {num_oc}) from {raw_dir}", flush=True)
    for si, sample_path in enumerate(paths):
        stats["processed_oc"] += 1
        oc_id = sample_path.stem
        if si % 5 == 0:
            print(f"    {grid_id} OC {si + 1}/{len(paths)} ({oc_id})", flush=True)
        row: dict = {
            "grid_id": grid_id,
            "scenario_id": oc_id,
            "topology_id": topology,
            "source_split": source_split,
            "json_path": str(sample_path),
            "requested": 1,
            "newton_iters_total": 0,
            "saved_iterations": 0,
            "converged": 0,
            "final_mismatch_inf": "",
            "status": "ok",
            "error": "",
        }
        try:
            pm = _load_json(sample_path)
            case = case_from_pfdelta_json(pm)
            steps, st = newton_trace(
                case["g"],
                case["b_mat"],
                case["types"],
                case["p_spec"],
                case["q_spec"],
                v_set=case["v_set"],
            )
            row["newton_iters_total"] = len(steps)
            row["final_mismatch_inf"] = st.get("final_mismatch_inf", "")
            row["converged"] = int(bool(st.get("converged")))
            stats["newton_iters_total"] += len(steps)
            if st.get("converged"):
                stats["converged_oc"] += 1
            else:
                stats["failed_oc"] += 1
            for step in steps:
                if step.lin_residual_rel > lin_tol:
                    continue
                bucket = assign_pfdelta_bucket(grid_id, source_split, oc_id, topology)
                meta = {
                    "grid_id": grid_id,
                    "topology_id": topology,
                    "scenario_id": oc_id,
                    "iteration": step.iteration,
                    "split_bucket": bucket,
                    "source_split": source_split,
                    "hf_repo": HF_REPO,
                    "n_reduced": step.j.shape[0],
                    "j_nnz": int(step.j.nnz),
                    "lin_residual_rel": step.lin_residual_rel,
                }
                rel = f"{bucket}/{grid_id}/{topology}/{oc_id}/iter{step.iteration:02d}.npz"
                save_triple_npz(out / rel, step.j, step.r, step.dx_star, meta)
                manifest.append({"path": rel, **meta})
                iter_rows.append(
                    {
                        "grid_id": grid_id,
                        "scenario_id": oc_id,
                        "topology_id": topology,
                        "split_bucket": bucket,
                        "iteration": step.iteration,
                        "n_reduced": step.j.shape[0],
                        "j_nnz": int(step.j.nnz),
                        "lin_residual_rel": step.lin_residual_rel,
                        "npz_path": rel,
                    }
                )
                row["saved_iterations"] += 1
                stats["saved_shards"] += 1
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            stats["failed_oc"] += 1
        oc_rows.append(row)
    return manifest, stats, oc_rows, iter_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Generate JMAT PFΔ Newton dataset + CSV reports.")
    p.add_argument("--vendor-root", type=Path, default=ROOT.parent / "Foundation Model" / "vendor")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "jmat" / "pfdelta_pilot")
    p.add_argument(
        "--num-oc",
        "--per-grid",
        type=int,
        dest="num_oc",
        default=500,
        help="Operating conditions (PFΔ scenarios) per grid",
    )
    p.add_argument("--pilot", action="store_true", help="20 OCs/grid smoke test")
    p.add_argument("--lin-tol", type=float, default=1e-8)
    p.add_argument(
        "--grids",
        type=str,
        nargs="*",
        default=None,
        help="Only these grid ids (e.g. case57). Default: all train + holdout grids.",
    )
    args = p.parse_args()
    num_oc = 20 if args.pilot else args.num_oc
    reports = args.out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    print(
        f"phase3_pfdelta_pilot: num_oc_per_grid={num_oc} pilot={args.pilot} vendor={args.vendor_root} out={args.out}",
        flush=True,
    )
    grids_train = list(TRAIN_GRIDS)
    grids_hold = list(HOLDOUT_GRIDS)
    if args.grids:
        want = {g if g.startswith("case") else f"case{g}" for g in args.grids}
        grids_train = [g for g in grids_train if g in want]
        grids_hold = [g for g in grids_hold if g in want]
        if not grids_train and not grids_hold:
            raise SystemExit(f"No matching grids in {want}; train={TRAIN_GRIDS} holdout={HOLDOUT_GRIDS}")
        print(f"Grid filter: train={grids_train} holdout={grids_hold}", flush=True)
    all_manifest: list[dict] = []
    all_oc: list[dict] = []
    all_iter: list[dict] = []
    grid_rows: list[dict] = []
    path_diag: list[dict] = []
    for grid in grids_train + grids_hold:
        raw = None
        used_vendor = args.vendor_root
        for vroot in resolve_vendor_for_grid(args.vendor_root, grid):
            raw = resolve_pfdelta_raw_dir(vroot, grid)
            if raw is not None:
                used_vendor = vroot
                break
        for row in diagnose_raw_dirs(args.vendor_root, grid):
            path_diag.append({"grid_id": grid, "vendor_root": str(args.vendor_root), **row})
        if raw is None:
            grid_rows.append(
                {
                    "grid_id": grid,
                    "raw_dir": "",
                    "vendor_used": "",
                    "requested_oc": num_oc,
                    "available_json_oc": 0,
                    "processed_oc": 0,
                    "saved_shards": 0,
                    "converged_oc": 0,
                    "failed_oc": 0,
                    "newton_iters_total": 0,
                    "status": "skipped_no_json",
                }
            )
            print(f"\n*** NO JSON for {grid} — path check ***", flush=True)
            for row in path_diag:
                if row["grid_id"] != grid:
                    continue
                print(
                    f"  exists={row['exists']}  n_json={row['n_json']}  {row['path'][:100]}",
                    flush=True,
                )
            print(
                f"  Fix: use --vendor-root with pfdelta_task1.1_{grid} or ensure train/.../raw has sample_*.json",
                flush=True,
            )
            continue
        available = len(list_raw_sample_paths(raw))
        src = "train" if grid in grids_train else "holdout"
        man, st, oc_rows, iter_rows = process_grid(raw, grid, src, "n", num_oc, args.out, lin_tol=args.lin_tol)
        all_manifest.extend(man)
        all_oc.extend(oc_rows)
        all_iter.extend(iter_rows)
        grid_rows.append(
            {
                "grid_id": grid,
                "raw_dir": str(raw),
                "vendor_used": str(used_vendor),
                "requested_oc": num_oc,
                "available_json_oc": available,
                "processed_oc": st["processed_oc"],
                "saved_shards": st["saved_shards"],
                "converged_oc": st["converged_oc"],
                "failed_oc": st["failed_oc"],
                "newton_iters_total": st["newton_iters_total"],
                "status": "ok" if st["saved_shards"] else "no_shards",
            }
        )
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_oc_per_grid": num_oc,
        "pilot": args.pilot,
        "vendor_root": str(args.vendor_root),
        "total_shards": len(all_manifest),
        "total_oc": len(all_oc),
    }
    write_manifest(args.out / "manifest.json", all_manifest, meta=meta)
    (args.out / "dataset_stats.json").write_text(
        json.dumps({"meta": meta, "grids": grid_rows}, indent=2),
        encoding="utf-8",
    )

    n_grid = write_csv(
        reports / "grid_summary.csv",
        [
            "grid_id",
            "raw_dir",
            "vendor_used",
            "requested_oc",
            "available_json_oc",
            "processed_oc",
            "saved_shards",
            "converged_oc",
            "failed_oc",
            "newton_iters_total",
            "status",
        ],
        grid_rows,
    )
    n_oc = write_csv(
        reports / "oc_summary.csv",
        [
            "grid_id",
            "scenario_id",
            "topology_id",
            "source_split",
            "json_path",
            "newton_iters_total",
            "saved_iterations",
            "converged",
            "final_mismatch_inf",
            "status",
            "error",
        ],
        all_oc,
    )
    n_iter = write_csv(
        reports / "iteration_log.csv",
        [
            "grid_id",
            "scenario_id",
            "topology_id",
            "split_bucket",
            "iteration",
            "n_reduced",
            "j_nnz",
            "lin_residual_rel",
            "npz_path",
        ],
        all_iter,
    )
    n_triple = manifest_to_triples_csv(all_manifest, reports / "training_triples_index.csv")
    write_csv(
        reports / "generation_run.csv",
        [
            "generated_at_utc",
            "num_oc_per_grid",
            "pilot",
            "vendor_root",
            "out_dir",
            "total_oc",
            "total_saved_iterations",
            "total_npz_shards",
        ],
        [
            {
                **meta,
                "out_dir": str(args.out),
                "total_saved_iterations": n_iter,
                "total_npz_shards": n_triple,
            }
        ],
    )

    write_csv(
        reports / "path_diagnostics.csv",
        ["grid_id", "vendor_root", "path", "exists", "n_json"],
        path_diag,
    )

    print("\n" + "=" * 60, flush=True)
    print("PFΔ → JMAT  INTERPRETABLE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for g in grid_rows:
        print(
            f"Grid {g['grid_id']}: status={g['status']}  "
            f"OCs processed={g.get('processed_oc', 0)}/{g.get('requested_oc', 0)}  "
            f"saved Newton steps={g.get('saved_shards', 0)}  "
            f"converged OC={g.get('converged_oc', 0)}",
            flush=True,
        )
        if g.get("raw_dir"):
            print(f"  raw_json: {g['raw_dir']}", flush=True)
    if all_oc:
        iters = [int(r["newton_iters_total"]) for r in all_oc if r.get("newton_iters_total")]
        saved = [int(r["saved_iterations"]) for r in all_oc if r.get("saved_iterations")]
        if iters:
            print(
                f"Per OC Newton iters: min={min(iters)} max={max(iters)} avg={sum(iters)/len(iters):.1f}",
                flush=True,
            )
        if saved:
            print(
                f"Saved training rows per OC: min={min(saved)} max={max(saved)} total={sum(saved)}",
                flush=True,
            )
        print("\nFirst 5 operating conditions (see oc_summary.csv):", flush=True)
        for r in all_oc[:5]:
            print(
                f"  {r['scenario_id']}: newton_iters={r['newton_iters_total']} "
                f"saved={r['saved_iterations']} converged={r['converged']}",
                flush=True,
            )
    else:
        print("\n>>> ZERO operating conditions processed — only CSV diagnostics written.", flush=True)
        print(f"    Open: {reports / 'path_diagnostics.csv'}", flush=True)
    print(f"\nCSV folder: {reports}", flush=True)
    print(f"NPZ training shards: {len(all_manifest)} under {args.out}", flush=True)
    print("=" * 60 + "\n", flush=True)


if __name__ == "__main__":
    main()