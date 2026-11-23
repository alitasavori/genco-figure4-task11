#!/usr/bin/env python3
"""Phase 3 §3 — PFΔ Newton traces (500 scenarios/grid pilot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.data.manifest import assign_pfdelta_bucket, write_manifest  # noqa: E402
from jmat.data.shard_io import save_triple_npz  # noqa: E402
from jmat.data.splits import HOLDOUT_GRIDS, TRAIN_GRIDS  # noqa: E402
from jmat.pf.newton_logger import newton_trace  # noqa: E402
from jmat.pf.pfdelta_case import case_from_pfdelta_json  # noqa: E402
from jmat.pf.pfdelta_paths import resolve_pfdelta_raw_dir  # noqa: E402
from pfdelta_case_parquet import HF_REPO, _load_json, list_raw_sample_paths  # noqa: E402


def process_grid(raw_dir: Path, grid_id: str, source_split: str, topology: str, max_samples: int, out: Path):
    manifest = []
    stats = {"scenarios": 0, "steps": 0, "failed_scenarios": 0, "j_dims": [], "nnz": [], "cond": []}
    paths = list_raw_sample_paths(raw_dir)[:max_samples]
    print(f"  {grid_id}: {len(paths)} scenarios from {raw_dir}", flush=True)
    for si, sample_path in enumerate(paths):
        stats["scenarios"] += 1
        if si % 5 == 0:
            print(f"    {grid_id} scenario {si + 1}/{len(paths)}", flush=True)
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
            if not st["converged"]:
                stats["failed_scenarios"] += 1
            for step in steps:
                if step.lin_residual_rel > 1e-8:
                    continue
                bucket = assign_pfdelta_bucket(grid_id, source_split, sample_path.stem, topology)
                meta = {
                    "grid_id": grid_id,
                    "topology_id": topology,
                    "scenario_id": sample_path.stem,
                    "iteration": step.iteration,
                    "split_bucket": bucket,
                    "source_split": source_split,
                    "hf_repo": HF_REPO,
                }
                rel = f"{bucket}/{grid_id}/{topology}/{sample_path.stem}/iter{step.iteration:02d}.npz"
                save_triple_npz(out / rel, step.j, step.r, step.dx_star, meta)
                manifest.append({"path": rel, **meta})
                stats["steps"] += 1
                stats["j_dims"].append(step.j.shape[0])
                stats["nnz"].append(int(step.j.nnz))
        except Exception:
            stats["failed_scenarios"] += 1
    return manifest, stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vendor-root", type=Path, default=ROOT.parent / "Foundation Model" / "vendor")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "jmat" / "pfdelta_pilot")
    p.add_argument("--per-grid", type=int, default=500)
    p.add_argument("--pilot", action="store_true", help="20 samples/grid for smoke")
    args = p.parse_args()
    n = 20 if args.pilot else args.per_grid
    print(f"phase3_pfdelta_pilot: pilot={args.pilot} per_grid={n} vendor={args.vendor_root}", flush=True)
    grids_train = list(TRAIN_GRIDS)
    grids_hold = list(HOLDOUT_GRIDS)
    all_manifest = []
    grid_stats = {}
    for grid in grids_train + grids_hold:
        raw = resolve_pfdelta_raw_dir(args.vendor_root, grid)
        if raw is None:
            print(f"skip {grid}: no sample_*.json under Task 1.1 paths (see jmat/pf/pfdelta_paths.py)", flush=True)
            continue
        src = "train" if grid in grids_train else "holdout"
        man, st = process_grid(raw, grid, src, "n", n, args.out)
        all_manifest.extend(man)
        grid_stats[grid] = st
    write_manifest(args.out / "manifest.json", all_manifest, meta={"per_grid": n, "stats": grid_stats})
    (args.out / "dataset_stats.json").write_text(json.dumps(grid_stats, indent=2), encoding="utf-8")
    print(f"manifest entries={len(all_manifest)} stats={grid_stats}")


if __name__ == "__main__":
    main()