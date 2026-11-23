#!/usr/bin/env python3
"""PFΔ JSON → Newton trajectory NPZ shards (Task 3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.data.shard_io import save_triple_npz  # noqa: E402
from jmat.data.splits import SplitPolicy  # noqa: E402
from jmat.pf.newton_logger import newton_trace, verify_linear_residual  # noqa: E402
from jmat.pf.pfdelta_case import case_from_pfdelta_json  # noqa: E402
from pfdelta_case_parquet import HF_REPO, _load_json, list_raw_sample_paths  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True, help="PFΔ case raw folder with sample_*.json")
    p.add_argument("--grid-id", type=str, default="case118")
    p.add_argument("--source-split", type=str, default="train")
    p.add_argument("--topology-id", type=str, default="n")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "jmat" / "pfdelta_traces")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--lin-tol", type=float, default=1e-8)
    args = p.parse_args()
    paths = list_raw_sample_paths(args.raw_dir)[: args.max_samples]
    manifest = []
    policy = SplitPolicy(grid_id=args.grid_id, source_split=args.source_split)
    bucket = policy.bucket()
    for sample_path in paths:
        pm = _load_json(sample_path)
        case = case_from_pfdelta_json(pm)
        steps, status = newton_trace(
            case["g"],
            case["b_mat"],
            case["types"],
            case["p_spec"],
            case["q_spec"],
            v_set=case["v_set"],
        )
        scenario_id = sample_path.stem
        for step in steps:
            if step.lin_residual_rel > args.lin_tol:
                continue
            meta = {
                "grid_id": args.grid_id,
                "topology_id": args.topology_id,
                "scenario_id": scenario_id,
                "iteration": step.iteration,
                "split_bucket": bucket,
                "source_split": args.source_split,
                "hf_repo": HF_REPO,
                "newton_converged": bool(status["converged"]),
                "lin_residual_rel": step.lin_residual_rel,
            }
            rel = f"{bucket}/{args.grid_id}/{args.topology_id}/{scenario_id}/iter{step.iteration:02d}.npz"
            save_triple_npz(args.out / rel, step.j, step.r, step.dx_star, meta)
            manifest.append({"path": rel, **meta})
    out_manifest = args.out / f"manifest_{args.grid_id}_{args.source_split}.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} linear steps from {len(paths)} scenarios")


if __name__ == "__main__":
    main()