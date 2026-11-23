#!/usr/bin/env python3
"""Generate synthetic (J, r, x*) shards with held-out matrix families/sizes (Task 2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.data.shard_io import save_triple_npz  # noqa: E402
from jmat.data.synthetic import SyntheticFamily, sample_matrix, sample_rhs_ensemble  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "results" / "jmat" / "synthetic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-sizes", type=int, default=5)
    p.add_argument("--matrices-per-family", type=int, default=4)
    p.add_argument("--rhs-per-matrix", type=int, default=8)
    p.add_argument("--holdout-n", type=int, nargs="*", default=[128, 256])
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    manifest = []
    train_sizes = [16, 24, 32, 48, 64, 96][: args.n_sizes]
    split_assign: dict[str, str] = {}
    for family in SyntheticFamily:
        for mid in range(args.matrices_per_family):
            for n in train_sizes:
                split = "train"
                if n in args.holdout_n and mid % 2 == 1:
                    split = "test"
                j, meta = sample_matrix(family, n, mid, seed=args.seed)
                pairs = sample_rhs_ensemble(j, rng, args.rhs_per_matrix)
                for rhs_id, (r, x_star) in enumerate(pairs):
                    rel_path = (
                        f"{split}/{family.value}/n{n}/mat{mid}/rhs{rhs_id}.npz"
                    )
                    out_path = args.out / rel_path
                    meta_dict = {
                        "family": family.value,
                        "n": n,
                        "matrix_id": mid,
                        "rhs_id": rhs_id,
                        "split": split,
                        "cond_hint": meta.cond_hint,
                    }
                    save_triple_npz(out_path, j, r, x_star, meta_dict)
                    manifest.append({"path": rel_path, **meta_dict})
    stats_path = args.out / "manifest.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({"samples": manifest, "split_assign": split_assign}, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} samples -> {args.out}")


if __name__ == "__main__":
    main()