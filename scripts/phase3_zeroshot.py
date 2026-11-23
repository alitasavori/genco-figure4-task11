#!/usr/bin/env python3
"""Phase 3 §7 — frozen checkpoint on holdout grids + SuiteSparse."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.data.manifest import load_manifest  # noqa: E402
from jmat.data.splits import HOLDOUT_GRIDS, SPLIT_HOLDOUT  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.train_loop import evaluate_paths  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pfdelta-root", type=Path, default=ROOT / "results" / "jmat" / "pfdelta_pilot")
    p.add_argument("--suitesparse", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JMAT(JMATConfig()).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    out = {}
    man = args.pfdelta_root / "manifest.json"
    if man.is_file():
        samples, _ = load_manifest(man)
        for grid in HOLDOUT_GRIDS:
            paths = [
                args.pfdelta_root / s["path"]
                for s in samples
                if s.get("grid_id") == grid or s.get("split_bucket") == SPLIT_HOLDOUT
            ]
            if paths:
                out[grid] = evaluate_paths(model, paths, slice_name=grid, device=device).__dict__
    if args.suitesparse:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "evaluate_jmat_suitesparse.py"), "--checkpoint", str(args.checkpoint)],
            cwd=ROOT,
            check=False,
        )
    dest = ROOT / "results" / "jmat" / "phase3" / "zeroshot.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(dest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()