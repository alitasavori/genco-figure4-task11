#!/usr/bin/env python3
"""Phase 3 §4 — train on PFΔ pilot (118 sanity → joint 4-grid)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.data.manifest import load_manifest  # noqa: E402
from jmat.data.splits import SPLIT_HOLDOUT, SPLIT_TRAIN, SPLIT_VAL  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.train_loop import train_epochs  # noqa: E402


def filter_samples(samples, data_root: Path, *, grids: set[str] | None, buckets: set[str]):
    paths = []
    for s in samples:
        if buckets and s.get("split_bucket") not in buckets:
            continue
        if grids and s.get("grid_id") not in grids:
            continue
        paths.append(data_root / s["path"])
    return paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "results" / "jmat" / "pfdelta_pilot")
    p.add_argument("--mode", choices=("case118", "joint"), default="case118")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    if args.pilot:
        args.epochs = min(args.epochs, 10)
    samples, _ = load_manifest(args.data_root / "manifest.json")
    grids = {"case118"} if args.mode == "case118" else {"case14", "case30", "case118", "case500"}
    train_b = {SPLIT_TRAIN}
    val_b = {SPLIT_VAL}
    train_paths = filter_samples(samples, args.data_root, grids=grids, buckets=train_b)
    val_paths = filter_samples(samples, args.data_root, grids=grids, buckets=val_b)
    if not val_paths:
        # fallback: 10% scenario holdout by hash
        val_paths = train_paths[::10]
        train_paths = [p for i, p in enumerate(train_paths) if i % 10 != 0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JMAT(JMATConfig())
    log = train_epochs(model, train_paths, val_paths, epochs=args.epochs, device=device)
    out = ROOT / "results" / "jmat" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / f"pfdelta_{args.mode}.pt"
    torch.save(model.state_dict(), ckpt)
    report = {
        "mode": args.mode,
        "grids": sorted(grids),
        "n_train": len(train_paths),
        "n_val": len(val_paths),
        "train_loss": [t.mean_loss for t in log.train],
        "val_loss": log.val_loss,
        "checkpoint": str(ckpt),
        "holdout_excluded": SPLIT_HOLDOUT,
    }
    (out / f"train_pfdelta_{args.mode}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()