#!/usr/bin/env python3
"""Phase 3 §2 — synthetic train/eval with slice metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.ablation import build_ablation  # noqa: E402
from jmat.data.manifest import load_manifest  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.train_loop import evaluate_paths, train_epochs  # noqa: E402


def _paths_by_bucket(samples: list[dict], data_root: Path, bucket: str) -> list[Path]:
    out = []
    for item in samples:
        b = item.get("eval_slice") or item.get("split") or item.get("split_bucket")
        if bucket == "train" and b not in ("train", "trainpool", "train"):
            if b != "train" and item.get("split") != "train":
                continue
        if bucket == "val" and item.get("split") != "train":
            # val carved from train pool via secondary tag
            if item.get("rhs_id", 0) % 10 != 0:
                continue
        elif bucket == "val":
            pass
        elif bucket.startswith("unseen") or bucket == "ill_conditioned":
            if item.get("eval_slice") != bucket and item.get("split") != "test":
                continue
        if bucket == "train" and item.get("split") == "test":
            continue
        if bucket == "val" and item.get("split") == "test":
            continue
        out.append(data_root / item["path"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "results" / "jmat" / "synthetic")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--pilot", action="store_true", help="Small synthetic + few epochs for Colab")
    p.add_argument("--ablation", type=str, default="jmat_full")
    args = p.parse_args()
    if args.pilot:
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_synthetic_jmat.py"),
                "--matrices-per-family",
                "3",
                "--rhs-per-matrix",
                "6",
                "--n-sizes",
                "4",
            ],
            cwd=ROOT,
        )
        args.epochs = min(args.epochs, 15)
    manifest_path = args.manifest or args.data_root / "manifest.json"
    samples, _ = load_manifest(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ablation(args.ablation) if args.ablation in (
        "mlp_only",
        "sparse_a_only",
        "sparse_a_at",
        "global_attn_only",
        "sparse_a_global",
        "jmat_full",
    ) else JMAT(JMATConfig())
    train_paths = [args.data_root / s["path"] for s in samples if s.get("split") == "train" and s.get("rhs_id", 0) % 10 != 0]
    val_paths = [args.data_root / s["path"] for s in samples if s.get("split") == "train" and s.get("rhs_id", 0) % 10 == 0]
    log = train_epochs(model, train_paths, val_paths, epochs=args.epochs, device=device)
    slices = {}
    for name, filt in (
        ("unseen_rhs", lambda s: s.get("split") == "test" and s.get("rhs_id", 0) >= 0),
        ("unseen_matrix", lambda s: s.get("split") == "test"),
        ("unseen_dim", lambda s: s.get("eval_slice") == "unseen_dim"),
        ("ill_conditioned", lambda s: s.get("family") == "ill_conditioned" and s.get("split") == "test"),
        ("unseen_family", lambda s: s.get("split") == "test"),
    ):
        paths = [args.data_root / s["path"] for s in samples if filt(s)]
        if paths:
            slices[name] = evaluate_paths(model, paths, slice_name=name, device=device).__dict__
    out_dir = ROOT / "results" / "jmat" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"synthetic_{args.ablation}.pt"
    torch.save(model.state_dict(), ckpt)
    curves = {
        "train_loss": [t.mean_loss for t in log.train],
        "val_loss": log.val_loss,
        "eval_slices": slices,
        "checkpoint": str(ckpt),
    }
    (out_dir / f"synthetic_{args.ablation}_report.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    print(json.dumps(curves, indent=2))


if __name__ == "__main__":
    main()