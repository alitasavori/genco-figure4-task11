#!/usr/bin/env python3
"""Train JMAT on NPZ shards with ablation flags (Task 5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.csr_io import scipy_csr_to_torch  # noqa: E402
from jmat.data.shard_io import load_triple_npz  # noqa: E402
from jmat.loss import jmat_combined_loss  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.normalize import normalize_linear_system  # noqa: E402


def _load_manifest_paths(manifest: Path, data_root: Path, want_split: str) -> list[Path]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    items = raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw
    out = []
    for item in items:
        split = item.get("split", item.get("split_bucket", "train"))
        if split != want_split:
            continue
        out.append(data_root / item["path"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "jmat" / "checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-global-attn", action="store_true")
    p.add_argument("--no-at", action="store_true")
    p.add_argument("--no-sparse-a", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=500)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    cfg = JMATConfig(
        use_global_attn=not args.no_global_attn,
        use_sparse_at=not args.no_at,
        use_sparse_a=not args.no_sparse_a,
    )
    model = JMAT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    paths = _load_manifest_paths(args.manifest, args.data_root, want_split="train")[: args.max_train]
    if not paths:
        raise SystemExit("No training paths in manifest (check split tags).")
    for epoch in range(args.epochs):
        losses = []
        for path in paths:
            j, r, dx, _meta = load_triple_npz(path)
            crow, col, val, n = scipy_csr_to_torch(j, dtype=torch.float32)
            rt = torch.tensor(r, dtype=torch.float32)
            norm = normalize_linear_system(crow, col, val, n, rt)
            if norm is None:
                continue
            z_star = torch.tensor(dx / norm.beta, dtype=torch.float32)
            opt.zero_grad()
            z_hat, _ = model.forward_normalized(norm)
            loss, _, _ = jmat_combined_loss(
                z_hat, z_star, norm.a_crow, norm.a_col, norm.a_val, n, norm.b_tilde, lam=1.0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        print(f"epoch {epoch} mean_loss={np.mean(losses):.4e} n={len(losses)}")
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"global{int(cfg.use_global_attn)}_a{int(cfg.use_sparse_a)}_at{int(cfg.use_sparse_at)}"
    ckpt = args.out / f"jmat_{tag}_seed{args.seed}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()