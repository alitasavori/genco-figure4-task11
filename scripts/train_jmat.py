#!/usr/bin/env python3
"""Smoke trainer on the brief §6.4 reference 3×3 system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.csr_io import dense_to_csr_torch  # noqa: E402
from jmat.loss import jmat_combined_loss  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.normalize import normalize_linear_system  # noqa: E402

REF_J = np.array([[4, -1, 0], [-2, 3, -1], [0, -1, 2]], dtype=np.float64)
REF_R = np.array([1.0, 0.0, 0.0], dtype=np.float64)
REF_DX = np.array([0.3125, 0.25, 0.125], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="JMAT reference overfit smoke test")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    crow, col, val, n = dense_to_csr_torch(REF_J)
    r = torch.as_tensor(REF_R, dtype=torch.float32)
    norm = normalize_linear_system(crow, col, val, n, r)
    assert norm is not None
    z_star = torch.as_tensor(REF_DX / norm.beta, dtype=torch.float32)
    model = JMAT(JMATConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    for step in range(args.steps):
        opt.zero_grad()
        z_hat, dx = model.forward_normalized(norm)
        loss, lz, lr = jmat_combined_loss(
            z_hat, z_star, norm.a_crow, norm.a_col, norm.a_val, n, norm.b_tilde, lam=1.0
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            rel = torch.linalg.vector_norm(dx - torch.as_tensor(REF_DX, dtype=dx.dtype)) / float(
                np.linalg.norm(REF_DX)
            )
            print(f"step {step:4d}  loss={float(loss):.4e}  rel_dx={float(rel):.4e}")


if __name__ == "__main__":
    main()