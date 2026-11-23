#!/usr/bin/env python3
"""Task 6 — evaluate pretrained JMAT on SuiteSparse ACTIVSg Jacobians (no fine-tuning)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from ssgetpy import ssget
except ImportError:
    ssget = None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrices", nargs="+", default=["ACTIVSg2000", "ACTIVSg10K"])
    p.add_argument("--n-rhs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "jmat" / "suitesparse_eval.csv")
    args = p.parse_args()
    if ssget is None:
        raise SystemExit("Install ssgetpy: pip install ssgetpy")
    rng = np.random.default_rng(args.seed)
    # Defer to baseline harness: build RHS r=Jx*, run evaluate_jmat_baselines logic inline.
    from scipy.io import mmread
    from scipy.sparse import csr_matrix

    from jmat.csr_io import scipy_csr_to_torch
    from jmat.metrics import relative_errors, summarize_errors
    from jmat.model import JMAT, JMATConfig
    import torch

    device = torch.device("cpu")
    model = JMAT(JMATConfig())
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    ex, er = [], []
    for name in args.matrices:
        mat = ssget(name, download=True)
        path = Path(mat.primary)
        j = csr_matrix(mmread(str(path)))
        j = j.tocsr()
        for _ in range(args.n_rhs):
            x = rng.standard_normal(j.shape[0])
            r = (j @ x).astype(np.float64)
            dx_ref = x
            crow, col, val, n = scipy_csr_to_torch(j, dtype=torch.float32, device=device)
            rt = torch.tensor(r, dtype=torch.float32, device=device)
            with torch.no_grad():
                dx_hat = model.forward_csr(crow, col, val, n, rt).cpu().numpy()
            erx, err = relative_errors(dx_hat, dx_ref, j, r)
            ex.append(erx)
            er.append(err)
    print(summarize_errors(ex, er))


if __name__ == "__main__":
    main()