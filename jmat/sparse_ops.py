"""Sparse matrix–dense feature ops for normalized Jacobians."""

from __future__ import annotations

import torch


def csr_matmul(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    n: int,
    dense: torch.Tensor,
) -> torch.Tensor:
    """Sparse (n,n) @ dense (n,d) without building a dense n×n matrix."""
    if dense.dim() == 1:
        dense = dense.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    d = dense.shape[-1]
    out = torch.zeros(n, d, dtype=dense.dtype, device=dense.device)
    for i in range(n):
        a, b = int(crow[i].item()), int(crow[i + 1].item())
        if b <= a:
            continue
        cols = col[a:b]
        vals = val[a:b]
        out[i] = (vals.unsqueeze(-1) * dense[cols]).sum(dim=0)
    if squeeze:
        return out.squeeze(-1)
    return out


def csr_transpose_matmul(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    n: int,
    dense: torch.Tensor,
) -> torch.Tensor:
    """Aᵀ @ dense using CSR storage of A."""
    if dense.dim() == 1:
        dense = dense.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    d = dense.shape[-1]
    out = torch.zeros(n, d, dtype=dense.dtype, device=dense.device)
    for i in range(n):
        a, b = int(crow[i].item()), int(crow[i + 1].item())
        if b <= a:
            continue
        cols = col[a:b]
        vals = val[a:b]
        for k, j in enumerate(cols.tolist()):
            out[j] = out[j] + vals[k] * dense[i]
    if squeeze:
        return out.squeeze(-1)
    return out


def index_features(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    n: int,
    b_tilde: torch.Tensor,
) -> torch.Tensor:
    """Per-index encoder input q_i = [b̃_i, A_ii, Σ|A_ij|, Σ|A_ji|] (n, 4)."""
    device = val.device
    dtype = val.dtype
    diag = torch.zeros(n, dtype=dtype, device=device)
    row_abs = torch.zeros(n, dtype=dtype, device=device)
    col_abs = torch.zeros(n, dtype=dtype, device=device)
    for i in range(n):
        a, b = int(crow[i].item()), int(crow[i + 1].item())
        if b <= a:
            continue
        c = col[a:b]
        v = val[a:b]
        av = v.abs()
        row_abs[i] = av.sum()
        for k, j in enumerate(c.tolist()):
            if j == i:
                diag[i] = v[k]
            col_abs[j] = col_abs[j] + av[k]
    return torch.stack([b_tilde, diag, row_abs, col_abs], dim=-1)