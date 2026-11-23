"""Training objectives for JMAT (brief §6.1)."""

from __future__ import annotations

import torch


def jmat_combined_loss(
    z_hat: torch.Tensor,
    z_star: torch.Tensor,
    a_crow: torch.Tensor,
    a_col: torch.Tensor,
    a_val: torch.Tensor,
    n: int,
    b_tilde: torch.Tensor,
    *,
    lam: float = 1.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scaled-space solution + residual terms; returns (total, L_z, L_res)."""
    from jmat.sparse_ops import csr_matmul

    l_z = torch.sum((z_hat - z_star) ** 2) / (torch.sum(z_star**2) + eps)
    azb = csr_matmul(a_crow, a_col, a_val, n, z_hat) - b_tilde
    l_r = torch.sum(azb**2) / (torch.sum(b_tilde**2) + eps)
    total = l_z + lam * l_r
    return total, l_z, l_r