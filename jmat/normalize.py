"""Global scalar scaling for (J, r) as in JMAT engineering brief §5.1."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NormalizedLinearSystem:
    """Scaled system with Az̃ ≈ b̃ and physical correction Δx = β z."""

    gamma: float
    beta: float
    a_crow: torch.Tensor
    a_col: torch.Tensor
    a_val: torch.Tensor
    n: int
    b_tilde: torch.Tensor  # (n,)
    r_original: torch.Tensor  # (n,) unscaled mismatch


def _gamma_from_csr(crow: torch.Tensor, col: torch.Tensor, val: torch.Tensor, n: int) -> float:
    row_sum = torch.zeros(n, dtype=val.dtype, device=val.device)
    for i in range(n):
        a, b = int(crow[i].item()), int(crow[i + 1].item())
        if b > a:
            row_sum[i] = val[a:b].abs().sum()
    return float(row_sum.max().item())


def normalize_linear_system(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    n: int,
    r: torch.Tensor,
    *,
    row_equilibrate: bool = False,
) -> NormalizedLinearSystem | None:
    """Return scaled CSR(A), b̃, and metadata; None if r == 0 or γ == 0."""
    r = r.reshape(-1).to(dtype=val.dtype, device=val.device)
    if r.numel() != n:
        raise ValueError(f"r length {r.numel()} != n {n}")
    if torch.all(r == 0):
        return None

    if row_equilibrate:
        raise NotImplementedError("row equilibration ablation not implemented in v1 scaffold")

    from jmat.errors import InvalidLinearSystemError
    from jmat.safeguards import is_zero_jacobian

    if is_zero_jacobian(crow, col, val, n):
        raise InvalidLinearSystemError(
            "Zero Jacobian with nonzero mismatch: cannot normalize a degenerate system."
        )

    gamma = _gamma_from_csr(crow, col, val, n)
    if gamma <= 0:
        raise InvalidLinearSystemError(
            "Zero Jacobian with nonzero mismatch: cannot normalize a degenerate system."
        )

    a_val = val / gamma
    b = r / gamma
    beta = float(torch.linalg.vector_norm(b).item())
    if beta <= 0:
        return None
    b_tilde = b / beta
    return NormalizedLinearSystem(
        gamma=gamma,
        beta=beta,
        a_crow=crow,
        a_col=col,
        a_val=a_val,
        n=n,
        b_tilde=b_tilde,
        r_original=r,
    )