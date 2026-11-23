"""Mathematical safeguards for (J, r) preprocessing (Phase 2 Task 1)."""

from __future__ import annotations

import torch


def is_zero_mismatch(r: torch.Tensor, *, atol: float = 0.0) -> bool:
    r = r.reshape(-1)
    if atol <= 0:
        return bool(torch.all(r == 0).item())
    return bool(torch.all(r.abs() <= atol).item())


def is_zero_jacobian(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    n: int,
    *,
    atol: float = 0.0,
) -> bool:
    """True when every stored entry of J is (numerically) zero."""
    if n == 0:
        return True
    if val.numel() == 0:
        return True
    if atol <= 0:
        return bool(torch.all(val == 0).item())
    return bool(torch.all(val.abs() <= atol).item())


def validate_linear_system(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    n: int,
    r: torch.Tensor,
) -> None:
    """Raise when J is zero but r is not (undefined correction)."""
    from jmat.errors import InvalidLinearSystemError

    if is_zero_mismatch(r):
        return
    if is_zero_jacobian(crow, col, val, n):
        raise InvalidLinearSystemError(
            "Zero Jacobian with nonzero mismatch: J Δx = r has no meaningful solution."
        )