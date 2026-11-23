"""Convert between SciPy CSR and torch index tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


def scipy_csr_to_torch(m: csr_matrix, *, device: torch.device | None = None, dtype=torch.float32):
    """Return (crow, col, val, n) int64/long indices and float values."""
    m = m.tocsr()
    n = m.shape[0]
    if m.shape[1] != n:
        raise ValueError("JMAT scaffold expects square Jacobians")
    device = device or torch.device("cpu")
    crow = torch.as_tensor(m.indptr.astype(np.int64), device=device)
    col = torch.as_tensor(m.indices.astype(np.int64), device=device)
    val = torch.as_tensor(m.data.astype(np.float64), dtype=dtype, device=device)
    return crow, col, val, n


def dense_to_csr_torch(mat: torch.Tensor | np.ndarray, *, device: torch.device | None = None):
    from scipy.sparse import csr_matrix

    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()
    return scipy_csr_to_torch(csr_matrix(mat), device=device)


def apply_permutation(
    crow: torch.Tensor,
    col: torch.Tensor,
    val: torch.Tensor,
    n: int,
    r: torch.Tensor,
    perm: torch.Tensor,
):
    """Return (J', r') for reordering unknowns/equations with P.

    ``perm[i_new] = i_old``: new index ``i_new`` carries the old label ``i_old``.
    Then ``J'_{i,j} = J_{perm[i], perm[j]}`` and ``r'_i = r_{perm[i]}``.
    A consistent correction satisfies ``Δx' = P Δx`` with
    ``Δx'[i_new] = Δx[perm[i_new]]``.
    """
    from scipy.sparse import csr_matrix

    p = perm.cpu().numpy().astype(np.int64)
    j_dense = csr_matrix(
        (
            val.detach().cpu().numpy(),
            col.detach().cpu().numpy(),
            crow.detach().cpu().numpy(),
        ),
        shape=(n, n),
    ).toarray()
    j_perm = j_dense[np.ix_(p, p)]
    crow_p, col_p, val_p, _ = scipy_csr_to_torch(
        csr_matrix(j_perm), device=r.device, dtype=r.dtype
    )
    r_perm = r[perm]
    return crow_p, col_p, val_p, r_perm