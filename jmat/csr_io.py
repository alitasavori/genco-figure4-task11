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
    """Return (PJPᵀ, Pr) as new CSR + r; perm maps new index → old index."""
    from scipy.sparse import csr_matrix

    p = perm.cpu().numpy()
    inv = np.empty(n, dtype=np.int64)
    inv[p] = np.arange(n)
    rows, cols, data = [], [], []
    for i in range(n):
        a, b = int(crow[i].item()), int(crow[i + 1].item())
        for k in range(a, b):
            rows.append(int(inv[i]))
            cols.append(int(inv[int(col[k].item())]))
            data.append(float(val[k].item()))
    j_perm = csr_matrix((data, (rows, cols)), shape=(n, n))
    crow_p, col_p, val_p, _ = scipy_csr_to_torch(j_perm, device=r.device, dtype=r.dtype)
    r_perm = r[perm]
    return crow_p, col_p, val_p, r_perm