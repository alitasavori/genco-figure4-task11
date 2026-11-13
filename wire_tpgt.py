"""TPGT-WIRE (v14): electrically weighted WIRE + connectivity-masked full attention.

Susceptance-weighted real Laplacian spectral coordinates drive rotary Q/K
embeddings (WIRE). Attention is standard scaled-dot-product softmax, masked to
the same energized component excluding self. No dense complex Zeff RPE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

NUM_REAL_TYPES = 3
NUM_TYPE_PAIRS = 9
DEFAULT_SPECTRAL_DIM = 8
WEIGHT_EPS = 1e-8
LAMBDA_EPS = 1e-8


def spectral_feature_dim(k_s: int = DEFAULT_SPECTRAL_DIM) -> int:
    return int(k_s)


# ---------------------------------------------------------------------------
# Electrically weighted Laplacian + spectral coordinates (NumPy, no grad)
# ---------------------------------------------------------------------------


def _series_susceptance_graph(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
    *,
    weight_eps: float = WEIGHT_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return weighted adjacency A_B, boolean physical adjacency, component labels.

    Edge weight ω_e = |β_e| + ε_w with β_e = -Im(1/(r+jx)) = x/(r²+x²).
    Shunts / charging are excluded (series branches only).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adjacency_w = np.zeros((bus_count, bus_count), dtype=np.float64)
    adjacency_phys = np.zeros((bus_count, bus_count), dtype=bool)
    table = np.asarray(branches, dtype=np.float64).reshape(-1, 4) if branches else np.zeros((0, 4))
    for origin, destination, resistance, reactance in table:
        i, j = int(origin), int(destination)
        if i < 0 or j < 0 or i >= bus_count or j >= bus_count or i == j:
            continue
        denom = float(resistance) ** 2 + float(reactance) ** 2
        if denom <= 0.0:
            continue
        beta = float(reactance) / denom  # -Im(y) = x/(r²+x²)
        omega = abs(beta) + float(weight_eps)
        adjacency_w[i, j] += omega
        adjacency_w[j, i] += omega
        adjacency_phys[i, j] = True
        adjacency_phys[j, i] = True

    graph = csr_matrix(adjacency_phys.astype(np.float64))
    _count, labels = connected_components(graph, directed=False)
    _unique, first_bus, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first_bus.size, dtype=np.int64)
    rank[np.argsort(first_bus, kind="stable")] = np.arange(first_bus.size)
    component = rank[inverse.reshape(-1)].astype(np.int64)
    return adjacency_w, adjacency_phys, component


def _component_spectral_coords(
    adjacency_w: np.ndarray,
    buses: np.ndarray,
    k_s: int,
    *,
    lambda_eps: float = LAMBDA_EPS,
) -> np.ndarray:
    """Return (n_local, k_s) coordinates ρ with ρ_k = u_k / sqrt(max(λ_k, ε))."""
    n = int(buses.size)
    out = np.zeros((n, k_s), dtype=np.float64)
    if n <= 1:
        return out
    sub_a = adjacency_w[np.ix_(buses, buses)]
    degree = np.sum(sub_a, axis=1)
    laplacian = np.diag(degree) - sub_a
    # Symmetric real eigensolve; eigenvalues ascending.
    evals, evecs = np.linalg.eigh(laplacian)
    # Drop near-zero (constant) mode; take next k_s.
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    # Skip λ≈0
    start = 0
    while start < n and evals[start] <= lambda_eps:
        start += 1
    taken = 0
    for k in range(start, n):
        if taken >= k_s:
            break
        lam = float(max(evals[k], lambda_eps))
        # Canonicalize eigenvector sign: first nonzero entry ≥ 0.
        vec = evecs[:, k].astype(np.float64, copy=True)
        for entry in vec:
            if abs(entry) > 1e-12:
                if entry < 0:
                    vec = -vec
                break
        out[:, taken] = vec / np.sqrt(lam)
        taken += 1
    return out


@dataclass
class WireFactors:
    """Per-topology WIRE spectral embeddings."""

    rho: np.ndarray  # (N, K_s) float32
    component: np.ndarray  # (N,) int64
    k_s: int


def compute_wire_factors(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
    *,
    k_s: int = DEFAULT_SPECTRAL_DIM,
    weight_eps: float = WEIGHT_EPS,
    lambda_eps: float = LAMBDA_EPS,
) -> WireFactors:
    """Build padded spectral coordinates without dense Zeff."""
    adjacency_w, _phys, component = _series_susceptance_graph(
        bus_count, branches, weight_eps=weight_eps
    )
    k_s = int(k_s)
    rho = np.zeros((bus_count, k_s), dtype=np.float32)
    n_comp = int(component.max()) + 1 if bus_count else 0
    for comp in range(n_comp):
        buses = np.flatnonzero(component == comp)
        local = _component_spectral_coords(adjacency_w, buses, k_s, lambda_eps=lambda_eps)
        rho[buses] = local.astype(np.float32)
    return WireFactors(rho=rho, component=component.astype(np.int64), k_s=k_s)


# ---------------------------------------------------------------------------
# Rotary application
# ---------------------------------------------------------------------------


def apply_wire_rotary(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply 2D block-diagonal rotations to last dim of ``x``.

    Parameters
    ----------
    x : (..., d) with d even
    theta : (..., d/2) rotation angles
    """
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"WIRE rotary expects even last dim, got {d}")
    n_blocks = d // 2
    if theta.shape[-1] != n_blocks:
        raise ValueError(f"theta last dim {theta.shape[-1]} != d/2={n_blocks}")
    pairs = x.reshape(*x.shape[:-1], n_blocks, 2)
    cos = torch.cos(theta).unsqueeze(-1)
    sin = torch.sin(theta).unsqueeze(-1)
    x0 = pairs[..., 0:1]
    x1 = pairs[..., 1:2]
    rotated = torch.cat((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return rotated.reshape(*x.shape)


def wire_messages(
    hidden: torch.Tensor,
    rho: torch.Tensor,
    types: torch.Tensor,
    component: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Full softmax attention with WIRE rotations and connectivity mask.

    Parameters
    ----------
    hidden : (B, N, D)
    rho : (N, K) or (B, N, K)
    types, component : (N,)
    w_q,w_k,w_v : (9, H, D_A, D)
    w_o : (H, D, D_A)
    omega : (H, n_blocks, K) shared across type pairs within each head
    """
    batch, buses, _hidden_dim = hidden.shape
    heads = int(w_q.shape[1])
    d_a = int(w_q.shape[2])
    device = hidden.device
    dtype = hidden.dtype

    if rho.dim() == 2:
        rho_b = rho.unsqueeze(0).expand(batch, -1, -1).to(dtype=dtype)
    else:
        rho_b = rho.to(dtype=dtype)

    # θ_i^{(h)} = Ω^{(h)} ρ_i → (B, H, N, n_blocks); omega: (H, n_blocks, K)
    theta = torch.einsum("bnk,hok->bhno", rho_b, omega.to(dtype=dtype))
    scores = hidden.new_full((batch, heads, buses, buses), float("-inf"))

    same = component.view(1, buses) == component.view(buses, 1)
    not_self = ~torch.eye(buses, dtype=torch.bool, device=device)
    allow_base = same & not_self

    for recv_t in range(NUM_REAL_TYPES):
        receivers = torch.nonzero(types == recv_t, as_tuple=False).view(-1)
        if receivers.numel() == 0:
            continue
        for send_t in range(NUM_REAL_TYPES):
            pair_id = recv_t * NUM_REAL_TYPES + send_t
            senders = torch.nonzero(types == send_t, as_tuple=False).view(-1)
            if senders.numel() == 0:
                continue
            allow = allow_base[receivers.view(-1, 1), senders.view(1, -1)]
            if not bool(allow.any()):
                continue
            q = torch.einsum("bnd,had->bhna", hidden[:, receivers], w_q[pair_id].to(dtype=dtype))
            k = torch.einsum("bnd,had->bhna", hidden[:, senders], w_k[pair_id].to(dtype=dtype))
            q_rot = apply_wire_rotary(q, theta[:, :, receivers, :])
            k_rot = apply_wire_rotary(k, theta[:, :, senders, :])
            score = torch.einsum("bhid,bhjd->bhij", q_rot, k_rot) / (float(d_a) ** 0.5)
            score = score.masked_fill(~allow.unsqueeze(0).unsqueeze(0), float("-inf"))
            ri = receivers.view(-1, 1)
            sj = senders.view(1, -1)
            scores[:, :, ri, sj] = torch.where(
                allow.unsqueeze(0).unsqueeze(0),
                score,
                scores[:, :, ri, sj],
            )

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    finite_row = torch.isfinite(scores).any(dim=-1, keepdim=True)
    weights = torch.where(finite_row, weights, torch.zeros_like(weights))

    mixed = torch.zeros(batch, heads, buses, d_a, device=device, dtype=dtype)
    for recv_t in range(NUM_REAL_TYPES):
        receivers = torch.nonzero(types == recv_t, as_tuple=False).view(-1)
        if receivers.numel() == 0:
            continue
        for send_t in range(NUM_REAL_TYPES):
            pair_id = recv_t * NUM_REAL_TYPES + send_t
            senders = torch.nonzero(types == send_t, as_tuple=False).view(-1)
            if senders.numel() == 0:
                continue
            allow = allow_base[receivers.view(-1, 1), senders.view(1, -1)]
            if not bool(allow.any()):
                continue
            v = torch.einsum("bnd,had->bhna", hidden[:, senders], w_v[pair_id].to(dtype=dtype))
            block_w = weights[:, :, receivers.view(-1, 1), senders.view(1, -1)]
            block_w = block_w.masked_fill(~allow.unsqueeze(0).unsqueeze(0), 0.0)
            gathered = torch.einsum("bhij,bhjd->bhid", block_w, v)
            mixed.index_add_(2, receivers, gathered)

    message = torch.einsum("bhna,hda->bnd", mixed, w_o.to(dtype=dtype))
    # Singleton islands: zero message.
    for c in component.unique().tolist():
        buses_c = torch.nonzero(component == c, as_tuple=False).view(-1)
        if buses_c.numel() <= 1:
            message[:, buses_c] = 0
    return message
