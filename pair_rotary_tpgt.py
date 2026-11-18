"""TPGT v15: electrically conditioned pairwise rotary attention.

Replaces WIRE's spectral per-node RoPE with pair-dependent rotations driven by
basis-invariant electrical features from the series-susceptance Laplacian L_B:

  d_ij = (e_i - e_j)^T L_B^† (e_i - e_j)
  e_ij = [log(1 + d_ij/s_d), 1_{(i,j) in E_on}, log(1 + w_ij/s_w)]

  θ_ij^(ℓ,h) = Ω^(ℓ,h) e_ij
  s_ij = q_i^T R(θ_ij) k_j / √D_A

Full softmax + same-component connectivity mask (no self). No dense Zeff RPE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import wire_tpgt as wire

NUM_REAL_TYPES = 3
NUM_TYPE_PAIRS = 9
PAIR_FEATURE_DIM = 3
WEIGHT_EPS = wire.WEIGHT_EPS
LAMBDA_EPS = wire.LAMBDA_EPS
SCALE_EPS = 1e-8


@dataclass
class PairRotaryFactors:
    """Per-topology pairwise electrical features for rotary attention."""

    pair_elec: np.ndarray  # (N, N, 3) float32
    component: np.ndarray  # (N,) int64
    s_d: float
    s_w: float


def _laplacian_pinv(laplacian: np.ndarray, *, lambda_eps: float = LAMBDA_EPS) -> np.ndarray:
    """Moore–Penrose pseudoinverse of a symmetric Laplacian (nullspace zeroed)."""
    n = int(laplacian.shape[0])
    if n == 0:
        return laplacian
    if n == 1:
        return np.zeros((1, 1), dtype=np.float64)
    evals, evecs = np.linalg.eigh(laplacian)
    inv = np.zeros_like(evals)
    for i, lam in enumerate(evals):
        if lam > lambda_eps:
            inv[i] = 1.0 / float(lam)
    return (evecs * inv) @ evecs.T


def _component_effective_distance(
    adjacency_w: np.ndarray,
    buses: np.ndarray,
    *,
    lambda_eps: float = LAMBDA_EPS,
) -> np.ndarray:
    """Return (n_local, n_local) effective electrical distances on one component."""
    n = int(buses.size)
    dist = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return dist
    sub_a = adjacency_w[np.ix_(buses, buses)]
    degree = np.sum(sub_a, axis=1)
    laplacian = np.diag(degree) - sub_a
    pinv = _laplacian_pinv(laplacian, lambda_eps=lambda_eps)
    diag = np.diag(pinv)
    # d_ij = L†_ii + L†_jj - 2 L†_ij
    dist = diag[:, None] + diag[None, :] - 2.0 * pinv
    np.maximum(dist, 0.0, out=dist)
    np.fill_diagonal(dist, 0.0)
    return dist


def raw_pair_electrical(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
    *,
    weight_eps: float = WEIGHT_EPS,
    lambda_eps: float = LAMBDA_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (d_ij, w_ij, component) before log/scale features.

    ``w_ij`` is total in-service series-susceptance weight of direct branches.
    """
    adjacency_w, adjacency_phys, component = wire._series_susceptance_graph(
        bus_count, branches, weight_eps=weight_eps
    )
    d = np.zeros((bus_count, bus_count), dtype=np.float64)
    n_comp = int(component.max()) + 1 if bus_count else 0
    for comp in range(n_comp):
        buses = np.flatnonzero(component == comp)
        local = _component_effective_distance(adjacency_w, buses, lambda_eps=lambda_eps)
        d[np.ix_(buses, buses)] = local
    w = adjacency_w.astype(np.float64, copy=True)
    # Zero cross-component accidental mass (should already be zero).
    same = component[:, None] == component[None, :]
    d = np.where(same, d, 0.0)
    w = np.where(same & adjacency_phys, w, 0.0)
    return d, w, component.astype(np.int64)


def fit_pair_rotary_scales_from_branch_lists(
    branch_lists: list[list[tuple[int, int, float, float]]],
    bus_count: int,
    *,
    weight_eps: float = WEIGHT_EPS,
) -> tuple[float, float]:
    """Median positive d_ij / w_ij over topologies (training-derived fixed scales)."""
    d_vals: list[float] = []
    w_vals: list[float] = []
    for branches in branch_lists:
        d, w, component = raw_pair_electrical(bus_count, branches, weight_eps=weight_eps)
        same = component[:, None] == component[None, :]
        eye = np.eye(bus_count, dtype=bool)
        mask = same & ~eye
        d_pos = d[mask]
        w_pos = w[mask & (w > 0)]
        if d_pos.size:
            d_vals.append(float(np.median(d_pos)))
        if w_pos.size:
            w_vals.append(float(np.median(w_pos)))
    s_d = float(np.median(d_vals)) if d_vals else 1.0
    s_w = float(np.median(w_vals)) if w_vals else 1.0
    return max(s_d, SCALE_EPS), max(s_w, SCALE_EPS)


def build_pair_elec_features(
    d: np.ndarray,
    w: np.ndarray,
    component: np.ndarray,
    *,
    s_d: float,
    s_w: float,
) -> np.ndarray:
    """Stack log-distance, direct-edge indicator, log-weight → (N, N, 3)."""
    n = int(d.shape[0])
    same = component[:, None] == component[None, :]
    eye = np.eye(n, dtype=bool)
    edge = (w > 0) & same & ~eye
    feat = np.zeros((n, n, PAIR_FEATURE_DIM), dtype=np.float32)
    feat[..., 0] = np.log1p(np.maximum(d, 0.0) / float(s_d)).astype(np.float32)
    feat[..., 1] = edge.astype(np.float32)
    feat[..., 2] = np.log1p(np.maximum(w, 0.0) / float(s_w)).astype(np.float32)
    # No self features.
    feat[eye] = 0.0
    # Cross-component zeros.
    feat[~same] = 0.0
    return feat


def compute_pair_rotary_factors(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
    *,
    s_d: float = 1.0,
    s_w: float = 1.0,
    weight_eps: float = WEIGHT_EPS,
    lambda_eps: float = LAMBDA_EPS,
) -> PairRotaryFactors:
    d, w, component = raw_pair_electrical(
        bus_count, branches, weight_eps=weight_eps, lambda_eps=lambda_eps
    )
    pair_elec = build_pair_elec_features(d, w, component, s_d=s_d, s_w=s_w)
    return PairRotaryFactors(
        pair_elec=pair_elec,
        component=component,
        s_d=float(s_d),
        s_w=float(s_w),
    )


def apply_pair_rotary(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply 2D block-diagonal rotations; same convention as ``wire.apply_wire_rotary``."""
    return wire.apply_wire_rotary(x, theta)


def pair_rotary_messages(
    hidden: torch.Tensor,
    pair_elec: torch.Tensor,
    types: torch.Tensor,
    component: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """Full softmax with pair-dependent rotary on K and connectivity mask.

    Parameters
    ----------
    hidden : (B, N, D)
    pair_elec : (N, N, 3) or (B, N, N, 3)
    types, component : (N,)
    w_q,w_k,w_v : (9, H, D_A, D)
    w_o : (H, D, D_A)
    omega : (H, n_blocks, 3)  — θ_ij = Ω e_ij per head
    """
    batch, buses, _hidden_dim = hidden.shape
    heads = int(w_q.shape[1])
    d_a = int(w_q.shape[2])
    device = hidden.device
    dtype = hidden.dtype

    if pair_elec.dim() == 3:
        elec = pair_elec.unsqueeze(0).expand(batch, -1, -1, -1).to(dtype=dtype)
    else:
        elec = pair_elec.to(dtype=dtype)

    # θ: (B, H, N, N, n_blocks); omega: (H, n_blocks, 3)
    theta_all = torch.einsum("bnmc,hoc->bhnmo", elec, omega.to(dtype=dtype))

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
            # Broadcast K over receivers: (B,H,Nr,Ns,D_A); rotate by θ_ij
            k_exp = k.unsqueeze(2).expand(-1, -1, receivers.numel(), -1, -1)
            theta = theta_all[:, :, receivers.view(-1, 1), senders.view(1, -1), :]
            k_rot = apply_pair_rotary(k_exp, theta)
            score = torch.einsum("bhid,bhijd->bhij", q, k_rot) / (float(d_a) ** 0.5)
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
    for c in component.unique().tolist():
        buses_c = torch.nonzero(component == c, as_tuple=False).view(-1)
        if buses_c.numel() <= 1:
            message[:, buses_c] = 0
    return message
