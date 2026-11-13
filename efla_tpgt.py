"""TPGT-EFLA (v13): electrically factorized linear attention helpers.

Landmark-based complex-Laplacian node embeddings plus factorized type-pair
attention with separable receiver-dependent values. No dense N×N Zeff or
attention matrices on the training/inference path.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

NUM_REAL_TYPES = 3
NUM_TYPE_PAIRS = 9
DEFAULT_LANDMARKS = 16
DEFAULT_NODE_KERNEL = 32
DEFAULT_ELEC_KERNEL = 4
DEFAULT_VALUE_RANK = 4
FEATURE_BOUND_BETA = 4.0
ATTENTION_EPS = 1e-8
GAMMA_INIT = 0.01


def elec_feature_dim(num_landmarks: int = DEFAULT_LANDMARKS) -> int:
    """Width of padded node electrical embedding a_i = [Re C, Im C, Re d, Im d]."""
    return 2 * int(num_landmarks) + 2


def combined_kernel_dim(
    node_kernel: int = DEFAULT_NODE_KERNEL,
    elec_kernel: int = DEFAULT_ELEC_KERNEL,
) -> int:
    return int(node_kernel) * int(elec_kernel)


def bound_preactivation(z: torch.Tensor, beta: float = FEATURE_BOUND_BETA) -> torch.Tensor:
    return beta * torch.tanh(z / beta)


def positive_feature_map(z: torch.Tensor, beta: float = FEATURE_BOUND_BETA) -> torch.Tensor:
    return torch.exp(bound_preactivation(z, beta))


def kronecker_features(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Outer product flattened on the last axis: (..., m) ⊗ (..., r) → (..., m*r)."""
    return (left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(*left.shape[:-1], left.shape[-1] * right.shape[-1])


# ---------------------------------------------------------------------------
# Electrical preprocessor (NumPy, no grad)
# ---------------------------------------------------------------------------


def _branch_admittance_graph(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complex W, boolean adjacency, and connected-component labels."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    admittance = np.zeros((bus_count, bus_count), dtype=np.complex128)
    adjacency = np.zeros((bus_count, bus_count), dtype=bool)
    table = np.asarray(branches, dtype=np.float64).reshape(-1, 4) if branches else np.zeros((0, 4))
    for origin, destination, resistance, reactance in table:
        impedance = complex(float(resistance), float(reactance))
        if abs(impedance) <= 0.0:
            continue
        branch_y = 1.0 / impedance
        i = int(origin)
        j = int(destination)
        if i < 0 or j < 0 or i >= bus_count or j >= bus_count or i == j:
            continue
        admittance[i, j] += branch_y
        admittance[j, i] += branch_y
        adjacency[i, j] = True
        adjacency[j, i] = True

    graph = csr_matrix(adjacency.astype(np.float64))
    _count, labels = connected_components(graph, directed=False)
    _unique, first_bus, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first_bus.size, dtype=np.int64)
    rank[np.argsort(first_bus, kind="stable")] = np.arange(first_bus.size)
    component = rank[inverse.reshape(-1)].astype(np.int64)
    return admittance, adjacency, component


def _bfs_order(adjacency: np.ndarray, buses: np.ndarray, root_local: int) -> np.ndarray:
    """BFS visit order over ``buses`` starting at local index ``root_local``."""
    n = int(buses.size)
    sub = adjacency[np.ix_(buses, buses)]
    order: list[int] = []
    seen = np.zeros(n, dtype=bool)
    queue: deque[int] = deque([int(root_local)])
    seen[root_local] = True
    while queue:
        u = queue.popleft()
        order.append(u)
        neighbors = np.flatnonzero(sub[u])
        # Stable: visit lower electrical-strength neighbors later via caller ranking.
        for v in neighbors.tolist():
            if not seen[v]:
                seen[v] = True
                queue.append(int(v))
    for i in range(n):
        if not seen[i]:
            order.append(i)
    return np.asarray(order, dtype=np.int64)


def select_landmarks(
    adjacency: np.ndarray,
    admittance: np.ndarray,
    buses: np.ndarray,
    num_landmarks: int,
) -> np.ndarray:
    """Permutation-insensitive landmark ranking within one component.

    Rank by electrical degree (sum |y|), then graph degree, then BFS order from
    the strongest bus. Returns local indices into ``buses``.
    """
    n = int(buses.size)
    if n <= 1 or num_landmarks <= 0:
        return np.zeros(0, dtype=np.int64)
    sub_y = admittance[np.ix_(buses, buses)]
    elec_deg = np.asarray(np.sum(np.abs(sub_y), axis=1)).reshape(-1)
    graph_deg = np.asarray(adjacency[np.ix_(buses, buses)].sum(axis=1)).reshape(-1).astype(np.float64)
    # Strongest bus: max electrical degree, then graph degree, then first in buses order.
    strength = list(zip((-elec_deg).tolist(), (-graph_deg).tolist(), list(range(n))))
    root = min(range(n), key=lambda i: strength[i])
    bfs = _bfs_order(adjacency, buses, root)
    bfs_rank = np.empty(n, dtype=np.int64)
    bfs_rank[bfs] = np.arange(n)
    keys = sorted(
        range(n),
        key=lambda i: (-elec_deg[i], -graph_deg[i], int(bfs_rank[i]), i),
    )
    r = min(int(num_landmarks), n - 1)
    return np.asarray(keys[:r], dtype=np.int64)


def _solve_zero_mean_columns(
    laplacian: np.ndarray,
    rhs_columns: np.ndarray,
) -> np.ndarray:
    """Solve L c = b with 1^T c = 0 by grounding the last bus, then zero-mean shift."""
    n = int(laplacian.shape[0])
    r = int(rhs_columns.shape[1])
    if n <= 1 or r == 0:
        return np.zeros((n, r), dtype=np.complex128)
    # Ground last equation.
    lg = laplacian[: n - 1, : n - 1].astype(np.complex128, copy=True)
    bg = rhs_columns[: n - 1].astype(np.complex128, copy=True)
    try:
        cg = np.linalg.solve(lg, bg)
    except np.linalg.LinAlgError:
        cg = np.linalg.pinv(lg) @ bg
    c = np.zeros((n, r), dtype=np.complex128)
    c[: n - 1] = cg
    c -= np.mean(c, axis=0, keepdims=True)
    return c


def _component_landmark_factors(
    admittance: np.ndarray,
    adjacency: np.ndarray,
    buses: np.ndarray,
    num_landmarks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (C padded local, landmarks local, B_L) for one component."""
    n = int(buses.size)
    r_max = int(num_landmarks)
    if n <= 1:
        return (
            np.zeros((n, r_max), dtype=np.complex128),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 0), dtype=np.complex128),
        )
    sub_y = admittance[np.ix_(buses, buses)]
    laplacian = np.diag(np.sum(sub_y, axis=1)) - sub_y
    landmarks = select_landmarks(adjacency, admittance, buses, num_landmarks)
    r_c = int(landmarks.size)
    if r_c == 0:
        return (
            np.zeros((n, r_max), dtype=np.complex128),
            landmarks,
            np.zeros((0, 0), dtype=np.complex128),
        )
    rhs = np.zeros((n, r_c), dtype=np.complex128)
    for a, loc in enumerate(landmarks.tolist()):
        rhs[:, a] = -1.0 / n
        rhs[loc, a] += 1.0
    c_full = _solve_zero_mean_columns(laplacian, rhs)
    w_l = c_full[landmarks, :]
    # Complex-symmetric: ordinary transpose; symmetrize for numerical noise.
    try:
        b_l = np.linalg.pinv(w_l)
    except np.linalg.LinAlgError:
        b_l = np.zeros_like(w_l)
    b_l = 0.5 * (b_l + b_l.T)
    c_pad = np.zeros((n, r_max), dtype=np.complex128)
    c_pad[:, :r_c] = c_full
    return c_pad, landmarks, b_l


def reconstruct_zeff_pair(
    c_i: np.ndarray,
    c_j: np.ndarray,
    b_l: np.ndarray,
) -> complex:
    """Eq. (6): Zhat_ij = (C_i - C_j) B_L (C_i - C_j)^T using active landmark width."""
    r = int(b_l.shape[0])
    if r == 0:
        return 0.0 + 0.0j
    diff = (c_i[:r] - c_j[:r]).reshape(1, -1)
    return complex((diff @ b_l @ diff.T)[0, 0])


@dataclass
class ElectricalFactors:
    """Per-topology electrical node features and scales."""

    elec_a: np.ndarray  # (N, elec_dim) float32
    component: np.ndarray  # (N,) int64
    landmarks_by_component: list[np.ndarray]
    c_by_component: list[np.ndarray]
    b_l_by_component: list[np.ndarray]
    buses_by_component: list[np.ndarray]
    s_r: float
    s_x: float
    s_c: float


def compute_electrical_factors(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
    *,
    num_landmarks: int = DEFAULT_LANDMARKS,
    s_r: float | None = None,
    s_x: float | None = None,
    s_c: float | None = None,
) -> ElectricalFactors:
    """Build padded a_i embeddings without materializing N×N Zeff."""
    admittance, adjacency, component = _branch_admittance_graph(bus_count, branches)
    n_comp = int(component.max()) + 1 if bus_count else 0
    r = int(num_landmarks)
    elec_dim = elec_feature_dim(r)
    c_global = np.zeros((bus_count, r), dtype=np.complex128)
    d_diag = np.zeros(bus_count, dtype=np.complex128)
    landmarks_by_component: list[np.ndarray] = []
    c_by_component: list[np.ndarray] = []
    b_l_by_component: list[np.ndarray] = []
    buses_by_component: list[np.ndarray] = []

    max_c = 0.0
    max_re = 0.0
    max_im = 0.0
    for comp in range(n_comp):
        buses = np.flatnonzero(component == comp)
        buses_by_component.append(buses)
        c_pad, landmarks, b_l = _component_landmark_factors(
            admittance, adjacency, buses, num_landmarks
        )
        landmarks_by_component.append(landmarks)
        c_by_component.append(c_pad)
        b_l_by_component.append(b_l)
        if buses.size == 0:
            continue
        c_global[buses] = c_pad
        r_c = int(landmarks.size)
        if r_c == 0:
            continue
        # d_i = C_i B_L C_i^T using active columns.
        c_act = c_pad[:, :r_c]
        # (n, r) @ (r, r) @ (r, n) diagonals via row-wise quadratic forms.
        cb = c_act @ b_l
        d_local = np.sum(cb * c_act, axis=1)
        d_diag[buses] = d_local
        max_c = max(max_c, float(np.max(np.abs(c_act))))
        # Approximate Zeff peaks from landmark pairs + a few random pairs for scale fit.
        for a in range(r_c):
            for b in range(a + 1, r_c):
                i = int(landmarks[a])
                j = int(landmarks[b])
                z = reconstruct_zeff_pair(c_pad[i], c_pad[j], b_l)
                max_re = max(max_re, abs(z.real))
                max_im = max(max_im, abs(z.imag))
        if buses.size > 2:
            rng = np.random.default_rng(comp + 17)
            sample = rng.choice(buses.size, size=min(32, buses.size), replace=False)
            for ii in range(len(sample)):
                for jj in range(ii + 1, len(sample)):
                    z = reconstruct_zeff_pair(c_pad[sample[ii]], c_pad[sample[jj]], b_l)
                    max_re = max(max_re, abs(z.real))
                    max_im = max(max_im, abs(z.imag))

    use_s_r = float(s_r) if s_r is not None else max(max_re, 1.0)
    use_s_x = float(s_x) if s_x is not None else max(max_im, 1.0)
    use_s_c = float(s_c) if s_c is not None else max(max_c, 1e-30)

    elec_a = np.zeros((bus_count, elec_dim), dtype=np.float32)
    if bus_count:
        elec_a[:, :r] = (c_global.real / use_s_c).astype(np.float32)
        elec_a[:, r : 2 * r] = (c_global.imag / use_s_c).astype(np.float32)
        elec_a[:, 2 * r] = (d_diag.real / use_s_r).astype(np.float32)
        elec_a[:, 2 * r + 1] = (d_diag.imag / use_s_x).astype(np.float32)

    return ElectricalFactors(
        elec_a=elec_a,
        component=component.astype(np.int64),
        landmarks_by_component=landmarks_by_component,
        c_by_component=c_by_component,
        b_l_by_component=b_l_by_component,
        buses_by_component=buses_by_component,
        s_r=use_s_r,
        s_x=use_s_x,
        s_c=use_s_c,
    )


def fit_electrical_scales_from_branch_lists(
    branch_lists: list[list[tuple[int, int, float, float]]],
    bus_count: int,
    *,
    num_landmarks: int = DEFAULT_LANDMARKS,
) -> tuple[float, float, float]:
    """Train-fit s_R, s_X, s_C across topologies."""
    max_re = 0.0
    max_im = 0.0
    max_c = 0.0
    for branches in branch_lists:
        if not branches:
            continue
        factors = compute_electrical_factors(
            bus_count, branches, num_landmarks=num_landmarks
        )
        max_re = max(max_re, factors.s_r)
        max_im = max(max_im, factors.s_x)
        max_c = max(max_c, factors.s_c)
    return max(max_re, 1.0), max(max_im, 1.0), max(max_c, 1e-30)


def approximate_zeff_matrix(
    factors: ElectricalFactors,
    bus_count: int | None = None,
) -> np.ndarray:
    """Diagnostic full-matrix reconstruction (tests only; not used in training)."""
    n = int(factors.elec_a.shape[0] if bus_count is None else bus_count)
    z = np.zeros((n, n), dtype=np.complex128)
    for buses, c_pad, b_l in zip(
        factors.buses_by_component,
        factors.c_by_component,
        factors.b_l_by_component,
    ):
        if buses.size <= 1 or b_l.size == 0:
            continue
        for ii, bi in enumerate(buses.tolist()):
            for jj, bj in enumerate(buses.tolist()):
                z[bi, bj] = reconstruct_zeff_pair(c_pad[ii], c_pad[jj], b_l)
    return z


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class PositiveFeatureMap(nn.Module):
    """Learned bounded exponential feature map: exp(β tanh((A x + b)/β))."""

    def __init__(self, in_dim: int, out_dim: int, num_pairs: int = NUM_TYPE_PAIRS, num_heads: int = 1) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_pairs, num_heads, out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(num_pairs, num_heads, out_dim))
        nn.init.xavier_uniform_(self.weight.view(-1, in_dim))

    def forward(self, pair_id: int, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, N, in) or (B, N, in) — we always pass (B, H, N, in)
        w = self.weight[pair_id]
        b = self.bias[pair_id]
        projected = torch.einsum("bhnd,hod->bhno", x, w) + b.view(1, w.shape[0], 1, -1)
        return positive_feature_map(projected)


class PairValueFactor(nn.Module):
    """Rank-R separable receiver/sender value correction."""

    def __init__(
        self,
        hidden_dim: int,
        elec_dim: int,
        value_dim: int,
        rank: int = DEFAULT_VALUE_RANK,
        num_pairs: int = NUM_TYPE_PAIRS,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        in_dim = hidden_dim + elec_dim
        self.rank = int(rank)
        self.m_mu = nn.Parameter(torch.empty(num_pairs, num_heads, rank, in_dim))
        self.m_nu = nn.Parameter(torch.empty(num_pairs, num_heads, rank, in_dim))
        self.c_v = nn.Parameter(torch.empty(num_pairs, num_heads, rank, value_dim))
        self.gamma = nn.Parameter(torch.full((num_pairs, num_heads), GAMMA_INIT))
        for param in (self.m_mu, self.m_nu, self.c_v):
            nn.init.xavier_uniform_(param.view(-1, param.shape[-1]))

    def scalars(self, pair_id: int, hidden: torch.Tensor, elec: torch.Tensor, which: str) -> torch.Tensor:
        # hidden/elec: (B, N, D) → (B, H, N, R)
        joint = torch.cat((hidden, elec), dim=-1)
        weight = self.m_mu[pair_id] if which == "mu" else self.m_nu[pair_id]
        heads = weight.shape[0]
        projected = torch.einsum("bnd,hrd->bhnr", joint, weight)
        return torch.tanh(projected)


def _amp_off(device: torch.device):
    """Disable CUDA autocast so EFLA stays float32 (avoids Half/Float index_put mismatches)."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def dense_efla_messages(
    hidden: torch.Tensor,
    elec_a: torch.Tensor,
    types: torch.Tensor,
    component: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_k: torch.Tensor,
    b_k: torch.Tensor,
    bq_elec: torch.Tensor,
    cq_elec: torch.Tensor,
    bk_elec: torch.Tensor,
    ck_elec: torch.Tensor,
    m_mu: torch.Tensor,
    m_nu: torch.Tensor,
    c_v: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = ATTENTION_EPS,
) -> torch.Tensor:
    """Explicit dense implementation of Eqs. (12)+(16) for parity tests."""
    out_dtype = hidden.dtype
    batch, buses, hidden_dim = hidden.shape
    if elec_a.dim() == 2:
        elec_a = elec_a.unsqueeze(0).expand(batch, -1, -1)
    heads = int(w_q.shape[1])
    d_a = int(w_q.shape[2])
    device = hidden.device
    with _amp_off(device):
        hidden = hidden.to(dtype=torch.float32)
        elec_a = elec_a.to(dtype=torch.float32)
        w_q, w_k, w_v, w_o = (t.to(dtype=torch.float32) for t in (w_q, w_k, w_v, w_o))
        a_q, b_q, a_k, b_k = (t.to(dtype=torch.float32) for t in (a_q, b_q, a_k, b_k))
        bq_elec, cq_elec, bk_elec, ck_elec = (
            t.to(dtype=torch.float32) for t in (bq_elec, cq_elec, bk_elec, ck_elec)
        )
        m_mu, m_nu, c_v, gamma = (t.to(dtype=torch.float32) for t in (m_mu, m_nu, c_v, gamma))
        dtype = torch.float32
        # w_o: (H, hidden_dim, d_a) → message in hidden space
        message = torch.zeros(batch, buses, hidden_dim, device=device, dtype=dtype)
        message = _dense_efla_messages_core(
            hidden,
            elec_a,
            types,
            component,
            message,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            w_o=w_o,
            a_q=a_q,
            b_q=b_q,
            a_k=a_k,
            b_k=b_k,
            bq_elec=bq_elec,
            cq_elec=cq_elec,
            bk_elec=bk_elec,
            ck_elec=ck_elec,
            m_mu=m_mu,
            m_nu=m_nu,
            c_v=c_v,
            gamma=gamma,
            eps=eps,
            batch=batch,
            heads=heads,
            d_a=d_a,
            device=device,
            dtype=dtype,
        )
    return message.to(dtype=out_dtype)


def _dense_efla_messages_core(
    hidden: torch.Tensor,
    elec_a: torch.Tensor,
    types: torch.Tensor,
    component: torch.Tensor,
    message: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_k: torch.Tensor,
    b_k: torch.Tensor,
    bq_elec: torch.Tensor,
    cq_elec: torch.Tensor,
    bk_elec: torch.Tensor,
    ck_elec: torch.Tensor,
    m_mu: torch.Tensor,
    m_nu: torch.Tensor,
    c_v: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    batch: int,
    heads: int,
    d_a: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    buses = int(hidden.shape[1])
    same = component.view(1, buses) == component.view(buses, 1)
    for recv_t in range(NUM_REAL_TYPES):
        receivers = torch.nonzero(types == recv_t, as_tuple=False).view(-1)
        if receivers.numel() == 0:
            continue
        numer = torch.zeros(batch, heads, receivers.numel(), d_a, device=device, dtype=dtype)
        denom = torch.zeros(batch, heads, receivers.numel(), device=device, dtype=dtype)
        for send_t in range(NUM_REAL_TYPES):
            pair_id = recv_t * NUM_REAL_TYPES + send_t
            senders = torch.nonzero(types == send_t, as_tuple=False).view(-1)
            if senders.numel() == 0:
                continue
            allow = same[receivers.view(-1, 1), senders.view(1, -1)]
            # No self: drop i==j
            if recv_t == send_t:
                for li, ri in enumerate(receivers.tolist()):
                    sj = torch.nonzero(senders == ri, as_tuple=False).view(-1)
                    if sj.numel():
                        allow[li, sj] = False
            if not bool(allow.any()):
                continue
            q = torch.einsum("bnd,had->bhna", hidden[:, receivers], w_q[pair_id])
            k = torch.einsum("bnd,had->bhna", hidden[:, senders], w_k[pair_id])
            v = torch.einsum("bnd,had->bhna", hidden[:, senders], w_v[pair_id])
            f = positive_feature_map(
                torch.einsum("bhnd,hod->bhno", q, a_q[pair_id]) + b_q[pair_id].view(1, heads, 1, -1)
            )
            g = positive_feature_map(
                torch.einsum("bhnd,hod->bhno", k, a_k[pair_id]) + b_k[pair_id].view(1, heads, 1, -1)
            )
            u = positive_feature_map(
                torch.einsum("bnd,hod->bhno", elec_a[:, receivers], bq_elec[pair_id])
                + cq_elec[pair_id].view(1, heads, 1, -1)
            )
            z = positive_feature_map(
                torch.einsum("bnd,hod->bhno", elec_a[:, senders], bk_elec[pair_id])
                + ck_elec[pair_id].view(1, heads, 1, -1)
            )
            F = kronecker_features(f, u)
            G = kronecker_features(g, z)
            scores = torch.einsum("bhir,bhjr->bhij", F, G)
            scores = scores.masked_fill(~allow.unsqueeze(0).unsqueeze(0), 0.0)

            joint_r = torch.cat((hidden[:, receivers], elec_a[:, receivers]), dim=-1)
            joint_s = torch.cat((hidden[:, senders], elec_a[:, senders]), dim=-1)
            mu = torch.tanh(torch.einsum("bnd,hrd->bhnr", joint_r, m_mu[pair_id]))
            nu = torch.tanh(torch.einsum("bnd,hrd->bhnr", joint_s, m_nu[pair_id]))
            # value_ij = v_j + gamma * (mu_i ⊙ nu_j)^T C_V
            # (mu ⊙ nu): (B,H,Nr,Ns,R)
            mu_exp = mu.unsqueeze(-2)
            nu_exp = nu.unsqueeze(-3)
            mix = mu_exp * nu_exp
            delta = torch.einsum("bhijr,hrd->bhijd", mix, c_v[pair_id]) * gamma[pair_id].view(
                1, heads, 1, 1, 1
            )
            values = v.unsqueeze(-3) + delta
            weighted = scores.unsqueeze(-1) * values
            numer = numer + weighted.sum(dim=-2)
            denom = denom + scores.sum(dim=-1)

        safe = denom.clamp_min(eps)
        head_msg = numer / safe.unsqueeze(-1)
        # isolated receivers (denom ~ 0 from masking): zero
        head_msg = torch.where(denom.unsqueeze(-1) > eps, head_msg, torch.zeros_like(head_msg))
        message[:, receivers] = torch.einsum("bhna,hda->bnd", head_msg, w_o)
    return message


def factorized_efla_messages(
    hidden: torch.Tensor,
    elec_a: torch.Tensor,
    types: torch.Tensor,
    component: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_k: torch.Tensor,
    b_k: torch.Tensor,
    bq_elec: torch.Tensor,
    cq_elec: torch.Tensor,
    bk_elec: torch.Tensor,
    ck_elec: torch.Tensor,
    m_mu: torch.Tensor,
    m_nu: torch.Tensor,
    c_v: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = ATTENTION_EPS,
) -> torch.Tensor:
    """Linear-complexity aggregation of Eqs. (17)–(22)."""
    out_dtype = hidden.dtype
    batch, buses, hidden_dim = hidden.shape
    if elec_a.dim() == 2:
        elec_a = elec_a.unsqueeze(0).expand(batch, -1, -1)
    heads = int(w_q.shape[1])
    d_a = int(w_q.shape[2])
    device = hidden.device

    if buses == 0:
        return torch.zeros(batch, 0, hidden_dim, device=device, dtype=out_dtype)

    with _amp_off(device):
        hidden = hidden.to(dtype=torch.float32)
        elec_a = elec_a.to(dtype=torch.float32)
        w_q, w_k, w_v, w_o = (t.to(dtype=torch.float32) for t in (w_q, w_k, w_v, w_o))
        a_q, b_q, a_k, b_k = (t.to(dtype=torch.float32) for t in (a_q, b_q, a_k, b_k))
        bq_elec, cq_elec, bk_elec, ck_elec = (
            t.to(dtype=torch.float32) for t in (bq_elec, cq_elec, bk_elec, ck_elec)
        )
        m_mu, m_nu, c_v, gamma = (t.to(dtype=torch.float32) for t in (m_mu, m_nu, c_v, gamma))
        dtype = torch.float32
        message = torch.zeros(batch, buses, hidden_dim, device=device, dtype=dtype)

        components = component.detach().cpu().numpy().astype(np.int64)
        n_comp = int(components.max()) + 1 if buses else 0

        for comp in range(n_comp):
            buses_c = torch.as_tensor(
                np.flatnonzero(components == comp), device=device, dtype=torch.long
            )
            if buses_c.numel() <= 1:
                continue
            types_c = types[buses_c]
            hidden_c = hidden[:, buses_c]
            elec_c = elec_a[:, buses_c]

            type_locals = [
                torch.nonzero(types_c == t, as_tuple=False).view(-1) for t in range(NUM_REAL_TYPES)
            ]

            for recv_t in range(NUM_REAL_TYPES):
                recv_local = type_locals[recv_t]
                if recv_local.numel() == 0:
                    continue
                numer = torch.zeros(batch, heads, recv_local.numel(), d_a, device=device, dtype=dtype)
                denom = torch.zeros(batch, heads, recv_local.numel(), device=device, dtype=dtype)

                h_recv = hidden_c[:, recv_local]
                e_recv = elec_c[:, recv_local]

                for send_t in range(NUM_REAL_TYPES):
                    pair_id = recv_t * NUM_REAL_TYPES + send_t
                    send_local = type_locals[send_t]
                    if send_local.numel() == 0:
                        continue
                    h_send = hidden_c[:, send_local]
                    e_send = elec_c[:, send_local]

                    q = torch.einsum("bnd,had->bhna", h_recv, w_q[pair_id])
                    k = torch.einsum("bnd,had->bhna", h_send, w_k[pair_id])
                    v = torch.einsum("bnd,had->bhna", h_send, w_v[pair_id])

                    f = positive_feature_map(
                        torch.einsum("bhnd,hod->bhno", q, a_q[pair_id])
                        + b_q[pair_id].view(1, heads, 1, -1)
                    )
                    g = positive_feature_map(
                        torch.einsum("bhnd,hod->bhno", k, a_k[pair_id])
                        + b_k[pair_id].view(1, heads, 1, -1)
                    )
                    u = positive_feature_map(
                        torch.einsum("bnd,hod->bhno", e_recv, bq_elec[pair_id])
                        + cq_elec[pair_id].view(1, heads, 1, -1)
                    )
                    z = positive_feature_map(
                        torch.einsum("bnd,hod->bhno", e_send, bk_elec[pair_id])
                        + ck_elec[pair_id].view(1, heads, 1, -1)
                    )
                    F = kronecker_features(f, u)
                    G = kronecker_features(g, z)

                    joint_r = torch.cat((h_recv, e_recv), dim=-1)
                    joint_s = torch.cat((h_send, e_send), dim=-1)
                    mu = torch.tanh(torch.einsum("bnd,hrd->bhnr", joint_r, m_mu[pair_id]))
                    nu = torch.tanh(torch.einsum("bnd,hrd->bhnr", joint_s, m_nu[pair_id]))

                    S = torch.einsum("bhjf,bhjd->bhfd", G, v)
                    z_sum = G.sum(dim=2)
                    T = torch.einsum("bhjf,bhjr->bhfr", G, nu)

                    base = torch.einsum("bhif,bhfd->bhid", F, S)
                    ft = torch.einsum("bhif,bhfr->bhir", F, T)
                    corr = torch.einsum("bhir,hrd->bhid", ft * mu, c_v[pair_id]) * gamma[pair_id].view(
                        1, heads, 1, 1
                    )
                    numer = numer + base + corr
                    denom = denom + torch.einsum("bhif,bhf->bhi", F, z_sum)

                    if recv_t == send_t:
                        send_pos = {int(s): i for i, s in enumerate(send_local.tolist())}
                        for li, rloc in enumerate(recv_local.tolist()):
                            sj = send_pos.get(int(rloc))
                            if sj is None:
                                continue
                            Fi = F[:, :, li, :]
                            Gj = G[:, :, sj, :]
                            w_ii = torch.einsum("bhf,bhf->bh", Fi, Gj)
                            v_i = v[:, :, sj, :]
                            mu_i = mu[:, :, li, :]
                            nu_i = nu[:, :, sj, :]
                            delta_i = torch.einsum(
                                "bhr,hrd->bhd", mu_i * nu_i, c_v[pair_id]
                            ) * gamma[pair_id].view(1, heads, 1)
                            numer[:, :, li, :] = numer[:, :, li, :] - w_ii.unsqueeze(-1) * (
                                v_i + delta_i
                            )
                            denom[:, :, li] = denom[:, :, li] - w_ii

                safe = denom.clamp_min(eps)
                head_msg = numer / safe.unsqueeze(-1)
                head_msg = torch.where(
                    denom.unsqueeze(-1) > eps, head_msg, torch.zeros_like(head_msg)
                )
                global_recv = buses_c[recv_local]
                message[:, global_recv] = torch.einsum("bhna,hda->bnd", head_msg, w_o)

    return message.to(dtype=out_dtype)


class ElectricalFactorizedAttention(nn.Module):
    """Nine type-pair × head EFLA with factorized aggregation."""

    def __init__(
        self,
        hidden_dim: int,
        value_dim: int,
        elec_dim: int,
        *,
        num_heads: int = 4,
        node_kernel: int = DEFAULT_NODE_KERNEL,
        elec_kernel: int = DEFAULT_ELEC_KERNEL,
        value_rank: int = DEFAULT_VALUE_RANK,
        num_pairs: int = NUM_TYPE_PAIRS,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value_dim = value_dim
        self.elec_dim = elec_dim
        self.num_heads = num_heads
        self.node_kernel = node_kernel
        self.elec_kernel = elec_kernel
        self.value_rank = value_rank
        shape = (num_pairs, num_heads)
        self.w_q = nn.Parameter(torch.empty(*shape, value_dim, hidden_dim))
        self.w_k = nn.Parameter(torch.empty(*shape, value_dim, hidden_dim))
        self.w_v = nn.Parameter(torch.empty(*shape, value_dim, hidden_dim))
        self.w_o = nn.Parameter(torch.empty(num_heads, hidden_dim, value_dim))
        self.a_q = nn.Parameter(torch.empty(*shape, node_kernel, value_dim))
        self.b_q = nn.Parameter(torch.zeros(*shape, node_kernel))
        self.a_k = nn.Parameter(torch.empty(*shape, node_kernel, value_dim))
        self.b_k = nn.Parameter(torch.zeros(*shape, node_kernel))
        self.bq_elec = nn.Parameter(torch.empty(*shape, elec_kernel, elec_dim))
        self.cq_elec = nn.Parameter(torch.zeros(*shape, elec_kernel))
        self.bk_elec = nn.Parameter(torch.empty(*shape, elec_kernel, elec_dim))
        self.ck_elec = nn.Parameter(torch.zeros(*shape, elec_kernel))
        in_pair = hidden_dim + elec_dim
        self.m_mu = nn.Parameter(torch.empty(*shape, value_rank, in_pair))
        self.m_nu = nn.Parameter(torch.empty(*shape, value_rank, in_pair))
        self.c_v = nn.Parameter(torch.empty(*shape, value_rank, value_dim))
        self.gamma = nn.Parameter(torch.full(shape, GAMMA_INIT))
        for param in (
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.a_q,
            self.a_k,
            self.bq_elec,
            self.bk_elec,
            self.m_mu,
            self.m_nu,
            self.c_v,
        ):
            nn.init.xavier_uniform_(param.reshape(param.shape[0] * param.shape[1], -1))

    def param_kwargs(self) -> dict[str, torch.Tensor]:
        return dict(
            w_q=self.w_q,
            w_k=self.w_k,
            w_v=self.w_v,
            w_o=self.w_o,
            a_q=self.a_q,
            b_q=self.b_q,
            a_k=self.a_k,
            b_k=self.b_k,
            bq_elec=self.bq_elec,
            cq_elec=self.cq_elec,
            bk_elec=self.bk_elec,
            ck_elec=self.ck_elec,
            m_mu=self.m_mu,
            m_nu=self.m_nu,
            c_v=self.c_v,
            gamma=self.gamma,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        elec_a: torch.Tensor,
        types: torch.Tensor,
        component: torch.Tensor,
        *,
        dense: bool = False,
    ) -> torch.Tensor:
        kwargs = self.param_kwargs()
        if dense:
            return dense_efla_messages(hidden, elec_a, types, component, **kwargs)
        return factorized_efla_messages(hidden, elec_a, types, component, **kwargs)
