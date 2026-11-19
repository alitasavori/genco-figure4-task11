"""JSGC-Net physics operators: reference Jacobian, block-Jacobi, coarse projection.

Topology-cached quantities for the two-level correction
``Δx = z_local + P (Pᵀ J⁰ P)⁻¹ Pᵀ (r − J⁰ z_local)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Bus types match train_tpgt_task11: 0=PQ, 1=PV, 2=Slack.
TYPE_PQ = 0
TYPE_PV = 1
TYPE_SLACK = 2
Y_ADMITTANCE_EPS = 1e-10
JACOBI_REG = 1e-8


@dataclass
class ReducedIndex:
    """Maps reduced Newton unknowns x = [θ_{I_θ}; v_{I_v}] ↔ buses."""

    non_slack: torch.Tensor  # (n_theta,) bus indices
    pq: torch.Tensor  # (n_v,) bus indices
    n_theta: int
    n_v: int
    n_x: int
    n_buses: int
    types: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.types.device


def build_reduced_index(types: torch.Tensor) -> ReducedIndex:
    """``types``: (N,) int64 on any device."""
    types = types.long().view(-1)
    non_slack = torch.nonzero(types != TYPE_SLACK, as_tuple=False).view(-1)
    pq = torch.nonzero(types == TYPE_PQ, as_tuple=False).view(-1)
    n_theta = int(non_slack.numel())
    n_v = int(pq.numel())
    return ReducedIndex(
        non_slack=non_slack,
        pq=pq,
        n_theta=n_theta,
        n_v=n_v,
        n_x=n_theta + n_v,
        n_buses=int(types.numel()),
        types=types,
    )


def ybus_to_gb(ybus: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split complex Ybus into G, B (float64). Accepts (N,N) complex."""
    y = ybus.to(torch.complex128)
    return y.real.contiguous(), y.imag.contiguous()


def physical_edge_index(
    ybus: torch.Tensor,
    *,
    eps: float = Y_ADMITTANCE_EPS,
) -> torch.Tensor:
    """Directed physical edges (2, E) from off-diagonal |Y_ij| > eps."""
    y = ybus.detach()
    if y.is_complex():
        mag = y.abs()
    else:
        mag = y.abs()
    n = mag.shape[0]
    mask = mag > eps
    mask.fill_diagonal_(False)
    recv, send = torch.nonzero(mask, as_tuple=True)
    return torch.stack((recv, send), dim=0)


def flat_start_theta_v(
    types: torch.Tensor,
    *,
    v_set: torch.Tensor | None = None,
    theta_slack: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat-start (θ, v) for all buses. PV/slack keep ``v_set`` if given else 1.0."""
    n = int(types.numel())
    device = types.device
    dtype = torch.float64
    theta = torch.zeros(n, device=device, dtype=dtype)
    v = torch.ones(n, device=device, dtype=dtype)
    if v_set is not None:
        v = v_set.to(device=device, dtype=dtype).clone()
    slack = types == TYPE_SLACK
    if slack.any():
        theta = torch.where(slack, torch.full_like(theta, float(theta_slack)), theta)
    return theta, v


def pack_reduced_x(theta: torch.Tensor, v: torch.Tensor, index: ReducedIndex) -> torch.Tensor:
    """Pack full bus (θ, v) → reduced x (n_x,)."""
    return torch.cat((theta[index.non_slack], v[index.pq]), dim=0)


def unpack_reduced_x(
    x: torch.Tensor,
    index: ReducedIndex,
    *,
    theta_base: torch.Tensor,
    v_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write reduced x into copies of base (θ, v)."""
    theta = theta_base.clone()
    v = v_base.clone()
    if index.n_theta:
        theta[index.non_slack] = x[: index.n_theta]
    if index.n_v:
        v[index.pq] = x[index.n_theta :]
    return theta, v


def ac_power_injections(
    theta: torch.Tensor,
    vm: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P_calc, Q_calc for theta/vm shaped (N,) or (B, N); G,B (N,N)."""
    single = theta.dim() == 1
    if single:
        theta = theta.unsqueeze(0)
        vm = vm.unsqueeze(0)
    vr = vm * torch.cos(theta)
    vi = vm * torch.sin(theta)
    # I = Y V with Y = G + jB
    ir = vr @ g.T - vi @ b.T
    ii = vr @ b.T + vi @ g.T
    p_calc = vr * ir + vi * ii
    q_calc = vi * ir - vr * ii
    if single:
        return p_calc[0], q_calc[0]
    return p_calc, q_calc


def reduced_ac_residual(
    theta: torch.Tensor,
    vm: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    p_spec: torch.Tensor,
    q_spec: torch.Tensor,
    index: ReducedIndex,
) -> torch.Tensor:
    """r = [P_spec − P_calc]_{I_θ}; [Q_spec − Q_calc]_{I_v}. Supports (N,) or (B,N) specs."""
    p_calc, q_calc = ac_power_injections(theta, vm, g, b)
    if theta.dim() == 1:
        r_p = p_spec[index.non_slack] - p_calc[index.non_slack]
        r_q = q_spec[index.pq] - q_calc[index.pq] if index.n_v else p_spec.new_zeros((0,))
        return torch.cat((r_p, r_q), dim=0)
    # batched theta (B,N) — p_spec (B,N)
    r_p = p_spec[:, index.non_slack] - p_calc[:, index.non_slack]
    if index.n_v:
        r_q = q_spec[:, index.pq] - q_calc[:, index.pq]
        return torch.cat((r_p, r_q), dim=-1)
    return r_p


def build_ac_jacobian(
    theta: torch.Tensor,
    vm: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    index: ReducedIndex,
    *,
    p_calc: torch.Tensor | None = None,
    q_calc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduced AC Jacobian J = ∂(P_calc, Q_calc)/∂(θ, v) at given state (n_x, n_x).

    Matches Newton form J Δx = r with r = spec − calc, so solve uses this J.
    """
    if p_calc is None or q_calc is None:
        p_calc, q_calc = ac_power_injections(theta, vm, g, b)
    n = index.n_buses
    device = theta.device
    dtype = theta.dtype
    # Full bus derivatives then reduce.
    # δ_ij = θ_i − θ_j
    dtheta = theta.view(n, 1) - theta.view(1, n)
    cos_d = torch.cos(dtheta)
    sin_d = torch.sin(dtheta)
    vi = vm.view(n, 1)
    vj = vm.view(1, n)

    # Off-diagonal i≠j formulas; diagonal overwritten below.
    dp_dth = vi * vj * (g * sin_d - b * cos_d)
    dp_dv = vi * (g * cos_d + b * sin_d)
    dq_dth = -vi * vj * (g * cos_d + b * sin_d)
    dq_dv = vi * (g * sin_d - b * cos_d)

    # Diagonal
    eye = torch.eye(n, device=device, dtype=dtype)
    dp_dth = dp_dth * (1.0 - eye) + eye * (-q_calc - b.diag() * vm * vm)
    dp_dv = dp_dv * (1.0 - eye) + eye * (p_calc / vm.clamp(min=1e-8) + g.diag() * vm)
    dq_dth = dq_dth * (1.0 - eye) + eye * (p_calc - g.diag() * vm * vm)
    dq_dv = dq_dv * (1.0 - eye) + eye * (q_calc / vm.clamp(min=1e-8) - b.diag() * vm)

    # Reduced blocks: rows P for non_slack, Q for pq; cols θ for non_slack, v for pq.
    j_pt = dp_dth[index.non_slack][:, index.non_slack]
    j_pv = dp_dv[index.non_slack][:, index.pq] if index.n_v else theta.new_zeros((index.n_theta, 0))
    if index.n_v:
        j_qt = dq_dth[index.pq][:, index.non_slack]
        j_qv = dq_dv[index.pq][:, index.pq]
        top = torch.cat((j_pt, j_pv), dim=1)
        bot = torch.cat((j_qt, j_qv), dim=1)
        return torch.cat((top, bot), dim=0)
    return j_pt


def extract_block_diagonal(j0: torch.Tensor, index: ReducedIndex) -> list[torch.Tensor]:
    """Per-bus local Jacobian blocks from reduced J⁰ (list of 1×1 or 2×2)."""
    blocks: list[torch.Tensor] = []
    # Map bus → θ-column / v-column in reduced coords.
    theta_col = {int(b): i for i, b in enumerate(index.non_slack.tolist())}
    v_col = {int(b): index.n_theta + i for i, b in enumerate(index.pq.tolist())}
    for bus in range(index.n_buses):
        t = int(index.types[bus].item())
        if t == TYPE_SLACK:
            continue
        if t == TYPE_PV:
            c = theta_col[bus]
            blocks.append(j0[c : c + 1, c : c + 1].clone())
        else:  # PQ: unknowns θ and v
            ct = theta_col[bus]
            cv = v_col[bus]
            # Rows for this bus: P-row at ct, Q-row at cv
            rows = torch.tensor([ct, cv], device=j0.device)
            cols = rows
            blocks.append(j0[rows][:, cols].clone())
    return blocks


def jacobi_solve(
    r: torch.Tensor,
    blocks: list[torch.Tensor],
    index: ReducedIndex,
    *,
    reg: float = JACOBI_REG,
) -> torch.Tensor:
    """z = D⁻¹ r using per-bus 1×1 / 2×2 solves. ``r`` is (n_x,) or (B, n_x)."""
    batched = r.dim() == 2
    if not batched:
        r = r.unsqueeze(0)
    batch = r.shape[0]
    z = torch.zeros_like(r)
    theta_col = {int(b): i for i, b in enumerate(index.non_slack.tolist())}
    v_col = {int(b): index.n_theta + i for i, b in enumerate(index.pq.tolist())}
    bi = 0
    for bus in range(index.n_buses):
        t = int(index.types[bus].item())
        if t == TYPE_SLACK:
            continue
        blk = blocks[bi]
        bi += 1
        if t == TYPE_PV:
            c = theta_col[bus]
            d = blk[0, 0] + reg
            z[:, c] = r[:, c] / d
        else:
            ct = theta_col[bus]
            cv = v_col[bus]
            local = blk + reg * torch.eye(2, device=blk.device, dtype=blk.dtype)
            rhs = torch.stack((r[:, ct], r[:, cv]), dim=-1)  # (B, 2)
            try:
                sol = torch.linalg.solve(local.expand(batch, 2, 2), rhs.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                sol = torch.linalg.lstsq(local.expand(batch, 2, 2), rhs.unsqueeze(-1)).solution.squeeze(-1)
            z[:, ct] = sol[:, 0]
            z[:, cv] = sol[:, 1]
    return z if batched else z[0]


def greedy_partition(n: int, adjacency: list[set[int]], n_parts: int) -> np.ndarray:
    """Greedy contiguous-ish partition of an undirected graph into ``n_parts`` labels."""
    n_parts = max(1, min(int(n_parts), n))
    labels = -np.ones(n, dtype=np.int64)
    # Seed BFS from evenly spaced starts among connected nodes.
    order = list(range(n))
    # Prefer high-degree seeds
    deg = sorted(order, key=lambda i: -len(adjacency[i]))
    seeds = deg[:n_parts] if len(deg) >= n_parts else deg + list(range(n_parts - len(deg)))
    for a, s in enumerate(seeds[:n_parts]):
        if labels[s] < 0:
            labels[s] = a
    # Grow by assigning unlabeled neighbors
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if labels[i] >= 0:
                continue
            neigh_labels = [labels[j] for j in adjacency[i] if labels[j] >= 0]
            if neigh_labels:
                # majority
                vals, counts = np.unique(neigh_labels, return_counts=True)
                labels[i] = int(vals[np.argmax(counts)])
                changed = True
        if not changed:
            # assign leftovers round-robin
            for i in range(n):
                if labels[i] < 0:
                    labels[i] = int(i % n_parts)
                    changed = True
            break
    # Ensure no empty parts: steal from largest
    for a in range(n_parts):
        if not np.any(labels == a):
            sizes = [(b, int(np.sum(labels == b))) for b in range(n_parts)]
            donor = max(sizes, key=lambda t: t[1])[0]
            idxs = np.flatnonzero(labels == donor)
            labels[idxs[0]] = a
    return labels


def build_interpolation(
    index: ReducedIndex,
    edge_index: torch.Tensor,
    *,
    r_theta: int = 8,
    r_v: int = 8,
) -> torch.Tensor:
    """Piecewise-constant P_T (n_x, r_θ+r_v) with orthonormal columns within aggregates."""
    n = index.n_buses
    # Undirected adjacency from directed edges
    adj: list[set[int]] = [set() for _ in range(n)]
    if edge_index.numel():
        for a, b in edge_index.t().tolist():
            if a != b:
                adj[a].add(b)
                adj[b].add(a)

    device = index.types.device
    dtype = torch.float64
    parts_t = greedy_partition(n, adj, r_theta) if index.n_theta else np.zeros(n, dtype=np.int64)
    parts_v = greedy_partition(n, adj, r_v) if index.n_v else np.zeros(n, dtype=np.int64)

    # Restrict partitions to participating buses and reindex to 0..r-1 densely
    def _dense_labels(bus_ids: torch.Tensor, parts: np.ndarray, r_target: int) -> tuple[np.ndarray, int]:
        if bus_ids.numel() == 0:
            return np.zeros(0, dtype=np.int64), 0
        raw = parts[bus_ids.cpu().numpy()]
        uniq = np.unique(raw)
        remap = {int(u): i for i, u in enumerate(uniq.tolist())}
        labels = np.array([remap[int(u)] for u in raw.tolist()], dtype=np.int64)
        return labels, len(uniq)

    lab_t, rt = _dense_labels(index.non_slack, parts_t, r_theta)
    lab_v, rv = _dense_labels(index.pq, parts_v, r_v)
    r = rt + rv
    p = torch.zeros((index.n_x, r), device=device, dtype=dtype)
    for i, a in enumerate(lab_t.tolist()):
        n_a = max(1, int(np.sum(lab_t == a)))
        p[i, a] = 1.0 / np.sqrt(n_a)
    for i, a in enumerate(lab_v.tolist()):
        n_a = max(1, int(np.sum(lab_v == a)))
        p[index.n_theta + i, rt + a] = 1.0 / np.sqrt(n_a)
    return p


def factor_coarse_jacobian(
    j0: torch.Tensor,
    p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Jc, LU factors, pivots for Jc = Pᵀ J⁰ P."""
    jc = p.T @ j0 @ p
    # Stabilize if needed
    eye = torch.eye(jc.shape[0], device=jc.device, dtype=jc.dtype)
    try:
        lu, pivots = torch.linalg.lu_factor(jc)
        # quick nonsingularity probe
        _ = torch.linalg.lu_solve(lu, pivots, eye)
    except RuntimeError:
        jc = jc + 1e-6 * eye
        lu, pivots = torch.linalg.lu_factor(jc)
    return jc, lu, pivots


def coarse_correct(
    r: torch.Tensor,
    z_local: torch.Tensor,
    j0: torch.Tensor,
    p: torch.Tensor,
    lu: torch.Tensor,
    pivots: torch.Tensor,
) -> torch.Tensor:
    """Δx = z_local + P Jc⁻¹ Pᵀ (r − J⁰ z_local). ``r``, ``z_local``: (n_x,) or (B, n_x)."""
    batched = r.dim() == 2
    if not batched:
        r = r.unsqueeze(0)
        z_local = z_local.unsqueeze(0)
    # defect = r - J0 @ z_local  (J0 acts on last dim)
    defect = r - torch.matmul(z_local, j0.T)
    b_coarse = torch.matmul(defect, p)  # (B, r)
    c = torch.linalg.lu_solve(lu, pivots, b_coarse.T.contiguous()).T
    z_global = torch.matmul(c, p.T)
    out = z_local + z_global
    return out if batched else out[0]


def projected_linear_residual(
    r: torch.Tensor,
    delta_x: torch.Tensor,
    j0: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """‖Pᵀ (r − J⁰ Δx)‖₂ — should be ≈0 for undamped two-level correction."""
    residual = r - torch.matmul(delta_x, j0.T) if r.dim() == 1 else r - torch.matmul(delta_x, j0.T)
    proj = residual @ p if r.dim() == 1 else residual @ p
    return torch.linalg.vector_norm(proj)


@dataclass
class TopologyCache:
    """Cached per-topology physics for JSGC."""

    index: ReducedIndex
    g: torch.Tensor
    b: torch.Tensor
    j0: torch.Tensor
    d_blocks: list[torch.Tensor]
    p: torch.Tensor
    jc: torch.Tensor
    lu: torch.Tensor
    pivots: torch.Tensor
    edge_index: torch.Tensor
    theta0: torch.Tensor
    v0: torch.Tensor
    r_coarse: int


def prepare_topology(
    ybus: torch.Tensor,
    types: torch.Tensor,
    *,
    r_theta: int = 8,
    r_v: int = 8,
    v_set: torch.Tensor | None = None,
) -> TopologyCache:
    """Build / cache all static operators for topology T."""
    types = types.long().view(-1).to(ybus.device)
    index = build_reduced_index(types)
    g, b = ybus_to_gb(ybus.to(device=ybus.device))
    g = g.to(dtype=torch.float64)
    b = b.to(dtype=torch.float64)
    theta0, v0 = flat_start_theta_v(types, v_set=v_set)
    j0 = build_ac_jacobian(theta0, v0, g, b, index)
    d_blocks = extract_block_diagonal(j0, index)
    edges = physical_edge_index(ybus)
    p = build_interpolation(index, edges, r_theta=r_theta, r_v=r_v)
    jc, lu, pivots = factor_coarse_jacobian(j0, p)
    return TopologyCache(
        index=index,
        g=g,
        b=b,
        j0=j0,
        d_blocks=d_blocks,
        p=p,
        jc=jc,
        lu=lu,
        pivots=pivots,
        edge_index=edges,
        theta0=theta0,
        v0=v0,
        r_coarse=int(p.shape[1]),
    )


def two_level_step(
    r: torch.Tensor,
    cache: TopologyCache,
    *,
    z_local: torch.Tensor | None = None,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One undamped/damped two-level correction. Returns (Δx, z_local used)."""
    if z_local is None:
        z_local = jacobi_solve(r, cache.d_blocks, cache.index)
    delta = coarse_correct(r, z_local, cache.j0, cache.p, cache.lu, cache.pivots)
    return eta * delta, z_local


def baseline_iterate(
    p_spec: torch.Tensor,
    q_spec: torch.Tensor,
    cache: TopologyCache,
    *,
    k: int = 4,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Physics-only Jacobi+coarse iterations. Returns final (θ, v) and residual history.

    ``p_spec``, ``q_spec``: (N,) or (B, N).
    """
    batched = p_spec.dim() == 2
    if not batched:
        p_spec = p_spec.unsqueeze(0)
        q_spec = q_spec.unsqueeze(0)
    batch = p_spec.shape[0]
    theta = cache.theta0.unsqueeze(0).expand(batch, -1).clone()
    v = cache.v0.unsqueeze(0).expand(batch, -1).clone()
    residuals: list[torch.Tensor] = []
    for _ in range(k):
        r = reduced_ac_residual(theta, v, cache.g, cache.b, p_spec, q_spec, cache.index)
        residuals.append(r)
        delta, _ = two_level_step(r, cache, eta=eta)
        # Apply Δx to θ, v
        if cache.index.n_theta:
            theta = theta.clone()
            theta[:, cache.index.non_slack] = theta[:, cache.index.non_slack] + delta[..., : cache.index.n_theta]
        if cache.index.n_v:
            v = v.clone()
            v[:, cache.index.pq] = v[:, cache.index.pq] + delta[..., cache.index.n_theta :]
            v = v.clamp(min=0.05, max=2.0)
    if not batched:
        return theta[0], v[0], [ri[0] for ri in residuals]
    return theta, v, residuals


def reduced_to_tpgt_unknown(
    theta: torch.Tensor,
    v: torch.Tensor,
    *,
    p_calc: torch.Tensor,
    q_calc: torch.Tensor,
    types: torch.Tensor,
) -> torch.Tensor:
    """Pack (θ,v) into TPGT unknown slots (B,N,2) or (N,2): PQ (V,θ), PV (Q,θ), Slack (P,Q)."""
    single = theta.dim() == 1
    if single:
        theta = theta.unsqueeze(0)
        v = v.unsqueeze(0)
        p_calc = p_calc.unsqueeze(0)
        q_calc = q_calc.unsqueeze(0)
    bsz, n = theta.shape
    unknown = torch.zeros(bsz, n, 2, device=theta.device, dtype=theta.dtype)
    types = types.view(-1)
    pq = types == TYPE_PQ
    pv = types == TYPE_PV
    slack = types == TYPE_SLACK
    unknown[:, :, 0] = torch.where(pq, v, torch.where(pv, q_calc, p_calc))
    unknown[:, :, 1] = torch.where(slack, q_calc, theta)
    return unknown[0] if single else unknown
