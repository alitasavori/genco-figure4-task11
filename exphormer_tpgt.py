"""Exphormer-style sparse attention helpers for TPGT v12 (dense-masked first, sparse later)."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
import torch.nn as nn

TYPE_VIRTUAL = 3
NUM_REAL_TYPES = 3
NUM_EXPH_PAIRS = 15  # 9 real←real + 6 virt↔real (no Virtual←Virtual)

EDGE_TAU_PHYS = 0
EDGE_TAU_EXP = 1
EDGE_TAU_VIRT = 2
NUM_EDGE_TAU = 3


def exph_pair_id(recv_type: int, send_type: int) -> int:
    """Map ordered types to relation id in ``{0..14}`` (raises if Virtual←Virtual)."""
    if recv_type < NUM_REAL_TYPES and send_type < NUM_REAL_TYPES:
        return recv_type * NUM_REAL_TYPES + send_type
    if recv_type < NUM_REAL_TYPES and send_type == TYPE_VIRTUAL:
        return 9 + recv_type
    if recv_type == TYPE_VIRTUAL and send_type < NUM_REAL_TYPES:
        return 12 + send_type
    raise ValueError(f"unsupported Exphormer relation {recv_type}←{send_type}")


def topology_expander_seed(branches: list | tuple, bus_count: int, base_seed: int = 0) -> int:
    """Stable seed from topology structure so expanders are fixed per feeder graph."""
    payload = np.int64(bus_count).tobytes() + np.int64(base_seed).tobytes()
    if branches:
        table = np.asarray(branches, dtype=np.float64).reshape(-1, 4)
        # Status is already filtered; hash endpoints + rounded r/x.
        packed = np.column_stack(
            (
                table[:, 0].astype(np.int64),
                table[:, 1].astype(np.int64),
                np.round(table[:, 2] * 1e9).astype(np.int64),
                np.round(table[:, 3] * 1e9).astype(np.int64),
            )
        )
        payload += packed.tobytes()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def build_exphormer_graph(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
    component: np.ndarray,
    *,
    d_exp: int = 4,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build dense allow-mask and edge-type ids on augmented nodes ``N + C``.

    Returns
    -------
    allow : bool (N+C, N+C)
    tau : int64 (N+C, N+C) with {-1, phys=0, exp=1, virt=2}
    n_virt : number of virtual nodes (= number of components)
    """
    if d_exp < 2 or d_exp % 2 != 0:
        raise ValueError(f"d_exp must be even and >= 2, got {d_exp}")
    component = np.asarray(component, dtype=np.int64).reshape(-1)
    if component.size != bus_count:
        raise ValueError(f"component length {component.size} != bus_count {bus_count}")
    n_comp = int(component.max()) + 1 if bus_count else 0
    n_aug = bus_count + n_comp
    allow = np.zeros((n_aug, n_aug), dtype=bool)
    tau = np.full((n_aug, n_aug), -1, dtype=np.int64)
    if bus_count == 0:
        return allow, tau, n_comp

    phys = np.zeros((bus_count, bus_count), dtype=bool)
    table = np.asarray(branches, dtype=np.float64).reshape(-1, 4) if branches else np.zeros((0, 4))
    for origin, destination, resistance, reactance in table:
        if abs(resistance + 1j * reactance) <= 0.0:
            continue
        i = int(origin)
        j = int(destination)
        phys[i, j] = True
        phys[j, i] = True

    exp = np.zeros((bus_count, bus_count), dtype=bool)
    rng_seed = topology_expander_seed(branches, bus_count) if seed is None else int(seed)
    rng = np.random.default_rng(rng_seed)
    half = d_exp // 2
    for comp in range(n_comp):
        nodes = np.flatnonzero(component == comp)
        m = int(nodes.size)
        if m <= 1:
            continue
        for _ in range(half):
            perm = rng.permutation(m)
            inv = np.empty(m, dtype=np.int64)
            inv[perm] = np.arange(m)
            for local in range(m):
                i = int(nodes[local])
                j_fwd = int(nodes[int(perm[local])])
                j_inv = int(nodes[int(inv[local])])
                if i != j_fwd:
                    exp[i, j_fwd] = True
                    exp[j_fwd, i] = True
                if i != j_inv:
                    exp[i, j_inv] = True
                    exp[j_inv, i] = True

    # Real–real: physical wins over expander; no self-loops.
    for i in range(bus_count):
        for j in range(bus_count):
            if i == j:
                continue
            if phys[i, j]:
                allow[i, j] = True
                tau[i, j] = EDGE_TAU_PHYS
            elif exp[i, j]:
                allow[i, j] = True
                tau[i, j] = EDGE_TAU_EXP

    # Virtual node u_c at index N+c: bidirectional with all buses in component c.
    for i in range(bus_count):
        u = bus_count + int(component[i])
        allow[i, u] = True
        allow[u, i] = True
        tau[i, u] = EDGE_TAU_VIRT
        tau[u, i] = EDGE_TAU_VIRT

    return allow, tau, n_comp


def augmented_types(types: torch.Tensor, n_virt: int) -> torch.Tensor:
    """Append ``n_virt`` virtual-type labels (3) after real bus types."""
    if types.dim() != 1:
        raise ValueError("augmented_types expects (N,) types")
    if n_virt <= 0:
        return types
    virt = types.new_full((n_virt,), TYPE_VIRTUAL)
    return torch.cat((types, virt), dim=0)


def build_exphormer_pair_features(
    rpe: torch.Tensor,
    allow: torch.Tensor,
    tau: torch.Tensor,
    types_aug: torch.Tensor,
    virt_rpe: torch.Tensor,
) -> torch.Tensor:
    """``(N+C, N+C, RPE_DIM)`` or batched ``(B,N+C,N+C,RPE_DIM)`` pair features.

    Real–real slots copy Zeff ``rpe``; virtual edges use learned ``virt_rpe``
    rows ``0..2 = t←U`` and ``3..5 = U←t``.
    """
    if rpe.dim() == 3:
        n_real = int(rpe.shape[0])
        n_aug = int(allow.shape[0])
        out = rpe.new_zeros((n_aug, n_aug, rpe.shape[-1]))
        out[:n_real, :n_real] = rpe
        virt_edges = allow & (tau == EDGE_TAU_VIRT)
        if bool(virt_edges.any()):
            ii, jj = torch.nonzero(virt_edges, as_tuple=True)
            ti = types_aug[ii]
            tj = types_aug[jj]
            real_from_virt = (ti < NUM_REAL_TYPES) & (tj == TYPE_VIRTUAL)
            virt_from_real = (ti == TYPE_VIRTUAL) & (tj < NUM_REAL_TYPES)
            if bool(real_from_virt.any()):
                out[ii[real_from_virt], jj[real_from_virt]] = virt_rpe[ti[real_from_virt]]
            if bool(virt_from_real.any()):
                out[ii[virt_from_real], jj[virt_from_real]] = virt_rpe[3 + tj[virt_from_real]]
        return out
    # Batched rpe (B, N, N, R)
    batch, n_real, _, rpe_dim = rpe.shape
    n_aug = int(allow.shape[-1])
    out = rpe.new_zeros((batch, n_aug, n_aug, rpe_dim))
    out[:, :n_real, :n_real] = rpe
    if allow.dim() == 2:
        allow_b = allow
        tau_b = tau
        types = types_aug
        virt_edges = allow_b & (tau_b == EDGE_TAU_VIRT)
        if bool(virt_edges.any()):
            ii, jj = torch.nonzero(virt_edges, as_tuple=True)
            ti = types[ii]
            tj = types[jj]
            real_from_virt = (ti < NUM_REAL_TYPES) & (tj == TYPE_VIRTUAL)
            virt_from_real = (ti == TYPE_VIRTUAL) & (tj < NUM_REAL_TYPES)
            if bool(real_from_virt.any()):
                feat = virt_rpe[ti[real_from_virt]]
                out[:, ii[real_from_virt], jj[real_from_virt]] = feat.unsqueeze(0).expand(batch, -1, -1)
            if bool(virt_from_real.any()):
                feat = virt_rpe[3 + tj[virt_from_real]]
                out[:, ii[virt_from_real], jj[virt_from_real]] = feat.unsqueeze(0).expand(batch, -1, -1)
        return out
    raise ValueError(f"unsupported allow/rpe shapes allow={tuple(allow.shape)} rpe={tuple(rpe.shape)}")


def _score_mlp_block(
    query: torch.Tensor,
    key: torch.Tensor,
    edge: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    """Joint MLP score for one type-pair: query/key (B,H,N*,D_A), edge (B,H,Nr,Ns,R)."""
    heads = query.shape[1]
    joint = torch.cat(
        (
            query.unsqueeze(-2).expand(-1, -1, -1, key.shape[-2], -1),
            key.unsqueeze(-3).expand(-1, -1, query.shape[-2], -1, -1),
            edge,
        ),
        dim=-1,
    )
    h1 = torch.relu(torch.einsum("bhnms,has->bhnma", joint, w1) + b1.view(1, heads, 1, 1, -1))
    h2 = torch.relu(torch.einsum("bhnma,hca->bhnmc", h1, w2) + b2.view(1, heads, 1, 1, -1))
    score = torch.einsum("bhnmc,hc->bhnm", h2, w3[:, 0]) + b3.view(1, heads, 1, 1)
    return score


def dense_exphormer_messages(
    hidden_real: torch.Tensor,
    z_virt: torch.Tensor,
    pair_aug: torch.Tensor,
    types_aug: torch.Tensor,
    allow: torch.Tensor,
    tau: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    score_w1: torch.Tensor,
    score_b1: torch.Tensor,
    score_w2: torch.Tensor,
    score_b2: torch.Tensor,
    score_w3: torch.Tensor,
    score_b3: torch.Tensor,
    edge_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense Exphormer attention with -inf mask.

    Returns ``(message_real, message_virt)`` with shapes matching ``hidden_real`` / ``z_virt``.
    """
    batch, n_real, width = hidden_real.shape
    n_virt = int(z_virt.shape[1])
    n_aug = n_real + n_virt
    heads = int(w_q.shape[1])
    d_a = int(w_q.shape[2])
    y = torch.cat((hidden_real, z_virt), dim=1)
    scores = y.new_full((batch, heads, n_aug, n_aug), float("-inf"))
    allow_b = allow if allow.dim() == 2 else allow
    tau_b = tau if tau.dim() == 2 else tau

    for recv_t in range(NUM_REAL_TYPES + 1):
        for send_t in range(NUM_REAL_TYPES + 1):
            if recv_t == TYPE_VIRTUAL and send_t == TYPE_VIRTUAL:
                continue
            try:
                p = exph_pair_id(recv_t, send_t)
            except ValueError:
                continue
            receivers = torch.nonzero(types_aug == recv_t, as_tuple=False).view(-1)
            senders = torch.nonzero(types_aug == send_t, as_tuple=False).view(-1)
            if receivers.numel() == 0 or senders.numel() == 0:
                continue
            block_allow = allow_b[receivers.view(-1, 1), senders.view(1, -1)]
            if not bool(block_allow.any()):
                continue
            query = torch.einsum("bnd,had->bhna", y[:, receivers], w_q[p])
            key = torch.einsum("bnd,had->bhna", y[:, senders], w_k[p])
            if pair_aug.dim() == 3:
                edge = pair_aug[receivers.view(-1, 1), senders.view(1, -1)].unsqueeze(0).unsqueeze(1)
                edge = edge.expand(batch, heads, -1, -1, -1)
            else:
                edge = pair_aug[:, receivers.view(-1, 1), senders.view(1, -1)].unsqueeze(1)
                edge = edge.expand(-1, heads, -1, -1, -1)
            edge = edge.to(dtype=query.dtype)
            score = _score_mlp_block(
                query,
                key,
                edge,
                score_w1[p],
                score_b1[p],
                score_w2[p],
                score_b2[p],
                score_w3[p],
                score_b3[p],
            )
            block_tau = tau_b[receivers.view(-1, 1), senders.view(1, -1)]
            # Edge-type bias (phys/exp/virt); non-edges stay masked.
            bias = edge_bias.new_zeros((heads, receivers.numel(), senders.numel()))
            for tau_id in range(NUM_EDGE_TAU):
                hit = block_tau == tau_id
                if bool(hit.any()):
                    bias = bias + hit.to(bias.dtype).unsqueeze(0) * edge_bias[tau_id].view(heads, 1, 1)
            score = score + bias.unsqueeze(0)
            score = score.masked_fill(~block_allow.unsqueeze(0).unsqueeze(0), float("-inf"))
            scores[:, :, receivers.view(-1, 1), senders.view(1, -1)] = score.to(scores.dtype)

    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    row_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)

    mixed = weights.new_zeros((batch, heads, n_aug, d_a))
    for recv_t in range(NUM_REAL_TYPES + 1):
        for send_t in range(NUM_REAL_TYPES + 1):
            if recv_t == TYPE_VIRTUAL and send_t == TYPE_VIRTUAL:
                continue
            try:
                p = exph_pair_id(recv_t, send_t)
            except ValueError:
                continue
            receivers = torch.nonzero(types_aug == recv_t, as_tuple=False).view(-1)
            senders = torch.nonzero(types_aug == send_t, as_tuple=False).view(-1)
            if receivers.numel() == 0 or senders.numel() == 0:
                continue
            block_allow = allow_b[receivers.view(-1, 1), senders.view(1, -1)]
            if not bool(block_allow.any()):
                continue
            block_w = weights[:, :, receivers.view(-1, 1), senders.view(1, -1)]
            block_w = block_w.masked_fill(~block_allow.unsqueeze(0).unsqueeze(0), 0.0)
            value = torch.einsum("bnd,had->bhna", y[:, senders], w_v[p])
            gathered = torch.einsum("bhnm,bhma->bhna", block_w, value)
            mixed = mixed.index_add(2, receivers, gathered.to(mixed.dtype))

    message = torch.einsum("bhna,hda->bnd", mixed, w_o)
    return message[:, :n_real], message[:, n_real:]


def sparse_edges_from_dense(
    allow: np.ndarray,
    tau: np.ndarray,
    types_aug: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """COO edges ``(recv, send, tau, pair_id)`` for Phase-2 sparse kernel."""
    recv_idx, send_idx = np.nonzero(allow)
    edge_tau = tau[recv_idx, send_idx]
    pair_ids = np.array(
        [exph_pair_id(int(types_aug[i]), int(types_aug[j])) for i, j in zip(recv_idx, send_idx)],
        dtype=np.int64,
    )
    return (
        recv_idx.astype(np.int64),
        send_idx.astype(np.int64),
        edge_tau.astype(np.int64),
        pair_ids,
    )


def sparse_exphormer_messages(
    hidden_real: torch.Tensor,
    z_virt: torch.Tensor,
    pair_aug: torch.Tensor,
    types_aug: torch.Tensor,
    recv_idx: torch.Tensor,
    send_idx: torch.Tensor,
    edge_tau: torch.Tensor,
    pair_ids: torch.Tensor,
    *,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    score_w1: torch.Tensor,
    score_b1: torch.Tensor,
    score_w2: torch.Tensor,
    score_b2: torch.Tensor,
    score_w3: torch.Tensor,
    score_b3: torch.Tensor,
    edge_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse edge-list Exphormer attention (should match dense within atol)."""
    batch, n_real, _ = hidden_real.shape
    n_virt = int(z_virt.shape[1])
    n_aug = n_real + n_virt
    heads = int(w_q.shape[1])
    d_a = int(w_q.shape[2])
    y = torch.cat((hidden_real, z_virt), dim=1)
    e = int(recv_idx.numel())
    if e == 0:
        return hidden_real.new_zeros(hidden_real.shape), z_virt.new_zeros(z_virt.shape)

    # Per-edge scores (B, H, E).
    scores = y.new_zeros((batch, heads, e))
    values = y.new_zeros((batch, heads, e, d_a))
    for p in range(NUM_EXPH_PAIRS):
        edge_mask = pair_ids == p
        if not bool(edge_mask.any()):
            continue
        local = torch.nonzero(edge_mask, as_tuple=False).view(-1)
        r = recv_idx[local]
        s = send_idx[local]
        # Unique receivers/senders for this pair, then map edges into the dense block path.
        # For parity and simplicity: evaluate each edge with einsum over gathered endpoints.
        q = torch.einsum("bed,had->bhea", y[:, r], w_q[p])  # B,H,E_p,D_A
        k = torch.einsum("bed,had->bhea", y[:, s], w_k[p])
        v = torch.einsum("bed,had->bhea", y[:, s], w_v[p])
        if pair_aug.dim() == 3:
            edge_feat = pair_aug[r, s].unsqueeze(0).unsqueeze(1).expand(batch, heads, -1, -1)
        else:
            edge_feat = pair_aug[:, r, s].unsqueeze(1).expand(-1, heads, -1, -1)
        # Match activation dtype under CUDA autocast (pair/RPE may stay float32).
        edge_feat = edge_feat.to(dtype=q.dtype)
        joint = torch.cat((q, k, edge_feat), dim=-1)
        h1 = torch.relu(
            torch.einsum("bhes,has->bhea", joint, score_w1[p]) + score_b1[p].view(1, heads, 1, -1)
        )
        h2 = torch.relu(
            torch.einsum("bhea,hca->bhec", h1, score_w2[p]) + score_b2[p].view(1, heads, 1, -1)
        )
        score = torch.einsum("bhec,hc->bhe", h2, score_w3[p, :, 0]) + score_b3[p].view(1, heads, 1)
        tau_e = edge_tau[local]
        bias = edge_bias[tau_e].transpose(0, 1).unsqueeze(0).to(dtype=score.dtype)  # (1,H,E_p)
        score = score + bias
        # Autocast: buffers follow y (often Half); MLP/bias math may promote to Float.
        scores[:, :, local] = score.to(dtype=scores.dtype)
        values[:, :, local] = v.to(dtype=values.dtype)

    # Softmax per receiver over its incident edges.
    weights = y.new_zeros((batch, heads, e))
    for node in range(n_aug):
        local = torch.nonzero(recv_idx == node, as_tuple=False).view(-1)
        if local.numel() == 0:
            continue
        logits = scores[:, :, local]
        w = torch.softmax(logits, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        weights[:, :, local] = w.to(dtype=weights.dtype)

    mixed = y.new_zeros((batch, heads, n_aug, d_a))
    # Scatter-add α * v onto receivers.
    for edge_i in range(e):
        r = int(recv_idx[edge_i])
        contrib = weights[:, :, edge_i].unsqueeze(-1) * values[:, :, edge_i]
        mixed[:, :, r] = mixed[:, :, r] + contrib.to(dtype=mixed.dtype)

    message = torch.einsum("bhna,hda->bnd", mixed, w_o)
    return message[:, :n_real], message[:, n_real:]
