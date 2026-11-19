"""JSGC-Net: type-pair Sparse GRIT (physical edges) + Jacobian coarse global correction.

Retains TPGT's type-pair GRIT parameterization (9 directed PQ/PV/Slack pairs with
pair-conditioned Q/K/V, W_Ew/W_Eb/w_A/W_Ev). Attention connectivity is restricted to
physical Ybus neighbors; long-range coupling comes from the coarse Jacobian correction.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from jsgc_physics import (
    TYPE_PQ,
    TYPE_PV,
    TYPE_SLACK,
    TopologyCache,
    ac_power_injections,
    coarse_correct,
    jacobi_solve,
    prepare_topology,
    reduced_ac_residual,
    reduced_to_tpgt_unknown,
    ybus_to_gb,
)

# Defaults from the JSGC-Net v18 spec.
JSGC_K = 4
JSGC_L = 2
JSGC_D = 64
JSGC_H = 2
JSGC_DA = 32
JSGC_DE = 32
JSGC_R_THETA = 8
JSGC_R_V = 8
NUM_TYPE_PAIRS = 9  # directed (t_recv, t_send); pair_id = t_recv * 3 + t_send


def rho_nonlinearity(values: torch.Tensor) -> torch.Tensor:
    """TPGT GRIT ρ: √ReLU(z) − √ReLU(−z)."""
    return torch.sqrt(torch.relu(values)) - torch.sqrt(torch.relu(-values))


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or max(d_in, d_out)
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.GELU(),
            nn.Linear(h, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _softmax_per_receiver(scores: torch.Tensor, recv: torch.Tensor, n_buses: int) -> torch.Tensor:
    """Softmax scores (B, H, E) over physical neighbors of each receiver."""
    # Numerically stable: subtract max per (batch, head, receiver), then exp / sum.
    batch, heads, n_edges = scores.shape
    if n_edges == 0:
        return scores
    # Scatter max
    neg_inf = scores.new_full((batch, heads, n_buses), float("-inf"))
    # index_reduce_amax if available; fallback loop-free via unique groups
    flat_recv = recv.long()
    # Use a large negative floor so empty receivers stay -inf
    max_buf = neg_inf.clone()
    # Expand receiver index for batched scatter
    # scores.transpose → (B*H, E); recv shared
    bh = batch * heads
    scores_flat = scores.reshape(bh, n_edges)
    max_flat = scores_flat.new_full((bh, n_buses), float("-inf"))
    max_flat.scatter_reduce_(1, flat_recv.unsqueeze(0).expand(bh, -1), scores_flat, reduce="amax", include_self=True)
    score_max = max_flat.gather(1, flat_recv.unsqueeze(0).expand(bh, -1))
    exp_scores = torch.exp(scores_flat - score_max)
    sum_flat = scores_flat.new_zeros((bh, n_buses))
    sum_flat.scatter_add_(1, flat_recv.unsqueeze(0).expand(bh, -1), exp_scores)
    denom = sum_flat.gather(1, flat_recv.unsqueeze(0).expand(bh, -1)).clamp(min=1e-12)
    alpha = (exp_scores / denom).reshape(batch, heads, n_edges)
    return torch.nan_to_num(alpha, nan=0.0)


class TypePairSparseGritLayer(nn.Module):
    """One sparse type-pair GRIT layer over directed physical edges.

    Parameter layout matches TPGT ``CorrectionStage`` GRIT (9 type-pairs × heads):
    ``W_Q/K/V[p]``, ``W_Ew/Eb[p]``, ``w_A[p]``, ``W_Ev[p]`` with
    ``p = t_recv * 3 + t_send``. Softmax is over physical neighbors only.
    """

    def __init__(self, d: int = JSGC_D, d_a: int = JSGC_DA, d_e: int = JSGC_DE, heads: int = JSGC_H) -> None:
        super().__init__()
        self.d = d
        self.d_a = d_a
        self.d_e = d_e
        self.heads = heads
        shape = (NUM_TYPE_PAIRS, heads)
        self.w_q = nn.Parameter(torch.empty(*shape, d_a, d))
        self.w_k = nn.Parameter(torch.empty(*shape, d_a, d))
        self.w_v = nn.Parameter(torch.empty(*shape, d_a, d))
        self.w_ew = nn.Parameter(torch.empty(*shape, d_a, d_e))
        self.w_eb = nn.Parameter(torch.empty(*shape, d_a, d_e))
        self.w_a = nn.Parameter(torch.empty(*shape, 1, d_a))
        self.w_ev = nn.Parameter(torch.empty(*shape, d_a, d_a))
        self.w_o = nn.Parameter(torch.empty(heads, d, d_a))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = MLP(d, d, hidden=2 * d)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.w_q, self.w_k, self.w_v, self.w_ew, self.w_eb, self.w_ev, self.w_o):
            nn.init.xavier_uniform_(p)
        nn.init.xavier_uniform_(self.w_a)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        types: torch.Tensor,
    ) -> torch.Tensor:
        """``h``: (B, N, D); ``edge_index``: (2, E); ``edge_attr``: (E, D_E) or (B, E, D_E); ``types``: (N,)."""
        batch, n, _ = h.shape
        if edge_index.numel() == 0:
            a = self.norm1(h)
            return self.norm2(a + self.ffn(a))

        recv, send = edge_index[0].long(), edge_index[1].long()
        types = types.long().view(-1)
        pair_ids = types[recv] * 3 + types[send]  # (E,)
        e = edge_attr
        if e.dim() == 2:
            e = e.unsqueeze(0).expand(batch, -1, -1)

        # Pass 1: type-pair scores + interactions on physical edges.
        scores = h.new_zeros(batch, self.heads, recv.numel())
        # Store ξ and V-projected sender features per edge for the message pass.
        xi_edges = h.new_zeros(batch, self.heads, recv.numel(), self.d_a)
        v_edges = h.new_zeros(batch, self.heads, recv.numel(), self.d_a)

        for p in range(NUM_TYPE_PAIRS):
            mask = pair_ids == p
            if not bool(mask.any()):
                continue
            e_idx = torch.nonzero(mask, as_tuple=False).view(-1)
            r = recv[e_idx]
            s = send[e_idx]
            e_p = e[:, e_idx, :]  # (B, Ep, De)

            query = torch.einsum("bnd,had->bhna", h[:, r], self.w_q[p])
            key = torch.einsum("bnd,had->bhna", h[:, s], self.w_k[p])
            mixed = rho_nonlinearity(query + key)
            projected = torch.einsum("bed,had->bhea", e_p, self.w_ew[p])
            bias = torch.einsum("bed,had->bhea", e_p, self.w_eb[p])
            interaction = torch.relu(mixed * projected + bias)
            score = torch.einsum("bhea,ha->bhe", interaction, self.w_a[p, :, 0])
            scores[:, :, e_idx] = score
            xi_edges[:, :, e_idx, :] = interaction
            v_edges[:, :, e_idx, :] = torch.einsum("bnd,had->bhna", h[:, s], self.w_v[p])

        alpha = _softmax_per_receiver(scores, recv, n)

        # Pass 2: messages with type-pair W_Ev (aggregate-then-project, matching TPGT).
        mixed_heads = h.new_zeros(batch, self.heads, n, self.d_a)
        for p in range(NUM_TYPE_PAIRS):
            mask = pair_ids == p
            if not bool(mask.any()):
                continue
            e_idx = torch.nonzero(mask, as_tuple=False).view(-1)
            r = recv[e_idx]
            a_p = alpha[:, :, e_idx]  # (B,H,Ep)
            v_p = v_edges[:, :, e_idx, :]
            xi_p = xi_edges[:, :, e_idx, :]

            msg_v = h.new_zeros(batch, self.heads, n, self.d_a)
            msg_xi = h.new_zeros(batch, self.heads, n, self.d_a)
            msg_v.index_add_(2, r, a_p.unsqueeze(-1) * v_p)
            msg_xi.index_add_(2, r, a_p.unsqueeze(-1) * xi_p)
            gathered = msg_v + torch.einsum("bhna,hxa->bhnx", msg_xi, self.w_ev[p])
            mixed_heads = mixed_heads + gathered

        # Output projection matches TPGT: einsum over heads with W_O (H, D, D_A).
        out = torch.einsum("bhna,hda->bnd", mixed_heads, self.w_o)
        a = self.norm1(h + out)
        return self.norm2(a + self.ffn(a))


class TypePairSparseGritStack(nn.Module):
    def __init__(
        self,
        d: int = JSGC_D,
        d_a: int = JSGC_DA,
        d_e: int = JSGC_DE,
        heads: int = JSGC_H,
        layers: int = JSGC_L,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TypePairSparseGritLayer(d=d, d_a=d_a, d_e=d_e, heads=heads) for _ in range(layers)
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        types: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, types)
        return h

# Back-compat aliases
SparseGritLayer = TypePairSparseGritLayer
SparseGritStack = TypePairSparseGritStack


class JSGCNet(nn.Module):
    """Jacobian-Guided Sparse–Global Correction Network (ARCH v18).

    Local pathway: **type-pair Sparse GRIT** on physical edges (same GRIT mechanism as
    TPGT v16, connectivity restricted to Ybus neighbors). Global pathway: coarse
    Jacobian correction. No distillation.

    Returns TPGT-compatible ``(unknown_prediction, residuals_list)`` where each residual
    is a full-bus (B, N, 2) P/Q mismatch for PBL / physics loss.
    """

    def __init__(
        self,
        *,
        d: int = JSGC_D,
        d_a: int = JSGC_DA,
        d_e: int = JSGC_DE,
        heads: int = JSGC_H,
        layers: int = JSGC_L,
        k_iter: int = JSGC_K,
        r_theta: int = JSGC_R_THETA,
        r_v: int = JSGC_R_V,
        beta: float = 0.1,
        gate_c: float = 1.0,
        use_learned_local: bool = True,
    ) -> None:
        super().__init__()
        self.d = d
        self.d_e = d_e
        self.k_iter = k_iter
        self.r_theta = r_theta
        self.r_v = r_v
        self.beta = beta
        self.gate_c = gate_c
        self.use_learned_local = use_learned_local
        # Node feature: P,Q,V,θ,rP,rQ,Vset,Gii,Bii (type via per-type encoders) = 9
        self.node_in = 9
        self.encoders = nn.ModuleList(MLP(self.node_in, d) for _ in range(3))
        self.edge_enc = MLP(7, d_e)
        self.gnn = TypePairSparseGritStack(d=d, d_a=d_a, d_e=d_e, heads=heads, layers=layers)
        self.head_theta = MLP(d, 1)
        self.head_v = MLP(d, 1)
        self.eta_param = nn.Parameter(torch.tensor(2.0))  # σ(2)≈0.88
        self._cache: dict[Any, TopologyCache] = {}

    def clear_topology_cache(self) -> None:
        self._cache.clear()

    def encode_pairs(self, rpe: torch.Tensor) -> torch.Tensor:
        """TPGT compatibility stub — JSGC does not use dense pair embeddings."""
        return rpe

    def clear_grit_elec_proj_caches(self) -> None:
        return None

    @property
    def stages(self) -> nn.ModuleList:
        """Empty stand-in so DEPTH-based diagnostics that inspect ``model.stages`` do not crash."""
        if not hasattr(self, "_stages_stub"):
            self._stages_stub = nn.ModuleList()
        return self._stages_stub

    def _topology_key(self, ybus: torch.Tensor, types: torch.Tensor) -> bytes:
        y = ybus.detach().cpu()
        t = types.detach().cpu().numpy().astype("int64").tobytes()
        return y.shape[0].to_bytes(4, "little") + t + y.real.numpy().tobytes()[:64]

    def get_cache(self, ybus: torch.Tensor, types: torch.Tensor) -> TopologyCache:
        key = self._topology_key(ybus, types)
        hit = self._cache.get(key)
        if hit is None:
            hit = prepare_topology(
                ybus.detach(),
                types.detach(),
                r_theta=self.r_theta,
                r_v=self.r_v,
            )
            # Move cache tensors to model device
            device = next(self.parameters()).device
            hit = prepare_topology(
                ybus.to(device=device),
                types.to(device=device),
                r_theta=self.r_theta,
                r_v=self.r_v,
            )
            self._cache[key] = hit
        return hit

    def _edge_features(self, ybus: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Build (E, 7) edge features from Ybus."""
        g, b = ybus_to_gb(ybus)
        recv, send = edge_index[0], edge_index[1]
        y_ij = ybus.to(torch.complex128)[recv, send]
        # series approx from off-diag; charging/tap defaults
        feat = torch.stack(
            (
                y_ij.real,
                y_ij.imag,
                -y_ij.real,  # series Re approx
                -y_ij.imag,
                torch.zeros_like(y_ij.real),  # charging
                torch.ones_like(y_ij.real),  # |tap|
                torch.zeros_like(y_ij.real),  # angle tap
            ),
            dim=-1,
        ).to(dtype=torch.float32)
        return feat

    def _node_features(
        self,
        *,
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        theta: torch.Tensor,
        v: torch.Tensor,
        r_full: torch.Tensor,
        types: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
        v_set: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N, 9) operating features; bus type handled by per-type encoders."""
        batch, n = theta.shape
        del n
        gii = g.diag().to(dtype=theta.dtype).unsqueeze(0).expand(batch, -1)
        bii = b.diag().to(dtype=theta.dtype).unsqueeze(0).expand(batch, -1)
        r_p = r_full[..., 0]
        r_q = r_full[..., 1]
        pq = types == TYPE_PQ
        pv = types == TYPE_PV
        slack = types == TYPE_SLACK
        r_p = torch.where(slack.unsqueeze(0), torch.zeros_like(r_p), r_p)
        r_q = torch.where((slack | pv).unsqueeze(0), torch.zeros_like(r_q), r_q)
        p_feat = torch.where(slack.unsqueeze(0), torch.zeros_like(p_spec), p_spec)
        q_feat = torch.where((slack | pv).unsqueeze(0), torch.zeros_like(q_spec), q_spec)
        return torch.stack(
            (
                p_feat,
                q_feat,
                v,
                theta,
                r_p,
                r_q,
                v_set.unsqueeze(0).expand(batch, -1),
                gii,
                bii,
            ),
            dim=-1,
        )

    def _encode_nodes(self, feats: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
        """Per-type node encoders (TPGT-style), ``feats`` (B, N, F)."""
        batch, n, _ = feats.shape
        h = feats.new_zeros(batch, n, self.d)
        types = types.long().view(-1)
        for type_id, enc in enumerate(self.encoders):
            buses = types == type_id
            if not bool(buses.any()):
                continue
            h[:, buses] = enc(feats[:, buses])
        return h

    def _full_bus_residual(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        p_spec: torch.Tensor,
        q_spec: torch.Tensor,
        g: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        p_calc, q_calc = ac_power_injections(theta, v, g, b)
        return torch.stack((p_spec - p_calc, q_spec - q_calc), dim=-1)

    def forward(
        self,
        specified: torch.Tensor,
        types: torch.Tensor,
        rpe: torch.Tensor | None = None,
        pair_type_masks: torch.Tensor | None = None,
        ybus: torch.Tensor | None = None,
        pair: torch.Tensor | None = None,
        layout: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """TPGT-compatible forward.

        ``specified``: (B, N, 2) packed as in train_tpgt_task11.
        Returns ``(prediction_unknown, residuals)`` with residuals (B,N,2) per iteration.
        """
        del rpe, pair_type_masks, pair, kwargs  # unused (dense-GRIT path args)
        import sys

        # Prefer the aliased training module (``__main__`` under %run).
        _tpgt = sys.modules.get("train_tpgt_task11") or sys.modules.get("__main__")
        if _tpgt is None or not hasattr(_tpgt, "assemble"):
            import train_tpgt_task11 as _tpgt  # type: ignore[no-redef]
        assemble = _tpgt.assemble
        flat_start = _tpgt.flat_start
        type_layout = _tpgt.type_layout

        if ybus is None:
            raise ValueError("JSGCNet.forward requires ybus")
        if layout is None:
            if torch.is_tensor(types):
                layout = type_layout(types if types.dim() == 1 else types[0])
            else:
                layout = types  # already a layout; do not re-wrap across module copies
        types_tensor = getattr(layout, "types", types)
        if not torch.is_tensor(types_tensor):
            raise TypeError("JSGCNet.forward could not resolve bus-type tensor")
        types_1d = types_tensor if types_tensor.dim() == 1 else types_tensor[0]
        cache = self.get_cache(ybus if ybus.dim() == 2 else ybus[0], types_1d)
        device = specified.device
        dtype = specified.dtype

        # Unpack specified into P,Q,V,θ using flat-start unknowns for free slots
        p0, q0, v_ass, th_ass = assemble(specified, flat_start(specified, layout), layout)
        # Spec injections: for residual use assembled P,Q at flat; V setpoints from specified
        p_spec = p0.to(dtype=torch.float64)
        q_spec = q0.to(dtype=torch.float64)
        # PV/Slack voltage setpoints
        v_set = cache.v0.to(device=device)
        if (types_1d == TYPE_PV).any() or (types_1d == TYPE_SLACK).any():
            # Prefer assembled voltages for set buses
            v_set = v_ass[0].to(dtype=torch.float64) if v_ass.shape[0] else cache.v0

        batch = specified.shape[0]
        theta = cache.theta0.to(device=device).unsqueeze(0).expand(batch, -1).clone()
        v = cache.v0.to(device=device).unsqueeze(0).expand(batch, -1).clone()
        # Init PQ voltages to 1, keep PV/slack setpoints from assemble
        pq = types_1d == TYPE_PQ
        pv = types_1d == TYPE_PV
        slack = types_1d == TYPE_SLACK
        v = v.clone()
        v[:, pq] = 1.0
        if pv.any():
            v[:, pv] = v_ass[:, pv].to(dtype=v.dtype)
        if slack.any():
            v[:, slack] = v_ass[:, slack].to(dtype=v.dtype)
            theta = theta.clone()
            theta[:, slack] = th_ass[:, slack].to(dtype=theta.dtype)

        edge_index = cache.edge_index.to(device=device)
        edge_feat = self._edge_features(ybus if ybus.dim() == 2 else ybus[0], edge_index)
        edge_attr = self.edge_enc(edge_feat.to(dtype=dtype))

        g = cache.g.to(device=device)
        b = cache.b.to(device=device)
        residuals: list[torch.Tensor] = []
        eta = torch.sigmoid(self.eta_param).to(dtype=torch.float64)

        for _t in range(self.k_iter):
            r_full = self._full_bus_residual(theta, v, p_spec, q_spec, g, b)
            r_red = reduced_ac_residual(
                theta, v, g, b, p_spec, q_spec, cache.index
            )  # (B, nx)

            z_j = jacobi_solve(r_red, cache.d_blocks, cache.index)

            if self.use_learned_local:
                feats = self._node_features(
                    p_spec=p_spec.to(dtype=dtype),
                    q_spec=q_spec.to(dtype=dtype),
                    theta=theta.to(dtype=dtype),
                    v=v.to(dtype=dtype),
                    r_full=r_full.to(dtype=dtype),
                    types=types_1d,
                    g=g,
                    b=b,
                    v_set=v_set.to(dtype=dtype),
                )
                h = self._encode_nodes(feats, types_1d)
                h = self.gnn(h, edge_index, edge_attr, types_1d)
                g_theta = self.head_theta(h).squeeze(-1)  # (B,N)
                g_v = self.head_v(h).squeeze(-1)
                g_psi = torch.cat(
                    (
                        g_theta[:, cache.index.non_slack],
                        g_v[:, cache.index.pq] if cache.index.n_v else g_theta.new_zeros(batch, 0),
                    ),
                    dim=-1,
                ).to(dtype=torch.float64)
                a_t = torch.linalg.vector_norm(r_red, dim=-1) / math.sqrt(max(cache.index.n_x, 1))
                gamma = (a_t / (a_t + self.gate_c)).unsqueeze(-1)
                z_local = z_j + self.beta * gamma * g_psi
            else:
                z_local = z_j

            delta = coarse_correct(
                r_red, z_local, cache.j0.to(device=device), cache.p.to(device=device),
                cache.lu.to(device=device), cache.pivots.to(device=device),
            )
            delta = eta * delta
            if cache.index.n_theta:
                theta = theta.clone()
                theta[:, cache.index.non_slack] = (
                    theta[:, cache.index.non_slack] + delta[:, : cache.index.n_theta]
                )
            if cache.index.n_v:
                v = v.clone()
                v[:, cache.index.pq] = (
                    v[:, cache.index.pq] + delta[:, cache.index.n_theta :]
                ).clamp(0.05, 2.0)
            # One residual per correction iteration (matches TPGT DEPTH staging).
            residuals.append(
                self._full_bus_residual(theta, v, p_spec, q_spec, g, b).to(dtype=dtype)
            )

        p_calc, q_calc = ac_power_injections(theta, v, g, b)
        prediction = reduced_to_tpgt_unknown(
            theta, v, p_calc=p_calc, q_calc=q_calc, types=types_1d
        ).to(dtype=dtype)
        return prediction, residuals


def build_jsgc_model(**kwargs: Any) -> JSGCNet:
    return JSGCNet(**kwargs)
