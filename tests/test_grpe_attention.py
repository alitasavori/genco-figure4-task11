"""Full Park GRPE (v11) and electrical GRPE (v11e) checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_tpgt_task11 as tpgt  # noqa: E402


def _activate_arch(name: str) -> None:
    os.environ["GENCO_TPGT_ARCH"] = name
    tpgt.apply_architecture_config()


def _restore_default_arch() -> None:
    os.environ.pop("GENCO_TPGT_ARCH", None)
    os.environ.pop("GENCO_TPGT_ATTENTION", None)
    os.environ.pop("GENCO_TPGT_GRPE_MAX_HOP", None)
    tpgt.D = 48
    tpgt.D_E = 24
    tpgt.D_A = 24
    tpgt.HEADS = 2
    tpgt.DEPTH = 4
    tpgt.RUN_TAG = "v7"
    tpgt.CHECKPOINT_VERSION = "meshed-rpe-2-d4-v7"
    tpgt.USE_NODE_SHUNT = False
    tpgt.ATTENTION_KIND = "grit"
    tpgt.ATTENTION_MLP_HIDDENS = (24, 12)
    tpgt.GRPE_MAX_HOP = 8
    tpgt.RUN_DIR_NAME = "tpgt_task11_v7"


def _line_branches(n: int = 4) -> list[tuple[int, int, float, float]]:
    return [(i, i + 1, 0.01, 0.1) for i in range(n - 1)]


def _tiny_layout(types: torch.Tensor, component: torch.Tensor | None = None):
    if component is None:
        component = torch.zeros(types.numel(), dtype=torch.long)
    masks = tpgt.type_pair_masks(types, component)
    return tpgt.shared_layout(types, masks), masks, component


def test_v11_attention_kind_is_full_grpe() -> None:
    try:
        _activate_arch("v11")
        assert tpgt.ATTENTION_KIND == "grpe"
        assert tpgt.HEADS == 2 and tpgt.DEPTH == 8 and tpgt.D_A == 48
        stage = tpgt.CorrectionStage()
        assert stage.attention_kind == "grpe"
        assert stage.u_q is None and stage.u_k is None and stage.u_v is None
        assert stage.score_w1 is None and stage.w_ew is None
        model = tpgt.TPGT()
        assert model.grpe_encodings is not None
        hop_bins = tpgt.grpe_num_hop_bins()
        assert model.grpe_encodings.p_q.shape == (hop_bins, tpgt.HEADS, tpgt.D_A)
        assert model.grpe_encodings.e_v.shape == (tpgt.GRPE_NUM_EDGE_TYPES, tpgt.HEADS, tpgt.D_A)
    finally:
        _restore_default_arch()


def test_v11e_has_stage_u_terms() -> None:
    try:
        _activate_arch("v11e")
        assert tpgt.ATTENTION_KIND == "grpe_elec"
        stage = tpgt.CorrectionStage()
        assert stage.u_q is not None and stage.u_k is not None and stage.u_v is not None
        assert stage.u_q.shape == (9, tpgt.HEADS, tpgt.D_A, tpgt.RPE_DIM)
        model = tpgt.TPGT()
        assert model.grpe_encodings is not None
        assert model.use_raw_rpe
    finally:
        _restore_default_arch()


def test_build_grpe_graph_ids_shapes_and_semantics() -> None:
    branches = _line_branches(4)
    hop, edge = tpgt.build_grpe_graph_ids(4, branches, max_hop=2)
    assert hop.shape == (4, 4) and edge.shape == (4, 4)
    assert hop[0, 0] == 0 and edge[0, 0] == tpgt.GRPE_EDGE_SELF
    assert edge[0, 1] == tpgt.GRPE_EDGE_BRANCH and edge[0, 2] == tpgt.GRPE_EDGE_NONADJ
    assert hop[0, 1] == 1 and hop[0, 2] == 2 and hop[0, 3] == 3  # clamped to overflow=3
    # Unreachable / overflow bin when max_hop=2: hop>2 → 3
    assert int(hop[0, 3]) == 3


def test_grpe_messages_finite_and_shaped() -> None:
    try:
        _activate_arch("v11")
        torch.manual_seed(0)
        buses = 12
        batch = 3
        types = torch.tensor([0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 0, 0], dtype=torch.long)
        component = torch.zeros(buses, dtype=torch.long)
        hidden = torch.randn(batch, buses, tpgt.D)
        pair = torch.randn(buses, buses, tpgt.RPE_DIM)
        hop, edge = tpgt.build_grpe_graph_ids(buses, _line_branches(buses))
        model = tpgt.TPGT()
        stage = model.stages[0]
        stage.eval()
        grpe = model.encode_grpe(torch.from_numpy(hop), torch.from_numpy(edge))
        blocks = tpgt.type_pair_blocks(types, tpgt.type_pair_masks(types, component))
        with torch.no_grad():
            out = stage._messages(hidden, pair, blocks, grpe=grpe)
        assert out.shape == hidden.shape
        assert torch.isfinite(out).all()
    finally:
        _restore_default_arch()


def test_grpe_shared_encoding_gradients() -> None:
    try:
        _activate_arch("v11")
        torch.manual_seed(1)
        buses = 9
        types = torch.tensor([0, 1, 2, 0, 1, 0, 0, 1, 0], dtype=torch.long)
        component = torch.zeros(buses, dtype=torch.long)
        hidden = torch.randn(2, buses, tpgt.D, requires_grad=True)
        pair = torch.randn(buses, buses, tpgt.RPE_DIM)
        hop, edge = tpgt.build_grpe_graph_ids(buses, _line_branches(buses))
        model = tpgt.TPGT()
        grpe = model.encode_grpe(torch.from_numpy(hop), torch.from_numpy(edge))
        stage = model.stages[0]
        blocks = tpgt.type_pair_blocks(types, tpgt.type_pair_masks(types, component))
        out = stage._messages(hidden, pair, blocks, grpe=grpe)
        out.square().sum().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert model.grpe_encodings is not None
        assert model.grpe_encodings.p_q.grad is not None
        assert torch.isfinite(model.grpe_encodings.p_q.grad).all()
        assert model.grpe_encodings.e_v.grad is not None
        assert stage.w_q.grad is not None and torch.isfinite(stage.w_q.grad).all()
        # Pure GRPE does not route grads through raw RPE pair.
        assert pair.grad is None
    finally:
        _restore_default_arch()


def test_grpe_elec_u_gradients() -> None:
    try:
        _activate_arch("v11e")
        torch.manual_seed(2)
        buses = 6
        types = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)
        component = torch.zeros(buses, dtype=torch.long)
        hidden = torch.randn(2, buses, tpgt.D, requires_grad=True)
        pair = torch.randn(buses, buses, tpgt.RPE_DIM, requires_grad=True)
        hop, edge = tpgt.build_grpe_graph_ids(buses, _line_branches(buses))
        model = tpgt.TPGT()
        grpe = model.encode_grpe(torch.from_numpy(hop), torch.from_numpy(edge))
        stage = model.stages[0]
        blocks = tpgt.type_pair_blocks(types, tpgt.type_pair_masks(types, component))
        out = stage._messages(hidden, pair, blocks, grpe=grpe)
        out.square().sum().backward()
        assert stage.u_q is not None and stage.u_q.grad is not None
        assert stage.u_v is not None and stage.u_v.grad is not None
        assert pair.grad is not None and torch.isfinite(pair.grad).all()
        assert model.grpe_encodings is not None
        assert model.grpe_encodings.p_k.grad is not None
    finally:
        _restore_default_arch()


def test_grpe_dense_reference_scores_and_values() -> None:
    """Hand-computed dense GRPE vs module for a tiny all-PQ/PV/slack graph."""
    try:
        # Small dims for a readable reference.
        os.environ["GENCO_TPGT_ARCH"] = "v11"
        tpgt.apply_architecture_config()
        tpgt.D = 4
        tpgt.D_A = 2
        tpgt.HEADS = 1
        tpgt.DEPTH = 1
        tpgt.ATTENTION_KIND = "grpe"
        torch.manual_seed(0)
        buses = 3
        types = torch.tensor([0, 1, 2], dtype=torch.long)
        component = torch.zeros(buses, dtype=torch.long)
        hidden = torch.randn(1, buses, tpgt.D)
        pair = torch.zeros(buses, buses, tpgt.RPE_DIM)
        hop = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.long)
        edge = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.long)
        model = tpgt.TPGT()
        assert model.grpe_encodings is not None
        # Deterministic embeddings / weights.
        with torch.no_grad():
            for name, table in (
                ("p_q", model.grpe_encodings.p_q),
                ("p_k", model.grpe_encodings.p_k),
                ("p_v", model.grpe_encodings.p_v),
                ("e_q", model.grpe_encodings.e_q),
                ("e_k", model.grpe_encodings.e_k),
                ("e_v", model.grpe_encodings.e_v),
            ):
                table.copy_(torch.arange(table.numel(), dtype=torch.float32).view_as(table) * 0.01)
            stage = model.stages[0]
            stage.w_q.copy_(torch.eye(tpgt.D_A, tpgt.D).unsqueeze(0).unsqueeze(0).expand_as(stage.w_q) * 0.5)
            stage.w_k.copy_(torch.eye(tpgt.D_A, tpgt.D).unsqueeze(0).unsqueeze(0).expand_as(stage.w_k) * 0.25)
            stage.w_v.copy_(torch.eye(tpgt.D_A, tpgt.D).unsqueeze(0).unsqueeze(0).expand_as(stage.w_v) * 0.125)
            stage.w_o.copy_(torch.eye(tpgt.D, tpgt.D_A).unsqueeze(0).expand_as(stage.w_o))

        grpe = model.encode_grpe(hop, edge)
        blocks = tpgt.type_pair_blocks(types, tpgt.type_pair_masks(types, component))
        stage = model.stages[0]
        out = stage._messages_grpe(hidden, pair, blocks, grpe)

        # Dense reference over all buses with type-pair weights (same as blocks).
        scale = float(tpgt.D_A) ** 0.5
        scores = torch.full((1, 1, buses, buses), float("-inf"))
        values = torch.zeros(1, 1, buses, buses, tpgt.D_A)
        for block in blocks:
            p = block.pair_id
            recv = block.receivers
            send = block.senders
            q = torch.einsum("bnd,had->bhna", hidden[:, recv], stage.w_q[p])
            k = torch.einsum("bnd,had->bhna", hidden[:, send], stage.w_k[p])
            v = torch.einsum("bnd,had->bhna", hidden[:, send], stage.w_v[p])
            pq = grpe.p_q[:, recv][:, :, send]
            pk = grpe.p_k[:, recv][:, :, send]
            pv = grpe.p_v[:, recv][:, :, send]
            eq = grpe.e_q[:, recv][:, :, send]
            ek = grpe.e_k[:, recv][:, :, send]
            ev = grpe.e_v[:, recv][:, :, send]
            node = torch.einsum("bhna,bhma->bhnm", q, k)
            q_t = torch.einsum("bhra,bhrsa->bhrs", q, (pq + eq).unsqueeze(0))
            k_t = torch.einsum("bhsa,bhrsa->bhrs", k, (pk + ek).unsqueeze(0))
            scores[:, :, recv.view(-1, 1), send.view(1, -1)] = (node + q_t + k_t) / scale
            values[:, :, recv.view(-1, 1), send.view(1, -1)] = (
                v.unsqueeze(-3) + pv.unsqueeze(0) + ev.unsqueeze(0)
            )
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        mixed = torch.einsum("bhnm,bhnma->bhna", weights, values)
        ref = torch.einsum("bhna,hda->bnd", mixed, stage.w_o)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    finally:
        _restore_default_arch()


def test_grpe_cross_component_mask_zero_weight() -> None:
    try:
        _activate_arch("v11")
        torch.manual_seed(3)
        buses = 4
        types = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        # Two components: {0,1} and {2,3}
        component = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        hidden = torch.randn(1, buses, tpgt.D)
        pair = torch.randn(buses, buses, tpgt.RPE_DIM)
        # Artificial fully-connected hop/edge so only masks silence cross-component.
        hop = torch.ones(buses, buses, dtype=torch.long)
        hop.fill_diagonal_(0)
        edge = torch.full((buses, buses), tpgt.GRPE_EDGE_BRANCH, dtype=torch.long)
        edge.fill_diagonal_(tpgt.GRPE_EDGE_SELF)
        model = tpgt.TPGT()
        grpe = model.encode_grpe(hop, edge)
        stage = model.stages[0]
        blocks = tpgt.type_pair_blocks(types, tpgt.type_pair_masks(types, component))
        # Probe scores via a forward that inspects softmax weights indirectly:
        # message from a receiver in component 0 should ignore senders in component 1.
        out = stage._messages(hidden, pair, blocks, grpe=grpe)
        assert torch.isfinite(out).all()
        # Explicit score scatter check.
        batch = 1
        scores = None
        for block in blocks:
            p = block.pair_id
            query = torch.einsum("bnd,had->bhna", hidden[:, block.receivers], stage.w_q[p])
            key = torch.einsum("bnd,had->bhna", hidden[:, block.senders], stage.w_k[p])
            pq = tpgt._gather_grpe_block(grpe.p_q, block.receivers, block.senders).unsqueeze(0)
            pk = tpgt._gather_grpe_block(grpe.p_k, block.receivers, block.senders).unsqueeze(0)
            eq = tpgt._gather_grpe_block(grpe.e_q, block.receivers, block.senders).unsqueeze(0)
            ek = tpgt._gather_grpe_block(grpe.e_k, block.receivers, block.senders).unsqueeze(0)
            scale = float(tpgt.D_A) ** 0.5
            score = (
                torch.einsum("bhna,bhma->bhnm", query, key)
                + torch.einsum("bhra,bhrsa->bhrs", query, pq + eq)
                + torch.einsum("bhsa,bhrsa->bhrs", key, pk + ek)
            ) / scale
            scores = stage._scatter_block_scores(scores, score, block, batch, buses)
        assert scores is not None
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        # Bus 0 (comp 0) must put zero mass on bus 2/3 (comp 1).
        assert float(weights[0, :, 0, 2].detach().abs().max()) == 0.0
        assert float(weights[0, :, 0, 3].detach().abs().max()) == 0.0
        # Bus 2 (comp 1) must put zero mass on bus 0/1.
        assert float(weights[0, :, 2, 0].detach().abs().max()) == 0.0
        assert float(weights[0, :, 2, 1].detach().abs().max()) == 0.0
    finally:
        _restore_default_arch()


def _closed_form_total(kind: str) -> int:
    """Closed-form param count matching ``scripts/count_tpgt_params.py`` formulas."""
    assert kind in ("grpe", "grpe_elec")

    def mlp(in_f: int, out_f: int, width: int) -> int:
        return in_f * width + width + width * width + width + width * out_f + out_f

    hop_bins = tpgt.grpe_num_hop_bins()
    per_hop = hop_bins * tpgt.HEADS * tpgt.D_A
    per_edge = tpgt.GRPE_NUM_EDGE_TYPES * tpgt.HEADS * tpgt.D_A
    shared = 3 * per_hop + 3 * per_edge
    attn = 9 * tpgt.HEADS * (3 * tpgt.D_A * tpgt.D) + tpgt.HEADS * tpgt.D * tpgt.D_A
    if kind == "grpe_elec":
        attn += 9 * tpgt.HEADS * 3 * tpgt.D_A * tpgt.RPE_DIM
    update = 3 * mlp(tpgt.D + tpgt.D + 2, tpgt.D, tpgt.D)
    decode = 3 * mlp(tpgt.D, 2, tpgt.D)
    ln = 2 * tpgt.D
    node_in = tpgt.NODE_FEATURE_DIM if tpgt.USE_NODE_SHUNT else tpgt.NODE_SPECIFIED_DIM
    enc = 3 * mlp(node_in, tpgt.D, tpgt.D)
    return enc + shared + tpgt.DEPTH * (attn + update + decode + ln)


def test_v11_param_count_matches_closed_form() -> None:
    try:
        _activate_arch("v11")
        model = tpgt.TPGT()
        total = sum(p.numel() for p in model.parameters())
        assert total == _closed_form_total("grpe")
    finally:
        _restore_default_arch()


def test_v11e_param_count_matches_closed_form() -> None:
    try:
        _activate_arch("v11e")
        model = tpgt.TPGT()
        total = sum(p.numel() for p in model.parameters())
        assert total == _closed_form_total("grpe_elec")
    finally:
        _restore_default_arch()


def test_v11_model_uses_raw_rpe_identity_and_grpe_module() -> None:
    try:
        _activate_arch("v11")
        model = tpgt.TPGT()
        assert model.use_raw_rpe
        assert isinstance(model.pair_encoder, torch.nn.Identity)
        rpe = torch.randn(4, 4, tpgt.RPE_DIM)
        assert torch.equal(model.encode_pairs(rpe), rpe)
        assert model.grpe_encodings is not None
        assert model.stages[0].attention_kind == "grpe"
        assert model.stages[0].u_q is None
    finally:
        _restore_default_arch()
