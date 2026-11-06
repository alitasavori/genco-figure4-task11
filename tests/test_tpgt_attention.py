"""Numerical checks for TPGT attention (no dataset required)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import (  # noqa: E402
    D,
    D_E,
    HEADS,
    CorrectionStage,
    rho_nonlinearity,
    type_pair_blocks,
    type_pair_masks,
)


def reference_messages(
    stage: CorrectionStage,
    hidden: torch.Tensor,
    pair: torch.Tensor,
    types: torch.Tensor,
    component: torch.Tensor,
) -> torch.Tensor:
    """Pre-batching loop semantics (3375d55 style)."""
    batch, buses, _ = hidden.shape
    scores = hidden.new_full((batch, HEADS, buses, buses), float("-inf"))
    d_a = stage.w_a.shape[-1]
    values = hidden.new_zeros((batch, HEADS, buses, buses, d_a))
    same_component = component.view(buses, 1) == component.view(1, buses)
    for pair_id in range(9):
        receiver, sender = divmod(pair_id, 3)
        type_mask = (types.view(buses, 1) == receiver) & (types.view(1, buses) == sender)
        mask = type_mask & same_component
        if not bool(mask.any()):
            continue
        for head in range(HEADS):
            query = hidden @ stage.w_q[pair_id, head].T
            key = hidden @ stage.w_k[pair_id, head].T
            mixed = torch.sqrt(torch.relu(query[:, :, None, :] + key[:, None, :, :])) - torch.sqrt(
                torch.relu(-(query[:, :, None, :] + key[:, None, :, :]))
            )
            projected = pair @ stage.w_ew[pair_id, head].T
            bias = pair @ stage.w_eb[pair_id, head].T
            interaction = torch.relu(mixed * projected.unsqueeze(0) + bias.unsqueeze(0))
            score = (interaction * stage.w_a[pair_id, head]).sum(dim=-1)
            scores[:, head] = torch.where(mask, score, scores[:, head])
            value = hidden @ stage.w_v[pair_id, head].T
            content = value[:, None, :, :] + interaction @ stage.w_ev[pair_id, head].T
            values[:, head] = torch.where(mask.unsqueeze(-1), content, values[:, head])
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    row_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)
    mixed_heads = (weights.unsqueeze(-1) * values).sum(dim=3)
    return torch.einsum("bhnd,hod->bno", mixed_heads, stage.w_o)


def dense_reference_messages(
    stage: CorrectionStage,
    hidden: torch.Tensor,
    pair: torch.Tensor,
    pair_type_masks: torch.Tensor,
) -> torch.Tensor:
    """Previous ``_messages`` (93662ca): all nine weight sets on all N x N pairs, then mask-select."""
    batch, buses, _ = hidden.shape
    mask_view = pair_type_masks.view(1, 9, 1, buses, buses)
    query = torch.einsum("bnd,phad->bphna", hidden, stage.w_q)
    key = torch.einsum("bnd,phad->bphna", hidden, stage.w_k)
    mixed = rho_nonlinearity(query.unsqueeze(-2) + key.unsqueeze(-3))
    projected = torch.einsum("nme,phae->phnma", pair, stage.w_ew)
    bias = torch.einsum("nme,phae->phnma", pair, stage.w_eb)
    interaction = torch.relu(mixed * projected.unsqueeze(0) + bias.unsqueeze(0))
    all_scores = (interaction * stage.w_a.unsqueeze(0).unsqueeze(-2)).sum(dim=-1)
    scores = all_scores.masked_fill(~mask_view, float("-inf")).amax(dim=1)
    value = torch.einsum("bnd,phad->bphna", hidden, stage.w_v)
    content = value.unsqueeze(3) + torch.einsum("bphnma,phxa->bphnmx", interaction, stage.w_ev)
    values = (content * mask_view.unsqueeze(-1)).sum(dim=1)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    row_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)
    mixed_heads = (weights.unsqueeze(-1) * values).sum(dim=3)
    return torch.einsum("bhna,hda->bnd", mixed_heads, stage.w_o)


def random_stage() -> CorrectionStage:
    stage = CorrectionStage()
    with torch.no_grad():
        for parameter in (stage.w_q, stage.w_k, stage.w_v, stage.w_ew, stage.w_eb, stage.w_a, stage.w_ev, stage.w_o):
            parameter.normal_(0.0, 0.3)
    stage.eval()
    return stage


def block_messages(stage: CorrectionStage, hidden, pair, types, component) -> torch.Tensor:
    return stage._messages(hidden, pair, type_pair_blocks(types, type_pair_masks(types, component)))


def test_batched_messages_match_reference() -> None:
    torch.manual_seed(0)
    buses = 12
    batch = 4
    types = torch.tensor([0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 0, 0], dtype=torch.long)
    component = torch.zeros(buses, dtype=torch.long)
    hidden = torch.randn(batch, buses, D)
    pair = torch.randn(buses, buses, D_E)
    stage = CorrectionStage()
    stage.eval()
    with torch.no_grad():
        ref = reference_messages(stage, hidden, pair, types, component)
        batched = block_messages(stage, hidden, pair, types, component)
    max_diff = (ref - batched).abs().max().item()
    assert max_diff < 1e-4, f"batched attention differs from reference by {max_diff}"


def test_block_messages_match_dense_two_components() -> None:
    torch.manual_seed(1)
    types = torch.tensor([0, 1, 2, 0, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0], dtype=torch.long)
    component = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    buses = types.numel()
    hidden = torch.randn(3, buses, D)
    pair = torch.randn(buses, buses, D_E)
    stage = random_stage()
    masks = type_pair_masks(types, component)
    with torch.no_grad():
        dense = dense_reference_messages(stage, hidden, pair, masks)
        block = block_messages(stage, hidden, pair, types, component)
        loop = reference_messages(stage, hidden, pair, types, component)
    assert torch.allclose(block, dense, rtol=1e-4, atol=1e-5), (block - dense).abs().max()
    assert torch.allclose(block, loop, rtol=1e-4, atol=1e-5), (block - loop).abs().max()


def test_block_messages_match_dense_118_bus() -> None:
    torch.manual_seed(2)
    buses = 118
    types = torch.zeros(buses, dtype=torch.long)
    types[torch.randperm(buses)[:53]] = 1
    types[7] = 2
    component = torch.zeros(buses, dtype=torch.long)
    component[torch.randperm(buses)[:5]] = 1
    hidden = torch.randn(2, buses, D)
    pair = torch.randn(buses, buses, D_E)
    stage = random_stage()
    masks = type_pair_masks(types, component)
    with torch.no_grad():
        dense = dense_reference_messages(stage, hidden, pair, masks)
        block = block_messages(stage, hidden, pair, types, component)
    assert torch.allclose(block, dense, rtol=1e-4, atol=1e-5), (block - dense).abs().max()


def test_block_messages_gradients_match_dense() -> None:
    torch.manual_seed(3)
    types = torch.tensor([0, 1, 2, 0, 1, 0, 0, 1, 0], dtype=torch.long)
    component = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    buses = types.numel()
    # float64: rho's sqrt amplifies fp32 rounding in the q/k gradients near zero.
    stage = random_stage().double()
    masks = type_pair_masks(types, component)
    hidden = torch.randn(2, buses, D, dtype=torch.float64)
    pair = torch.randn(buses, buses, D_E, dtype=torch.float64)
    grads = []
    for compute in (
        lambda: dense_reference_messages(stage, hidden, pair, masks),
        lambda: block_messages(stage, hidden, pair, types, component),
    ):
        stage.zero_grad()
        compute().square().sum().backward()
        grads.append({name: param.grad.clone() for name, param in stage.named_parameters() if param.grad is not None})
    dense_grads, block_grads = grads
    for name in ("w_q", "w_k", "w_v", "w_ew", "w_eb", "w_a", "w_ev", "w_o"):
        assert torch.allclose(block_grads[name], dense_grads[name], rtol=1e-9, atol=1e-9), name


def test_type_pair_blocks_partition_same_component_pairs() -> None:
    types = torch.tensor([0, 1, 2, 0, 1, 0], dtype=torch.long)
    component = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    masks = type_pair_masks(types, component)
    covered = torch.zeros(6, 6, dtype=torch.long)
    for block in type_pair_blocks(types, masks):
        receiver, sender = divmod(block.pair_id, 3)
        assert bool((types[block.receivers] == receiver).all())
        assert bool((types[block.senders] == sender).all())
        covered[block.receivers.view(-1, 1), block.senders.view(1, -1)] += block.mask.long()
    same_component = component.view(-1, 1) == component.view(1, -1)
    assert torch.equal(covered, same_component.long())


if __name__ == "__main__":
    test_batched_messages_match_reference()
    print("test_tpgt_attention OK")
