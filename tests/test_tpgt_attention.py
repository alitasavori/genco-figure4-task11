"""Numerical checks for TPGT attention (no dataset required)."""

from __future__ import annotations

import math
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
    masks = type_pair_masks(types, component)
    with torch.no_grad():
        ref = reference_messages(stage, hidden, pair, types, component)
        batched = stage._messages(hidden, pair, masks)
    max_diff = (ref - batched).abs().max().item()
    assert max_diff < 1e-4, f"batched attention differs from reference by {max_diff}"


if __name__ == "__main__":
    test_batched_messages_match_reference()
    print("test_tpgt_attention OK")
