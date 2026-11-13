"""Adam checkpoint buffers must match live parameter shapes on resume."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import optim

from train_tpgt_task11 import (
    live_optimizer_state_shape_mismatch,
    optimizer_state_shape_mismatch,
    reset_optimizer_state,
)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(9, 4, 48, 96))
        self.v = nn.Parameter(torch.randn(9, 4, 96, 48))


def test_optimizer_state_shape_mismatch_detects_swapped_buffers() -> None:
    model = _Toy()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    (model.w.sum() + model.v.sum()).backward()
    optimizer.step()
    payload = optimizer.state_dict()
    # Swap serialized momentum for the two params (group order).
    state = payload["state"]
    ids = list(payload["param_groups"][0]["params"])
    state[ids[0]]["exp_avg"], state[ids[1]]["exp_avg"] = (
        state[ids[1]]["exp_avg"].clone(),
        state[ids[0]]["exp_avg"].clone(),
    )
    fresh = optim.Adam(model.parameters(), lr=1e-3)
    reason = optimizer_state_shape_mismatch(fresh, payload)
    assert reason is not None
    assert "exp_avg shape" in reason


def test_reset_optimizer_state_clears_momentum() -> None:
    model = _Toy()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.w.sum().backward()
    optimizer.step()
    assert optimizer.state
    reset_optimizer_state(optimizer)
    assert live_optimizer_state_shape_mismatch(optimizer) is None
    assert not optimizer.state
