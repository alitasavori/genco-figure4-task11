"""Joint MLP attention (v10 / Approach C) smoke and shape checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_tpgt_task11 as tpgt  # noqa: E402


def _activate_v10() -> None:
    os.environ["GENCO_TPGT_ARCH"] = "v10"
    tpgt.apply_architecture_config()


def _restore_default_arch() -> None:
    os.environ.pop("GENCO_TPGT_ARCH", None)
    # Reset module defaults to grit v7-like dims used by other unit tests.
    tpgt.D = 48
    tpgt.D_E = 24
    tpgt.D_A = 24
    tpgt.HEADS = 2
    tpgt.DEPTH = 4
    tpgt.RUN_TAG = "v7"
    tpgt.CHECKPOINT_VERSION = "meshed-rpe-2-d4-v7"
    tpgt.USE_NODE_SHUNT = False
    tpgt.ATTENTION_KIND = "grit"
    tpgt.ATTENTION_MLP_HIDDEN = 24
    tpgt.RUN_DIR_NAME = "tpgt_task11_v7"


def test_v10_attention_kind_is_joint_mlp() -> None:
    try:
        _activate_v10()
        assert tpgt.ATTENTION_KIND == "joint_mlp"
        stage = tpgt.CorrectionStage()
        assert stage.attention_kind == "joint_mlp"
        assert stage.score_w1 is not None
        assert stage.score_w1.shape[0] == 9 and stage.score_w1.shape[1] == tpgt.HEADS
        assert stage.score_w1.shape[-1] == 2 * tpgt.D_A + tpgt.RPE_DIM
        assert stage.w_ew is None
    finally:
        _restore_default_arch()


def test_joint_mlp_messages_finite_and_shaped() -> None:
    try:
        _activate_v10()
        torch.manual_seed(0)
        buses = 12
        batch = 3
        types = torch.tensor([0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 0, 0], dtype=torch.long)
        component = torch.zeros(buses, dtype=torch.long)
        hidden = torch.randn(batch, buses, tpgt.D)
        pair = torch.randn(buses, buses, tpgt.RPE_DIM)
        stage = tpgt.CorrectionStage()
        stage.eval()
        blocks = tpgt.type_pair_blocks(types, tpgt.type_pair_masks(types, component))
        with torch.no_grad():
            out = stage._messages(hidden, pair, blocks)
        assert out.shape == hidden.shape
        assert torch.isfinite(out).all()
    finally:
        _restore_default_arch()


def test_joint_mlp_messages_gradients() -> None:
    try:
        _activate_v10()
        torch.manual_seed(1)
        buses = 9
        types = torch.tensor([0, 1, 2, 0, 1, 0, 0, 1, 0], dtype=torch.long)
        component = torch.zeros(buses, dtype=torch.long)
        hidden = torch.randn(2, buses, tpgt.D, requires_grad=True)
        pair = torch.randn(buses, buses, tpgt.RPE_DIM, requires_grad=True)
        stage = tpgt.CorrectionStage()
        blocks = tpgt.type_pair_blocks(types, tpgt.type_pair_masks(types, component))
        out = stage._messages(hidden, pair, blocks)
        out.square().sum().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        assert pair.grad is not None and torch.isfinite(pair.grad).all()
        assert stage.score_w1 is not None and stage.score_w1.grad is not None
        assert torch.isfinite(stage.score_w1.grad).all()
        assert stage.w_q.grad is not None and torch.isfinite(stage.w_q.grad).all()
    finally:
        _restore_default_arch()


def test_v10_model_uses_raw_rpe_no_pair_encoder() -> None:
    try:
        _activate_v10()
        model = tpgt.TPGT()
        assert model.use_raw_rpe
        assert isinstance(model.pair_encoder, torch.nn.Identity)
        rpe = torch.randn(4, 4, tpgt.RPE_DIM)
        assert torch.equal(model.encode_pairs(rpe), rpe)
        total = sum(p.numel() for p in model.parameters())
        assert total > 1_000_000
        stage = model.stages[0]
        assert stage.attention_kind == "joint_mlp"
        # 9 type-pairs × HEADS score MLPs per stage × DEPTH stages.
        assert stage.score_w1 is not None
        assert stage.score_w1.shape[:2] == (9, tpgt.HEADS)
        assert len(model.stages) == tpgt.DEPTH
    finally:
        _restore_default_arch()
