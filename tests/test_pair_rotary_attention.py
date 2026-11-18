"""TPGT v15 pairwise electrical rotary attention smoke tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pair_rotary_tpgt as pr  # noqa: E402
import train_tpgt_task11 as tpgt  # noqa: E402


def _activate_arch(name: str) -> None:
    os.environ["GENCO_TPGT_ARCH"] = name
    tpgt.apply_architecture_config()


def _restore_default_arch() -> None:
    os.environ.pop("GENCO_TPGT_ARCH", None)
    os.environ.pop("GENCO_TPGT_ATTENTION", None)
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
    tpgt.RUN_DIR_NAME = "tpgt_task11_v7"


def _mesh_branches() -> list[tuple[int, int, float, float]]:
    return [
        (0, 1, 0.02, 0.2),
        (1, 2, 0.01, 0.15),
        (2, 3, 0.03, 0.25),
        (3, 0, 0.02, 0.18),
        (0, 2, 0.04, 0.3),
        (3, 4, 0.05, 0.4),
    ]


def test_v15_preset() -> None:
    try:
        _activate_arch("v15")
        assert tpgt.ATTENTION_KIND == "pair_rotary"
        assert tpgt.HEADS == 4 and tpgt.DEPTH == 8 and tpgt.D == 96
        assert tpgt.USE_NODE_SHUNT is True
        stage = tpgt.CorrectionStage()
        assert stage.pair_rotary_omega is not None
        assert stage.pair_rotary_omega.shape == (4, 24, 3)
        assert stage.wire_omega is None
        model = tpgt.TPGT()
        assert model.attention_kind == "pair_rotary"
        assert "pair_rotary" in tpgt.architecture_summary()
    finally:
        _restore_default_arch()


def test_pair_features_outage_changes_distance() -> None:
    branches = _mesh_branches()
    full = pr.compute_pair_rotary_factors(5, branches, s_d=1.0, s_w=1.0)
    # Drop branch 0-2
    outaged = [b for b in branches if not (b[0] == 0 and b[1] == 2)]
    cut = pr.compute_pair_rotary_factors(5, outaged, s_d=1.0, s_w=1.0)
    assert full.pair_elec.shape == (5, 5, 3)
    assert full.pair_elec[0, 2, 1] == 1.0  # direct edge present
    assert cut.pair_elec[0, 2, 1] == 0.0  # edge indicator cleared
    # Electrical distance feature should change for at least some pairs.
    assert not np.allclose(full.pair_elec[..., 0], cut.pair_elec[..., 0])


def test_no_cross_component_attention() -> None:
    torch.manual_seed(0)
    branches = [
        (0, 1, 0.01, 0.1),
        (1, 2, 0.01, 0.1),
        (3, 4, 0.02, 0.2),
        (4, 5, 0.02, 0.2),
    ]
    factors = pr.compute_pair_rotary_factors(6, branches, s_d=1.0, s_w=1.0)
    types = torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.long)
    component = torch.from_numpy(factors.component)
    pair_elec = torch.from_numpy(factors.pair_elec).float()
    hidden = torch.zeros(1, 6, 8)
    hidden[0, 5] = 5.0
    w_q = torch.randn(9, 1, 4, 8) * 0.1
    w_k = torch.randn(9, 1, 4, 8) * 0.1
    w_v = torch.zeros(9, 1, 4, 8)
    w_v[:] = torch.eye(4, 8)
    w_o = torch.eye(8, 4).unsqueeze(0)
    omega = torch.zeros(1, 2, 3)
    msg = pr.pair_rotary_messages(
        hidden,
        pair_elec,
        types,
        component,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        omega=omega,
    )
    assert torch.allclose(msg[0, :3], torch.zeros(3, 8), atol=1e-5), msg[0, :3]


def test_forward_smoke() -> None:
    try:
        _activate_arch("v15")
        tpgt.DEPTH = 1
        torch.manual_seed(0)
        n = 5
        branches = _mesh_branches()
        factors = pr.compute_pair_rotary_factors(n, branches, s_d=1.0, s_w=1.0)
        types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
        component = torch.from_numpy(factors.component)
        pair_elec = torch.from_numpy(factors.pair_elec).float()
        rpe = torch.zeros(n, n, tpgt.RPE_DIM)
        masks = tpgt.type_pair_masks(types, component)
        specified = torch.randn(2, n, 2)
        shunt = torch.zeros(2, n)
        ybus = torch.eye(n, dtype=torch.complex128) * (1 + 1j)
        model = tpgt.TPGT()
        pred, residuals = model(
            specified,
            types,
            rpe,
            masks,
            ybus,
            shunt_b=shunt,
            pair_elec=pair_elec,
            component=component,
        )
        assert pred.shape == (2, n, 2)
        assert len(residuals) == 1
        assert torch.isfinite(pred).all()
    finally:
        _restore_default_arch()
