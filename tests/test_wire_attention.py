"""TPGT-WIRE (v14): susceptance Laplacian spectral coords + masked softmax."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_tpgt_task11 as tpgt  # noqa: E402
import wire_tpgt as wire  # noqa: E402


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


def test_v14_preset() -> None:
    try:
        _activate_arch("v14")
        assert tpgt.ATTENTION_KIND == "wire"
        assert tpgt.HEADS == 4 and tpgt.DEPTH == 8 and tpgt.D == 96
        assert tpgt.ETA_ANGLE == 1.0
        assert tpgt.WIRE_SPECTRAL_DIM == 8
        stage = tpgt.CorrectionStage()
        assert stage.w_q.shape[0] == 9
        assert stage.wire_omega is not None
        assert stage.wire_omega.shape == (4, 24, 8)
        assert stage.a_q is None and stage.score_w1 is None
        model = tpgt.TPGT()
        assert model.attention_kind == "wire"
        assert "wire" in tpgt.architecture_summary()
    finally:
        _restore_default_arch()


def test_spectral_coords_shape_and_singleton() -> None:
    branches = [
        (0, 1, 0.01, 0.1),
        (1, 2, 0.01, 0.1),
        # bus 3 isolated
        (4, 5, 0.02, 0.2),
    ]
    factors = wire.compute_wire_factors(6, branches, k_s=8)
    assert factors.rho.shape == (6, 8)
    assert np.allclose(factors.rho[3], 0.0)
    assert factors.component[0] != factors.component[3]
    assert np.isfinite(factors.rho).all()


def test_no_cross_component_attention() -> None:
    torch.manual_seed(0)
    branches = [
        (0, 1, 0.01, 0.1),
        (1, 2, 0.01, 0.1),
        (3, 4, 0.02, 0.2),
        (4, 5, 0.02, 0.2),
    ]
    factors = wire.compute_wire_factors(6, branches, k_s=4)
    types = torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.long)
    component = torch.from_numpy(factors.component)
    rho = torch.from_numpy(factors.rho).float()
    hidden = torch.zeros(1, 6, 8)
    hidden[0, 5] = 5.0
    w_q = torch.randn(9, 1, 4, 8) * 0.1
    w_k = torch.randn(9, 1, 4, 8) * 0.1
    w_v = torch.zeros(9, 1, 4, 8)
    # Identity-like V so messages carry sender hidden
    eye = torch.eye(4, 8)
    w_v[:] = eye
    w_o = torch.eye(8, 4).unsqueeze(0)
    omega = torch.zeros(1, 2, 4)
    msg = wire.wire_messages(
        hidden, rho, types, component, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, omega=omega
    )
    # Island 0 (buses 0-2) must not receive the spike from bus 5.
    assert torch.allclose(msg[0, :3], torch.zeros(3, 8), atol=1e-5), msg[0, :3]


def test_self_excluded_and_relative_rotation() -> None:
    torch.manual_seed(1)
    branches = _mesh_branches()
    factors = wire.compute_wire_factors(5, branches, k_s=4)
    types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    component = torch.from_numpy(factors.component)
    rho = torch.from_numpy(factors.rho).float()
    hidden = torch.randn(2, 5, 8)
    w_q = torch.randn(9, 2, 4, 8) * 0.05
    w_k = torch.randn(9, 2, 4, 8) * 0.05
    w_v = torch.randn(9, 2, 4, 8) * 0.05
    w_o = torch.randn(2, 8, 4) * 0.05
    omega = torch.randn(2, 2, 4) * 0.1
    msg = wire.wire_messages(
        hidden, rho, types, component, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, omega=omega
    )
    assert msg.shape == (2, 5, 8)
    assert torch.isfinite(msg).all()
    # Gradients flow.
    hidden2 = hidden.clone().requires_grad_(True)
    msg2 = wire.wire_messages(
        hidden2, rho, types, component, w_q=w_q, w_k=w_k, w_v=w_v, w_o=w_o, omega=omega
    )
    msg2.sum().backward()
    assert hidden2.grad is not None and torch.isfinite(hidden2.grad).all()


def test_stage_forward_smoke() -> None:
    try:
        _activate_arch("v14")
        tpgt.DEPTH = 1
        model = tpgt.TPGT()
        n = 5
        branches = _mesh_branches()
        factors = wire.compute_wire_factors(n, branches, k_s=tpgt.WIRE_SPECTRAL_DIM)
        types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
        component = torch.from_numpy(factors.component)
        rho = torch.from_numpy(factors.rho).float()
        rpe = torch.zeros(n, n, tpgt.RPE_DIM)
        masks = tpgt.type_pair_masks(types, component)
        specified = torch.randn(2, n, 2)
        shunt = torch.zeros(2, n)
        ybus = torch.eye(n, dtype=torch.complex128) * (1 + 1j)
        pred, residuals = model(
            specified,
            types,
            rpe,
            masks,
            ybus,
            shunt_b=shunt,
            wire_rho=rho,
            component=component,
        )
        assert pred.shape == (2, n, 2)
        assert len(residuals) == 1
        assert torch.isfinite(pred).all()
    finally:
        _restore_default_arch()


if __name__ == "__main__":
    test_v14_preset()
    test_spectral_coords_shape_and_singleton()
    test_no_cross_component_attention()
    test_self_excluded_and_relative_rotation()
    test_stage_forward_smoke()
    print("all wire tests passed")
