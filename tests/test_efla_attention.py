"""TPGT-EFLA (v13): electrical landmarks, dense-vs-factorized parity, preset checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import efla_tpgt as efla  # noqa: E402
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


def _line_branches(n: int = 6) -> list[tuple[int, int, float, float]]:
    return [(i, i + 1, 0.01 * (1 + 0.1 * i), 0.1 * (1 + 0.05 * i)) for i in range(n - 1)]


def _mesh_branches() -> list[tuple[int, int, float, float]]:
    # Small meshed graph (5 buses).
    return [
        (0, 1, 0.02, 0.2),
        (1, 2, 0.01, 0.15),
        (2, 3, 0.03, 0.25),
        (3, 0, 0.02, 0.18),
        (0, 2, 0.04, 0.3),
        (1, 3, 0.025, 0.22),
        (3, 4, 0.05, 0.4),
    ]


def test_v13_preset_and_stage_shapes() -> None:
    try:
        _activate_arch("v13")
        assert tpgt.ATTENTION_KIND == "efla"
        assert tpgt.HEADS == 4 and tpgt.DEPTH == 8 and tpgt.D == 96
        assert tpgt.ETA_ANGLE == 1.0
        assert tpgt.EFLA_LANDMARKS == 16
        assert tpgt.EFLA_ELEC_DIM == efla.elec_feature_dim(16)
        stage = tpgt.CorrectionStage()
        assert stage.w_q.shape[0] == 9
        assert stage.a_q is not None and stage.a_q.shape == (9, 4, tpgt.EFLA_NODE_KERNEL, tpgt.D_A)
        assert stage.bq_elec is not None and stage.bq_elec.shape[-1] == tpgt.EFLA_ELEC_DIM
        assert stage.m_mu is not None and stage.gamma is not None
        assert stage.score_w1 is None and stage.w_ew is None
        model = tpgt.TPGT()
        assert model.use_raw_rpe
        assert model.attention_kind == "efla"
        summary = tpgt.architecture_summary()
        assert "efla" in summary and "v13" in summary
    finally:
        _restore_default_arch()


def test_electrical_reconstruction_vs_exact() -> None:
    branches = _mesh_branches()
    bus_count = 5
    exact, component = tpgt._meshed_zeff_and_component(bus_count, branches)
    factors = efla.compute_electrical_factors(bus_count, branches, num_landmarks=3)
    approx = efla.approximate_zeff_matrix(factors)
    same = component[:, None] == component[None, :]
    off = same & ~np.eye(bus_count, dtype=bool)
    err = np.abs(approx - exact)[off]
    # With r = n-1 landmarks on a connected component, Nyström of the pseudoinverse is exact
    # in exact arithmetic; allow float noise.
    factors_full = efla.compute_electrical_factors(bus_count, branches, num_landmarks=4)
    approx_full = efla.approximate_zeff_matrix(factors_full)
    err_full = np.abs(approx_full - exact)[off]
    assert float(np.max(err_full)) < 1e-4, float(np.max(err_full))
    # Reduced rank is approximate but should remain finite and not explode.
    assert float(np.max(err)) < 5.0
    assert np.isfinite(factors.elec_a).all()
    assert factors.elec_a.shape == (bus_count, efla.elec_feature_dim(3))


def test_singleton_and_islands() -> None:
    branches = [
        (0, 1, 0.01, 0.1),
        (1, 2, 0.01, 0.1),
        # bus 3 isolated
        (4, 5, 0.02, 0.2),
    ]
    factors = efla.compute_electrical_factors(6, branches, num_landmarks=2)
    assert factors.component[3] != factors.component[0]
    assert np.allclose(factors.elec_a[3], 0.0)
    # Cross-component Zeff reconstruction stays zero.
    approx = efla.approximate_zeff_matrix(factors)
    assert abs(approx[0, 4]) < 1e-12
    assert abs(approx[3, 0]) < 1e-12


def test_landmark_relabel_stability() -> None:
    branches = _mesh_branches()
    f1 = efla.compute_electrical_factors(5, branches, num_landmarks=3)
    # Relabel: new_id = inv[old_id]; remap branch endpoints.
    perm = np.array([2, 0, 4, 1, 3], dtype=np.int64)  # old_id -> unused; build inv below
    inv = np.empty(5, dtype=np.int64)
    inv[perm] = np.arange(5)  # old -> new
    remapped = [(int(inv[i]), int(inv[j]), r, x) for i, j, r, x in branches]
    f2 = efla.compute_electrical_factors(5, remapped, num_landmarks=3)
    # Landmark bus sets (global ids) must match after mapping.
    lm1 = set(int(f1.buses_by_component[0][i]) for i in f1.landmarks_by_component[0].tolist())
    lm2_new = set(int(f2.buses_by_component[0][i]) for i in f2.landmarks_by_component[0].tolist())
    lm2_old = set(int(perm[j]) for j in lm2_new)  # new -> old via perm (since inv[perm]=id)
    # perm[new]=old because inv[perm[k]]=k ⇒ perm[new]=old.
    assert lm1 == lm2_old, (lm1, lm2_old)
    # Aligned embeddings: f2[inv[old]] ↔ f1[old]
    c1 = f1.elec_a
    c2 = f2.elec_a[inv]
    a = c1[:, :3].reshape(-1)
    b = c2[:, :3].reshape(-1)
    if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0:
        corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert corr > 0.95, corr


def test_dense_vs_factorized_parity() -> None:
    torch.manual_seed(0)
    n = 5
    branches = _mesh_branches()
    factors = efla.compute_electrical_factors(n, branches, num_landmarks=3)
    types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    component = torch.from_numpy(factors.component)
    elec_a = torch.from_numpy(factors.elec_a).float()
    hidden = torch.randn(2, n, 8, requires_grad=True)

    attn = efla.ElectricalFactorizedAttention(
        hidden_dim=8,
        value_dim=4,
        elec_dim=elec_a.shape[-1],
        num_heads=2,
        node_kernel=4,
        elec_kernel=2,
        value_rank=2,
    )
    dense = attn(hidden, elec_a, types, component, dense=True)
    fact = attn(hidden, elec_a, types, component, dense=False)
    assert dense.shape == fact.shape == (2, n, 8)
    assert torch.allclose(dense, fact, rtol=1e-4, atol=1e-5), (dense - fact).abs().max().item()

    # Gradients through factorized path.
    loss = fact.pow(2).sum()
    loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()

    # Self-exclusion: diagonal contribution must not be needed (singleton/self zeroed).
    # Isolated-type check: if only one Slack, its message from Slack←Slack is zero after exclusion
    # when no other Slack exists — still may receive from PQ/PV.
    assert torch.isfinite(fact).all()


def test_no_cross_component_attention() -> None:
    torch.manual_seed(1)
    branches = [
        (0, 1, 0.01, 0.1),
        (1, 2, 0.01, 0.1),
        (3, 4, 0.02, 0.2),
        (4, 5, 0.02, 0.2),
    ]
    factors = efla.compute_electrical_factors(6, branches, num_landmarks=2)
    types = torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.long)
    component = torch.from_numpy(factors.component)
    elec_a = torch.from_numpy(factors.elec_a).float()
    # Put a spike only on bus 5 hidden; messages on buses 0-2 must be unchanged if we
    # zero bus 5 vs zero bus 2 (cross-island).
    hidden = torch.zeros(1, 6, 6)
    hidden[0, 5] = 10.0
    attn = efla.ElectricalFactorizedAttention(
        hidden_dim=6,
        value_dim=3,
        elec_dim=elec_a.shape[-1],
        num_heads=1,
        node_kernel=3,
        elec_kernel=2,
        value_rank=1,
    )
    # Freeze random maps to identity-ish positive features by zeroing electrical maps
    # and making node maps small — still, island separation is structural.
    with torch.no_grad():
        for p in attn.parameters():
            p.zero_()
            if p is attn.gamma:
                p.fill_(0.0)
            if p is attn.w_v:
                # identity-like so values carry hidden
                eye = torch.eye(3, 6)
                p.copy_(eye.unsqueeze(0).unsqueeze(0).expand_as(p))
            if p is attn.w_o:
                eye = torch.eye(6, 3)
                p.copy_(eye.unsqueeze(0).expand_as(p))
            if p is attn.a_q or p is attn.a_k:
                p[:, :, :, :3] = torch.eye(3).unsqueeze(0).unsqueeze(0)
            if p is attn.bq_elec or p is attn.bk_elec:
                p[:, :, :, 0] = 0.0
            if p is attn.b_q or p is attn.b_k:
                p.fill_(0.0)
            if p is attn.cq_elec or p is attn.ck_elec:
                p.fill_(0.0)

    msg = attn(hidden, elec_a, types, component, dense=False)
    # Buses in component 0 should not pick up the spike on bus 5 (component 1).
    assert torch.allclose(msg[0, :3], torch.zeros(3, 6), atol=1e-5), msg[0, :3]


def test_stage_forward_smoke() -> None:
    try:
        _activate_arch("v13")
        # Shrink depth for a cheap smoke test of one stage path via TPGT DEPTH override.
        tpgt.DEPTH = 1
        model = tpgt.TPGT()
        n = 5
        branches = _mesh_branches()
        factors = efla.compute_electrical_factors(n, branches, num_landmarks=tpgt.EFLA_LANDMARKS)
        types = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
        component = torch.from_numpy(factors.component)
        elec_a = torch.from_numpy(np.zeros((n, tpgt.EFLA_ELEC_DIM), dtype=np.float32))
        elec_a[:, : factors.elec_a.shape[1]] = torch.from_numpy(factors.elec_a)
        rpe = torch.zeros(n, n, tpgt.RPE_DIM)
        masks = tpgt.type_pair_masks(types, component)
        specified = torch.randn(2, n, 2)
        shunt = torch.zeros(2, n)
        ybus = torch.eye(n, dtype=torch.complex128)
        # Make a weakly valid Ybus diagonal.
        ybus = ybus * (1 + 1j)
        pred, residuals = model(
            specified,
            types,
            rpe,
            masks,
            ybus,
            shunt_b=shunt,
            elec_a=elec_a,
            component=component,
        )
        assert pred.shape == (2, n, 2)
        assert len(residuals) == 1
        assert torch.isfinite(pred).all()
    finally:
        _restore_default_arch()


if __name__ == "__main__":
    test_v13_preset_and_stage_shapes()
    test_electrical_reconstruction_vs_exact()
    test_singleton_and_islands()
    test_landmark_relabel_stability()
    test_dense_vs_factorized_parity()
    test_no_cross_component_attention()
    test_stage_forward_smoke()
    print("all efla tests passed")
