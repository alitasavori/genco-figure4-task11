"""Forward diagnostics helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_tpgt_task11 as tpgt  # noqa: E402
from train_tpgt_task11 import (  # noqa: E402
    RPE_DIM,
    RPE_FEATURE_NAMES,
    TPGT,
    Network,
    Split,
    architecture_summary,
    audit_residual_correctness,
    audit_rpe_correctness,
    bus_shunt_b,
    bus_shunt_gb,
    node_encoder_features,
    contingency_class,
    evaluate_pbl,
    nodal_residual,
    pack_case,
    format_bus_shunt_ablation_diagnostics,
    format_rpe_effectiveness_diagnostics,
    meshed_rpe,
    pair_encoder_input_weight_norms,
    pbl_stats,
    print_contingency_diagnostics,
    summarize_rpe,
    train_probe_split,
    type_pair_masks,
)


def test_pbl_stats_on_zero_residual():
    residual = torch.zeros(4, 118, 2)
    stats = pbl_stats(residual)
    assert stats["mean"] == 0.0
    assert stats["max"] == 0.0


def test_summarize_rpe_shape():
    rpe = torch.randn(5, 5, RPE_DIM)
    rows = summarize_rpe(rpe)
    assert len(rows) == RPE_DIM
    assert rows[0]["off_mean"] == rows[0]["off_mean"]  # finite


def test_rpe_is_zeff_re_im():
    assert RPE_DIM == 2
    assert RPE_FEATURE_NAMES == ("Zeff_Re_norm", "Zeff_Im_norm")
    branches = [(0, 1, 0.1, 0.1), (1, 2, 0.1, 0.1), (0, 2, 0.1, 0.2)]
    ybus = np.zeros((3, 3), dtype=np.complex64)
    features, _component = meshed_rpe(3, branches, ybus)
    assert features.shape == (3, 3, 2)
    assert TPGT().pair_encoder.net[0].in_features == 2
    for channel in range(2):
        assert np.allclose(np.diagonal(features[..., channel]), 0.0)
        assert np.allclose(features[..., channel], features[..., channel].T, atol=1e-4)


def test_v17_generator_grounded_support_rpe():
    """Radial PV→i→j: C built for ρ; RPE is Im+ρ only; shared 2→D_E encoder."""
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    previous_d = tpgt.D
    previous_de = tpgt.D_E
    try:
        tpgt.apply_rpe_channel_config("im_rho")
        tpgt.D = 96
        tpgt.D_E = 48
        assert tpgt.RPE_DIM == 2
        assert tpgt.RPE_FEATURE_NAMES == ("Zeff_Im_norm", "rho_ij")
        assert not tpgt.USE_TYPE_CONDITIONED_RPE
        assert tpgt.uses_generator_support_rpe()
        xg, xij = 0.2, 0.5
        # bus 0 = PV, 1 = i (PQ), 2 = j (PQ)
        branches = [(0, 1, 0.0, xg), (1, 2, 0.0, xij)]
        types = np.array([1, 0, 0], dtype=np.int64)
        built = tpgt.build_generator_grounded_support(3, branches, types)
        c = built.c_full
        assert not np.any(built.unsupported_pq)
        assert not np.any(built.invalid_support)
        assert built.skipped_nonpositive_x == 0
        assert np.allclose(c[1:, 1:], np.array([[xg, xg], [xg, xg + xij]]), atol=1e-6)
        assert np.allclose(c[0, :], 0.0) and np.allclose(c[:, 0], 0.0)
        for check in tpgt.validate_generator_support_matrix(
            c, types, unsupported_pq=built.unsupported_pq
        ):
            assert check.passed, check.detail
        ybus = np.zeros((3, 3), dtype=np.complex64)
        features, _, invalid = meshed_rpe(
            3, branches, ybus, types=types, c_scale=1.0, return_invalid=True
        )
        assert features.shape == (3, 3, 2)
        assert invalid is not None and not np.any(invalid)
        rho = xg / np.sqrt(xg * (xg + xij))
        # PQ–PQ: ρ present; mixed / PV pairs: ρ = 0 (Im may be nonzero).
        assert np.isclose(features[1, 2, 1], rho, atol=1e-5)
        assert np.isclose(features[2, 1, 1], rho, atol=1e-5)
        assert np.allclose(features[1, 0, 1], 0.0)
        assert np.allclose(features[0, 1, 1], 0.0)
        model = TPGT()
        assert isinstance(model.pair_encoder, tpgt.MLP)
        assert model.pair_encoder.net[0].in_features == 2
        assert model.pair_encoder.net[0].out_features == 96
        assert model.pair_encoder.net[2].out_features == 96
        assert model.pair_encoder.net[-1].out_features == 48
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc
        tpgt.D = previous_d
        tpgt.D_E = previous_de


def test_v17_type_conditioned_rpe():
    """Default v17: Im+ρ stored; E_PQ on PQ–PQ, E_other(Im) on other type pairs."""
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    previous_d = tpgt.D
    previous_de = tpgt.D_E
    previous_cv = tpgt.CHECKPOINT_VERSION
    previous_run = tpgt.RUN_TAG
    try:
        tpgt.apply_rpe_channel_config("type_cond")
        tpgt.D = 96
        tpgt.D_E = 48
        tpgt.RUN_TAG = "v17"
        tpgt.CHECKPOINT_VERSION = "meshed-rpe-1-d12-w96-h4-pqvt-busBS-rho-v17"
        tpgt.retag_checkpoint_version_for_rpe()
        assert tpgt.RPE_DIM == 2
        assert tpgt.RPE_FEATURE_NAMES == ("Zeff_Im_norm", "rho_ij")
        assert tpgt.USE_TYPE_CONDITIONED_RPE
        assert tpgt.uses_generator_support_rpe()
        assert tpgt.rpe_channels_slug() == "type-cond"
        assert "meshed-rpe-2-" in tpgt.CHECKPOINT_VERSION
        assert tpgt.CHECKPOINT_VERSION.endswith("-type-cond-v17")

        xg, xij = 0.2, 0.5
        branches = [(0, 1, 0.0, xg), (1, 2, 0.0, xij)]
        types_np = np.array([1, 0, 0], dtype=np.int64)  # PV, PQ, PQ
        ybus = np.zeros((3, 3), dtype=np.complex64)
        features, _, invalid = meshed_rpe(
            3, branches, ybus, types=types_np, c_scale=1.0, return_invalid=True
        )
        assert features.shape == (3, 3, 2)
        assert invalid is not None and not np.any(invalid)

        model = TPGT()
        assert model.use_type_conditioned_rpe
        enc = model.pair_encoder
        assert isinstance(enc, tpgt.TypeConditionedPairEncoder)
        assert enc.pq_encoder.net[0].in_features == 2
        assert enc.other_encoder.net[0].in_features == 1
        assert enc.pq_encoder.net[-1].out_features == 48
        assert enc.other_encoder.net[-1].out_features == 48

        types = torch.as_tensor(types_np, dtype=torch.long)
        rpe = torch.as_tensor(features, dtype=torch.float32)
        out = model.encode_pairs(rpe, types=types)
        assert out.shape == (3, 3, 48)

        # PQ–PQ uses E_PQ; PV–PQ / other pairs use E_other(Im only).
        e_pq = enc.pq_encoder(rpe)
        e_other = enc.other_encoder(rpe[..., :1])
        assert torch.allclose(out[1, 2], e_pq[1, 2])
        assert torch.allclose(out[2, 1], e_pq[2, 1])
        assert torch.allclose(out[0, 1], e_other[0, 1])
        assert torch.allclose(out[1, 0], e_other[1, 0])
        assert torch.allclose(out[0, 0], e_other[0, 0])

        # Zero diagonal convention preserved in encoder inputs (ρ_ii stored as 0).
        assert float(rpe[1, 1, 1]) == 0.0
        assert float(rpe[2, 2, 1]) == 0.0

        # Zeroing ρ changes PQ–PQ embeds only.
        rpe_no_rho = rpe.clone()
        rpe_no_rho[..., 1] = 0.0
        out_no_rho = model.encode_pairs(rpe_no_rho, types=types)
        assert not torch.allclose(out[1, 2], out_no_rho[1, 2])
        assert torch.allclose(out[0, 1], out_no_rho[0, 1])

        # Gradients reach both encoders.
        loss = out.sum()
        loss.backward()
        assert enc.pq_encoder.net[0].weight.grad is not None
        assert enc.other_encoder.net[0].weight.grad is not None
        assert float(enc.pq_encoder.net[0].weight.grad.abs().sum()) > 0.0
        assert float(enc.other_encoder.net[0].weight.grad.abs().sum()) > 0.0

        weight_rows = pair_encoder_input_weight_norms(model)
        assert len(weight_rows) == 3
        assert {row["name"] for row in weight_rows} == {
            "Zeff_Im_norm(other)",
            "Zeff_Im_norm(pq)",
            "rho_ij(pq)",
        }
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc
        tpgt.D = previous_d
        tpgt.D_E = previous_de
        tpgt.CHECKPOINT_VERSION = previous_cv
        tpgt.RUN_TAG = previous_run


def test_v17_type_conditioned_cij_rpe():
    """type_cond_cij: Im+Cij stored; E_PQ on PQ–PQ, E_other(Im) elsewhere; distinct slug."""
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    previous_d = tpgt.D
    previous_de = tpgt.D_E
    previous_cv = tpgt.CHECKPOINT_VERSION
    previous_run = tpgt.RUN_TAG
    try:
        tpgt.apply_rpe_channel_config("type_cond_cij")
        tpgt.D = 96
        tpgt.D_E = 48
        tpgt.RUN_TAG = "v17"
        tpgt.CHECKPOINT_VERSION = "meshed-rpe-1-d12-w96-h4-pqvt-busBS-type-cond-v17"
        tpgt.retag_checkpoint_version_for_rpe()
        assert tpgt.RPE_DIM == 2
        assert tpgt.RPE_FEATURE_NAMES == ("Zeff_Im_norm", "Cij_norm")
        assert tpgt.USE_TYPE_CONDITIONED_RPE
        assert tpgt.uses_generator_support_rpe()
        assert tpgt.rpe_channels_slug() == "type-cond-cij"
        assert "meshed-rpe-2-" in tpgt.CHECKPOINT_VERSION
        assert tpgt.CHECKPOINT_VERSION.endswith("-type-cond-cij-v17")
        # Switching back to type_cond must not leave a trailing -cij on the slug.
        tpgt.apply_rpe_channel_config("type_cond")
        tpgt.retag_checkpoint_version_for_rpe()
        assert tpgt.CHECKPOINT_VERSION.endswith("-type-cond-v17")
        assert not tpgt.CHECKPOINT_VERSION.endswith("-type-cond-cij-v17")
        tpgt.apply_rpe_channel_config("type_cond_cij")
        tpgt.retag_checkpoint_version_for_rpe()
        assert tpgt.CHECKPOINT_VERSION.endswith("-type-cond-cij-v17")

        xg, xij = 0.2, 0.5
        branches = [(0, 1, 0.0, xg), (1, 2, 0.0, xij)]
        types_np = np.array([1, 0, 0], dtype=np.int64)  # PV, PQ, PQ
        ybus = np.zeros((3, 3), dtype=np.complex64)
        features, _, invalid = meshed_rpe(
            3, branches, ybus, types=types_np, c_scale=1.0, return_invalid=True
        )
        assert features.shape == (3, 3, 2)
        assert invalid is not None and not np.any(invalid)
        # Channel 1 matches cij-only conventions (gen rows/cols 0; C_UU values).
        assert np.isclose(features[1, 1, 1], xg, atol=1e-5)
        assert np.isclose(features[1, 2, 1], xg, atol=1e-5)
        assert np.isclose(features[2, 2, 1], xg + xij, atol=1e-5)
        assert np.allclose(features[0, :, 1], 0.0)
        assert np.allclose(features[:, 0, 1], 0.0)

        model = TPGT()
        assert model.use_type_conditioned_rpe
        enc = model.pair_encoder
        assert isinstance(enc, tpgt.TypeConditionedPairEncoder)
        assert enc.pq_encoder.net[0].in_features == 2
        assert enc.other_encoder.net[0].in_features == 1

        types = torch.as_tensor(types_np, dtype=torch.long)
        rpe = torch.as_tensor(features, dtype=torch.float32)
        out = model.encode_pairs(rpe, types=types)
        assert out.shape == (3, 3, 48)

        e_pq = enc.pq_encoder(rpe)
        e_other = enc.other_encoder(rpe[..., :1])
        assert torch.allclose(out[1, 2], e_pq[1, 2])
        assert torch.allclose(out[2, 1], e_pq[2, 1])
        assert torch.allclose(out[0, 1], e_other[0, 1])
        assert torch.allclose(out[1, 0], e_other[1, 0])

        # Zeroing Cij changes PQ–PQ embeds only.
        rpe_no_cij = rpe.clone()
        rpe_no_cij[..., 1] = 0.0
        out_no_cij = model.encode_pairs(rpe_no_cij, types=types)
        assert not torch.allclose(out[1, 2], out_no_cij[1, 2])
        assert torch.allclose(out[0, 1], out_no_cij[0, 1])

        weight_rows = pair_encoder_input_weight_norms(model)
        assert {row["name"] for row in weight_rows} == {
            "Zeff_Im_norm(other)",
            "Zeff_Im_norm(pq)",
            "Cij_norm(pq)",
        }
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc
        tpgt.D = previous_d
        tpgt.D_E = previous_de
        tpgt.CHECKPOINT_VERSION = previous_cv
        tpgt.RUN_TAG = previous_run


def test_v17_rho_only_rpe():
    """Selectable ρ-only layout: 1 channel; pair encoder 1→D→D→D_E."""
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    previous_d = tpgt.D
    previous_de = tpgt.D_E
    previous_cv = tpgt.CHECKPOINT_VERSION
    try:
        tpgt.apply_rpe_channel_config("rho")
        tpgt.D = 96
        tpgt.D_E = 48
        tpgt.CHECKPOINT_VERSION = "meshed-rpe-2-d12-w96-h4-pqvt-busBS-im-rho-v17"
        tpgt.retag_checkpoint_version_for_rpe()
        assert tpgt.RPE_DIM == 1
        assert tpgt.RPE_FEATURE_NAMES == ("rho_ij",)
        assert not tpgt.USE_TYPE_CONDITIONED_RPE
        assert tpgt.uses_generator_support_rpe()
        assert tpgt.rpe_channels_slug() == "rho"
        assert "meshed-rpe-1-" in tpgt.CHECKPOINT_VERSION
        assert tpgt.CHECKPOINT_VERSION.endswith("-rho-v17")
        xg, xij = 0.2, 0.5
        branches = [(0, 1, 0.0, xg), (1, 2, 0.0, xij)]
        types = np.array([1, 0, 0], dtype=np.int64)
        ybus = np.zeros((3, 3), dtype=np.complex64)
        features, _, invalid = meshed_rpe(
            3, branches, ybus, types=types, c_scale=1.0, return_invalid=True
        )
        assert features.shape == (3, 3, 1)
        assert invalid is not None and not np.any(invalid)
        rho = xg / np.sqrt(xg * (xg + xij))
        assert np.isclose(features[1, 2, 0], rho, atol=1e-5)
        assert np.allclose(features[1, 0, 0], 0.0)
        model = TPGT()
        assert model.pair_encoder.net[0].in_features == 1
        assert model.pair_encoder.net[-1].out_features == 48
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc
        tpgt.D = previous_d
        tpgt.D_E = previous_de
        tpgt.CHECKPOINT_VERSION = previous_cv


def test_v17_cij_only_rpe():
    """Selectable Cij_norm = C/s_C (1 channel); gen rows/cols stay 0."""
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    previous_d = tpgt.D
    previous_de = tpgt.D_E
    previous_cv = tpgt.CHECKPOINT_VERSION
    try:
        tpgt.apply_rpe_channel_config("cij")
        tpgt.D = 96
        tpgt.D_E = 48
        tpgt.CHECKPOINT_VERSION = "meshed-rpe-1-d12-w96-h4-pqvt-busBS-rho-v17"
        tpgt.retag_checkpoint_version_for_rpe()
        assert tpgt.RPE_DIM == 1
        assert tpgt.RPE_FEATURE_NAMES == ("Cij_norm",)
        assert not tpgt.USE_TYPE_CONDITIONED_RPE
        assert tpgt.uses_generator_support_rpe()
        assert tpgt.rpe_channels_slug() == "cij"
        assert tpgt.CHECKPOINT_VERSION.endswith("-cij-v17")
        xg, xij = 0.2, 0.5
        branches = [(0, 1, 0.0, xg), (1, 2, 0.0, xij)]
        types = np.array([1, 0, 0], dtype=np.int64)
        ybus = np.zeros((3, 3), dtype=np.complex64)
        features, _, invalid = meshed_rpe(
            3, branches, ybus, types=types, c_scale=1.0, return_invalid=True
        )
        assert features.shape == (3, 3, 1)
        assert invalid is not None and not np.any(invalid)
        # C_UU = [[xg, xg], [xg, xg+xij]] with PV grounded.
        assert np.isclose(features[1, 1, 0], xg, atol=1e-5)
        assert np.isclose(features[1, 2, 0], xg, atol=1e-5)
        assert np.isclose(features[2, 2, 0], xg + xij, atol=1e-5)
        assert np.allclose(features[0, :], 0.0)
        assert np.allclose(features[:, 0], 0.0)
        model = TPGT()
        assert model.pair_encoder.net[0].in_features == 1
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc
        tpgt.D = previous_d
        tpgt.D_E = previous_de
        tpgt.CHECKPOINT_VERSION = previous_cv


def test_v17_gen_support_no_pinv_on_pq_island():
    """PQ island without Slack/PV → invalid_support; C=0 is numerical fallback only."""
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    try:
        tpgt.apply_rpe_channel_config("im_rho")
        # Two PQ buses linked; no generator in the component.
        branches = [(0, 1, 0.0, 0.4)]
        types = np.array([0, 0], dtype=np.int64)
        built = tpgt.build_generator_grounded_support(2, branches, types)
        assert np.allclose(built.c_full, 0.0)
        assert bool(np.all(built.unsupported_pq))
        assert bool(np.all(built.invalid_support))
        assert any("invalid_support" in m for m in built.messages)
        assert any("not physical zero sensitivity" in m for m in built.messages)
        assert any("no pinv" in m for m in built.messages)
        checks = {
            c.name: c
            for c in tpgt.validate_generator_support_matrix(
                built.c_full, types, unsupported_pq=built.unsupported_pq
            )
        }
        assert checks["gen_support_invalid_flag"].passed
        assert checks["gen_support_invalid_flag"].value == 2.0
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc


def test_v17_gen_support_flags_nonpositive_x():
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    try:
        tpgt.apply_rpe_channel_config("im_rho")
        # PV—PQ with negative x (excluded) plus positive spur so PQ stays gen-supported.
        branches = [(0, 1, 0.0, -0.2), (0, 1, 0.0, 0.5)]
        types = np.array([1, 0], dtype=np.int64)
        built = tpgt.build_generator_grounded_support(2, branches, types)
        assert built.skipped_nonpositive_x >= 1
        assert any("non-positive" in m for m in built.messages)
        assert np.isclose(built.c_full[1, 1], 0.5, atol=1e-6)
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc


def test_v17_rpe_zeros_disconnected_pairs():
    """Im+ρ are zero across components; isolated PQ is invalid_support (ρ=0)."""
    previous_names = tpgt.RPE_FEATURE_NAMES
    previous_dim = tpgt.RPE_DIM
    previous_tc = tpgt.USE_TYPE_CONDITIONED_RPE
    try:
        tpgt.apply_rpe_channel_config("im_rho")
        # Component A: PV—PQ; component B: isolated PQ (invalid_support → ρ=0).
        branches = [(0, 1, 0.0, 0.3)]
        types = np.array([1, 0, 0], dtype=np.int64)
        ybus = np.zeros((3, 3), dtype=np.complex64)
        features, component, invalid = meshed_rpe(
            3, branches, ybus, types=types, c_scale=1.0, return_invalid=True
        )
        assert component[0] == component[1]
        assert component[2] != component[1]
        assert invalid is not None and bool(invalid[2]) and not bool(invalid[1])
        assert np.allclose(features[1, 2], 0.0)
        assert np.allclose(features[2, 1], 0.0)
        # Same-component PQ–PV: Im may be nonzero; ρ must be 0 (not both PQ).
        assert np.allclose(features[1, 0, 1], 0.0)
    finally:
        tpgt.RPE_FEATURE_NAMES = previous_names
        tpgt.RPE_DIM = previous_dim
        tpgt.USE_TYPE_CONDITIONED_RPE = previous_tc


def test_bus_shunt_b_row_sum_includes_charging_unlike_device_bs():
    ybus = torch.zeros(3, 3, dtype=torch.complex64)
    ybus[0, 0] = 1.5 - 0.25j
    ybus[0, 1] = -1.0 + 0.1j
    ybus[1, 0] = -1.0 + 0.1j
    ybus[1, 1] = 2.0 + 0.5j
    ybus[2, 2] = 0.3j
    shunt = bus_shunt_gb(ybus)
    assert shunt.shape == (3, 2)
    assert torch.allclose(shunt[0], torch.tensor([0.5, -0.15]))
    assert torch.allclose(bus_shunt_b(ybus), shunt[:, 1])


def test_v9_node_features_are_pqvt_plus_device_bs():
    previous = tpgt.USE_NODE_SHUNT
    previous_sd = tpgt.USE_NODE_SUPPLY_DEMAND
    try:
        tpgt.USE_NODE_SUPPLY_DEMAND = False
        tpgt.USE_NODE_SHUNT = True
        tpgt.refresh_node_feature_config()
        model = TPGT()
        assert model.encoders[0].net[0].in_features == 5
        assert model.encoders[1].net[0].in_features == 5
        assert model.encoders[2].net[0].in_features == 5
        types = torch.tensor([0, 1, 2])
        specified = torch.tensor([[[0.1, 0.2], [0.3, 1.0], [1.0, 0.0]]])
        # Device-only BS (not Im(row-sum Y), which would also include line charging).
        device_bs = torch.tensor([-0.4, 0.14, 0.0])
        layout = tpgt.shared_layout(types, tpgt.type_pair_masks(types, torch.zeros(3, dtype=torch.long)))
        feats = node_encoder_features(specified, layout, device_bs)
        assert feats.shape == (1, 3, 5)
        assert torch.allclose(feats[0, 0], torch.tensor([0.1, 0.2, 1.0, 0.0, -0.4]))
        assert torch.allclose(feats[0, 1], torch.tensor([0.3, 0.0, 1.0, 0.0, 0.14]))
        assert torch.allclose(feats[0, 2], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    finally:
        tpgt.USE_NODE_SHUNT = previous
        tpgt.USE_NODE_SUPPLY_DEMAND = previous_sd
        tpgt.refresh_node_feature_config()


def test_v19_node_features_pq5_pv_slack7():
    previous = tpgt.USE_NODE_SHUNT
    previous_sd = tpgt.USE_NODE_SUPPLY_DEMAND
    try:
        tpgt.USE_NODE_SHUNT = True
        tpgt.USE_NODE_SUPPLY_DEMAND = True
        tpgt.refresh_node_feature_config()
        assert tpgt.NODE_FEATURE_DIM == 5
        assert tpgt.NODE_FEATURE_DIM_PV_SLACK == 7
        assert tpgt.NODE_ENCODER_FEATURE_NAMES == ("P", "Q", "V", "theta", "busBS")
        assert tpgt.NODE_ENCODER_FEATURE_NAMES_PV_SLACK == ("Pd", "Qd", "Ps", "Qs", "V", "theta", "busBS")
        model = TPGT()
        assert model.use_node_supply_demand is True
        assert model.encoders[0].net[0].in_features == 5  # PQ
        assert model.encoders[1].net[0].in_features == 7  # PV
        assert model.encoders[2].net[0].in_features == 7  # Slack
        types = torch.tensor([0, 1, 2])
        # PQ specified (P,Q)=(0.1,0.2); PV (P,V)=(0.3,1.0); Slack (V,θ)=(1.05,0.0)
        specified = torch.tensor([[[0.1, 0.2], [0.3, 1.0], [1.05, 0.0]]])
        device_bs = torch.tensor([-0.4, 0.14, 0.0])
        demand_p = torch.tensor([[0.5, 0.0, 0.2]])
        demand_q = torch.tensor([[0.1, 0.0, 0.05]])
        supply_p = torch.tensor([[0.0, 0.3, 1.1]])
        supply_q = torch.tensor([[0.0, 0.0, 0.4]])
        layout = tpgt.shared_layout(types, tpgt.type_pair_masks(types, torch.zeros(3, dtype=torch.long)))
        pq_feats = node_encoder_features(specified, layout, device_bs, type_id=0)
        assert pq_feats.shape == (1, 3, 5)
        assert torch.allclose(pq_feats[0, 0], torch.tensor([0.1, 0.2, 1.0, 0.0, -0.4]))
        pv_feats = node_encoder_features(
            specified,
            layout,
            device_bs,
            demand_p=demand_p,
            demand_q=demand_q,
            supply_p=supply_p,
            supply_q=supply_q,
            type_id=1,
        )
        assert pv_feats.shape == (1, 3, 7)
        assert torch.allclose(pv_feats[0, 1], torch.tensor([0.0, 0.0, 0.3, 0.0, 1.0, 0.0, 0.14]))
        slack_feats = node_encoder_features(
            specified,
            layout,
            device_bs,
            demand_p=demand_p,
            demand_q=demand_q,
            supply_p=supply_p,
            supply_q=supply_q,
            type_id=2,
        )
        assert slack_feats.shape == (1, 3, 7)
        assert torch.allclose(slack_feats[0, 2], torch.tensor([0.2, 0.05, 1.1, 0.4, 1.05, 0.0, 0.0]))
    finally:
        tpgt.USE_NODE_SHUNT = previous
        tpgt.USE_NODE_SUPPLY_DEMAND = previous_sd
        tpgt.refresh_node_feature_config()


def test_v20_grit_pair_update_flag():
    previous = tpgt.USE_NODE_SHUNT
    previous_sd = tpgt.USE_NODE_SUPPLY_DEMAND
    previous_pair = tpgt.GRIT_UPDATE_PAIR
    previous_kind = tpgt.ATTENTION_KIND
    try:
        tpgt.USE_NODE_SHUNT = True
        tpgt.USE_NODE_SUPPLY_DEMAND = True
        tpgt.GRIT_UPDATE_PAIR = True
        tpgt.ATTENTION_KIND = "grit"
        tpgt.refresh_node_feature_config()
        model = TPGT()
        assert model.stages[0].update_pair is True
        assert model.stages[0].w_oe is not None
        assert model.stages[0].pair_norm is not None
        assert "pair_upd" in architecture_summary()
    finally:
        tpgt.USE_NODE_SHUNT = previous
        tpgt.USE_NODE_SUPPLY_DEMAND = previous_sd
        tpgt.GRIT_UPDATE_PAIR = previous_pair
        tpgt.ATTENTION_KIND = previous_kind
        tpgt.refresh_node_feature_config()


def test_pv_slack_gen_pack_residual_subtracts_demand():
    """PV/Slack pack Ps/Qs; residual injects Ps−Pd / Qs−Qd (matches supply/demand features)."""
    previous_sd = tpgt.USE_NODE_SUPPLY_DEMAND
    try:
        tpgt.USE_NODE_SUPPLY_DEMAND = True
        # buses: 0=PQ, 1=PV, 2=Slack
        types = torch.tensor([0, 1, 2])
        demand_p = torch.tensor([[0.5, 0.2, 0.3]])
        demand_q = torch.tensor([[0.1, 0.05, 0.08]])
        supply_p = torch.tensor([[0.0, 0.7, 1.2]])
        supply_q = torch.tensor([[0.0, 0.4, 0.6]])
        # Net injections (= Ps−Pd) used for PQ packing and as the physical truth.
        power_p = supply_p - demand_p
        power_q = supply_q - demand_q
        voltage = torch.tensor([[1.0, 1.02, 1.05]])
        angle = torch.tensor([[0.1, -0.05, 0.0]])
        specified, target = pack_case(
            power_p, power_q, voltage, angle, types, supply_p=supply_p, supply_q=supply_q
        )
        # PV specified P is generation Ps, not net.
        assert torch.allclose(specified[0, 1, 0], supply_p[0, 1])
        assert not torch.allclose(specified[0, 1, 0], power_p[0, 1])
        # Slack targets are generation Ps/Qs.
        assert torch.allclose(target[0, 2, 0], supply_p[0, 2])
        assert torch.allclose(target[0, 2, 1], supply_q[0, 2])
        # Synthetic Ybus: choose flow so net injection matches power_*; residual ~0 with demand.
        # Build diagonal-ish ybus so S = V conj(Y V) is controllable via a identity-scaled network.
        # Use assemble+identity residual path: for V∠θ with Y=0, flow=0, residual = inj.
        ybus = torch.zeros(3, 3, dtype=torch.complex64)
        residual = nodal_residual(
            specified, target, types, ybus, demand_p=demand_p, demand_q=demand_q
        )
        # With Y=0, flow=0 → residual equals net injection (Ps−Pd, Qs−Qd).
        assert torch.allclose(residual[0, 0, 0], power_p[0, 0], atol=1e-5)
        assert torch.allclose(residual[0, 0, 1], power_q[0, 0], atol=1e-5)
        assert torch.allclose(residual[0, 1, 0], power_p[0, 1], atol=1e-5)  # Ps−Pd
        assert torch.allclose(residual[0, 1, 1], power_q[0, 1], atol=1e-5)
        assert torch.allclose(residual[0, 2, 0], power_p[0, 2], atol=1e-5)
        assert torch.allclose(residual[0, 2, 1], power_q[0, 2], atol=1e-5)
        # Forgetting demand on PV would leave residual at Ps (wrong).
        residual_no_dem = nodal_residual(specified, target, types, ybus)
        assert torch.allclose(residual_no_dem[0, 1, 0], supply_p[0, 1], atol=1e-5)
        assert abs(float(residual_no_dem[0, 1, 0] - residual[0, 1, 0])) > 0.1
    finally:
        tpgt.USE_NODE_SUPPLY_DEMAND = previous_sd


def test_pair_encoder_weight_norms():
    model = TPGT()
    rows = pair_encoder_input_weight_norms(model)
    assert len(rows) == RPE_DIM
    assert all(row["weight_l2"] >= 0.0 for row in rows)


def test_format_rpe_effectiveness_without_ablation():
    weights = [{"name": "adjacency", "weight_l2": 1.0}]
    text = format_rpe_effectiveness_diagnostics(weights, None, None)
    assert "not run" in text
    assert "adjacency" in text


def test_format_bus_shunt_ablation_with_zero_row():
    weights = [{"type": "PQ", "name": "busBS", "weight_l2": 0.5}]
    baseline = {"pbl_mean": 0.03, "state_mse": 0.01, "total_loss": 0.02}
    rows = [
        {
            "name": "busBS",
            "delta_pbl_mean": 0.002,
            "delta_state_mse": 0.001,
            "delta_total_loss": 0.001,
            "mean_abs_bs": 0.14,
        }
    ]
    text = format_bus_shunt_ablation_diagnostics(weights, baseline, rows)
    assert "busBS" in text
    assert "zero-ablation" in text
    assert "ΔPBL +0.002" in text


def test_audit_rpe_correctness_small_graph():
    branches = [(0, 1, 0.1, 0.1), (1, 2, 0.1, 0.1)]
    ybus = np.zeros((3, 3), dtype=np.complex64)
    ybus[0, 1] = ybus[1, 0] = 2.0 + 1.0j
    features, component = meshed_rpe(3, branches, ybus)
    network = Network(
        ybus=torch.from_numpy(ybus),
        rpe=torch.from_numpy(features),
        component=torch.from_numpy(component.astype(np.int64)),
        types=torch.tensor([0, 1, 2], dtype=torch.long),
        branches=tuple(branches),
    )
    checks = audit_rpe_correctness(network)
    assert all(check.passed for check in checks)


def test_move_to_preserves_global_rpe_scales_for_audit():
    """Device copies must keep s_R/s_X; otherwise audit falls back to per-component norms and fails."""
    branches = [(0, 1, 0.02, 0.06), (1, 2, 0.03, 0.08), (0, 2, 0.04, 0.1)]
    ybus = np.zeros((3, 3), dtype=np.complex64)
    # Global unit scales (train floor) differ from per-component peaks on this graph.
    features, component = meshed_rpe(3, branches, ybus, re_scale=1.0, im_scale=1.0)
    host = Network(
        ybus=torch.from_numpy(ybus),
        rpe=torch.from_numpy(features),
        component=torch.from_numpy(component.astype(np.int64)),
        types=torch.tensor([0, 1, 2], dtype=torch.long),
        branches=tuple(branches),
        rpe_re_scale=1.0,
        rpe_im_scale=1.0,
    )
    assert all(check.passed for check in audit_rpe_correctness(host))
    moved = tpgt.move_to(host, torch.device("cpu"))
    assert moved.rpe_re_scale == 1.0 and moved.rpe_im_scale == 1.0
    assert all(check.passed for check in audit_rpe_correctness(moved))
    # Regression: dropping scales makes recompute_from_branches fail against globally scaled RPE.
    dropped = Network(
        ybus=moved.ybus,
        rpe=moved.rpe,
        component=moved.component,
        types=moved.types,
        branches=moved.branches,
        rpe_re_scale=None,
        rpe_im_scale=None,
    )
    recompute = next(check for check in audit_rpe_correctness(dropped) if check.name == "recompute_from_branches")
    assert not recompute.passed


def tiny_consistent_split(cases: int = 4) -> tuple[Split, Network]:
    """3-bus case whose P/Q labels exactly satisfy AC power flow for the sampled V, theta."""
    branches = [(0, 1, 0.02, 0.06), (1, 2, 0.03, 0.08), (0, 2, 0.04, 0.1)]
    ybus = np.zeros((3, 3), dtype=np.complex128)
    for origin, destination, resistance, reactance in branches:
        admittance = 1.0 / complex(resistance, reactance)
        ybus[origin, origin] += admittance
        ybus[destination, destination] += admittance
        ybus[origin, destination] -= admittance
        ybus[destination, origin] -= admittance
    rng = np.random.default_rng(0)
    voltage = 1.0 + 0.03 * rng.standard_normal((cases, 3))
    angle = 0.05 * rng.standard_normal((cases, 3))
    angle[:, 2] = 0.0
    solved = voltage * np.exp(1j * angle)
    injection = solved * np.conj(solved @ ybus.T)
    features, component = meshed_rpe(3, branches, ybus.astype(np.complex64))
    types = torch.tensor([0, 1, 2], dtype=torch.long)
    component_t = torch.from_numpy(component.astype(np.int64))
    network = Network(
        ybus=torch.from_numpy(ybus.astype(np.complex64)),
        rpe=torch.from_numpy(features),
        component=component_t,
        types=types,
        pair_type_masks=type_pair_masks(types, component_t),
        branches=tuple(branches),
    )
    split = Split(
        name="tiny",
        count=cases,
        power_p=torch.from_numpy(injection.real.astype(np.float32)),
        power_q=torch.from_numpy(injection.imag.astype(np.float32)),
        voltage=torch.from_numpy(voltage.astype(np.float32)),
        angle=torch.from_numpy(angle.astype(np.float32)),
        network_index=torch.zeros(cases, dtype=torch.long),
        networks=[network],
    )
    return split, network


def test_audit_residual_correctness_on_consistent_labels():
    cases = 4
    split, network = tiny_consistent_split(cases)
    checks = audit_residual_correctness(TPGT(), split, network, np.arange(cases), torch.device("cpu"))
    assert [check.name for check in checks] == [
        "oracle_labels_torch",
        "torch_vs_numpy_reference",
        "model_residual_matches_output",
    ]
    assert all(check.passed for check in checks), [check.detail for check in checks]


def test_train_probe_split_is_fixed_evenly_spaced_subset():
    split, _network = tiny_consistent_split(cases=10)
    probe = train_probe_split(split, 4)
    assert probe.count == 4
    assert torch.equal(probe.voltage[0], split.voltage[0])
    assert torch.equal(probe.voltage[-1], split.voltage[-1])
    assert probe.networks is split.networks
    assert train_probe_split(split, 50).count == 10


def test_contingency_diagnostics_report(capsys):
    split, network = tiny_consistent_split(cases=6)
    model = TPGT()
    per_network: dict[int, list[float]] = {}
    evaluate_pbl(model, split, [network], torch.device("cpu"), batch_size=4, per_network_out=per_network)
    assert per_network[0][1] == 6
    print_contingency_diagnostics(
        model, "tiny", split, [network], torch.device("cpu"), network, per_network
    )
    text = capsys.readouterr().out
    assert "Per-topology PBL" in text
    assert "Generator outages change bus types" in text
    assert "C/ρ generally move" in text
    assert "RPE correctness on 1 topologies: OK" in text
    assert "Residual correctness on 1 topologies: OK" in text
    assert "shift 0" in text


def test_contingency_class_separates_line_and_generator_outages():
    _split, base = tiny_consistent_split()
    line = Network(
        ybus=base.ybus * 0.5, rpe=base.rpe, component=base.component, types=base.types
    )
    generator = Network(
        ybus=base.ybus, rpe=base.rpe, component=base.component, types=torch.tensor([0, 0, 2])
    )
    assert contingency_class(base, base) == "unchanged"
    assert contingency_class(line, base) == "line outage"
    assert contingency_class(generator, base) == "generator outage"


def test_nodal_residual_returns_float32_and_is_exact_on_labels():
    split, network = tiny_consistent_split()
    specified, target = pack_case(split.power_p, split.power_q, split.voltage, split.angle, network.types)
    residual = nodal_residual(specified, target, network.types, network.ybus)
    assert residual.dtype == torch.float32
    assert float(residual.abs().max()) < 1e-5
