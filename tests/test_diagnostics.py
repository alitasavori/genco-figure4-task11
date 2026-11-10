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
    audit_residual_correctness,
    audit_rpe_correctness,
    bus_shunt_b,
    bus_shunt_gb,
    node_encoder_features,
    contingency_class,
    evaluate_pbl,
    nodal_residual,
    pack_case,
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
    try:
        tpgt.USE_NODE_SHUNT = True
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
        assert torch.allclose(feats[0, 0], torch.tensor([0.1, 0.2, 0.0, 0.0, -0.4]))
        assert torch.allclose(feats[0, 1], torch.tensor([0.3, 0.0, 1.0, 0.0, 0.14]))
        assert torch.allclose(feats[0, 2], torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    finally:
        tpgt.USE_NODE_SHUNT = previous


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
    assert "unchanged" in text
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
