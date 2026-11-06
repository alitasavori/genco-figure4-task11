"""Forward diagnostics helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import (  # noqa: E402
    RPE_DIM,
    TPGT,
    Network,
    Split,
    audit_residual_correctness,
    audit_rpe_correctness,
    evaluate_pbl,
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
    rpe = torch.randn(5, 5, 10)
    rows = summarize_rpe(rpe)
    assert len(rows) == 10
    assert rows[0]["off_mean"] == rows[0]["off_mean"]  # finite


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
        model, "tiny", split, [network], torch.device("cpu"), network.rpe, per_network
    )
    text = capsys.readouterr().out
    assert "Per-topology PBL" in text
    assert "RPE correctness on 1 topologies: OK" in text
    assert "Residual correctness on 1 topologies: OK" in text
    assert "shift 0" in text
