"""Vectorized ``meshed_rpe`` reproduces the original loop implementation exactly."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import RPE_DIM, component_admittance_scale, meshed_rpe  # noqa: E402


def loop_meshed_rpe(
    bus_count: int,
    branches: list[tuple[int, int, float, float]],
    ybus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``meshed_rpe`` as of a707d1c (pure Python loops)."""
    y = ybus.astype(np.complex128)
    y_abs = np.abs(y)
    adjacency: list[set[int]] = [set() for _ in range(bus_count)]
    weight = np.zeros((bus_count, bus_count), dtype=np.float64)
    for origin, destination, resistance, reactance in branches:
        magnitude = math.hypot(resistance, reactance)
        if magnitude <= 0:
            continue
        branch_weight = 1.0 / magnitude
        weight[origin, destination] += branch_weight
        weight[destination, origin] += branch_weight
        adjacency[origin].add(destination)
        adjacency[destination].add(origin)

    component = np.full(bus_count, -1, dtype=np.int64)
    component_id = 0
    for start in range(bus_count):
        if component[start] >= 0:
            continue
        queue = [start]
        component[start] = component_id
        head = 0
        while head < len(queue):
            bus = queue[head]
            head += 1
            for neighbor in adjacency[bus]:
                if component[neighbor] >= 0:
                    continue
                component[neighbor] = component_id
                queue.append(neighbor)
        component_id += 1

    hop = np.zeros((bus_count, bus_count), dtype=np.float64)
    for start in range(bus_count):
        if not adjacency[start]:
            continue
        distances = np.full(bus_count, -1, dtype=np.int64)
        distances[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            bus = queue[head]
            head += 1
            for neighbor in adjacency[bus]:
                if distances[neighbor] >= 0:
                    continue
                distances[neighbor] = distances[bus] + 1
                queue.append(neighbor)
        for destination in range(bus_count):
            if distances[destination] >= 0:
                hop[start, destination] = float(distances[destination])

    scale_y = np.ones(component_id, dtype=np.float64)
    hop_scale = np.ones(component_id, dtype=np.float64)
    resistance_scale = np.ones(component_id, dtype=np.float64)
    effective = np.zeros((bus_count, bus_count), dtype=np.float64)
    for comp in range(component_id):
        buses = np.flatnonzero(component == comp)
        if buses.size == 0:
            continue
        scale_y[comp] = component_admittance_scale(y_abs, component, comp)
        finite_hop = hop[np.ix_(buses, buses)]
        max_hop = float(finite_hop.max()) if buses.size else 0.0
        hop_scale[comp] = max_hop if max_hop > 0 else 1.0
        sub_weight = weight[np.ix_(buses, buses)]
        sub_degree = np.sum(sub_weight, axis=1)
        sub_laplacian = np.diag(sub_degree) - sub_weight
        if buses.size == 1:
            continue
        pseudoinverse = np.linalg.pinv(sub_laplacian)
        for local_i, bus_i in enumerate(buses):
            for local_j, bus_j in enumerate(buses):
                delta = pseudoinverse[local_i, local_i] + pseudoinverse[local_j, local_j]
                delta -= 2.0 * pseudoinverse[local_i, local_j]
                effective[bus_i, bus_j] = max(delta, 0.0)
        max_resistance = float(effective[np.ix_(buses, buses)].max())
        resistance_scale[comp] = max_resistance if max_resistance > 0 else 1.0

    adjacency_flag = np.zeros((bus_count, bus_count), dtype=np.float32)
    for origin in range(bus_count):
        for destination in adjacency[origin]:
            adjacency_flag[origin, destination] = 1.0

    features = np.zeros((bus_count, bus_count, RPE_DIM), dtype=np.float32)
    for origin in range(bus_count):
        comp = int(component[origin])
        if comp < 0:
            continue
        y_scale = scale_y[comp]
        h_scale = hop_scale[comp]
        r_scale = resistance_scale[comp]
        for destination in range(bus_count):
            if component[destination] != comp:
                continue
            features[origin, destination, 0] = y.real[origin, destination] / y_scale
            features[origin, destination, 1] = y.imag[origin, destination] / y_scale
            features[origin, destination, 2] = adjacency_flag[origin, destination]
            if origin == destination:
                features[origin, destination, 3] = 0.0
                features[origin, destination, 4] = 0.0
            else:
                features[origin, destination, 3] = hop[origin, destination] / h_scale
                features[origin, destination, 4] = effective[origin, destination] / r_scale
    return features, component


def random_grid(seed: int, buses: int, *, outages: int, isolated: int) -> tuple[list, np.ndarray]:
    """Meshed random grid with parallel, reversed and zero-impedance branches, then ``outages`` removed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(buses)
    branches = []
    for position in range(1, buses):
        branches.append((int(order[position]), int(order[rng.integers(0, position)])))
    for _ in range(buses // 2):
        first, second = rng.choice(buses, size=2, replace=False)
        branches.append((int(first), int(second)))
    for first, second in branches[: buses // 10]:
        branches.append((second, first))
    table = [(a, b, float(rng.uniform(0.001, 0.05)), float(rng.uniform(0.01, 0.3))) for a, b in branches]
    table.append((table[0][0], table[0][1], 0.0, 0.0))
    removed = set(rng.choice(len(table), size=outages, replace=False).tolist())
    lonely = set(rng.choice(buses, size=isolated, replace=False).tolist())
    kept = [row for index, row in enumerate(table) if index not in removed and row[0] not in lonely and row[1] not in lonely]
    ybus = np.zeros((buses, buses), dtype=np.complex128)
    for origin, destination, resistance, reactance in kept:
        if resistance == 0 and reactance == 0:
            continue
        admittance = 1.0 / complex(resistance, reactance)
        ybus[origin, origin] += admittance
        ybus[destination, destination] += admittance
        ybus[origin, destination] -= admittance
        ybus[destination, origin] -= admittance
    ybus[np.diag_indices(buses)] += 1j * rng.uniform(-0.2, 0.2, buses)
    return kept, ybus.astype(np.complex64)


def test_vectorized_meshed_rpe_matches_loop_bitwise():
    cases = [
        (0, 30, 0, 0),
        (1, 30, 25, 0),
        (2, 60, 40, 3),
        (3, 118, 0, 0),
        (4, 118, 60, 2),
        (5, 118, 120, 5),
        (6, 12, 10, 4),
    ]
    for seed, buses, outages, isolated in cases:
        branches, ybus = random_grid(seed, buses, outages=outages, isolated=isolated)
        expected, expected_component = loop_meshed_rpe(buses, branches, ybus)
        features, component = meshed_rpe(buses, branches, ybus)
        assert features.dtype == expected.dtype == np.float32
        assert np.array_equal(component, expected_component), seed
        assert np.array_equal(features, expected), (seed, float(np.max(np.abs(features - expected))))


def test_vectorized_meshed_rpe_covers_disconnected_grids():
    branches, ybus = random_grid(5, 118, outages=120, isolated=5)
    _features, component = meshed_rpe(118, branches, ybus)
    assert component.max() >= 5


def test_meshed_rpe_without_branches():
    ybus = np.diag(np.ones(4, dtype=np.complex64))
    features, component = meshed_rpe(4, [], ybus)
    expected, expected_component = loop_meshed_rpe(4, [], ybus)
    assert np.array_equal(component, expected_component)
    assert np.array_equal(features, expected)


def test_vectorized_meshed_rpe_is_faster():
    branches, ybus = random_grid(7, 118, outages=2, isolated=0)
    started = time.perf_counter()
    loop_meshed_rpe(118, branches, ybus)
    loop_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(5):
        meshed_rpe(118, branches, ybus)
    vector_seconds = (time.perf_counter() - started) / 5
    assert vector_seconds < loop_seconds
