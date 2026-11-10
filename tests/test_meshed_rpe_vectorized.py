"""Vectorized ``meshed_rpe`` matches loop reference (complex Z_eff Re/Im)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import RPE_DIM, _effective_impedance_matrix, meshed_rpe  # noqa: E402


def loop_meshed_rpe(
    bus_count: int,
    branches: list[tuple[int, int, float, float]],
    ybus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference loop form of complex Z_eff ``meshed_rpe``."""
    del ybus
    admittance = np.zeros((bus_count, bus_count), dtype=np.complex128)
    adjacency: list[set[int]] = [set() for _ in range(bus_count)]
    for origin, destination, resistance, reactance in branches:
        impedance = resistance + 1j * reactance
        if abs(impedance) <= 0.0:
            continue
        y = 1.0 / impedance
        i, j = int(origin), int(destination)
        admittance[i, j] += y
        admittance[j, i] += y
        adjacency[i].add(j)
        adjacency[j].add(i)

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

    re_scale = np.ones(component_id, dtype=np.float64)
    im_scale = np.ones(component_id, dtype=np.float64)
    z_eff = np.zeros((bus_count, bus_count), dtype=np.complex128)
    for comp in range(component_id):
        buses = np.flatnonzero(component == comp)
        if buses.size <= 1:
            continue
        sub_y = admittance[np.ix_(buses, buses)]
        sub_laplacian = np.diag(np.sum(sub_y, axis=1)) - sub_y
        block = _effective_impedance_matrix(sub_laplacian)
        z_eff[np.ix_(buses, buses)] = block
        off = block.copy()
        np.fill_diagonal(off, 0.0)
        re_scale[comp] = max(float(np.max(np.abs(off.real))), 1e-30) if off.size else 1.0
        im_scale[comp] = max(float(np.max(np.abs(off.imag))), 1e-30) if off.size else 1.0

    features = np.zeros((bus_count, bus_count, RPE_DIM), dtype=np.float32)
    for origin in range(bus_count):
        comp = int(component[origin])
        if comp < 0:
            continue
        for destination in range(bus_count):
            if component[destination] != comp or origin == destination:
                continue
            features[origin, destination, 0] = z_eff[origin, destination].real / re_scale[comp]
            features[origin, destination, 1] = z_eff[origin, destination].imag / im_scale[comp]
    return features, component


def random_grid(seed: int, buses: int, *, outages: int, isolated: int) -> tuple[list, np.ndarray]:
    """Meshed random grid with parallel branches, then ``outages`` removed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(buses)
    branches = []
    for position in range(1, buses):
        branches.append((int(order[position]), int(order[rng.integers(0, position)])))
    for _ in range(buses // 2):
        first, second = rng.choice(buses, size=2, replace=False)
        branches.append((int(first), int(second)))
    for index in range(len(branches)):
        origin, destination = branches[index]
        resistance = float(rng.uniform(0.01, 0.2))
        reactance = float(rng.uniform(0.05, 0.4))
        branches[index] = (origin, destination, resistance, reactance)
    for _ in range(outages):
        if not branches:
            break
        branches.pop(int(rng.integers(0, len(branches))))
    for bus in rng.choice(buses, size=isolated, replace=False):
        branches = [branch for branch in branches if bus not in branch[:2]]
    ybus = np.zeros((buses, buses), dtype=np.complex128)
    for origin, destination, resistance, reactance in branches:
        z = resistance + 1j * reactance
        if abs(z) <= 0:
            continue
        y = 1.0 / z
        ybus[origin, destination] -= y
        ybus[destination, origin] -= y
        ybus[origin, origin] += y
        ybus[destination, destination] += y
    return branches, ybus


def test_parallel_lines_complex_zeff_toy():
    branches = [(0, 1, 1.0, 1.0), (0, 1, 1.0, 3.0)]
    y1 = 1.0 / (1 + 1j)
    y2 = 1.0 / (1 + 3j)
    ys = y1 + y2
    z_block = _effective_impedance_matrix(np.array([[ys, -ys], [-ys, ys]], dtype=np.complex128))
    z_par = (1 + 1j) * (1 + 3j) / (2 + 4j)  # 0.6 + 0.8j
    assert np.allclose(z_block[0, 1], z_par, atol=1e-5)
    features, _ = meshed_rpe(2, branches, np.zeros((2, 2), dtype=np.complex64))
    # Only one off-diagonal pair: both channels divide by their own max → 1.
    assert np.allclose(features[0, 1], [1.0, 1.0], atol=1e-5)


def test_vectorized_matches_loop_on_random_grids():
    for seed in range(8):
        branches, ybus = random_grid(seed, buses=24, outages=seed % 5, isolated=seed % 3)
        expected, expected_component = loop_meshed_rpe(24, branches, ybus)
        got, got_component = meshed_rpe(24, branches, ybus)
        assert np.array_equal(got_component, expected_component)
        assert np.allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_meshed_rpe_runtime_smoke():
    branches, ybus = random_grid(0, buses=64, outages=3, isolated=1)
    started = time.perf_counter()
    features, _component = meshed_rpe(64, branches, ybus)
    elapsed = time.perf_counter() - started
    assert features.shape == (64, 64, RPE_DIM)
    assert elapsed < 5.0
