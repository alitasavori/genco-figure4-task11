"""Power-balance metric, seed aggregation, and result checks.

Model evaluation in the notebooks calls the authors' code. This module is the
reference used by the tests and by the CSV aggregation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

BASE_MVA = 100.0

# PowerModels / PFDelta bus_type codes.
BUS_PQ = 1
BUS_PV = 2
BUS_REF = 3


def specified_quantities(bus_type: int) -> tuple[str, ...]:
    """Quantities fixed by the power-flow specification at this bus.

    PQ buses specify active and reactive demand. PV buses specify active power
    and voltage magnitude. The reference bus specifies voltage magnitude and
    angle. The other two quantities are solved for. Power-balance loss is still
    computed at every bus, including PV and reference buses.
    """
    if bus_type == BUS_PQ:
        return ("P", "Q")
    if bus_type == BUS_PV:
        return ("P", "V")
    if bus_type == BUS_REF:
        return ("V", "theta")
    raise ValueError(f"Unknown bus type: {bus_type}")


def sample_std(values: Sequence[float]) -> float:
    """Unbiased sample standard deviation, matching torch.Tensor.std."""
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        raise ValueError("Standard deviation over seeds needs at least two values.")
    if not np.isfinite(array).all():
        raise ValueError("Seed metrics contain a non-finite value.")
    return float(np.std(array, ddof=1))


def sample_mean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("No seed metrics to average.")
    if not np.isfinite(array).all():
        raise ValueError("Seed metrics contain a non-finite value.")
    return float(np.mean(array))


def _as_1d(name: str, values: Iterable[float], size: int | None = None) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if size is not None and array.size != size:
        raise ValueError(f"{name} length {array.size} does not match {size}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value. It is excluded from the metric.")
    return array


def branch_flows(
    voltage: np.ndarray,
    angle: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    resistance: np.ndarray,
    reactance: np.ndarray,
    g_from: np.ndarray,
    b_from: np.ndarray,
    g_to: np.ndarray,
    b_to: np.ndarray,
    tap: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Line flows in the same form as PFDelta's PowerBalanceAnalysis."""
    admittance = 1.0 / (resistance + 1j * reactance)
    conductance = np.real(admittance)
    susceptance = np.imag(admittance)
    delta_forward = angle[src] - angle[dst]
    delta_backward = angle[dst] - angle[src]
    v_src = voltage[src]
    v_dst = voltage[dst]
    p_src = v_src * v_dst / tap * (
        -conductance * np.cos(delta_forward - shift) - susceptance * np.sin(delta_forward - shift)
    ) + conductance * (v_src / tap) ** 2
    p_dst = v_dst * v_src / tap * (
        -conductance * np.cos(delta_backward - shift) - susceptance * np.sin(delta_backward - shift)
    ) + conductance * v_dst**2
    q_src = v_src * v_dst / tap * (
        -conductance * np.sin(delta_forward - shift) + susceptance * np.cos(delta_forward - shift)
    ) - (susceptance + b_from) * (v_src / tap) ** 2
    q_dst = v_dst * v_src / tap * (
        -conductance * np.sin(delta_backward - shift) + susceptance * np.cos(delta_backward - shift)
    ) - (susceptance + b_to) * v_dst**2
    # g_from / g_to are part of the PFDelta edge attribute layout. The reference
    # implementation records them and does not add a second shunt term here.
    del g_from, g_to
    p_bus = np.zeros_like(voltage)
    q_bus = np.zeros_like(voltage)
    np.add.at(p_bus, src, p_src)
    np.add.at(p_bus, dst, p_dst)
    np.add.at(q_bus, src, q_src)
    np.add.at(q_bus, dst, q_dst)
    return p_bus, q_bus


def mean_power_balance_loss(
    voltage: Sequence[float],
    angle: Sequence[float],
    p_net: Sequence[float],
    q_net: Sequence[float],
    shunt_g: Sequence[float],
    shunt_b: Sequence[float],
    src: Sequence[int],
    dst: Sequence[int],
    resistance: Sequence[float],
    reactance: Sequence[float],
    g_from: Sequence[float],
    b_from: Sequence[float],
    g_to: Sequence[float],
    b_to: Sequence[float],
    tap: Sequence[float],
    shift: Sequence[float],
    bus_types: Sequence[int] | None = None,
) -> float:
    """Mean over buses of sqrt(dP^2 + dQ^2), including every bus type."""
    v = _as_1d("voltage", voltage)
    n_bus = v.size
    theta = _as_1d("angle", angle, n_bus)
    pnet = _as_1d("p_net", p_net, n_bus)
    qnet = _as_1d("q_net", q_net, n_bus)
    gs = _as_1d("shunt_g", shunt_g, n_bus)
    bs = _as_1d("shunt_b", shunt_b, n_bus)
    if bus_types is not None:
        kinds = _as_1d("bus_types", bus_types, n_bus)
        for kind in kinds:
            specified_quantities(int(kind))
    src_i = np.asarray(list(src), dtype=int)
    dst_i = np.asarray(list(dst), dtype=int)
    p_bus, q_bus = branch_flows(
        v,
        theta,
        src_i,
        dst_i,
        _as_1d("resistance", resistance),
        _as_1d("reactance", reactance),
        _as_1d("g_from", g_from),
        _as_1d("b_from", b_from),
        _as_1d("g_to", g_to),
        _as_1d("b_to", b_to),
        _as_1d("tap", tap),
        _as_1d("shift", shift),
    )
    delta_p = pnet - p_bus - (v**2) * gs
    delta_q = qnet - q_bus + (v**2) * bs
    if not np.isfinite(delta_p).all() or not np.isfinite(delta_q).all():
        raise ValueError("Power-balance residual is non-finite.")
    per_bus = np.sqrt(delta_p**2 + delta_q**2)
    return float(per_bus.mean())


def aggregate_c2i(category_means: dict[str, float], counts: dict[str, int] | None = None) -> float:
    """Close-to-infeasible score across N, N-1, and N-2 nose sets.

    PFDelta's scripts/test_error.py takes the unweighted mean of the three
    topology means. When each topology has the same number of samples (200 in
    the official test split) that equals the mean over samples. If counts are
    supplied and differ, this returns the sample-weighted mean, which is the
    GENCO paper's "averaged across all samples" definition.
    """
    keys = ("C2I-N", "C2I-N-1", "C2I-N-2")
    missing = [key for key in keys if key not in category_means]
    if missing:
        raise ValueError(f"C2I is incomplete, missing {missing}. The published C2I value is not a substitute.")
    values = [float(category_means[key]) for key in keys]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("C2I category mean is non-finite.")
    if not counts:
        return float(np.mean(values))
    weights = [int(counts[key]) for key in keys]
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"C2I sample counts must be positive, got {weights}.")
    if len(set(weights)) == 1:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def assert_no_overlap(train_ids: Iterable[str], test_ids: Iterable[str]) -> None:
    overlap = set(map(str, train_ids)) & set(map(str, test_ids))
    if overlap:
        sample = sorted(overlap)[:10]
        raise AssertionError(f"Training and test ids overlap, for example {sample}.")


def assert_split_counts(loaded: dict[tuple[str, str], int], *, split_name: str) -> None:
    if split_name == "test":
        expected = EXPECTED_TEST_FROM_COMMON()
        if loaded != expected:
            raise AssertionError(f"Test counts {loaded} do not match the official split {expected}.")
        return
    raise ValueError(split_name)


def EXPECTED_TEST_FROM_COMMON() -> dict[tuple[str, str], int]:
    from common import EXPECTED_TEST_COUNTS

    return dict(EXPECTED_TEST_COUNTS)
