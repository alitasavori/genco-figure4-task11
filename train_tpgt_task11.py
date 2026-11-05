"""Train the unshared type-pair graph Transformer on IEEE 118-bus Task 1.1.

Option II from the draft: each correction stage has its own weights.
The hidden sizes, depth, batch, and epoch cap below are implementation
choices so a Colab GPU can finish a run. They are not paper hyperparameters.

IEEE 118 is meshed, so pair features use the full in-service topology:
admittance, adjacency, hop distance, effective resistance, and weighted
random walks. See `meshed_rpe`.

Performance notes (Colab / large train split)
---------------------------------------------
* **Drive I/O**: The 48600-case train/valid tensors live under the checkpoint
  tree on Google Drive. ``torch.load`` from Drive is often minutes slower than
  from ``/content`` local disk. On Colab, ``split_cache_dirs`` mirrors ``.pt``
  caches to ``/content/tpgt_task11_cache`` after the first Drive read or build.
* **``meshed_rpe``**: Computed only when building or refreshing the split cache,
  once per *distinct* in-service topology (see ``read_split`` representatives),
  not per scenario and not per training batch.
* **``pair_encoder``**: Runs once per forward on the static ``118×118`` RPE for
  the batch's topology. Weights train, so training must re-encode every step;
  in ``eval`` mode encoded pairs are cached on ``Network.pair_embed``.
* **Type-pair attention**: ``CorrectionStage._messages`` fuses all nine
  receiver–sender masks and both heads into batched ``einsum`` ops over
  ``118×118`` (one kernel graph per stage instead of eighteen Python loops).
  Optional ``GENCO_TPGT_COMPILE=1`` or ``--compile`` wraps the model in
  ``torch.compile`` (may differ slightly in numerics).
* **Batch 64 vs 16**: Four times more cases per step increases matmul work
  roughly linearly; per-case time should be similar if the GPU is saturated.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import (
    CATEGORIES,
    SEED_COLUMNS,
    SUMMARY_COLUMNS,
    ensure_dirs,
    read_csv,
    upsert_csv,
)
from metrics import aggregate_c2i, sample_mean, sample_std
from plot_task1_1 import PALETTE, PUBLISHED_TASK11
from storage import ROOT, in_colab, mount_drive

# Implementation choices, not values from the draft.
D = 32
D_E = 16
D_A = 16
HEADS = 2
DEPTH = 3
ETA = 1.0
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100
# Drive checkpoints (and results CSV / comparison plot) every 3rd epoch and at the final epoch.
# The four held-out test PBL scores are printed every epoch (see on_epoch_end in train_seed).
CHECKPOINT_EVERY = 3
SEEDS = (0, 1, 2)
BASE_MVA = 100.0
LAMBDA_STATE = 0.85
LAMBDA_PHYS = 0.15
MODEL_NAME = "TPGT"
TYPE_NAMES = ("PQ", "PV", "Slack")
CACHE_VERSION = "tpgt-cache-2"
RPE_WALK_STEPS = 5
RPE_DIM = RPE_WALK_STEPS + 5
# Bumped when pair-encoder input size or architecture changes; old Drive checkpoints are ignored.
CHECKPOINT_VERSION = "meshed-rpe-10-v3"
# PQ voltage unknown is clamped to this band when evaluating Ybus injections (p.u.).
V_MIN_PU = 0.05
V_MAX_PU = 2.0
# Step size on angle unknowns; full ETA on P, Q, and V magnitudes.
ETA_ANGLE = 0.25

TEST_PARTS = (
    ("N", "n", "feasible"),
    ("N-1", "n-1", "feasible"),
    ("N-2", "n-2", "feasible"),
    ("C2I-N", "n", "nose"),
    ("C2I-N-1", "n-1", "nose"),
    ("C2I-N-2", "n-2", "nose"),
)
EXPECTED_COUNTS = {
    ("train", "n", "feasible"): 48600,
    ("valid", "n", "feasible"): 5400,
    ("test", "n", "feasible"): 2000,
    ("test", "n-1", "feasible"): 2000,
    ("test", "n-2", "feasible"): 2000,
    ("test", "n", "nose"): 200,
    ("test", "n-1", "nose"): 200,
    ("test", "n-2", "nose"): 200,
}


def ensure_runtime_packages() -> None:
    missing = []
    for name in ("pyarrow", "huggingface_hub", "matplotlib"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print("Installing", " ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_angle(delta: torch.Tensor) -> torch.Tensor:
    return torch.remainder(delta + math.pi, 2 * math.pi) - math.pi


def rho_nonlinearity(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.relu(values)) - torch.sqrt(torch.relu(-values))


def type_pair_masks(types: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
    """Boolean masks (9, buses, buses) for each receiver–sender type pair within a component."""
    buses = types.shape[0]
    same_component = component.view(buses, 1) == component.view(1, buses)
    masks = []
    for pair_id in range(9):
        receiver, sender = divmod(pair_id, 3)
        type_mask = (types.view(buses, 1) == receiver) & (types.view(1, buses) == sender)
        masks.append(type_mask & same_component)
    return torch.stack(masks, dim=0)


def wants_torch_compile() -> bool:
    if "--compile" in sys.argv or "--fast" in sys.argv:
        return True
    flag = os.environ.get("GENCO_TPGT_COMPILE", os.environ.get("GENCO_TPGT_FAST", ""))
    return flag.lower() in ("1", "true", "yes")


class MLP(nn.Module):
    """Two-layer map. The hidden width `D` is an implementation choice."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, D), nn.ReLU(), nn.Linear(D, out_features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def _init_matrices(weight: torch.Tensor) -> None:
    if weight.dim() < 2:
        nn.init.zeros_(weight)
        return
    view = weight.view(-1, weight.shape[-2], weight.shape[-1])
    for matrix in view:
        nn.init.xavier_uniform_(matrix)


class CorrectionStage(nn.Module):
    """One unshared GRIT correction. Nine type pairs, each with its own head weights."""

    def __init__(self) -> None:
        super().__init__()
        shape = (9, HEADS)
        self.w_q = nn.Parameter(torch.empty(*shape, D_A, D))
        self.w_k = nn.Parameter(torch.empty(*shape, D_A, D))
        self.w_v = nn.Parameter(torch.empty(*shape, D_A, D))
        self.w_ew = nn.Parameter(torch.empty(*shape, D_A, D_E))
        self.w_eb = nn.Parameter(torch.empty(*shape, D_A, D_E))
        self.w_a = nn.Parameter(torch.empty(*shape, 1, D_A))
        self.w_ev = nn.Parameter(torch.empty(*shape, D_A, D_A))
        self.w_o = nn.Parameter(torch.empty(HEADS, D, D_A))
        for parameter in (self.w_q, self.w_k, self.w_v, self.w_ew, self.w_eb, self.w_a, self.w_ev, self.w_o):
            _init_matrices(parameter)
        self.update = nn.ModuleList(MLP(D + D + 2, D) for _ in range(3))
        self.decode = nn.ModuleList(MLP(D, 2) for _ in range(3))
        self.norm = nn.LayerNorm(D)

    def forward(
        self,
        hidden: torch.Tensor,
        unknown: torch.Tensor,
        specified: torch.Tensor,
        pair: torch.Tensor,
        types: torch.Tensor,
        pair_type_masks: torch.Tensor,
        ybus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_in = nodal_residual(specified, unknown, types, ybus)
        message = self._messages(hidden, pair, pair_type_masks)
        updated = torch.zeros_like(hidden)
        for type_id, update in enumerate(self.update):
            mask = types == type_id
            if not bool(mask.any()):
                continue
            features = torch.cat((hidden[:, mask], message[:, mask], residual_in[:, mask]), dim=-1)
            updated[:, mask] = update(features)
        hidden_next = self.norm(hidden + updated)
        delta = torch.zeros_like(unknown)
        for type_id, decoder in enumerate(self.decode):
            mask = types == type_id
            if not bool(mask.any()):
                continue
            delta[:, mask] = decoder(hidden_next[:, mask])
        pq = types == 0
        pv = types == 1
        if bool(pq.any()):
            delta[:, pq, 1] = delta[:, pq, 1] * ETA_ANGLE
        if bool(pv.any()):
            delta[:, pv, 1] = delta[:, pv, 1] * ETA_ANGLE
        unknown_next = constrain_unknown(unknown + ETA * delta, types)
        residual_out = nodal_residual(specified, unknown_next, types, ybus)
        return hidden_next, unknown_next, residual_out

    def _messages(
        self,
        hidden: torch.Tensor,
        pair: torch.Tensor,
        pair_type_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch, buses, _ = hidden.shape
        mask_view = pair_type_masks.view(1, 9, 1, buses, buses)
        # Contract hidden dim D with w_*; attention lives in D_A (same as the pre-batch matmul loops).
        query = torch.einsum("bnd,phad->bphna", hidden, self.w_q)
        key = torch.einsum("bnd,phad->bphna", hidden, self.w_k)
        mixed = rho_nonlinearity(query.unsqueeze(-2) + key.unsqueeze(-3))
        projected = torch.einsum("nme,phae->phnma", pair, self.w_ew)
        bias = torch.einsum("nme,phae->phnma", pair, self.w_eb)
        interaction = torch.relu(mixed * projected.unsqueeze(0) + bias.unsqueeze(0))
        all_scores = (interaction * self.w_a.unsqueeze(0).unsqueeze(-2)).sum(dim=-1)
        scores = all_scores.masked_fill(~mask_view, float("-inf")).amax(dim=1)
        value = torch.einsum("bnd,phad->bphna", hidden, self.w_v)
        content = value.unsqueeze(3) + torch.einsum("bphnma,phxa->bphnmx", interaction, self.w_ev)
        values = (content * mask_view.unsqueeze(-1)).sum(dim=1)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        row_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)
        mixed_heads = (weights.unsqueeze(-1) * values).sum(dim=3)
        return torch.einsum("bhna,hda->bnd", mixed_heads, self.w_o)


class TPGT(nn.Module):
    """Fixed-depth type-pair graph Transformer with independently learned stages."""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(MLP(2, D) for _ in range(3))
        self.pair_encoder = MLP(RPE_DIM, D_E)
        self.stages = nn.ModuleList(CorrectionStage() for _ in range(DEPTH))

    def forward(
        self,
        specified: torch.Tensor,
        types: torch.Tensor,
        rpe: torch.Tensor,
        pair_type_masks: torch.Tensor,
        ybus: torch.Tensor,
        pair: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if pair is None:
            pair = self.pair_encoder(rpe)
        hidden = specified.new_zeros(specified.shape[0], specified.shape[1], D)
        unknown = flat_start(specified, types)
        for type_id, encoder in enumerate(self.encoders):
            mask = types == type_id
            if bool(mask.any()):
                hidden[:, mask] = encoder(specified[:, mask])
        residuals = []
        for stage in self.stages:
            hidden, unknown, residual = stage(hidden, unknown, specified, pair, types, pair_type_masks, ybus)
            residuals.append(residual)
        return unknown, residuals


def constrain_unknown(unknown: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Keep states in a range where the AC equations are meaningful."""
    out = unknown.clone()
    pq = types == 0
    if bool(pq.any()):
        out[:, pq, 0] = unknown[:, pq, 0].clamp(min=V_MIN_PU, max=V_MAX_PU)
    angle_buses = (types == 0) | (types == 1)
    if bool(angle_buses.any()):
        out[:, angle_buses, 1] = wrap_angle(unknown[:, angle_buses, 1])
    return out


def flat_start(specified: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    unknown = torch.zeros_like(specified)
    unknown[:, types == 0, 0] = 1.0
    return unknown


def pack_case(
    power_p: torch.Tensor,
    power_q: torch.Tensor,
    voltage: torch.Tensor,
    angle: torch.Tensor,
    types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Specified quantities stay fixed. The target is the unknown part of the solution."""
    specified = torch.zeros(*power_p.shape, 2, device=power_p.device, dtype=power_p.dtype)
    target = torch.zeros_like(specified)
    pq = types == 0
    pv = types == 1
    slack = types == 2
    specified[:, pq, 0] = power_p[:, pq]
    specified[:, pq, 1] = power_q[:, pq]
    target[:, pq, 0] = voltage[:, pq]
    target[:, pq, 1] = angle[:, pq]
    specified[:, pv, 0] = power_p[:, pv]
    specified[:, pv, 1] = voltage[:, pv]
    target[:, pv, 0] = power_q[:, pv]
    target[:, pv, 1] = angle[:, pv]
    specified[:, slack, 0] = voltage[:, slack]
    specified[:, slack, 1] = angle[:, slack]
    target[:, slack, 0] = power_p[:, slack]
    target[:, slack, 1] = power_q[:, slack]
    return specified, target


def assemble(
    specified: torch.Tensor,
    unknown: torch.Tensor,
    types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    power_p = torch.empty(specified.shape[:2], device=specified.device, dtype=specified.dtype)
    power_q = torch.empty_like(power_p)
    voltage = torch.empty_like(power_p)
    angle = torch.empty_like(power_p)
    pq = types == 0
    pv = types == 1
    slack = types == 2
    power_p[:, pq] = specified[:, pq, 0]
    power_q[:, pq] = specified[:, pq, 1]
    voltage[:, pq] = unknown[:, pq, 0]
    angle[:, pq] = unknown[:, pq, 1]
    power_p[:, pv] = specified[:, pv, 0]
    voltage[:, pv] = specified[:, pv, 1]
    power_q[:, pv] = unknown[:, pv, 0]
    angle[:, pv] = unknown[:, pv, 1]
    voltage[:, slack] = specified[:, slack, 0]
    angle[:, slack] = specified[:, slack, 1]
    power_p[:, slack] = unknown[:, slack, 0]
    power_q[:, slack] = unknown[:, slack, 1]
    return power_p, power_q, voltage, angle


def nodal_residual(
    specified: torch.Tensor,
    unknown: torch.Tensor,
    types: torch.Tensor,
    ybus: torch.Tensor,
) -> torch.Tensor:
    """AC mismatch in per unit. Ybus is already per unit, so this does not divide by baseMVA again."""
    physical = constrain_unknown(unknown, types)
    power_p, power_q, voltage, angle = assemble(specified, physical, types)
    voltage_complex = voltage.to(torch.complex64) * torch.exp(1j * angle.to(torch.complex64))
    current = torch.matmul(ybus, voltage_complex.transpose(0, 1)).transpose(0, 1)
    flow = voltage_complex * torch.conj(current)
    return torch.stack((power_p - flow.real, power_q - flow.imag), dim=-1)


@dataclass
class TaskLossBreakdown:
    state_mse: torch.Tensor
    physics_sum: torch.Tensor
    physics_stages: tuple[torch.Tensor, ...]
    total: torch.Tensor


def task_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    residuals: list[torch.Tensor],
    types: torch.Tensor,
) -> TaskLossBreakdown:
    error = prediction - target
    angle_unknown = (types == 0) | (types == 1)
    error[:, angle_unknown, 1] = wrap_angle(error[:, angle_unknown, 1])
    state_mse = torch.mean(torch.sum(error * error, dim=-1))
    physics_stages = tuple(
        torch.mean(torch.sum(residual * residual, dim=-1)) for residual in residuals
    )
    physics_sum = torch.stack(physics_stages).sum()
    total = LAMBDA_STATE * state_mse + LAMBDA_PHYS * physics_sum
    return TaskLossBreakdown(state_mse, physics_sum, physics_stages, total)


def mean_pbl(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2))


@dataclass
class Network:
    ybus: torch.Tensor
    rpe: torch.Tensor
    component: torch.Tensor
    types: torch.Tensor
    pair_type_masks: torch.Tensor | None = field(default=None, repr=False)
    pair_embed: torch.Tensor | None = field(default=None, repr=False)


@dataclass
class Split:
    name: str
    count: int
    power_p: torch.Tensor
    power_q: torch.Tensor
    voltage: torch.Tensor
    angle: torch.Tensor
    network_index: torch.Tensor
    networks: list[Network]


def meshed_rpe(
    bus_count: int,
    branches: list[tuple[int, int, float, float]],
    ybus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ten pair features on the full in-service meshed graph.

    Order: Re(Y)/s_Y, Im(Y)/s_Y, A_ij, hop/d_max, eff_res/r_max, then
    (P^k)_ij for k = 1..RPE_WALK_STEPS. Branch r and x build W; Y comes
    from the case Ybus used in the power-flow residual.

    For a non-adjacent pair, Y_ij is zero but hop and effective resistance
    are positive. That is expected: direct admittance is local, while hop
    and resistance summarize the full meshed graph. They are not contradictory.
    """
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

    degree = np.sum(weight, axis=1)
    transition = np.zeros_like(weight)
    positive_degree = degree > 0
    transition[positive_degree] = weight[positive_degree] / degree[positive_degree, None]
    walk_powers = [transition.copy()]
    for _ in range(2, RPE_WALK_STEPS + 1):
        walk_powers.append(walk_powers[-1] @ transition)

    scale_y = np.ones(component_id, dtype=np.float64)
    hop_scale = np.ones(component_id, dtype=np.float64)
    resistance_scale = np.ones(component_id, dtype=np.float64)
    effective = np.zeros((bus_count, bus_count), dtype=np.float64)
    for comp in range(component_id):
        buses = np.flatnonzero(component == comp)
        if buses.size == 0:
            continue
        off_diagonal = y_abs[np.ix_(buses, buses)]
        mean_admittance = float(off_diagonal.mean()) if off_diagonal.size else 0.0
        scale_y[comp] = mean_admittance if mean_admittance > 0 else 1.0
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
            for step, power in enumerate(walk_powers, start=5):
                features[origin, destination, step] = power[origin, destination]

    hop_symmetry = np.max(np.abs(hop - hop.T))
    if hop_symmetry > 1e-5:
        raise RuntimeError("Hop distance on the meshed graph is not symmetric.")
    resistance_symmetry = np.max(np.abs(effective - effective.T))
    if resistance_symmetry > 1e-4:
        raise RuntimeError("Effective resistance is not symmetric.")
    if not np.isfinite(features).all():
        raise RuntimeError("Meshed RPE contains non-finite values.")
    return features, component


def _blocks(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[0] % width != 0:
        raise RuntimeError(f"{values.shape[0]} rows do not divide into blocks of {width}.")
    return values.reshape(-1, width)


def scenario_count(raw: Path) -> int:
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(raw / "bus_data.parquet").metadata.num_rows
    if rows % 118 != 0:
        raise RuntimeError(f"{raw} bus rows {rows} are not a multiple of 118.")
    return rows // 118


def read_split(raw: Path, name: str, limit: int | None = None) -> Split:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    full = scenario_count(raw)
    count = full if limit is None else min(limit, full)
    branch_rows = pq.ParquetFile(raw / "branch_data.parquet").metadata.num_rows
    if branch_rows % full != 0:
        raise RuntimeError(f"{raw} branch rows are not one block per scenario.")
    branch_count = branch_rows // full
    bus = pq.read_table(
        raw / "bus_data.parquet",
        columns=["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"],
    ).slice(0, count * 118)
    buses = _blocks(bus.column("bus").to_numpy(), 118)
    scenario_blocks = _blocks(bus.column("scenario").to_numpy(), 118)
    if not np.all(scenario_blocks[:, :1] == scenario_blocks):
        raise RuntimeError(f"{raw} bus rows are not grouped by scenario.")
    if not np.all(buses == np.arange(118)):
        raise RuntimeError(f"{raw} buses are not ordered 0..117 within each scenario.")
    pq_flag = _blocks(bus.column("PQ").to_numpy(), 118)
    pv_flag = _blocks(bus.column("PV").to_numpy(), 118)
    ref_flag = _blocks(bus.column("REF").to_numpy(), 118)
    type_code = np.zeros((count, 118), dtype=np.float32)
    type_code[pv_flag > 0.5] = 1
    type_code[ref_flag > 0.5] = 2
    if np.any((pq_flag > 0.5) & (pv_flag > 0.5)) or np.any((type_code == 0) & (pq_flag <= 0.5)):
        raise RuntimeError(f"{name} has a bus that is not exactly one of PQ, PV, or slack.")
    power_p = (_blocks(bus.column("Pg").to_numpy(), 118) - _blocks(bus.column("Pd").to_numpy(), 118)) / BASE_MVA
    power_q = (_blocks(bus.column("Qg").to_numpy(), 118) - _blocks(bus.column("Qd").to_numpy(), 118)) / BASE_MVA
    voltage = _blocks(bus.column("Vm").to_numpy(), 118)
    angle = np.deg2rad(_blocks(bus.column("Va").to_numpy(), 118))

    branch = pq.read_table(
        raw / "branch_data.parquet",
        columns=["scenario", "idx", "from_bus", "to_bus", "r", "x", "br_status"],
    ).slice(0, count * branch_count)
    branch_scenario = _blocks(branch.column("scenario").to_numpy(), branch_count)
    bus_scenario = _blocks(bus.column("scenario").to_numpy(), 118)
    if not np.array_equal(branch_scenario[:, 0], bus_scenario[:, 0]):
        raise RuntimeError(f"{raw} bus and branch scenarios are not in the same order.")
    status = _blocks(branch.column("br_status").to_numpy(), branch_count)
    resistance = _blocks(branch.column("r").to_numpy(), branch_count)
    reactance = _blocks(branch.column("x").to_numpy(), branch_count)
    origin = _blocks(branch.column("from_bus").to_numpy(), branch_count)
    destination = _blocks(branch.column("to_bus").to_numpy(), branch_count)
    signature = np.concatenate(
        [status, np.round(resistance, 6), np.round(reactance, 6), type_code],
        axis=1,
    ).astype(np.float32)
    _, network_ids = np.unique(signature, axis=0, return_inverse=True)
    print(
        f"{name}: {count} scenarios loaded"
        f"{'' if limit is None else f' of {full}'} across {int(network_ids.max()) + 1} network(s).",
        flush=True,
    )

    wanted = bus_scenario[:, 0]
    representatives = [int(np.flatnonzero(network_ids == net)[0]) for net in range(int(network_ids.max()) + 1)]
    y_table = ds.dataset(raw / "y_bus_data.parquet").to_table(
        filter=pc.field("scenario").isin([int(wanted[local]) for local in representatives]),
        columns=["scenario", "index1", "index2", "G", "B"],
    )
    y_scenario = y_table.column("scenario").to_numpy()
    y_rows = {
        "index1": y_table.column("index1").to_numpy(),
        "index2": y_table.column("index2").to_numpy(),
        "G": y_table.column("G").to_numpy(),
        "B": y_table.column("B").to_numpy(),
    }
    networks: list[Network] = []
    for local in representatives:
        types = type_code[local].astype(np.int64)
        if int(np.sum(types == 2)) != 1:
            raise RuntimeError(f"{name} scenario {int(wanted[local])} has {int(np.sum(types == 2))} slack buses.")
        scenario_id = int(wanted[local])
        chosen = y_scenario == scenario_id
        rows_i = y_rows["index1"][chosen].astype(np.int64)
        rows_j = y_rows["index2"][chosen].astype(np.int64)
        ybus = np.zeros((118, 118), dtype=np.complex64)
        ybus[rows_i, rows_j] = y_rows["G"][chosen] + 1j * y_rows["B"][chosen]
        branches = []
        for branch_id in range(branch_count):
            if status[local, branch_id] < 0.5:
                continue
            first = int(origin[local, branch_id])
            second = int(destination[local, branch_id])
            branches.append(
                (
                    first,
                    second,
                    float(resistance[local, branch_id]),
                    float(reactance[local, branch_id]),
                )
            )
        features, component = meshed_rpe(118, branches, ybus)
        if len(networks) == 0:
            print(f"  meshed_rpe computed once for {name} topology (not per scenario)", flush=True)
        solved = np.exp(1j * angle[local]) * voltage[local]
        injection = solved * np.conj(ybus.astype(np.complex128) @ solved)
        mismatch = np.sqrt((power_p[local] - injection.real) ** 2 + (power_q[local] - injection.imag) ** 2)
        print(f"  reference PBL for {name} network {len(networks)}: {float(mismatch.mean()):.3e}", flush=True)
        if float(mismatch.mean()) > 1e-2:
            raise RuntimeError(f"Solved cases in {name} do not satisfy per-unit power balance.")
        networks.append(
            Network(
                ybus=torch.from_numpy(ybus),
                rpe=torch.from_numpy(features),
                component=torch.from_numpy(component.astype(np.int64)),
                types=torch.from_numpy(types),
            )
        )
    return Split(
        name=name,
        count=count,
        power_p=torch.from_numpy(power_p.astype(np.float32)),
        power_q=torch.from_numpy(power_q.astype(np.float32)),
        voltage=torch.from_numpy(voltage.astype(np.float32)),
        angle=torch.from_numpy(angle.astype(np.float32)),
        network_index=torch.from_numpy(network_ids.astype(np.int64)),
        networks=networks,
    )


def log_counts(data_root: Path, results: Path) -> dict:
    report = {}
    folders = [("train", "n", "feasible"), ("valid", "n", "feasible")]
    folders.extend(("test", topology, condition) for _label, topology, condition in TEST_PARTS)
    for split, topology, condition in folders:
        raw = data_root / "case118" / split / "case118" / topology / condition / "raw"
        count = scenario_count(raw)
        expected = EXPECTED_COUNTS[(split, topology, condition)]
        key = f"{split}/{topology}/{condition}"
        report[key] = {"loaded": count, "expected": expected}
        print(f"count {key}: {count} (expected {expected})", flush=True)
        if count != expected:
            print(f"  count differs from the PFΔ Task 1.1 split. Using the {count} loaded scenarios.", flush=True)
    path = results / "task1_1_tpgt_counts.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def raw_dir(data_root: Path, split: str, topology: str, condition: str) -> Path:
    return data_root / "case118" / split / "case118" / topology / condition / "raw"


def split_cache_dirs(checkpoints: Path) -> tuple[Path, Path]:
    """Return (fast_cache, persistent_cache). On Colab, fast_cache is local SSD."""
    persistent = checkpoints / "cache"
    if in_colab():
        return COLAB_LOCAL_CACHE, persistent
    return persistent, persistent


def _mirror_cache_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or source.stat().st_mtime > destination.stat().st_mtime:
        shutil.copy2(source, destination)


def sync_split_cache_from_drive(persistent: Path, active: Path) -> None:
    """Copy Drive split caches to local disk so ``torch.load`` is fast on Colab."""
    if persistent == active or not persistent.is_dir():
        return
    active.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(persistent.iterdir()):
        if path.is_file():
            _mirror_cache_file(path, active / path.name)
            copied += 1
    if copied:
        print(
            f"Mirrored {copied} split-cache file(s) from Drive ({persistent}) to {active}",
            flush=True,
        )


def sync_split_cache_to_drive(active: Path, persistent: Path) -> None:
    """Persist freshly built caches back to Drive checkpoints."""
    if persistent == active:
        return
    persistent.mkdir(parents=True, exist_ok=True)
    for path in sorted(active.iterdir()):
        if path.is_file():
            _mirror_cache_file(path, persistent / path.name)


def cached_split(active: Path, persistent: Path, raw: Path, name: str) -> Split:
    sync_split_cache_from_drive(persistent, active)
    stamp = {
        "version": CACHE_VERSION,
        "bytes": raw.joinpath("bus_data.parquet").stat().st_size,
    }
    stamp_path = active / f"{name}.json"
    tensor_path = active / f"{name}.pt"
    if stamp_path.exists() and tensor_path.exists():
        saved = json.loads(stamp_path.read_text(encoding="utf-8"))
        if saved == stamp:
            load_started = time.perf_counter()
            print(f"Loading cached {name} split from {tensor_path}", flush=True)
            payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
            elapsed = time.perf_counter() - load_started
            print(f"  loaded {name} split in {elapsed:.1f}s ({tensor_path})", flush=True)
            sync_split_cache_to_drive(active, persistent)
            return payload
    active.mkdir(parents=True, exist_ok=True)
    build_started = time.perf_counter()
    split = read_split(raw, name)
    print(f"  built {name} split from parquet in {time.perf_counter() - build_started:.1f}s", flush=True)
    save_started = time.perf_counter()
    torch.save(split, tensor_path)
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    print(f"  saved {name} cache in {time.perf_counter() - save_started:.1f}s", flush=True)
    sync_split_cache_to_drive(active, persistent)
    return split


DATASET_RELATIVE = Path("Foundation Model") / "vendor" / "pfdelta_task1.1"
WINDOWS_DRIVE = Path(r"K:\My Drive")
COLAB_DRIVE = Path("/content/drive/MyDrive")
# Fast split-cache mirror on Colab runtime disk (not synced across disconnects).
COLAB_LOCAL_CACHE = Path("/content/tpgt_task11_cache")


def dataset_request() -> Path:
    return Path(os.environ.get("GENCO_PFDELTA_TASK11", str(DATASET_RELATIVE)))


def google_drive_root() -> Path | None:
    """Colab MyDrive after mount, or the Windows Google Drive client root."""
    if in_colab():
        return mount_drive()
    if WINDOWS_DRIVE.is_dir():
        return WINDOWS_DRIVE
    return None


def drive_dataset(drive: Path | None) -> Path | None:
    """Return the saved Task 1.1 root when case118 is already on Drive.

    This only checks directory names. It does not copy the dataset.
    """
    requested = dataset_request()
    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    elif drive is not None:
        candidates.append(drive / requested)
    for extra in (WINDOWS_DRIVE / DATASET_RELATIVE, COLAB_DRIVE / DATASET_RELATIVE):
        if extra not in candidates:
            candidates.append(extra)
    for candidate in candidates:
        if (candidate / "case118").is_dir():
            return candidate
    return None


def legacy_locations(drive: Path | None) -> tuple[Path, Path]:
    """Old download and results directories, used only when the Drive copy is absent."""
    if in_colab():
        if drive is not None:
            work = drive / "genco_figure4_task11"
        else:
            work = Path("/content/genco_figure4_task11")
        return work, work / "results"
    work = Path(os.environ.get("GENCO_WORK_ROOT", str(ROOT / "vendor")))
    results = Path(os.environ.get("GENCO_RESULTS_DIR", str(ROOT / "results")))
    return work, results


def drive_output_dirs(drive: Path | None) -> tuple[Path, Path] | None:
    """Checkpoint directory and TPGT result directory on Drive.

    Colab writes under MyDrive/Foundation Model once Drive is mounted.
    Windows writes under K:\\My Drive\\Foundation Model when that folder exists.
    """
    if drive is None:
        return None
    foundation = drive / "Foundation Model"
    if not in_colab() and not foundation.is_dir():
        return None
    checkpoints = foundation / "tpgt_task11"
    results = foundation / "results"
    checkpoints.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return checkpoints, results


def ensure_dataset(work: Path, drive: Path | None = None) -> Path:
    saved = drive_dataset(drive)
    if saved is not None:
        print(f"Dataset already present: {saved}", flush=True)
        print("case118 is already on Drive; skipping download.", flush=True)
        return saved
    data = work / "pfdelta_task1.1"
    marker = data / "case118" / "train" / "case118" / "n" / "feasible" / "raw" / "bus_data.parquet"
    if marker.exists():
        print(f"Dataset already present: {data}", flush=True)
        return data
    from huggingface_hub import snapshot_download

    print("Downloading gridfm/pfdelta_task1.1", flush=True)
    snapshot_download(repo_id="gridfm/pfdelta_task1.1", repo_type="dataset", local_dir=str(data))
    if not marker.exists():
        raise FileNotFoundError(f"Download finished without {marker}")
    return data


def move_to(network: Network, device: torch.device) -> Network:
    types = network.types.to(device)
    component = network.component.to(device)
    pair_embed = network.pair_embed.to(device) if network.pair_embed is not None else None
    masks = network.pair_type_masks
    if masks is None:
        masks = type_pair_masks(types, component)
    else:
        masks = masks.to(device)
    return Network(
        ybus=network.ybus.to(device),
        rpe=network.rpe.to(device),
        component=component,
        types=types,
        pair_type_masks=masks,
        pair_embed=pair_embed,
    )


def clear_pair_embed_cache(networks: list[Network]) -> None:
    for network in networks:
        network.pair_embed = None


def pair_features(model: TPGT, network: Network) -> torch.Tensor:
    """Edge embeddings for this topology; cached in eval, fresh each step in train."""
    if not model.training:
        if network.pair_embed is not None:
            return network.pair_embed
        network.pair_embed = model.pair_encoder(network.rpe)
        return network.pair_embed
    return model.pair_encoder(network.rpe)


def run_batch(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
) -> tuple[TaskLossBreakdown, torch.Tensor]:
    chosen = torch.as_tensor(index, dtype=torch.long)
    power_p = split.power_p[chosen].to(device)
    power_q = split.power_q[chosen].to(device)
    voltage = split.voltage[chosen].to(device)
    angle = split.angle[chosen].to(device)
    specified, target = pack_case(power_p, power_q, voltage, angle, network.types)
    pair = pair_features(model, network)
    prediction, residuals = model(
        specified,
        network.types,
        network.rpe,
        network.pair_type_masks,
        network.ybus,
        pair=pair,
    )
    breakdown = task_loss(prediction, target, residuals, network.types)
    return breakdown, mean_pbl(residuals[-1])


def iterate_indexes(split: Split, seed: int, epoch: int, shuffle: bool):
    generator = torch.Generator()
    generator.manual_seed(seed * 1000 + epoch)
    groups = []
    for network_id in range(len(split.networks)):
        index = torch.nonzero(split.network_index == network_id, as_tuple=False).flatten()
        if shuffle and index.numel() > 1:
            order = torch.randperm(index.numel(), generator=generator)
            index = index[order]
        groups.append((network_id, index.numpy()))
    for network_id, index in groups:
        for start in range(0, len(index), BATCH_SIZE):
            yield network_id, index[start : start + BATCH_SIZE]


def evaluate_pbl(model: TPGT, split: Split, networks: list[Network], device: torch.device) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for network_id, index in iterate_indexes(split, seed=0, epoch=0, shuffle=False):
            _breakdown, score = run_batch(model, split, networks[network_id], index, device)
            total += float(score) * len(index)
            seen += len(index)
    if seen != split.count:
        raise RuntimeError(f"Evaluated {seen} of {split.count} scenarios in {split.name}.")
    return total / seen


def numbered_epoch_checkpoints(model_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in model_dir.glob("epoch_*.pt"):
        suffix = path.stem.removeprefix("epoch_")
        if suffix.isdigit():
            found.append((int(suffix), path))
    found.sort()
    return found


def checkpoint_version_reason(payload: dict) -> str | None:
    """Return a short reason when ``payload`` must not be restored; else None."""
    found = payload.get("checkpoint_version")
    if found is None:
        return f"missing checkpoint_version (expected {CHECKPOINT_VERSION})"
    if found != CHECKPOINT_VERSION:
        return f"checkpoint_version {found!r} != {CHECKPOINT_VERSION!r}"
    return None


def pair_encoder_input_mismatch(payload: dict) -> str | None:
    """Detect checkpoints from an older RPE width (e.g. 6-feature tree RPE) before load."""
    state = payload.get("model")
    if not isinstance(state, dict):
        return None
    weight = state.get("pair_encoder.net.0.weight")
    if weight is None or not hasattr(weight, "shape"):
        return None
    in_features = int(weight.shape[1])
    if in_features == RPE_DIM:
        return None
    return f"pair_encoder expects {RPE_DIM} RPE features, checkpoint has {in_features}"


def try_load_checkpoint_file(path: Path, device: torch.device) -> dict | None:
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def log_checkpoint_obsolete(seed: int, reason: str) -> None:
    print(f"Seed {seed}: checkpoint obsolete ({reason}); training from epoch 0.", flush=True)


def load_resume_checkpoint(
    model_dir: Path,
    device: torch.device,
) -> tuple[dict | None, Path | None, str | None]:
    """Pick the newest compatible checkpoint; ignore wrong-version numbered files."""
    last_path = model_dir / "last.pt"
    obsolete_reason: str | None = None
    candidates: list[tuple[int, Path, dict]] = []

    if last_path.is_file():
        last_payload = try_load_checkpoint_file(last_path, device)
        if last_payload is None:
            obsolete_reason = "last.pt unreadable or corrupt"
        else:
            version_reason = checkpoint_version_reason(last_payload)
            if version_reason is not None:
                obsolete_reason = version_reason
            else:
                candidates.append((int(last_payload.get("epoch", 0)), last_path, last_payload))

    for epoch_number, epoch_path in numbered_epoch_checkpoints(model_dir):
        epoch_payload = try_load_checkpoint_file(epoch_path, device)
        if epoch_payload is None:
            continue
        if checkpoint_version_reason(epoch_payload) is not None:
            continue
        candidates.append((epoch_number, epoch_path, epoch_payload))

    if candidates:
        _epoch, chosen_path, payload = max(candidates, key=lambda item: item[0])
        return payload, chosen_path, None

    if obsolete_reason is None and any(model_dir.glob("epoch_*.pt")):
        obsolete_reason = f"no checkpoint with checkpoint_version {CHECKPOINT_VERSION!r}"
    return None, last_path if last_path.is_file() else None, obsolete_reason


def read_completed_epochs(done_path: Path) -> int:
    """Epochs finished for the current code version only (old complete.json counts as 0)."""
    try:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            return 0
        return int(payload.get("epochs_completed", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def restore_checkpoint_payload(
    model: TPGT,
    optimizer: torch.optim.Optimizer,
    payload: dict | None,
) -> tuple[dict | None, str | None]:
    """Load weights when compatible; return (restored, obsolete_reason)."""
    if payload is None:
        return None, None
    version_reason = checkpoint_version_reason(payload)
    if version_reason is not None:
        return None, version_reason
    rpe_reason = pair_encoder_input_mismatch(payload)
    if rpe_reason is not None:
        return None, rpe_reason
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
    except RuntimeError as error:
        detail = str(error).split("\n", maxsplit=1)[0]
        return None, f"state_dict incompatible with current model ({detail})"
    return (
        {
            "epoch": int(payload["epoch"]),
            "best_val": float(payload["best_val"]),
            "wait": int(payload["wait"]),
            "best_state": payload["best_state"],
        },
        None,
    )


def saves_checkpoint_epoch(epoch_number: int) -> bool:
    """True on epochs 3, 6, ..., 99 and on the final epoch (Drive checkpoint + results CSV)."""
    if epoch_number <= 0:
        return False
    return epoch_number == EPOCHS or epoch_number % CHECKPOINT_EVERY == 0


def test_pbl_line(seed: int, epoch_number: int, scores: dict[str, float]) -> str:
    return (
        f"seed {seed} epoch {epoch_number} test PBL  "
        f"N {scores['N']:.6f}  N-1 {scores['N-1']:.6f}  "
        f"N-2 {scores['N-2']:.6f}  C2I {scores['C2I']:.6f}"
    )


def train_epoch_loss_line(
    seed: int,
    epoch_number: int,
    batches: int,
    state_mse: float,
    physics_sum: float,
    physics_stages: list[float],
    train_total: float,
    val_pbl: float,
) -> str:
    n = max(batches, 1)
    stage_text = "  ".join(f"phys_s{i + 1} {value / n:.4f}" for i, value in enumerate(physics_stages))
    return (
        f"seed {seed} epoch {epoch_number}/{EPOCHS}  "
        f"state {state_mse / n:.4f}  phys {physics_sum / n:.4f}  {stage_text}  "
        f"train {train_total / n:.4f}  val_pbl {val_pbl:.6f}"
    )


def train_seed(
    seed: int,
    model_dir: Path,
    train_split: Split,
    val_split: Split,
    device: torch.device,
    on_epoch_end: Callable[[TPGT, int], tuple[dict[str, float], dict[str, int]]] | None = None,
    after_periodic_checkpoint: Callable[
        [TPGT, Path, int, dict[str, float], dict[str, int]], None
    ]
    | None = None,
) -> TPGT:
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "last.pt"
    done_path = model_dir / "complete.json"
    seed_all(seed)
    model = TPGT().to(device)
    if wants_torch_compile():
        print(
            "torch.compile enabled (GENCO_TPGT_COMPILE=1, GENCO_TPGT_FAST=1, or --compile / --fast)",
            flush=True,
        )
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_networks = [move_to(network, device) for network in train_split.networks]
    val_networks = [move_to(network, device) for network in val_split.networks]
    start_epoch = 0
    best_val = float("inf")
    wait = 0
    best_state = None
    saved, resumed_from, load_obsolete = load_resume_checkpoint(model_dir, device)
    restored, restore_obsolete = restore_checkpoint_payload(model, optimizer, saved)
    if restored is not None:
        start_epoch = restored["epoch"]
        best_val = restored["best_val"]
        wait = restored["wait"]
        best_state = restored["best_state"]
    else:
        obsolete_reason = restore_obsolete or load_obsolete
        if obsolete_reason is not None:
            log_checkpoint_obsolete(seed, obsolete_reason)
    if done_path.exists() and read_completed_epochs(done_path) >= EPOCHS:
        print(f"Seed {seed} already completed {EPOCHS} epochs. Skipping.", flush=True)
        state = best_state
        if state is None:
            fallback, _path, _load_obsolete = load_resume_checkpoint(model_dir, device)
            if fallback is not None:
                restored_done, _restore_obsolete = restore_checkpoint_payload(model, optimizer, fallback)
                if restored_done is not None:
                    state = restored_done["best_state"]
        if state is None:
            raise RuntimeError(f"Seed {seed} finished without a compatible checkpoint.")
        model.load_state_dict(state, strict=True)
        return model

    def save_epoch(
        epoch_number: int,
        test_scores: tuple[dict[str, float], dict[str, int]] | None = None,
    ) -> None:
        epoch_path = model_dir / f"epoch_{epoch_number}.pt"
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_number,
            "best_val": best_val,
            "wait": wait,
            "best_state": best_state,
        }
        torch.save(payload, epoch_path)
        torch.save(payload, checkpoint_path)
        print(
            f"Saved seed {seed} epoch {epoch_number} checkpoint to {epoch_path} and {checkpoint_path}",
            flush=True,
        )
        if after_periodic_checkpoint is not None:
            if test_scores is None:
                raise RuntimeError(
                    f"Seed {seed} epoch {epoch_number}: checkpoint publish requires test scores "
                    "(run on_epoch_end before save_epoch)."
                )
            scores, counts = test_scores
            after_periodic_checkpoint(model, epoch_path, epoch_number, scores, counts)
        (model_dir / f"epoch_{epoch_number}.compared").write_text(
            json.dumps({"epoch": epoch_number}),
            encoding="utf-8",
        )

    def finish_checkpoint_epoch(epoch_number: int) -> None:
        """Run held-out test (print every epoch) then save Drive checkpoint when due."""
        test_scores: tuple[dict[str, float], dict[str, int]] | None = None
        if on_epoch_end is not None:
            test_scores = on_epoch_end(model, epoch_number)
        if saves_checkpoint_epoch(epoch_number):
            save_epoch(epoch_number, test_scores)

    pending_test = saves_checkpoint_epoch(start_epoch) and not (
        model_dir / f"epoch_{start_epoch}.compared"
    ).exists()
    finished = start_epoch >= EPOCHS
    if finished and pending_test:
        print(
            f"Seed {seed} epoch {start_epoch} checkpoint exists; running the missing test scores.",
            flush=True,
        )
        finish_checkpoint_epoch(start_epoch)
    elif finished:
        print(f"Seed {seed} already completed {EPOCHS} epochs. Skipping.", flush=True)
    elif restored is not None:
        print(
            f"Resuming seed {seed} from {resumed_from} after finished epoch {start_epoch}",
            flush=True,
        )
        if pending_test:
            finish_checkpoint_epoch(start_epoch)

    if not finished:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            clear_pair_embed_cache(train_networks)
            clear_pair_embed_cache(val_networks)
            running_total = 0.0
            running_state = 0.0
            running_phys = 0.0
            running_phys_stages = [0.0] * DEPTH
            batches = 0
            first_batch_started: float | None = None
            batch_durations: list[float] = []
            for network_id, index in iterate_indexes(train_split, seed, epoch, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                if first_batch_started is None:
                    first_batch_started = time.perf_counter()
                step_started = time.perf_counter()
                breakdown, _score = run_batch(model, train_split, train_networks[network_id], index, device)
                breakdown.total.backward()
                optimizer.step()
                if first_batch_started is not None:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - first_batch_started
                    print(
                        f"  seed {seed} epoch {epoch + 1} first batch ({len(index)} cases) "
                        f"forward+backward {elapsed:.2f}s",
                        flush=True,
                    )
                    first_batch_started = None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                batch_durations.append(time.perf_counter() - step_started)
                running_total += float(breakdown.total.detach())
                running_state += float(breakdown.state_mse.detach())
                running_phys += float(breakdown.physics_sum.detach())
                for stage_idx, stage_mse in enumerate(breakdown.physics_stages):
                    running_phys_stages[stage_idx] += float(stage_mse.detach())
                batches += 1
                if batches % 250 == 0:
                    window = batch_durations[-250:]
                    avg_sec = sum(window) / len(window)
                    print(
                        f"  seed {seed} epoch {epoch + 1} batch {batches} train {running_total / batches:.4f} "
                        f"avg {avg_sec:.3f}s/batch (last {len(window)})",
                        flush=True,
                    )
            val_score = evaluate_pbl(model, val_split, val_networks, device)
            print(
                train_epoch_loss_line(
                    seed,
                    epoch + 1,
                    batches,
                    running_state,
                    running_phys,
                    running_phys_stages,
                    running_total,
                    val_score,
                ),
                flush=True,
            )
            if val_score < best_val:
                best_val = val_score
                wait = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                wait += 1
            start_epoch = epoch + 1
            finish_checkpoint_epoch(start_epoch)
    if best_state is None:
        raise RuntimeError(f"Seed {seed} produced no checkpoint.")
    model.load_state_dict(best_state)
    done_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "checkpoint_version": CHECKPOINT_VERSION,
                "epochs_completed": start_epoch,
                "best_val_pbl": best_val,
            }
        ),
        encoding="utf-8",
    )
    return model


def record_seed(
    results: Path,
    seed: int,
    scores: dict[str, float],
    counts: dict[str, int],
    checkpoint: Path,
    epoch: int,
) -> None:
    c2i_counts = {"C2I-N": counts["C2I-N"], "C2I-N-1": counts["C2I-N-1"], "C2I-N-2": counts["C2I-N-2"]}
    scores = dict(scores)
    scores["C2I"] = aggregate_c2i(scores, c2i_counts)
    columns = [*SEED_COLUMNS, "epoch"]
    rows = []
    for category in (*CATEGORIES,):
        rows.append(
            {
                "model": MODEL_NAME,
                "task": "1.1",
                "test_category": category,
                "seed": str(seed),
                "mean_pbl_pu": f"{scores[category]:.8g}",
                "checkpoint_path": str(checkpoint),
                "source": "independent_training",
                "epoch": str(epoch),
            }
        )
    upsert_csv(
        results / "task1_1_tpgt_seed_metrics.csv",
        columns,
        rows,
        ["model", "task", "test_category", "seed", "epoch"],
    )


def write_summary(results: Path) -> list[dict]:
    latest: dict[tuple[str, str], tuple[int, float]] = {}
    for row in read_csv(results / "task1_1_tpgt_seed_metrics.csv"):
        if row.get("model") != MODEL_NAME:
            continue
        raw_epoch = (row.get("epoch") or "").strip()
        try:
            epoch_number = int(raw_epoch) if raw_epoch else 0
        except ValueError:
            epoch_number = 0
        key = (row.get("seed", ""), row.get("test_category", ""))
        value = float(row["mean_pbl_pu"])
        previous = latest.get(key)
        if previous is None or epoch_number >= previous[0]:
            latest[key] = (epoch_number, value)
    grouped: dict[str, list[float]] = {}
    for (_seed, category), (_epoch_number, value) in latest.items():
        grouped.setdefault(category, []).append(value)
    summary = []
    for category in CATEGORIES:
        values = grouped.get(category, [])
        if not values:
            continue
        summary.append(
            {
                "model": MODEL_NAME,
                "test_category": category,
                "mean_across_seeds": f"{sample_mean(values):.8g}",
                "std_across_seeds": f"{sample_std(values):.8g}" if len(values) >= 2 else "0",
                "n_seeds": str(len(values)),
            }
        )
    path = results / "task1_1_tpgt_summary.csv"
    if summary:
        upsert_csv(path, SUMMARY_COLUMNS, summary, ["model", "test_category"])
    return summary


def print_comparison(summary: list[dict]) -> None:
    print("\nTask 1.1 mean power-balance loss (per unit)", flush=True)
    print("PUBLISHED", flush=True)
    for model, scores in PUBLISHED_TASK11.items():
        parts = [f"{category} {scores[category][0]:.4g} ± {scores[category][1]:.4g}" for category in CATEGORIES]
        print(f"  {model}: " + "; ".join(parts), flush=True)
    print("TPGT this run", flush=True)
    by_category = {row["test_category"]: row for row in summary}
    if any(category not in by_category for category in CATEGORIES):
        print("  TPGT has no finished four-category seed yet.", flush=True)
        return
    parts = []
    for category in CATEGORIES:
        row = by_category[category]
        parts.append(
            f"{category} {float(row['mean_across_seeds']):.4g} ± {float(row['std_across_seeds']):.4g} (n={row['n_seeds']})"
        )
    print("  TPGT: " + "; ".join(parts), flush=True)


def plot_comparison(results: Path, summary: list[dict]) -> None:
    by_category = {row["test_category"]: row for row in summary}
    if any(category not in by_category for category in CATEGORIES):
        return
    if any(int(by_category[category]["n_seeds"]) < 1 for category in CATEGORIES):
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "path"

    labels = [f"{model} (published)" for model in ("GENCO Base", "CANOS-PF", "GNS-S", "PFNet")] + ["TPGT (this run)"]
    colors = [PALETTE[model] for model in ("GENCO Base", "CANOS-PF", "GNS-S", "PFNet")] + ["#1F4B99"]
    means = []
    stds = []
    for model in ("GENCO Base", "CANOS-PF", "GNS-S", "PFNet"):
        means.append([PUBLISHED_TASK11[model][category][0] for category in CATEGORIES])
        stds.append([PUBLISHED_TASK11[model][category][1] for category in CATEGORIES])
    means.append([float(by_category[category]["mean_across_seeds"]) for category in CATEGORIES])
    stds.append([float(by_category[category]["std_across_seeds"]) for category in CATEGORIES])
    figure, axis = plt.subplots(figsize=(10, 8))
    positions = np.arange(len(CATEGORIES))
    width = 0.15
    for index, label in enumerate(labels):
        offset = (index - (len(labels) - 1) / 2) * width
        axis.bar(positions + offset, means[index], width, yerr=stds[index], label=label, color=colors[index], capsize=2)
    axis.set_xticks(positions)
    axis.set_xticklabels(list(CATEGORIES))
    axis.set_ylabel("Power Balance Loss (Mean) [p.u.]")
    axis.set_ylim(0, 1.8)
    axis.yaxis.grid(True)
    axis.legend(frameon=False, fontsize=9)
    axis.set_title("Task 1.1")
    figure.tight_layout()
    output = results / "figure4_task1.1_tpgt.svg"
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {output}", flush=True)
    try:
        from IPython.display import SVG, display

        display(SVG(filename=str(output)))
    except Exception:
        pass


def test_model(
    model: TPGT,
    data_root: Path,
    device: torch.device,
    active_cache: Path,
    persistent_cache: Path,
) -> tuple[dict[str, float], dict[str, int]]:
    scores = {}
    counts = {}
    model.eval()
    for label, topology, condition in TEST_PARTS:
        split = cached_split(
            active_cache,
            persistent_cache,
            raw_dir(data_root, "test", topology, condition),
            f"test_{topology}_{condition}",
        )
        networks = [move_to(network, device) for network in split.networks]
        scores[label] = evaluate_pbl(model, split, networks, device)
        counts[label] = split.count
        print(f"test {label}: PBL {scores[label]:.6g} on {split.count} scenarios", flush=True)
    return scores, counts


def main() -> None:
    ensure_runtime_packages()
    drive = google_drive_root()
    outputs = drive_output_dirs(drive)
    work, legacy_results = legacy_locations(drive)
    if outputs is not None:
        checkpoints, results = outputs
    else:
        work.mkdir(parents=True, exist_ok=True)
        legacy_results.mkdir(parents=True, exist_ok=True)
        checkpoints = work / "tpgt_task11"
        checkpoints.mkdir(parents=True, exist_ok=True)
        results = legacy_results
    os.environ["GENCO_RESULTS_DIR"] = str(results)
    os.environ["GENCO_WORK_ROOT"] = str(checkpoints.parent)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    from common import use_selected_results

    use_selected_results()
    ensure_dirs()
    results.mkdir(parents=True, exist_ok=True)
    data = ensure_dataset(work, drive)
    print(f"Dataset: {data}", flush=True)
    print(f"Checkpoints: {checkpoints}", flush=True)
    print(f"Results directory: {results}", flush=True)
    log_counts(data, results)
    active_cache, persistent_cache = split_cache_dirs(checkpoints)
    print(f"Split cache (active): {active_cache}", flush=True)
    if persistent_cache != active_cache:
        print(f"Split cache (Drive): {persistent_cache}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    load_started = time.perf_counter()
    train_split = cached_split(
        active_cache,
        persistent_cache,
        raw_dir(data, "train", "n", "feasible"),
        "train_n_feasible",
    )
    val_split = cached_split(
        active_cache,
        persistent_cache,
        raw_dir(data, "valid", "n", "feasible"),
        "valid_n_feasible",
    )
    print(
        f"Train+valid splits ready in {time.perf_counter() - load_started:.1f}s "
        f"({train_split.count} train, {val_split.count} valid cases)",
        flush=True,
    )
    for seed in SEEDS:
        seed_dir = checkpoints / f"seed_{seed}"
        if (seed_dir / "complete.json").exists() and read_completed_epochs(seed_dir / "complete.json") >= EPOCHS:
            print(f"Seed {seed} already completed {EPOCHS} epochs. Skipping.", flush=True)
            continue

        def on_epoch_end(model: TPGT, epoch_number: int, seed: int = seed) -> tuple[dict[str, float], dict[str, int]]:
            scores, counts = test_model(model, data, device, active_cache, persistent_cache)
            c2i_counts = {
                "C2I-N": counts["C2I-N"],
                "C2I-N-1": counts["C2I-N-1"],
                "C2I-N-2": counts["C2I-N-2"],
            }
            reported = dict(scores)
            reported["C2I"] = aggregate_c2i(reported, c2i_counts)
            print(test_pbl_line(seed, epoch_number, reported), flush=True)
            return scores, counts

        def publish_checkpoint(
            model: TPGT,
            checkpoint: Path,
            epoch_number: int,
            scores: dict[str, float],
            counts: dict[str, int],
            seed: int = seed,
        ) -> None:
            # record_seed / summary CSV / plot only on Drive checkpoint epochs (not every epoch).
            record_seed(results, seed, scores, counts, checkpoint, epoch_number)
            summary = write_summary(results)
            print_comparison(summary)
            plot_comparison(results, summary)

        train_seed(
            seed,
            seed_dir,
            train_split,
            val_split,
            device,
            on_epoch_end=on_epoch_end,
            after_periodic_checkpoint=publish_checkpoint,
        )


def sanity_check_meshed_rpe(data_root: Path) -> None:
    """Load one 118-bus case and verify meshed RPE shape and invariants."""
    split = read_split(raw_dir(data_root, "train", "n", "feasible"), "sanity", limit=1)
    rpe = split.networks[0].rpe.numpy()
    if rpe.shape != (118, 118, RPE_DIM):
        raise RuntimeError(f"Expected RPE shape (118, 118, {RPE_DIM}), got {rpe.shape}.")
    if not np.isfinite(rpe).all():
        raise RuntimeError("RPE sanity check failed: non-finite values.")
    diagonal = np.arange(118)
    if not np.allclose(rpe[diagonal, diagonal, 3], 0.0):
        raise RuntimeError("Diagonal hop feature must be zero.")
    if not np.allclose(rpe[diagonal, diagonal, 4], 0.0):
        raise RuntimeError("Diagonal effective-resistance feature must be zero.")
    if not np.allclose(rpe[diagonal, diagonal, 2], 0.0):
        raise RuntimeError("Diagonal adjacency feature must be zero.")
    neighbor = int(np.argmax(rpe[:, :, 2]))
    row, column = divmod(neighbor, 118)
    if rpe[row, column, 2] < 0.5:
        raise RuntimeError("Expected at least one directly connected bus pair.")
    if not np.allclose(rpe[:, :, 3], rpe[:, :, 3].T, atol=1e-5):
        raise RuntimeError("Normalized hop features are not symmetric.")
    if not np.allclose(rpe[:, :, 4], rpe[:, :, 4].T, atol=1e-4):
        raise RuntimeError("Normalized effective-resistance features are not symmetric.")
    print(
        f"meshed RPE sanity OK: shape {rpe.shape}, neighbor ({row},{column}) A={rpe[row, column, 2]:.0f}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sanity-rpe":
        ensure_runtime_packages()
        drive = google_drive_root()
        work, _legacy = legacy_locations(drive)
        data = ensure_dataset(work, drive)
        sanity_check_meshed_rpe(data)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-attention":
        from tests.test_tpgt_attention import test_batched_messages_match_reference

        test_batched_messages_match_reference()
        print("test_tpgt_attention OK", flush=True)
    else:
        main()
