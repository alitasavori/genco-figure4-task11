"""Train the unshared type-pair graph Transformer on IEEE 118-bus Task 1.1.

Option II from the draft: each correction stage has its own weights.
The hidden sizes, depth, batch, and epoch cap below are implementation
choices so a Colab GPU can finish a run. They are not paper hyperparameters.

IEEE 118 is meshed, so pair features use the full in-service topology:
admittance (real, imaginary), adjacency, hop distance, and effective
resistance (5 channels). See `meshed_rpe`.

Architecture (checkpoint_version ``meshed-rpe-5-d4-v7``)
--------------------------------------------------------
* **4 correction stages** (``DEPTH``), each with its own weights; the physics
  loss is the unweighted sum over stages.
* **5-channel RPE** (random-walk channels removed).
* **RPE channel dropout**: during training each channel is zeroed with
  probability ``RPE_CHANNEL_DROPOUT`` per optimizer step; never at eval.
* **Versioned run folder**: checkpoints go to ``Foundation Model/tpgt_task11_<RUN_TAG>``
  and tables/plot to ``Foundation Model/results/tpgt_task11_<RUN_TAG>``, so runs
  of older architectures (``tpgt_task11/``) are left untouched.

Performance notes (Colab / large train split)
---------------------------------------------
* **Drive I/O**: The 48600-case train/valid tensors live under the checkpoint
  tree on Google Drive. ``torch.load`` from Drive is often minutes slower than
  from ``/content`` local disk. On Colab, ``split_cache_dirs`` mirrors ``.pt``
  caches to ``/content/tpgt_task11_<RUN_TAG>_cache`` after the first Drive read or build.
* **``meshed_rpe``**: Computed only when building or refreshing the split cache,
  once per *distinct* in-service topology (see ``read_split`` representatives),
  not per scenario and not per training batch.
* **``pair_encoder``**: Runs once per forward on the static ``118×118`` RPE for
  the batch's topology. Weights train, so training must re-encode every step;
  in ``eval`` mode encoded pairs are cached on ``Network.pair_embed``.
* **Type-pair attention**: ``CorrectionStage._messages`` evaluates each of the
  nine type-pair weight sets only on its own receiver-type × sender-type block
  (both heads batched), then takes one softmax over all senders per receiver.
  Optional ``GENCO_TPGT_COMPILE=1`` or ``--compile`` wraps the model in
  ``torch.compile`` (may differ slightly in numerics).
* **Batch 32**: Optimizer steps use 32 cases per batch. Default **micro-batch 32**
  (one forward/backward per step when memory allows). Override with
  ``GENCO_TPGT_MICRO_BATCH`` or ``--micro-batch``.
* **Eval batch 64**: Validation and test use forward-only batches up to 64
  (``GENCO_TPGT_EVAL_BATCH`` / ``--eval-batch``) — no backward, usually safe
  on the same GPU.
* **CUDA**: ``cudnn.benchmark``, TF32, and high matmul precision when training
  on GPU. Optional ``torch.compile`` via ``--fast``, ``GENCO_TPGT_FAST=1``, or
  ``--compile``.
* **Less sync**: GPU ``synchronize`` only for timed log lines and validation,
  not after every training batch.
* **Optional**: ``GENCO_TPGT_VAL_EVERY=3`` or ``--val-every 3`` skips most
  validation passes (faster; coarser early stopping).
* **LR schedule**: Adam at ``LEARNING_RATE`` with ``ReduceLROnPlateau`` on
  validation PBL (x0.7 after 5 non-improving validations) and early stopping
  after 100, as in GENCO. The epoch line shows ``lr``; scheduler state is saved
  in checkpoints, and resumed runs keep their saved optimizer learning rate.
* **Debug (on by default)**: every validation pass also scores a fixed training
  subset (``train_pbl`` and ``gap`` in the epoch line). On checkpoint epochs the
  run prints the forward report, RPE zero-ablation, RPE and residual correctness
  audits on the validation topology, and contingency diagnostics on test N-1/N-2
  (per-topology PBL spread, audits and ablation on the largest outage topologies,
  RPE shift from the training topology). Disable with ``GENCO_TPGT_DEBUG=0`` or
  ``--no-debug``; skip only the ablation with ``GENCO_TPGT_RPE_ABLATION=0``.
  ``--diagnose`` runs the same on a saved checkpoint (``--no-contingency``,
  ``--no-rpe-ablation`` to trim). ``--sanity-rpe`` audits one train case.
"""

from __future__ import annotations

import hashlib
import json
import gc
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
D = 48
D_E = 24
D_A = 24
HEADS = 2
DEPTH = 4
ETA = 1.0
BATCH_SIZE = 32
# Optimizer schedule follows GENCO's Task 1.1 config (HGNS_PF_pfdelta_bs64_seed1.yaml):
# Adam at 5e-4, ReduceLROnPlateau on validation PBL (x0.7, patience 5), early stop after 100.
LEARNING_RATE = 5e-4
LR_PLATEAU_FACTOR = 0.7
LR_PLATEAU_PATIENCE = 5
EARLY_STOP_PATIENCE = 100
EPOCHS = 100
# Drive checkpoints (and results CSV / comparison plot) every 3rd epoch and at the final epoch.
# The four held-out test PBL scores run on checkpoint epochs (3, 6, …) and the final epoch.
CHECKPOINT_EVERY = 3
SEEDS = (0, 1, 2)
BASE_MVA = 100.0
LAMBDA_STATE = 0.85
LAMBDA_PHYS = 0.15
MODEL_NAME = "TPGT"
TYPE_NAMES = ("PQ", "PV", "Slack")
CACHE_VERSION = "tpgt-cache-4"
RPE_FEATURE_NAMES = (
    "ReY/sY",
    "ImY/sY",
    "adjacency",
    "hop_norm",
    "Reff_norm",
)
RPE_DIM = len(RPE_FEATURE_NAMES)
# Per optimizer step, each RPE channel is zeroed (whole 118x118 plane, no rescaling) with this
# probability. Training only; validation, test and diagnostics always see the full RPE.
RPE_CHANNEL_DROPOUT = 0.1
# Bumped when pair-encoder input size or architecture changes; old Drive checkpoints are ignored.
CHECKPOINT_VERSION = "meshed-rpe-5-d4-v7"
# Checkpoints, split caches and result tables live in tpgt_task11_<RUN_TAG> so runs of an
# older architecture (e.g. the 2-stage, 10-channel run in tpgt_task11/) are never overwritten.
RUN_TAG = "v7"
RUN_DIR_NAME = f"tpgt_task11_{RUN_TAG}"
# Fixed training subset scored with val's PBL metric each validation pass (overfitting check).
TRAIN_PROBE_CASES = 2000
# Reference labels must satisfy nodal balance within this mean PBL (p.u.) when checked at load.
REFERENCE_PBL_TOLERANCE = 1e-2
# Off-diagonal |Y_pq| below this are treated as disconnected for admittance scaling.
Y_ADMITTANCE_EPS = 1e-10
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
CONTINGENCY_DIAGNOSTIC_LABELS = ("N-1", "N-2")
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
# Fraction of each split folder to load (first scenarios). 1.0 = full PFΔ split.
DATA_FRACTION = 1.0
# Forward/backward chunk size during training (gradient accumulation to BATCH_SIZE).
MICRO_BATCH_SIZE = 32
# Validation/test forward-only batches (no backward); can exceed BATCH_SIZE when memory allows.
EVAL_BATCH_SIZE = 64
# Run validation PBL every N epochs (1 = each epoch, used for best checkpoint).
VAL_EVERY = 1


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


def autocast_activation_dtype(reference: torch.Tensor) -> torch.dtype:
    if reference.is_cuda and torch.is_autocast_enabled():
        return torch.get_autocast_dtype("cuda")
    return reference.dtype


def masked_write(base: torch.Tensor, mask: torch.Tensor, values: torch.Tensor) -> None:
    base[:, mask] = values.to(dtype=base.dtype)


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


@dataclass
class TypePairBlock:
    """Receivers of one type and senders of one type; ``mask`` keeps same-component pairs only."""

    pair_id: int
    receivers: torch.Tensor
    senders: torch.Tensor
    mask: torch.Tensor


def type_pair_blocks(types: torch.Tensor, pair_type_masks: torch.Tensor) -> list[TypePairBlock]:
    """Non-empty (receiver type, sender type) blocks of the bus-pair matrix, one per type pair."""
    by_type = [torch.nonzero(types == type_id, as_tuple=False).flatten() for type_id in range(3)]
    blocks = []
    for pair_id in range(9):
        receiver, sender = divmod(pair_id, 3)
        receivers = by_type[receiver]
        senders = by_type[sender]
        if receivers.numel() == 0 or senders.numel() == 0:
            continue
        mask = pair_type_masks[pair_id][receivers.view(-1, 1), senders.view(1, -1)]
        blocks.append(TypePairBlock(pair_id, receivers, senders, mask))
    return blocks


def wants_torch_compile() -> bool:
    if "--compile" in sys.argv or "--fast" in sys.argv:
        return True
    flag = os.environ.get("GENCO_TPGT_COMPILE", os.environ.get("GENCO_TPGT_FAST", ""))
    return flag.lower() in ("1", "true", "yes")


def configure_cuda_training(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


class MLP(nn.Module):
    """Three-layer map (in → D → D → out). Width `D` is an implementation choice."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, out_features),
        )

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
        blocks: list[TypePairBlock],
        ybus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_in = nodal_residual(specified, unknown, types, ybus)
        message = self._messages(hidden, pair, blocks)
        updated = torch.zeros_like(hidden)
        for type_id, update in enumerate(self.update):
            mask = types == type_id
            if not bool(mask.any()):
                continue
            features = torch.cat(
                (
                    hidden[:, mask],
                    message[:, mask],
                    residual_in[:, mask].to(hidden.dtype),
                ),
                dim=-1,
            )
            masked_write(updated, mask, update(features))
        hidden_next = self.norm(hidden + updated)
        delta = torch.zeros_like(unknown)
        for type_id, decoder in enumerate(self.decode):
            mask = types == type_id
            if not bool(mask.any()):
                continue
            masked_write(delta, mask, decoder(hidden_next[:, mask]))
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
        blocks: list[TypePairBlock],
    ) -> torch.Tensor:
        """Each type pair's weights see only its own receiver-type x sender-type block.

        Block scores are scattered into one (batch, heads, N, N) score matrix so the softmax
        runs over all senders of a receiver; cross-component entries stay -inf.
        """
        batch, buses, _ = hidden.shape
        scores: torch.Tensor | None = None
        interactions = []
        for block in blocks:
            p = block.pair_id
            receivers = block.receivers.view(-1, 1)
            senders = block.senders.view(1, -1)
            query = torch.einsum("bnd,had->bhna", hidden[:, block.receivers], self.w_q[p])
            key = torch.einsum("bnd,had->bhna", hidden[:, block.senders], self.w_k[p])
            mixed = rho_nonlinearity(query.unsqueeze(-2) + key.unsqueeze(-3))
            edge = pair[receivers, senders]
            projected = torch.einsum("nme,hae->hnma", edge, self.w_ew[p])
            bias = torch.einsum("nme,hae->hnma", edge, self.w_eb[p])
            interaction = torch.relu(mixed * projected.unsqueeze(0) + bias.unsqueeze(0))
            score = torch.einsum("bhnma,ha->bhnm", interaction, self.w_a[p, :, 0])
            score = score.masked_fill(~block.mask, float("-inf"))
            if scores is None:
                scores = score.new_full((batch, HEADS, buses, buses), float("-inf"))
            scores[:, :, receivers, senders] = score.to(scores.dtype)
            interactions.append(interaction)
        if scores is None:
            return hidden.new_zeros(hidden.shape)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        row_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)
        mixed_heads = weights.new_zeros((batch, HEADS, buses, D_A))
        for block, interaction in zip(blocks, interactions):
            p = block.pair_id
            block_weights = weights[:, :, block.receivers.view(-1, 1), block.senders.view(1, -1)]
            value = torch.einsum("bnd,had->bhna", hidden[:, block.senders], self.w_v[p])
            # sum_j w_ij (W_V h_j + W_Ev x_ij) without materializing the per-pair values.
            gathered = torch.einsum("bhnm,bhma->bhna", block_weights, value)
            pooled = torch.einsum("bhnm,bhnma->bhna", block_weights, interaction)
            gathered = gathered + torch.einsum("bhna,hxa->bhnx", pooled, self.w_ev[p])
            mixed_heads = mixed_heads.index_add(2, block.receivers, gathered.to(mixed_heads.dtype))
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
        act_dtype = autocast_activation_dtype(specified)
        hidden = torch.zeros(
            specified.shape[0],
            specified.shape[1],
            D,
            device=specified.device,
            dtype=act_dtype,
        )
        unknown = flat_start(specified, types)
        for type_id, encoder in enumerate(self.encoders):
            mask = types == type_id
            if bool(mask.any()):
                masked_write(hidden, mask, encoder(specified[:, mask]))
        blocks = type_pair_blocks(types, pair_type_masks)
        residuals = []
        for stage in self.stages:
            hidden, unknown, residual = stage(hidden, unknown, specified, pair, types, blocks, ybus)
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
    """AC mismatch in per unit. Ybus is already per unit, so this does not divide by baseMVA again.

    Evaluated in complex128: with TF32 enabled, a complex64 Ybus matmul on CUDA has a ~0.02 p.u. mean
    error floor on case118 (|Y| up to ~390 with heavy cancellation), 10^4 x the label accuracy.
    """
    physical = constrain_unknown(unknown, types)
    power_p, power_q, voltage, angle = assemble(specified, physical, types)
    voltage_complex = voltage.double() * torch.exp(1j * angle.double())
    current = torch.matmul(ybus.to(torch.complex128), voltage_complex.transpose(0, 1)).transpose(0, 1)
    flow = voltage_complex * torch.conj(current)
    residual = torch.stack((power_p.double() - flow.real, power_q.double() - flow.imag), dim=-1)
    return residual.float()


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


def pbl_stats(residual: torch.Tensor) -> dict[str, float]:
    """Per-bus PBL (p.u.) summary; residual shape (batch, buses, 2)."""
    per_bus = torch.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2)
    return {
        "mean": float(per_bus.mean()),
        "max": float(per_bus.max()),
        "p95": float(torch.quantile(per_bus.flatten(), 0.95)),
    }


def pbl_from_unknown(
    specified: torch.Tensor,
    unknown: torch.Tensor,
    types: torch.Tensor,
    ybus: torch.Tensor,
) -> dict[str, float]:
    return pbl_stats(nodal_residual(specified, unknown, types, ybus))


def state_rmse_by_type(
    prediction: torch.Tensor,
    target: torch.Tensor,
    types: torch.Tensor,
) -> dict[str, float]:
    """RMSE on unknown slots (same packing as pack_case)."""
    error = prediction - target
    angle_unknown = (types == 0) | (types == 1)
    error = error.clone()
    error[:, angle_unknown, 1] = wrap_angle(error[:, angle_unknown, 1])
    out: dict[str, float] = {}
    labels = (
        ("PQ_V", 0, 0),
        ("PQ_theta", 0, 1),
        ("PV_Q", 1, 0),
        ("PV_theta", 1, 1),
        ("slack_P", 2, 0),
        ("slack_Q", 2, 1),
    )
    for name, type_id, slot in labels:
        mask = types == type_id
        if not bool(mask.any()):
            continue
        diff = error[:, mask, slot]
        out[name] = float(torch.sqrt(torch.mean(diff * diff)))
    return out


def summarize_rpe(rpe: torch.Tensor) -> list[dict[str, float]]:
    """Per-channel stats; separate diagonal and off-diagonal pairs."""
    values = rpe.detach().cpu().numpy()
    buses = values.shape[0]
    diagonal = np.eye(buses, dtype=bool)
    off = ~diagonal
    rows: list[dict[str, float]] = []
    for channel, name in enumerate(RPE_FEATURE_NAMES):
        plane = values[:, :, channel]
        rows.append(
            {
                "name": name,
                "off_mean": float(plane[off].mean()),
                "off_max": float(plane[off].max()),
                "diag_mean": float(plane[diagonal].mean()),
            }
        )
    return rows


def pair_encoder_input_weight_norms(model: TPGT) -> list[dict[str, float]]:
    """L2 norm of each column of the pair_encoder first layer (how strongly each RPE channel is wired in)."""
    first = model.pair_encoder.net[0]
    if not isinstance(first, nn.Linear):
        raise TypeError("pair_encoder first layer must be Linear")
    norms = torch.linalg.vector_norm(first.weight.detach(), dim=0)
    return [
        {"name": RPE_FEATURE_NAMES[channel], "weight_l2": float(norms[channel])}
        for channel in range(RPE_DIM)
    ]


def eval_batch_with_pair(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    pair: torch.Tensor,
) -> dict[str, float]:
    """Loss and final-stage PBL on a fixed batch with precomputed pair embeddings."""
    model.eval()
    chosen = np.asarray(index, dtype=np.int64)
    power_p = split.power_p[chosen].to(device, non_blocking=True)
    power_q = split.power_q[chosen].to(device, non_blocking=True)
    voltage = split.voltage[chosen].to(device, non_blocking=True)
    angle = split.angle[chosen].to(device, non_blocking=True)
    specified, target = pack_case(power_p, power_q, voltage, angle, network.types)
    with torch.no_grad():
        prediction, residuals = model(
            specified,
            network.types,
            network.rpe,
            network.pair_type_masks,
            network.ybus,
            pair=pair,
        )
        breakdown = task_loss(prediction, target, residuals, network.types)
        final_pbl = pbl_stats(residuals[-1])
    return {
        "pbl_mean": float(final_pbl["mean"]),
        "state_mse": float(breakdown.state_mse),
        "total_loss": float(breakdown.total),
    }


def collect_rpe_channel_ablation(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Zero one RPE channel at a time; compare batch metrics to the full-RPE baseline."""
    model.eval()
    clear_pair_embed_cache([network])
    baseline_pair = pair_features(model, network)
    baseline = eval_batch_with_pair(model, split, network, index, device, baseline_pair)
    rows: list[dict[str, float]] = []
    for channel, name in enumerate(RPE_FEATURE_NAMES):
        rpe_ablated = network.rpe.clone()
        rpe_ablated[..., channel] = 0.0
        ablated_pair = model.pair_encoder(rpe_ablated)
        metrics = eval_batch_with_pair(model, split, network, index, device, ablated_pair)
        rows.append(
            {
                "name": name,
                "delta_pbl_mean": metrics["pbl_mean"] - baseline["pbl_mean"],
                "delta_state_mse": metrics["state_mse"] - baseline["state_mse"],
                "delta_total_loss": metrics["total_loss"] - baseline["total_loss"],
            }
        )
    clear_pair_embed_cache([network])
    return baseline, rows


def format_rpe_effectiveness_diagnostics(
    weight_rows: list[dict[str, float]],
    baseline: dict[str, float] | None,
    ablation_rows: list[dict[str, float]] | None,
) -> str:
    lines = ["Pair-encoder input weight L2 per RPE channel (structural; not causal):"]
    for row in sorted(weight_rows, key=lambda item: item["weight_l2"], reverse=True):
        lines.append(f"  {row['name']:12s} weight_l2 {row['weight_l2']:.4g}")
    if baseline is None or ablation_rows is None:
        lines.append(
            "RPE channel ablation: not run (use --diagnose, --rpe-ablation, or GENCO_TPGT_RPE_ABLATION=1)."
        )
        return "\n".join(lines)
    lines.append(
        f"RPE zero-ablation on same batch (baseline PBL mean {baseline['pbl_mean']:.6g}, "
        f"state_mse {baseline['state_mse']:.6g}):"
    )
    lines.append(
        "  Positive ΔPBL after zeroing => channel helped; negative ΔPBL => zeroing improved metrics (hurting/noisy)."
    )
    for row in sorted(ablation_rows, key=lambda item: item["delta_pbl_mean"], reverse=True):
        lines.append(
            f"  {row['name']:12s} ΔPBL {row['delta_pbl_mean']:+.4g}  "
            f"Δstate_mse {row['delta_state_mse']:+.4g}  Δloss {row['delta_total_loss']:+.4g}"
        )
    return "\n".join(lines)


def summarize_pair_embed(pair: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        norms = torch.linalg.vector_norm(pair, dim=-1)
    return {
        "mean_norm": float(norms.mean()),
        "max_norm": float(norms.max()),
    }


@dataclass
class ForwardDiagnostics:
    oracle_pbl: dict[str, float]
    flat_start_pbl: dict[str, float]
    model_pbl: dict[str, float]
    stage_pbl: list[dict[str, float]]
    state_rmse: dict[str, float]
    train_state_mse: float
    train_phys_stages: tuple[float, ...]
    pair_embed: dict[str, float]
    rpe_channels: list[dict[str, float]]
    ybus_abs_mean: float


def collect_forward_diagnostics(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
) -> ForwardDiagnostics:
    model.eval()
    clear_pair_embed_cache([network])
    chosen = np.asarray(index, dtype=np.int64)
    power_p = split.power_p[chosen].to(device, non_blocking=True)
    power_q = split.power_q[chosen].to(device, non_blocking=True)
    voltage = split.voltage[chosen].to(device, non_blocking=True)
    angle = split.angle[chosen].to(device, non_blocking=True)
    specified, target = pack_case(power_p, power_q, voltage, angle, network.types)
    types = network.types
    ybus = network.ybus
    with torch.no_grad():
        oracle = pbl_from_unknown(specified, target, types, ybus)
        flat = flat_start(specified, types)
        flat_pbl = pbl_from_unknown(specified, flat, types, ybus)
        pair = pair_features(model, network)
        prediction, residuals = model(
            specified,
            types,
            network.rpe,
            network.pair_type_masks,
            ybus,
            pair=pair,
        )
        breakdown = task_loss(prediction, target, residuals, types)
        stage_pbl = [pbl_stats(r) for r in residuals]
        model_pbl = stage_pbl[-1] if stage_pbl else pbl_from_unknown(specified, prediction, types, ybus)
        state_rmse = state_rmse_by_type(prediction, target, types)
        ybus_abs = float(torch.abs(ybus).mean())
    return ForwardDiagnostics(
        oracle_pbl=oracle,
        flat_start_pbl=flat_pbl,
        model_pbl=model_pbl,
        stage_pbl=stage_pbl,
        state_rmse=state_rmse,
        train_state_mse=float(breakdown.state_mse),
        train_phys_stages=tuple(float(v) for v in breakdown.physics_stages),
        pair_embed=summarize_pair_embed(pair),
        rpe_channels=summarize_rpe(network.rpe),
        ybus_abs_mean=ybus_abs,
    )


def format_forward_diagnostics(
    tag: str,
    diagnostics: ForwardDiagnostics,
) -> str:
    lines = [f"=== TPGT diagnostics {tag} ==="]
    o = diagnostics.oracle_pbl
    lines.append(
        f"Oracle PBL (target state plugged in): mean {o['mean']:.6e} max {o['max']:.6e} "
        f"(expect ~1e-6 if labels match Ybus)"
    )
    f = diagnostics.flat_start_pbl
    lines.append(f"Flat-start PBL: mean {f['mean']:.6g} max {f['max']:.6g}")
    m = diagnostics.model_pbl
    lines.append(f"Model final-stage PBL: mean {m['mean']:.6g} max {m['max']:.6g}")
    for stage_idx, stats in enumerate(diagnostics.stage_pbl, start=1):
        lines.append(
            f"  stage {stage_idx} PBL: mean {stats['mean']:.6g} max {stats['max']:.6g} p95 {stats['p95']:.6g}"
        )
    lines.append(
        f"Loss: state_mse {diagnostics.train_state_mse:.6g}  "
        + "  ".join(
            f"phys_s{i + 1} {diagnostics.train_phys_stages[i]:.6g}" for i in range(len(diagnostics.train_phys_stages))
        )
    )
    rmse = diagnostics.state_rmse
    lines.append(
        "State RMSE vs labels (pu or rad): "
        + "  ".join(f"{key} {value:.6g}" for key, value in sorted(rmse.items()))
    )
    pe = diagnostics.pair_embed
    lines.append(f"Pair embed |e_ij|: mean {pe['mean_norm']:.6g} max {pe['max_norm']:.6g}")
    lines.append(f"Mean |Ybus| entry: {diagnostics.ybus_abs_mean:.6g}")
    lines.append("RPE channels (static feature scale; does not measure model usefulness):")
    for row in diagnostics.rpe_channels:
        lines.append(
            f"  {row['name']:12s} off_mean {row['off_mean']:+.4g} off_max {row['off_max']:+.4g} "
            f"diag_mean {row['diag_mean']:+.4g}"
        )
    lines.append(
        "Read the oracle line first: if it is not ~1e-6, fix data/units before judging the model."
    )
    return "\n".join(lines)


def audit_residual_correctness(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    *,
    agreement_atol: float = 1e-4,
) -> list[CorrectnessCheck]:
    """Labels satisfy the torch residual; torch matches the numpy reference; model residuals match its output."""
    model.eval()
    chosen = np.asarray(index, dtype=np.int64)
    power_p = split.power_p[chosen]
    power_q = split.power_q[chosen]
    voltage = split.voltage[chosen]
    angle = split.angle[chosen]
    types = network.types
    ybus = network.ybus
    specified, target = pack_case(
        power_p.to(device), power_q.to(device), voltage.to(device), angle.to(device), types
    )
    checks: list[CorrectnessCheck] = []
    with torch.no_grad():
        oracle_residual = nodal_residual(specified, target, types, ybus)
        oracle_per_case = torch.sqrt(oracle_residual[..., 0] ** 2 + oracle_residual[..., 1] ** 2).mean(dim=1)
        oracle_mean = float(oracle_per_case.mean())
        checks.append(
            CorrectnessCheck(
                "oracle_labels_torch",
                oracle_mean <= REFERENCE_PBL_TOLERANCE,
                f"torch residual on ground-truth labels: mean PBL {oracle_mean:.3e} "
                f"(tol {REFERENCE_PBL_TOLERANCE:.0e}; ~1e-6 expected)",
                value=oracle_mean,
            )
        )

        ybus_np = ybus.detach().cpu().numpy()
        numpy_per_case = np.array(
            [
                reference_mean_pbl(
                    power_p[row].numpy().astype(np.float64),
                    power_q[row].numpy().astype(np.float64),
                    voltage[row].numpy().astype(np.float64),
                    angle[row].numpy().astype(np.float64),
                    ybus_np,
                )
                for row in range(len(chosen))
            ]
        )
        gap = float(np.max(np.abs(numpy_per_case - oracle_per_case.cpu().numpy())))
        checks.append(
            CorrectnessCheck(
                "torch_vs_numpy_reference",
                gap <= agreement_atol,
                f"max per-case |torch PBL - numpy PBL| = {gap:.3e} (tol {agreement_atol:.0e})",
                value=gap,
            )
        )

        pair = pair_features(model, network)
        prediction, residuals = model(
            specified, types, network.rpe, network.pair_type_masks, ybus, pair=pair
        )
        recomputed = nodal_residual(specified, prediction, types, ybus)
        scale = float(residuals[-1].abs().max().clamp(min=1.0))
        drift = float((recomputed - residuals[-1]).abs().max())
        checks.append(
            CorrectnessCheck(
                "model_residual_matches_output",
                drift <= agreement_atol * scale,
                f"max |residual(final state) - stage residual| = {drift:.3e} "
                f"(tol {agreement_atol * scale:.0e})",
                value=drift,
            )
        )
    return checks


def format_correctness_checks(title: str, checks: list[CorrectnessCheck], failure_hint: str) -> str:
    lines = [title]
    for check in checks:
        mark = "OK" if check.passed else "FAIL"
        lines.append(f"  [{mark}] {check.name}: {check.detail}")
    if all(check.passed for check in checks):
        lines.append("  => all checks passed.")
    else:
        lines.append(f"  => {failure_hint}")
    return "\n".join(lines)


def print_forward_diagnostics(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    tag: str,
    *,
    rpe_ablation: bool = False,
) -> None:
    report = collect_forward_diagnostics(model, split, network, index, device)
    text = format_forward_diagnostics(tag, report)
    weight_rows = pair_encoder_input_weight_norms(model)
    baseline: dict[str, float] | None = None
    ablation_rows: list[dict[str, float]] | None = None
    if rpe_ablation:
        baseline, ablation_rows = collect_rpe_channel_ablation(model, split, network, index, device)
    text = text + "\n" + format_rpe_effectiveness_diagnostics(weight_rows, baseline, ablation_rows)
    text = text + "\n" + format_rpe_correctness_checks(audit_rpe_correctness(network))
    text = text + "\n" + format_correctness_checks(
        "Residual correctness audit (power-balance computation):",
        audit_residual_correctness(model, split, network, index, device),
        "Fix nodal_residual, units, or Ybus grouping before trusting PBL numbers.",
    )
    print(text, flush=True)


def wants_debug_diagnostics() -> bool:
    """On by default; disable with ``--no-debug`` or ``GENCO_TPGT_DEBUG=0``."""
    if "--no-debug" in sys.argv:
        return False
    if "--debug" in sys.argv:
        return True
    flag = os.environ.get("GENCO_TPGT_DEBUG", "").strip().lower()
    return flag not in ("0", "false", "no")


def wants_rpe_ablation(*, default: bool) -> bool:
    if "--no-rpe-ablation" in sys.argv:
        return False
    if "--rpe-ablation" in sys.argv:
        return True
    flag = os.environ.get("GENCO_TPGT_RPE_ABLATION", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return default


def debug_sample_index(split: Split, network_id: int = 0, count: int = 32) -> np.ndarray:
    index = torch.nonzero(split.network_index == network_id, as_tuple=False).flatten()
    take = min(int(count), int(index.numel()))
    return index[:take].numpy()


def subset_split(split: Split, index: np.ndarray, name: str) -> Split:
    """Rows ``index`` of ``split``; keeps the full network list so network ids stay valid."""
    chosen = torch.as_tensor(np.asarray(index, dtype=np.int64))
    return Split(
        name=name,
        count=int(chosen.numel()),
        power_p=split.power_p[chosen],
        power_q=split.power_q[chosen],
        voltage=split.voltage[chosen],
        angle=split.angle[chosen],
        network_index=split.network_index[chosen],
        networks=split.networks,
    )


def train_probe_split(split: Split, count: int) -> Split:
    """Evenly spaced fixed subset of training cases, scored with the same PBL metric as validation."""
    take = max(1, min(int(count), split.count))
    index = np.unique(np.linspace(0, split.count - 1, take).round().astype(np.int64))
    return subset_split(split, index, f"{split.name}_probe")


def dominant_network_id(split: Split) -> int:
    counts = torch.bincount(split.network_index, minlength=len(split.networks))
    return int(torch.argmax(counts))


def network_case_counts(split: Split) -> list[tuple[int, int]]:
    """(network_id, case count) sorted by count, largest first."""
    counts = torch.bincount(split.network_index, minlength=len(split.networks))
    order = torch.argsort(counts, descending=True)
    return [(int(net), int(counts[net])) for net in order if int(counts[net]) > 0]


def rpe_shift_from_reference(rpe: torch.Tensor, reference: torch.Tensor) -> list[float]:
    """Mean |RPE - reference RPE| per channel over off-diagonal pairs (how far a topology is from training)."""
    diff = (rpe.detach().float().cpu() - reference.detach().float().cpu()).abs()
    buses = diff.shape[0]
    off = ~torch.eye(buses, dtype=torch.bool)
    return [float(diff[..., channel][off].mean()) for channel in range(diff.shape[-1])]


CONTINGENCY_CLASSES = ("line outage", "generator outage", "line+generator", "unchanged")


def contingency_class(network: Network, reference: Network) -> str:
    """Line outages change Ybus (and so the RPE); generator outages only change bus types (PV -> PQ)."""
    line = not torch.allclose(
        network.ybus.detach().cpu(), reference.ybus.detach().cpu(), rtol=1e-5, atol=1e-6
    )
    generator = not torch.equal(network.types.detach().cpu(), reference.types.detach().cpu())
    if line and generator:
        return "line+generator"
    if line:
        return "line outage"
    if generator:
        return "generator outage"
    return "unchanged"


def print_contingency_diagnostics(
    model: TPGT,
    label: str,
    split: Split,
    networks: list[Network],
    device: torch.device,
    reference: Network,
    per_network_pbl: dict[int, list[float]],
    *,
    networks_per_class: int = 3,
    cases_per_network: int = 8,
) -> None:
    """PBL by contingency class, plus RPE/residual audits and RPE ablation on the largest groups of each class."""
    lines = [f"=== Contingency diagnostics {label} ({split.count} cases, {len(networks)} topologies) ==="]
    classes = [contingency_class(network, reference) for network in networks]
    scored = sorted(
        ((total / count, net, int(count)) for net, (total, count) in per_network_pbl.items() if count > 0),
        reverse=True,
    )
    if scored:
        values = np.array([score for score, _net, _count in scored])
        worst_score, worst_net, worst_count = scored[0]
        lines.append(
            f"Per-topology PBL: median {np.median(values):.4g}  p90 {np.quantile(values, 0.9):.4g}  "
            f"best {values.min():.4g}  worst {worst_score:.4g} "
            f"(topology {worst_net}, {classes[worst_net]}, {worst_count} cases)"
        )
        lines.append("PBL by contingency class (line outage changes Ybus/RPE; generator outage changes bus types):")
        for name in CONTINGENCY_CLASSES:
            members = [(score, count) for score, net, count in scored if classes[net] == name]
            if not members:
                continue
            cases = sum(count for _score, count in members)
            mean = sum(score * count for score, count in members) / cases
            lines.append(f"  {name:16s} {len(members):4d} topologies  {cases:5d} cases  mean PBL {mean:.4g}")

    chosen: list[tuple[int, str]] = []
    for name in CONTINGENCY_CLASSES:
        picked = [net for net, _count in network_case_counts(split) if classes[net] == name][:networks_per_class]
        chosen.extend((net, name) for net in picked)

    deltas: dict[str, np.ndarray] = {}
    shifts: dict[str, np.ndarray] = {}
    weights: dict[str, int] = {}
    rpe_failures: dict[str, int] = {}
    residual_failures: dict[str, int] = {}
    worst_values: dict[str, float] = {}
    for net, name in chosen:
        network = networks[net]
        index = debug_sample_index(split, network_id=net, count=cases_per_network)
        weight = len(index)
        _baseline, rows = collect_rpe_channel_ablation(model, split, network, index, device)
        deltas[name] = deltas.get(name, np.zeros(RPE_DIM)) + np.array([row["delta_pbl_mean"] for row in rows]) * weight
        shifts[name] = shifts.get(name, np.zeros(RPE_DIM)) + np.array(
            rpe_shift_from_reference(network.rpe, reference.rpe)
        ) * weight
        weights[name] = weights.get(name, 0) + weight
        for check in audit_rpe_correctness(network):
            if not check.passed:
                rpe_failures[check.name] = rpe_failures.get(check.name, 0) + 1
        for check in audit_residual_correctness(model, split, network, index, device):
            if check.value is not None:
                worst_values[check.name] = max(worst_values.get(check.name, 0.0), check.value)
            if not check.passed:
                residual_failures[check.name] = residual_failures.get(check.name, 0) + 1
        clear_pair_embed_cache([network])

    checked = len(chosen)
    if rpe_failures:
        detail = ", ".join(f"{name} x{count}" for name, count in sorted(rpe_failures.items()))
        lines.append(f"RPE correctness on {checked} topologies: FAIL ({detail})")
    else:
        lines.append(f"RPE correctness on {checked} topologies: OK")
    worst_text = ", ".join(f"worst {name} {value:.3e}" for name, value in sorted(worst_values.items()))
    if residual_failures:
        detail = ", ".join(f"{name} x{count}" for name, count in sorted(residual_failures.items()))
        lines.append(f"Residual correctness on {checked} topologies: FAIL ({detail}; {worst_text})")
    else:
        lines.append(f"Residual correctness on {checked} topologies: OK ({worst_text})")

    present = [name for name in CONTINGENCY_CLASSES if weights.get(name)]
    if present:
        mean_delta = {name: deltas[name] / weights[name] for name in present}
        mean_shift = {name: shifts[name] / weights[name] for name in present}
        lines.append(
            "RPE zero-ablation by class (ΔPBL when the channel is zeroed; shift = mean |RPE - training RPE|):"
        )
        lines.append(
            "  " + " " * 12 + "".join(
                f"  {name} ({sum(1 for _net, item in chosen if item == name)} topo)".ljust(32) for name in present
            )
        )
        order = np.argsort(-np.mean([mean_delta[name] for name in present], axis=0))
        for channel in order:
            cells = "".join(
                f"  ΔPBL {mean_delta[name][channel]:+.4g}  shift {mean_shift[name][channel]:.3g}".ljust(32)
                for name in present
            )
            lines.append(f"  {RPE_FEATURE_NAMES[channel]:12s}{cells}")
        lines.append(
            "  Generator outages leave the RPE unchanged (shift 0): their error comes from the bus-type change, "
            "not the pair features. For line outages, large shift plus large ΔPBL marks the channels that "
            "move under outages and that the model depends on."
        )
    print("\n".join(lines), flush=True)


@dataclass
class Network:
    ybus: torch.Tensor
    rpe: torch.Tensor
    component: torch.Tensor
    types: torch.Tensor
    pair_type_masks: torch.Tensor | None = field(default=None, repr=False)
    pair_embed: torch.Tensor | None = field(default=None, repr=False)
    branches: tuple[tuple[int, int, float, float], ...] | None = field(default=None, repr=False)


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
    """Five pair features on the full in-service meshed graph.

    Order: Re(Y)/s_Y, Im(Y)/s_Y, A_ij, hop/d_max, eff_res/r_max. Branch r and
    x build W (for effective resistance); Y comes from the case Ybus used in
    the power-flow residual.

    s_Y(c) is the mean of off-diagonal |Y_pq| with p != q and |Y_pq| > eps
    (zeros from non-adjacent pairs are excluded).

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

    hop_symmetry = np.max(np.abs(hop - hop.T))
    if hop_symmetry > 1e-5:
        raise RuntimeError("Hop distance on the meshed graph is not symmetric.")
    resistance_symmetry = np.max(np.abs(effective - effective.T))
    if resistance_symmetry > 1e-4:
        raise RuntimeError("Effective resistance is not symmetric.")
    if not np.isfinite(features).all():
        raise RuntimeError("Meshed RPE contains non-finite values.")
    return features, component


@dataclass
class CorrectnessCheck:
    name: str
    passed: bool
    detail: str
    value: float | None = None


def audit_rpe_correctness(network: Network, *, recompute_atol: float = 1e-4) -> list[CorrectnessCheck]:
    """Structural checks plus optional full recompute from stored Ybus and in-service branches."""
    rpe = network.rpe.detach().cpu().numpy()
    ybus = network.ybus.detach().cpu().numpy()
    component = network.component.detach().cpu().numpy().astype(np.int64)
    bus_count = rpe.shape[0]
    checks: list[CorrectnessCheck] = []

    if rpe.shape != (bus_count, bus_count, RPE_DIM):
        checks.append(
            CorrectnessCheck("shape", False, f"expected ({bus_count}, {bus_count}, {RPE_DIM}), got {rpe.shape}")
        )
        return checks
    checks.append(CorrectnessCheck("shape", True, str(tuple(rpe.shape))))

    finite = bool(np.isfinite(rpe).all())
    checks.append(
        CorrectnessCheck(
            "finite",
            finite,
            "all channels finite" if finite else "non-finite values in cached RPE",
        )
    )

    diagonal = np.arange(bus_count)
    diag_adj = bool(np.allclose(rpe[diagonal, diagonal, 2], 0.0))
    checks.append(
        CorrectnessCheck(
            "diag_adjacency_zero",
            diag_adj,
            "diagonal adjacency is 0" if diag_adj else "diagonal adjacency must be 0",
        )
    )
    diag_hop = bool(np.allclose(rpe[diagonal, diagonal, 3], 0.0))
    checks.append(
        CorrectnessCheck(
            "diag_hop_zero",
            diag_hop,
            "diagonal normalized hop is 0" if diag_hop else "diagonal hop must be 0",
        )
    )
    diag_reff = bool(np.allclose(rpe[diagonal, diagonal, 4], 0.0))
    checks.append(
        CorrectnessCheck(
            "diag_reff_zero",
            diag_reff,
            "diagonal normalized Reff is 0" if diag_reff else "diagonal Reff must be 0",
        )
    )

    hop_sym = bool(np.allclose(rpe[:, :, 3], rpe[:, :, 3].T, atol=1e-5))
    checks.append(
        CorrectnessCheck(
            "hop_symmetric",
            hop_sym,
            "normalized hop symmetric" if hop_sym else "hop channel not symmetric",
        )
    )
    reff_sym = bool(np.allclose(rpe[:, :, 4], rpe[:, :, 4].T, atol=1e-4))
    checks.append(
        CorrectnessCheck(
            "reff_symmetric",
            reff_sym,
            "normalized Reff symmetric" if reff_sym else "Reff channel not symmetric",
        )
    )

    y_abs = np.abs(ybus.astype(np.complex128))
    admittance_errors: list[float] = []
    for origin in range(bus_count):
        comp = int(component[origin])
        if comp < 0:
            continue
        y_scale = component_admittance_scale(y_abs, component, comp)
        for destination in range(bus_count):
            if int(component[destination]) != comp:
                continue
            admittance_errors.append(abs(rpe[origin, destination, 0] - ybus.real[origin, destination] / y_scale))
            admittance_errors.append(abs(rpe[origin, destination, 1] - ybus.imag[origin, destination] / y_scale))
    max_adm = float(max(admittance_errors)) if admittance_errors else 0.0
    adm_ok = max_adm <= recompute_atol
    checks.append(
        CorrectnessCheck(
            "ybus_admittance_channels",
            adm_ok,
            f"max |RPE0/1 - Y/s_Y| = {max_adm:.3e} (tol {recompute_atol:.0e})",
        )
    )

    if getattr(network, "branches", None) is not None:
        branches = network.branches
        assert branches is not None
        adjacency = np.zeros((bus_count, bus_count), dtype=np.float32)
        for origin, destination, _resistance, _reactance in branches:
            adjacency[origin, destination] = 1.0
            adjacency[destination, origin] = 1.0
        adj_ok = bool(np.allclose(rpe[:, :, 2], adjacency, atol=1e-6))
        checks.append(
            CorrectnessCheck(
                "adjacency_vs_branches",
                adj_ok,
                f"{len(branches)} in-service branches"
                + (" match channel 2" if adj_ok else " disagree with channel 2"),
            )
        )

        expected, expected_component = meshed_rpe(bus_count, list(branches), ybus)
        max_diff = float(np.max(np.abs(expected - rpe)))
        recompute_ok = max_diff <= recompute_atol and np.array_equal(expected_component, component)
        checks.append(
            CorrectnessCheck(
                "recompute_from_ybus_branches",
                recompute_ok,
                f"max |recomputed - cached| = {max_diff:.3e} (tol {recompute_atol:.0e})",
            )
        )
    else:
        checks.append(
            CorrectnessCheck(
                "recompute_from_ybus_branches",
                True,
                "skipped (branches not stored on Network; rebuild split cache to enable)",
            )
        )

    return checks


def format_rpe_correctness_checks(checks: list[CorrectnessCheck]) -> str:
    lines = ["RPE correctness audit (cached tensor vs Ybus/branches invariants):"]
    all_pass = all(check.passed for check in checks)
    for check in checks:
        mark = "OK" if check.passed else "FAIL"
        lines.append(f"  [{mark}] {check.name}: {check.detail}")
    if all_pass:
        lines.append("  => RPE calculation checks passed for this topology.")
    else:
        lines.append("  => Fix meshed_rpe or split cache before trusting pair features.")
    return "\n".join(lines)


def component_admittance_scale(
    y_abs: np.ndarray,
    component: np.ndarray,
    comp: int,
    epsilon: float = Y_ADMITTANCE_EPS,
) -> float:
    """Mean off-diagonal |Y_pq| in component ``comp``, excluding entries <= ``epsilon``."""
    buses = np.flatnonzero(component == comp)
    if buses.size == 0:
        return 1.0
    block = y_abs[np.ix_(buses, buses)]
    off_mask = ~np.eye(buses.size, dtype=bool)
    magnitudes = block[off_mask]
    nonzero = magnitudes > epsilon
    if not np.any(nonzero):
        return 1.0
    return float(magnitudes[nonzero].mean())


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


def parse_data_fraction() -> float:
    for index, arg in enumerate(sys.argv):
        if arg == "--data-fraction" and index + 1 < len(sys.argv):
            return float(sys.argv[index + 1])
        if arg.startswith("--data-fraction="):
            return float(arg.split("=", 1)[1])
    env = os.environ.get("GENCO_TPGT_DATA_FRACTION", "").strip()
    if env:
        return float(env)
    return 1.0


def parse_micro_batch_size() -> int:
    for index, arg in enumerate(sys.argv):
        if arg == "--micro-batch" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--micro-batch="):
            return int(arg.split("=", 1)[1])
    env = os.environ.get("GENCO_TPGT_MICRO_BATCH", "").strip()
    if env:
        return int(env)
    return MICRO_BATCH_SIZE


def parse_eval_batch_size() -> int:
    for index, arg in enumerate(sys.argv):
        if arg == "--eval-batch" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--eval-batch="):
            return int(arg.split("=", 1)[1])
    env = os.environ.get("GENCO_TPGT_EVAL_BATCH", "").strip()
    if env:
        return int(env)
    return EVAL_BATCH_SIZE


def parse_val_every() -> int:
    for index, arg in enumerate(sys.argv):
        if arg == "--val-every" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--val-every="):
            return int(arg.split("=", 1)[1])
    env = os.environ.get("GENCO_TPGT_VAL_EVERY", "").strip()
    if env:
        return int(env)
    return VAL_EVERY


def scenario_limit(raw: Path) -> int | None:
    """Number of scenarios kept from the start of each split folder (prefix, not a random subset)."""
    if DATA_FRACTION >= 1.0:
        return None
    if DATA_FRACTION <= 0.0:
        raise ValueError(f"data fraction must be in (0, 1], got {DATA_FRACTION}")
    full = scenario_count(raw)
    return max(1, min(full, int(full * DATA_FRACTION)))


def networks_match(left: Network, right: Network) -> bool:
    return bool(
        torch.equal(left.types, right.types)
        and torch.allclose(left.ybus, right.ybus, rtol=1e-5, atol=1e-6)
    )


def merge_splits(base: Split, extra: Split) -> Split:
    """Append ``extra`` scenarios after ``base``; reuse matching ``Network`` objects."""
    network_map: dict[int, int] = {}
    merged_networks = list(base.networks)
    for local_id, network in enumerate(extra.networks):
        matched = next(
            (index for index, existing in enumerate(merged_networks) if networks_match(existing, network)),
            None,
        )
        if matched is None:
            matched = len(merged_networks)
            merged_networks.append(network)
        network_map[local_id] = matched
    extra_index = extra.network_index.numpy()
    mapped = np.array([network_map[int(value)] for value in extra_index], dtype=np.int64)
    return Split(
        name=base.name,
        count=base.count + extra.count,
        power_p=torch.cat([base.power_p, extra.power_p]),
        power_q=torch.cat([base.power_q, extra.power_q]),
        voltage=torch.cat([base.voltage, extra.voltage]),
        angle=torch.cat([base.angle, extra.angle]),
        network_index=torch.cat([base.network_index, torch.from_numpy(mapped)]),
        networks=merged_networks,
    )


def extend_split_prefix(raw: Path, name: str, existing: Split, new_prefix_end: int) -> Split:
    full = scenario_count(raw)
    new_end = min(full, new_prefix_end)
    if new_end <= existing.count:
        raise ValueError(f"Cannot extend {name}: new prefix {new_end} <= existing {existing.count}.")
    print(
        f"Extending {name} prefix {existing.count} -> {new_end} (+{new_end - existing.count} scenarios); "
        f"first {existing.count} scenarios unchanged.",
        flush=True,
    )
    extra = read_split(raw, name, prefix_start=existing.count, prefix_end=new_end)
    return merge_splits(existing, extra)


YBusEntry = tuple[int, int, float, float]


def ybus_from_entries(entries: list[YBusEntry], bus_count: int = 118) -> np.ndarray:
    ybus = np.zeros((bus_count, bus_count), dtype=np.complex64)
    for row, column, conductance, susceptance in entries:
        ybus[row, column] = conductance + 1j * susceptance
    return ybus


def ybus_entries_fingerprint(entries: list[YBusEntry]) -> float:
    """Stable hash of sparse Y-bus entries for network grouping."""
    if not entries:
        return 0.0
    packed = np.array(
        sorted(
            (int(row), int(column), round(float(conductance), 6), round(float(susceptance), 6))
            for row, column, conductance, susceptance in entries
        ),
        dtype=np.float64,
    )
    digest = hashlib.sha256(packed.tobytes()).digest()
    return float(np.frombuffer(digest[:8], dtype=np.float64)[0])


def load_ybus_entries_by_scenario(
    raw: Path,
    scenario_ids: list[int],
) -> dict[int, list[YBusEntry]]:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    if not scenario_ids:
        return {}
    table = ds.dataset(raw / "y_bus_data.parquet").to_table(
        filter=pc.field("scenario").isin(scenario_ids),
        columns=["scenario", "index1", "index2", "G", "B"],
    )
    grouped: dict[int, list[YBusEntry]] = {}
    scenarios = table.column("scenario").to_numpy()
    rows_i = table.column("index1").to_numpy()
    rows_j = table.column("index2").to_numpy()
    conductance = table.column("G").to_numpy()
    susceptance = table.column("B").to_numpy()
    for index in range(len(scenarios)):
        scenario_id = int(scenarios[index])
        grouped.setdefault(scenario_id, []).append(
            (
                int(rows_i[index]),
                int(rows_j[index]),
                float(conductance[index]),
                float(susceptance[index]),
            )
        )
    missing = [scenario_id for scenario_id in scenario_ids if scenario_id not in grouped]
    if missing:
        raise RuntimeError(f"{raw} y_bus_data missing {len(missing)} scenario(s), e.g. {missing[:3]}.")
    return grouped


def reference_mean_pbl(
    power_p: np.ndarray,
    power_q: np.ndarray,
    voltage: np.ndarray,
    angle: np.ndarray,
    ybus: np.ndarray,
) -> float:
    solved = np.exp(1j * angle) * voltage
    injection = solved * np.conj(ybus.astype(np.complex128) @ solved)
    mismatch = np.sqrt((power_p - injection.real) ** 2 + (power_q - injection.imag) ** 2)
    return float(mismatch.mean())


def read_split(
    raw: Path,
    name: str,
    *,
    prefix_end: int | None = None,
    prefix_start: int = 0,
) -> Split:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    full = scenario_count(raw)
    start = prefix_start
    end = full if prefix_end is None else min(full, prefix_end)
    if start < 0 or start >= end:
        raise ValueError(f"{name}: invalid scenario prefix [{start}, {end}) for {full} scenarios.")
    count = end - start
    branch_rows = pq.ParquetFile(raw / "branch_data.parquet").metadata.num_rows
    if branch_rows % full != 0:
        raise RuntimeError(f"{raw} branch rows are not one block per scenario.")
    branch_count = branch_rows // full
    bus = pq.read_table(
        raw / "bus_data.parquet",
        columns=["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"],
    ).slice(start * 118, count * 118)
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
    ).slice(start * branch_count, count * branch_count)
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

    wanted = bus_scenario[:, 0].astype(np.int64)
    scenario_ids = [int(wanted[local]) for local in range(count)]
    y_by_scenario = load_ybus_entries_by_scenario(raw, scenario_ids)
    y_fingerprint = np.array(
        [ybus_entries_fingerprint(y_by_scenario[scenario_ids[local]]) for local in range(count)],
        dtype=np.float64,
    ).reshape(count, 1)
    signature = np.concatenate([signature, y_fingerprint], axis=1)
    _, network_ids = np.unique(signature, axis=0, return_inverse=True)
    print(
        f"{name}: {count} scenarios loaded"
        f"{'' if prefix_end is None else f' of {full}'}"
        f"{'' if start == 0 else f' (prefix offset {start})'}"
        f" across {int(network_ids.max()) + 1} network(s).",
        flush=True,
    )

    max_reference_pbl = 0.0
    worst_local = 0
    for local in range(count):
        scenario_id = scenario_ids[local]
        ybus = ybus_from_entries(y_by_scenario[scenario_id])
        pbl_mean = reference_mean_pbl(
            power_p[local],
            power_q[local],
            voltage[local],
            angle[local],
            ybus,
        )
        if pbl_mean > max_reference_pbl:
            max_reference_pbl = pbl_mean
            worst_local = local
        if pbl_mean > REFERENCE_PBL_TOLERANCE:
            raise RuntimeError(
                f"{name} scenario {scenario_id} reference PBL {pbl_mean:.6g} "
                f"exceeds tolerance {REFERENCE_PBL_TOLERANCE}."
            )
    print(
        f"  reference PBL for {name}: max {max_reference_pbl:.3e} over {count} scenario(s)"
        f" (worst local index {worst_local})",
        flush=True,
    )

    representatives = [int(np.flatnonzero(network_ids == net)[0]) for net in range(int(network_ids.max()) + 1)]
    networks: list[Network] = []
    for local in representatives:
        types = type_code[local].astype(np.int64)
        if int(np.sum(types == 2)) != 1:
            raise RuntimeError(f"{name} scenario {scenario_ids[local]} has {int(np.sum(types == 2))} slack buses.")
        scenario_id = scenario_ids[local]
        ybus = ybus_from_entries(y_by_scenario[scenario_id])
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
        networks.append(
            Network(
                ybus=torch.from_numpy(ybus),
                rpe=torch.from_numpy(features),
                component=torch.from_numpy(component.astype(np.int64)),
                types=torch.from_numpy(types),
                branches=tuple(branches),
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


def _mirror_cache_file(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or source.stat().st_mtime > destination.stat().st_mtime:
        shutil.copy2(source, destination)
        return True
    return False


def sync_split_cache_from_drive(persistent: Path, active: Path) -> None:
    """Copy Drive split caches to local disk so ``torch.load`` is fast on Colab."""
    if persistent == active or not persistent.is_dir():
        return
    active.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(persistent.iterdir()):
        if path.is_file() and _mirror_cache_file(path, active / path.name):
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


def cached_split(active: Path, persistent: Path, raw: Path, name: str, limit: int | None = None) -> Split:
    sync_split_cache_from_drive(persistent, active)
    full = scenario_count(raw)
    if limit is None:
        limit = scenario_limit(raw)
    if limit is None:
        limit = full
    stamp = {
        "version": CACHE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "bytes": raw.joinpath("bus_data.parquet").stat().st_size,
        "scenario_limit": limit,
    }
    stamp_path = active / f"{name}.json"
    tensor_path = active / f"{name}.pt"
    if stamp_path.exists() and tensor_path.exists():
        saved = json.loads(stamp_path.read_text(encoding="utf-8"))
        version_ok = (
            saved.get("version") == stamp["version"]
            and saved.get("checkpoint_version") == stamp["checkpoint_version"]
            and saved.get("bytes") == stamp["bytes"]
        )
        old_limit = saved.get("scenario_limit")
        if old_limit is None:
            old_limit = full
        if version_ok and limit == old_limit:
            load_started = time.perf_counter()
            print(f"Loading cached {name} split from {tensor_path}", flush=True)
            payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
            elapsed = time.perf_counter() - load_started
            print(f"  loaded {name} split in {elapsed:.1f}s ({tensor_path})", flush=True)
            sync_split_cache_to_drive(active, persistent)
            return payload
        if version_ok and limit > old_limit:
            existing = torch.load(tensor_path, map_location="cpu", weights_only=False)
            if isinstance(existing, Split) and existing.count == old_limit:
                build_started = time.perf_counter()
                split = extend_split_prefix(raw, name, existing, limit)
                print(
                    f"  extended {name} split in {time.perf_counter() - build_started:.1f}s",
                    flush=True,
                )
                torch.save(split, tensor_path)
                stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
                sync_split_cache_to_drive(active, persistent)
                return split
        if version_ok and limit < old_limit:
            print(
                f"Shrinking {name} data prefix {old_limit} -> {limit}; rebuilding split cache.",
                flush=True,
            )
    active.mkdir(parents=True, exist_ok=True)
    build_started = time.perf_counter()
    split = read_split(raw, name, prefix_end=limit)
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
COLAB_LOCAL_CACHE = Path(f"/content/{RUN_DIR_NAME}_cache")


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
    """Checkpoint directory and TPGT result directory on Drive, both versioned by ``RUN_DIR_NAME``.

    Colab writes under MyDrive/Foundation Model once Drive is mounted.
    Windows writes under K:\\My Drive\\Foundation Model when that folder exists.
    """
    if drive is None:
        return None
    foundation = drive / "Foundation Model"
    if not in_colab() and not foundation.is_dir():
        return None
    checkpoints = foundation / RUN_DIR_NAME
    results = foundation / "results" / RUN_DIR_NAME
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
        branches=getattr(network, "branches", None),
    )


def clear_pair_embed_cache(networks: list[Network]) -> None:
    for network in networks:
        network.pair_embed = None


def pair_features(model: TPGT, network: Network) -> torch.Tensor:
    """Edge embeddings for this topology; cached in eval on that Network object only."""
    if not model.training and network.pair_embed is not None:
        return network.pair_embed
    encoded = model.pair_encoder(network.rpe)
    if not model.training:
        network.pair_embed = encoded
    return encoded


def run_batch(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    pair: torch.Tensor | None = None,
) -> tuple[TaskLossBreakdown, torch.Tensor]:
    chosen = torch.as_tensor(index, dtype=torch.long)
    power_p = split.power_p[chosen].to(device, non_blocking=True)
    power_q = split.power_q[chosen].to(device, non_blocking=True)
    voltage = split.voltage[chosen].to(device, non_blocking=True)
    angle = split.angle[chosen].to(device, non_blocking=True)
    specified, target = pack_case(power_p, power_q, voltage, angle, network.types)
    if pair is None:
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


def _weighted_loss_breakdown(
    parts: list[tuple[TaskLossBreakdown, float]],
) -> TaskLossBreakdown:
    weight = sum(weight for _breakdown, weight in parts)
    if weight <= 0:
        raise RuntimeError("empty training batch")
    state = sum(float(b.state_mse.detach()) * w for b, w in parts) / weight
    phys = sum(float(b.physics_sum.detach()) * w for b, w in parts) / weight
    stages = tuple(
        sum(float(b.physics_stages[stage_idx].detach()) * w for b, w in parts) / weight
        for stage_idx in range(DEPTH)
    )
    total = sum(float(b.total.detach()) * w for b, w in parts) / weight
    return TaskLossBreakdown(
        torch.tensor(state),
        torch.tensor(phys),
        tuple(torch.tensor(value) for value in stages),
        torch.tensor(total),
    )


def rpe_channel_dropout(rpe: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    """Zero whole RPE channels independently with ``probability`` (no rescaling); identity unless training."""
    if not training or probability <= 0.0:
        return rpe
    keep = torch.rand(rpe.shape[-1], device=rpe.device) >= probability
    return rpe * keep.to(rpe.dtype)


def train_batch(
    model: TPGT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
) -> TaskLossBreakdown:
    """One optimizer step over ``index`` (up to BATCH_SIZE cases), using micro-batches on CUDA."""
    cases = np.asarray(index)
    n = len(cases)
    micro = min(MICRO_BATCH_SIZE, n)
    use_amp = device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    logged_parts: list[tuple[TaskLossBreakdown, float]] = []
    rpe = rpe_channel_dropout(network.rpe, RPE_CHANNEL_DROPOUT, model.training)
    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
        pair_embed = model.pair_encoder(rpe)
    for start in range(0, n, micro):
        chunk = cases[start : start + micro]
        weight = len(chunk) / n
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            breakdown, _score = run_batch(model, split, network, chunk, device, pair=pair_embed)
        scaled = breakdown.total * weight
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()
        logged_parts.append((breakdown, weight))
        if device.type == "cuda":
            del breakdown, scaled
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return _weighted_loss_breakdown(logged_parts)


def iterate_indexes(
    split: Split,
    seed: int,
    epoch: int,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
):
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
        for start in range(0, len(index), batch_size):
            yield network_id, index[start : start + batch_size]


def evaluate_pbl(
    model: TPGT,
    split: Split,
    networks: list[Network],
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
    per_network_out: dict[int, list[float]] | None = None,
) -> float:
    model.eval()
    clear_pair_embed_cache(networks)
    total = 0.0
    seen = 0
    with torch.no_grad():
        for network_id, index in iterate_indexes(split, seed=0, epoch=0, shuffle=False, batch_size=batch_size):
            _breakdown, score = run_batch(model, split, networks[network_id], index, device)
            weighted = float(score) * len(index)
            total += weighted
            seen += len(index)
            if per_network_out is not None:
                bucket = per_network_out.setdefault(network_id, [0.0, 0.0])
                bucket[0] += weighted
                bucket[1] += len(index)
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
    del device  # load on CPU; model/optimizer load_state_dict moves weights as needed
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
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
    """Epochs finished for the current code version only (old complete.json counts as 0).

    An early-stopped seed counts as EPOCHS so reruns skip it.
    """
    try:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
            return 0
        if payload.get("early_stopped"):
            return EPOCHS
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
    best_state = payload.get("best_state")
    if isinstance(best_state, dict):
        best_state = {
            key: value.cpu() if torch.is_tensor(value) else value for key, value in best_state.items()
        }
    return (
        {
            "epoch": int(payload["epoch"]),
            "best_val": float(payload["best_val"]),
            "wait": int(payload["wait"]),
            "best_state": best_state,
            "data_fraction": payload.get("data_fraction"),
            "scheduler": payload.get("scheduler"),
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
    val_display: str,
    lr: float | None = None,
) -> str:
    n = max(batches, 1)
    stage_text = "  ".join(f"phys_s{i + 1} {value / n:.4f}" for i, value in enumerate(physics_stages))
    lr_text = "" if lr is None else f"  lr {lr:.1e}"
    return (
        f"seed {seed} epoch {epoch_number}/{EPOCHS}  "
        f"state {state_mse / n:.4f}  phys {physics_sum / n:.4f}  {stage_text}  "
        f"train {train_total / n:.4f}  val_pbl {val_display}{lr_text}"
    )


def make_lr_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
    )


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def train_seed(
    seed: int,
    model_dir: Path,
    train_split: Split,
    val_split: Split,
    device: torch.device,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    val_every: int = VAL_EVERY,
    debug_diagnostics: bool = False,
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
    configure_cuda_training(device)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    model = TPGT().to(device)
    if wants_torch_compile():
        print(
            "torch.compile enabled (GENCO_TPGT_COMPILE=1, GENCO_TPGT_FAST=1, or --compile / --fast)",
            flush=True,
        )
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = make_lr_scheduler(optimizer)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_networks = [move_to(network, device) for network in train_split.networks]
    val_networks = [move_to(network, device) for network in val_split.networks]
    train_probe = train_probe_split(train_split, min(val_split.count, TRAIN_PROBE_CASES))
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
        saved_fraction = restored["data_fraction"]
        if saved_fraction is None or float(saved_fraction) != DATA_FRACTION:
            previous = "unknown" if saved_fraction is None else f"{float(saved_fraction):g}"
            print(
                f"Seed {seed}: checkpoint data fraction {previous} != current {DATA_FRACTION:g}; "
                "keeping weights and optimizer, resetting best-val tracking (val set changed).",
                flush=True,
            )
            best_val = float("inf")
            wait = 0
            best_state = None
        elif restored["scheduler"] is None:
            print(f"Seed {seed}: checkpoint has no LR scheduler state; starting a fresh scheduler.", flush=True)
        else:
            try:
                scheduler.load_state_dict(restored["scheduler"])
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                scheduler = make_lr_scheduler(optimizer)
                print(f"Seed {seed}: LR scheduler state not restored ({error}); starting fresh.", flush=True)
        print(f"Seed {seed}: resumed optimizer lr {current_lr(optimizer):.1e}", flush=True)
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
            "scheduler": scheduler.state_dict(),
            "epoch": epoch_number,
            "best_val": best_val,
            "wait": wait,
            "best_state": best_state,
            "data_fraction": DATA_FRACTION,
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

    def finish_checkpoint_epoch(epoch_number: int, force: bool = False) -> None:
        """Run held-out test on checkpoint epochs, then save Drive checkpoint when due."""
        due = force or saves_checkpoint_epoch(epoch_number)
        test_scores: tuple[dict[str, float], dict[str, int]] | None = None
        if on_epoch_end is not None and due:
            test_scores = on_epoch_end(model, epoch_number)
        if due:
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

    stopped_early = False
    if not finished:
        for epoch in range(start_epoch, EPOCHS):
            if wait >= EARLY_STOP_PATIENCE:
                print(
                    f"Seed {seed}: early stopping after epoch {start_epoch} "
                    f"({wait} validations without improvement, best val_pbl {best_val:.6f}).",
                    flush=True,
                )
                stopped_early = True
                break
            model.train()
            clear_pair_embed_cache(train_networks)
            clear_pair_embed_cache(val_networks)
            running_total = 0.0
            running_state = 0.0
            running_phys = 0.0
            running_phys_stages = [0.0] * DEPTH
            batches = 0
            logged_first_batch = False
            batch_durations: list[float] = []
            for network_id, index in iterate_indexes(train_split, seed, epoch, shuffle=True):
                if not logged_first_batch:
                    first_batch_started = time.perf_counter()
                step_started = time.perf_counter()
                breakdown = train_batch(
                    model,
                    optimizer,
                    grad_scaler if device.type == "cuda" else None,
                    train_split,
                    train_networks[network_id],
                    index,
                    device,
                )
                if not logged_first_batch:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - first_batch_started
                    print(
                        f"  seed {seed} epoch {epoch + 1} first batch ({len(index)} cases) "
                        f"forward+backward {elapsed:.2f}s",
                        flush=True,
                    )
                    logged_first_batch = True
                batch_durations.append(time.perf_counter() - step_started)
                running_total += float(breakdown.total.detach())
                running_state += float(breakdown.state_mse.detach())
                running_phys += float(breakdown.physics_sum.detach())
                for stage_idx, stage_mse in enumerate(breakdown.physics_stages):
                    running_phys_stages[stage_idx] += float(stage_mse.detach())
                batches += 1
                if batches % 250 == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    window = batch_durations[-250:]
                    avg_sec = sum(window) / len(window)
                    print(
                        f"  seed {seed} epoch {epoch + 1} batch {batches} train {running_total / batches:.4f} "
                        f"avg {avg_sec:.3f}s/batch (last {len(window)})",
                        flush=True,
                    )
            run_validation = val_every <= 1 or (epoch + 1) % val_every == 0 or (epoch + 1) == EPOCHS
            if run_validation:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                val_score = evaluate_pbl(
                    model,
                    val_split,
                    val_networks,
                    device,
                    batch_size=eval_batch_size,
                )
                train_probe_score = evaluate_pbl(
                    model,
                    train_probe,
                    train_networks,
                    device,
                    batch_size=eval_batch_size,
                )
                val_display = (
                    f"{val_score:.6f}  train_pbl {train_probe_score:.6f}  "
                    f"gap {val_score - train_probe_score:+.4f}"
                )
                if val_score < best_val:
                    best_val = val_score
                    wait = 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    wait += 1
                lr_before = current_lr(optimizer)
                scheduler.step(val_score)
                lr_after = current_lr(optimizer)
                if lr_after < lr_before:
                    print(
                        f"Seed {seed} epoch {epoch + 1}: val_pbl plateaued; "
                        f"lr reduced {lr_before:.1e} -> {lr_after:.1e}",
                        flush=True,
                    )
            else:
                val_display = "skipped"
            print(
                train_epoch_loss_line(
                    seed,
                    epoch + 1,
                    batches,
                    running_state,
                    running_phys,
                    running_phys_stages,
                    running_total,
                    val_display,
                    lr=current_lr(optimizer),
                ),
                flush=True,
            )
            if debug_diagnostics and saves_checkpoint_epoch(epoch + 1):
                val_net = dominant_network_id(val_split)
                sample = debug_sample_index(val_split, network_id=val_net)
                print_forward_diagnostics(
                    model,
                    val_split,
                    val_networks[val_net],
                    sample,
                    device,
                    tag=f"seed {seed} epoch {epoch + 1} ({len(sample)} valid cases, network {val_net})",
                    rpe_ablation=wants_rpe_ablation(default=True),
                )
            start_epoch = epoch + 1
            finish_checkpoint_epoch(start_epoch)
    if stopped_early and not (model_dir / f"epoch_{start_epoch}.compared").exists():
        finish_checkpoint_epoch(start_epoch, force=True)
    if best_state is None:
        raise RuntimeError(f"Seed {seed} produced no checkpoint.")
    model.load_state_dict(best_state)
    done_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "checkpoint_version": CHECKPOINT_VERSION,
                "epochs_completed": start_epoch,
                "early_stopped": stopped_early,
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
    eval_batch_size: int = EVAL_BATCH_SIZE,
    after_split: Callable[[str, Split, list[Network], dict[int, list[float]]], None] | None = None,
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
        clear_pair_embed_cache(networks)
        per_network: dict[int, list[float]] = {}
        scores[label] = evaluate_pbl(
            model, split, networks, device, batch_size=eval_batch_size, per_network_out=per_network
        )
        counts[label] = split.count
        print(f"test {label}: PBL {scores[label]:.6g} on {split.count} scenarios", flush=True)
        if after_split is not None:
            after_split(label, split, networks, per_network)
    return scores, counts


def main() -> None:
    global DATA_FRACTION, MICRO_BATCH_SIZE, EVAL_BATCH_SIZE, VAL_EVERY
    ensure_runtime_packages()
    DATA_FRACTION = parse_data_fraction()
    MICRO_BATCH_SIZE = parse_micro_batch_size()
    EVAL_BATCH_SIZE = parse_eval_batch_size()
    VAL_EVERY = parse_val_every()
    if MICRO_BATCH_SIZE < 1 or MICRO_BATCH_SIZE > BATCH_SIZE:
        raise ValueError(f"micro-batch must be in [1, {BATCH_SIZE}], got {MICRO_BATCH_SIZE}")
    if EVAL_BATCH_SIZE < 1:
        raise ValueError(f"eval-batch must be >= 1, got {EVAL_BATCH_SIZE}")
    if VAL_EVERY < 1:
        raise ValueError(f"val-every must be >= 1, got {VAL_EVERY}")
    batch_note = (
        f"Training batch {BATCH_SIZE}, micro {MICRO_BATCH_SIZE}, eval batch {EVAL_BATCH_SIZE}, "
        f"val every {VAL_EVERY} epoch(s), AMP on CUDA."
    )
    if DATA_FRACTION < 1.0:
        print(
            f"Data fraction {DATA_FRACTION}: fixed prefix of scenarios in each folder "
            f"(0..N-1 in parquet order; e.g. 0.30 keeps all of 0.25 plus the next 5%). {batch_note}",
            flush=True,
        )
    elif DATA_FRACTION > 1.0:
        raise ValueError(f"data fraction must be <= 1.0, got {DATA_FRACTION}")
    else:
        print(batch_note, flush=True)
    drive = google_drive_root()
    outputs = drive_output_dirs(drive)
    work, legacy_results = legacy_locations(drive)
    if outputs is not None:
        checkpoints, results = outputs
    else:
        work.mkdir(parents=True, exist_ok=True)
        checkpoints = work / RUN_DIR_NAME
        checkpoints.mkdir(parents=True, exist_ok=True)
        results = legacy_results / RUN_DIR_NAME
        results.mkdir(parents=True, exist_ok=True)
    os.environ["GENCO_RESULTS_DIR"] = str(results)
    os.environ["GENCO_WORK_ROOT"] = str(checkpoints.parent)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    from common import use_selected_results

    use_selected_results()
    ensure_dirs()
    results.mkdir(parents=True, exist_ok=True)
    data = ensure_dataset(work, drive)
    print(f"Dataset: {data}", flush=True)
    print(f"Run tag {RUN_TAG} (checkpoint_version {CHECKPOINT_VERSION}, {DEPTH} stages, {RPE_DIM}-channel RPE)", flush=True)
    print(f"Checkpoints: {checkpoints}", flush=True)
    print(f"Results directory: {results}", flush=True)
    log_counts(data, results)
    active_cache, persistent_cache = split_cache_dirs(checkpoints)
    print(f"Split cache (active): {active_cache}", flush=True)
    if persistent_cache != active_cache:
        print(f"Split cache (Drive): {persistent_cache}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
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
    debug_diagnostics = wants_debug_diagnostics()
    print(
        "Debug diagnostics ON at checkpoint epochs (disable with GENCO_TPGT_DEBUG=0 or --no-debug)."
        if debug_diagnostics
        else "Debug diagnostics OFF.",
        flush=True,
    )
    reference_network = train_split.networks[dominant_network_id(train_split)]
    for seed in SEEDS:
        seed_dir = checkpoints / f"seed_{seed}"
        if (seed_dir / "complete.json").exists() and read_completed_epochs(seed_dir / "complete.json") >= EPOCHS:
            print(f"Seed {seed} already completed {EPOCHS} epochs. Skipping.", flush=True)
            continue

        def on_epoch_end(model: TPGT, epoch_number: int, seed: int = seed) -> tuple[dict[str, float], dict[str, int]]:
            def contingency_hook(
                label: str,
                split: Split,
                networks: list[Network],
                per_network: dict[int, list[float]],
            ) -> None:
                if label in CONTINGENCY_DIAGNOSTIC_LABELS:
                    print_contingency_diagnostics(
                        model,
                        f"test {label} seed {seed} epoch {epoch_number}",
                        split,
                        networks,
                        device,
                        reference_network,
                        per_network,
                    )

            scores, counts = test_model(
                model,
                data,
                device,
                active_cache,
                persistent_cache,
                eval_batch_size=EVAL_BATCH_SIZE,
                after_split=contingency_hook if debug_diagnostics else None,
            )
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
            eval_batch_size=EVAL_BATCH_SIZE,
            val_every=VAL_EVERY,
            debug_diagnostics=debug_diagnostics,
            on_epoch_end=on_epoch_end,
            after_periodic_checkpoint=publish_checkpoint,
        )


def cli_diagnose(
    checkpoint: Path | None = None,
    seed: int = 0,
    cases: int = 32,
    *,
    rpe_ablation: bool | None = None,
    contingency: bool = True,
) -> None:
    global DATA_FRACTION
    ensure_runtime_packages()
    DATA_FRACTION = parse_data_fraction()
    drive = google_drive_root()
    outputs = drive_output_dirs(drive)
    work, legacy_results = legacy_locations(drive)
    if outputs is not None:
        checkpoints, results = outputs
    else:
        checkpoints = work / RUN_DIR_NAME
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_training(device)
    data = ensure_dataset(work, drive)
    active_cache, persistent_cache = split_cache_dirs(checkpoints)
    val_split = cached_split(
        active_cache,
        persistent_cache,
        raw_dir(data, "valid", "n", "feasible"),
        "valid_n_feasible",
    )
    val_net = dominant_network_id(val_split)
    network = move_to(val_split.networks[val_net], device)
    model = TPGT().to(device)
    seed_dir = checkpoints / f"seed_{seed}"
    ckpt = checkpoint or (seed_dir / "last.pt")
    if ckpt.exists():
        payload = try_load_checkpoint_file(ckpt, device)
        if payload and isinstance(payload.get("model"), dict):
            version_reason = checkpoint_version_reason(payload)
            if version_reason is None:
                model.load_state_dict(payload["model"], strict=True)
                print(f"Loaded weights from {ckpt}", flush=True)
            else:
                print(f"Skipping weights ({version_reason}); random init.", flush=True)
        else:
            print(f"Could not load {ckpt}; random init.", flush=True)
    else:
        print(f"No checkpoint at {ckpt}; random init.", flush=True)
    sample = debug_sample_index(val_split, network_id=val_net, count=cases)
    if rpe_ablation is None:
        rpe_ablation = wants_rpe_ablation(default=True)
    print_forward_diagnostics(
        model,
        val_split,
        network,
        sample,
        device,
        tag=f"diagnose seed {seed} ({len(sample)} valid cases)",
        rpe_ablation=rpe_ablation,
    )
    if not contingency:
        return
    for label, topology, condition in TEST_PARTS:
        if label not in CONTINGENCY_DIAGNOSTIC_LABELS:
            continue
        split = cached_split(
            active_cache,
            persistent_cache,
            raw_dir(data, "test", topology, condition),
            f"test_{topology}_{condition}",
        )
        networks = [move_to(item, device) for item in split.networks]
        per_network: dict[int, list[float]] = {}
        score = evaluate_pbl(model, split, networks, device, per_network_out=per_network)
        print(f"test {label}: PBL {score:.6g} on {split.count} scenarios", flush=True)
        print_contingency_diagnostics(
            model,
            f"test {label} diagnose seed {seed}",
            split,
            networks,
            device,
            val_split.networks[val_net],
            per_network,
        )


def verify_task11_splits(data_root: Path) -> None:
    """Load every Task 1.1 split; checks reference PBL and Y-aware network grouping."""
    checks = [
        ("train_n_feasible", ("train", "n", "feasible")),
        ("valid_n_feasible", ("valid", "n", "feasible")),
        ("test_n_feasible", ("test", "n", "feasible")),
        ("test_n-1_feasible", ("test", "n-1", "feasible")),
        ("test_n-2_feasible", ("test", "n-2", "feasible")),
        ("test_n_nose", ("test", "n", "nose")),
        ("test_n-1_nose", ("test", "n-1", "nose")),
        ("test_n-2_nose", ("test", "n-2", "nose")),
    ]
    for name, parts in checks:
        raw = data_root.joinpath(*parts)
        read_split(raw, name, prefix_end=scenario_limit(raw))
    print(f"verify_task11_splits OK ({len(checks)} folders)", flush=True)


def cli_verify_splits() -> None:
    global DATA_FRACTION
    ensure_runtime_packages()
    DATA_FRACTION = parse_data_fraction()
    drive = google_drive_root()
    work, _legacy = legacy_locations(drive)
    data = ensure_dataset(work, drive)
    if DATA_FRACTION < 1.0:
        print(f"verify-splits with DATA_FRACTION={DATA_FRACTION} (scenario cap per folder)", flush=True)
    verify_task11_splits(data)


def sanity_check_meshed_rpe(data_root: Path) -> None:
    """Load one 118-bus case and verify meshed RPE via the shared correctness audit."""
    split = read_split(raw_dir(data_root, "train", "n", "feasible"), "sanity", prefix_end=1)
    network = split.networks[0]
    checks = audit_rpe_correctness(network)
    text = format_rpe_correctness_checks(checks)
    print(text, flush=True)
    failed = [check for check in checks if not check.passed]
    if failed:
        raise RuntimeError(f"meshed RPE sanity failed: {failed[0].name} — {failed[0].detail}")
    neighbor = int(np.argmax(network.rpe.numpy()[:, :, 2]))
    row, column = divmod(neighbor, 118)
    print(
        f"meshed RPE sanity OK: shape {tuple(network.rpe.shape)}, "
        f"sample neighbor ({row},{column}) A={network.rpe[row, column, 2].item():.0f}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        ckpt: Path | None = None
        seed = 0
        cases = 32
        contingency = True
        args = sys.argv[2:]
        index = 0
        while index < len(args):
            if args[index] == "--checkpoint" and index + 1 < len(args):
                ckpt = Path(args[index + 1])
                index += 2
            elif args[index] == "--seed" and index + 1 < len(args):
                seed = int(args[index + 1])
                index += 2
            elif args[index] == "--cases" and index + 1 < len(args):
                cases = int(args[index + 1])
                index += 2
            elif args[index] == "--data-fraction" and index + 1 < len(args):
                index += 2
            elif args[index].startswith("--data-fraction=") or args[index] in ("--rpe-ablation", "--no-rpe-ablation"):
                index += 1
            elif args[index] == "--no-contingency":
                contingency = False
                index += 1
            else:
                raise SystemExit(f"Unknown diagnose argument: {args[index]}")
        cli_diagnose(checkpoint=ckpt, seed=seed, cases=cases, contingency=contingency)
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify-splits":
        cli_verify_splits()
    elif len(sys.argv) > 1 and sys.argv[1] == "--sanity-rpe":
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
