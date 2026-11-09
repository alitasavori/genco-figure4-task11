"""Train the unshared type-pair graph Transformer on IEEE 118-bus PFΔ tasks (default Task 1.1).

Option II from the draft: each correction stage has its own weights.

Tasks (``--task`` or ``GENCO_TPGT_TASK``; see ``TASKS``)
----------------------------------------------------
1.1 (default), 1.2, 1.3 (alias 2.1), 2.3, 4.1, 4.2, 4.3. Data: ``Foundation Model/vendor/pfdelta_task<id>``
(override ``GENCO_PFDELTA_TASK<id without dot>``, e.g. ``GENCO_PFDELTA_TASK42``), downloaded from
``gridfm/pfdelta_task<id>`` when missing. Training and validation use the union of the task's
train/valid folders (each cached and cut to ``DATA_FRACTION`` on its own, then merged); every task
is tested on N, N-1, N-2 and C2I. Task 1.1 keeps ``tpgt_task11_<RUN_TAG>``; other tasks write to
``tpgt_task<id>_<RUN_TAG>`` (results inside it). Each optimizer step takes the next 32 cases of
one global shuffle and runs them in one forward/backward: a single topology uses the shared
block-wise path, topologies with equal bus types (line outages) share it with per-case pair
embeddings and Ybus, and topologies with different bus types (generator outages) run per-case
type masks (``TypeLayout``). Device copies of the networks (``DeviceNetworks``) stay resident when
they fit the memory budget, otherwise in an LRU. Validation and test batch the same way. The
250-batch and epoch log lines break the step time into network moves, pair encoder, forward,
backward and optimizer (``StepTimer``). Checkpoint epochs also write a bar plot against the
published numbers and IEEE 118 bus error maps.
The hidden sizes, depth, batch, and epoch cap below are implementation
choices so a Colab GPU can finish a run. They are not paper hyperparameters.

IEEE 118 is meshed, so pair features use the full in-service topology:
admittance (real, imaginary), adjacency, hop distance, and effective
resistance (5 channels). See `meshed_rpe`.

Architecture (checkpoint_version ``meshed-rpe-5-d4-v7``)
--------------------------------------------------------
* **4 correction stages** (``DEPTH``), each with its own weights; the physics
  loss sums stage residuals with GENCO weights ``0.5**(K-i)`` (last stage 1).
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
* **``pair_encoder``**: Runs once per forward on the static ``118×118`` RPEs of
  the batch's topologies (stacked when several share bus types). Weights train, so
  training must re-encode every step; in ``eval`` mode a single-topology batch
  caches its encoded pairs on ``Network.pair_embed``.
* **Type-pair attention**: ``CorrectionStage._messages`` evaluates each of the
  nine type-pair weight sets only on its own receiver-type × sender-type block
  (both heads batched), then takes one softmax over all senders per receiver. With per-case
  bus types a block spans the buses of that type in any case and per-case masks keep each
  case's own pairs (the dense masked formulation restricted to those buses).
  Optional ``GENCO_TPGT_COMPILE=1`` or ``--compile`` wraps the model in
  ``torch.compile`` (may differ slightly in numerics).
* **Batch / micro-batch**: Optimizer steps use ``BATCH_SIZE`` cases (default 32).
  Default **micro-batch** matches that (one forward/backward per step when memory
  allows). Override with ``GENCO_TPGT_BATCH`` / ``--batch`` and
  ``GENCO_TPGT_MICRO_BATCH`` / ``--micro-batch`` (e.g. both 64 on an L4).
* **Eval batch 64**: Validation and test use forward-only batches up to 64
  (``GENCO_TPGT_EVAL_BATCH`` / ``--eval-batch``); no backward, usually safe
  on the same GPU.
* **CUDA**: ``cudnn.benchmark``, TF32, and high matmul precision when training
  on GPU. Optional ``torch.compile`` via ``--fast``, ``GENCO_TPGT_FAST=1``, or
  ``--compile``.
* **Less sync**: GPU ``synchronize`` only for timed log lines and validation,
  not after every training batch. Bus-type indices, attention blocks and type presence are
  precomputed per network (``move_to``) or per pass on the CPU, host-to-GPU copies are pinned,
  and step losses accumulate on the device, so a training forward/backward has no device sync.
* **Device network cache**: ``DeviceNetworks`` keeps every topology of a split on the GPU when
  they fit min(``DEVICE_CACHE_FREE_FRACTION`` of free memory, ``DEVICE_CACHE_MAX_BYTES``), about
  0.5 MB each on case118; otherwise an LRU of that size. The decision is logged at seed start.
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

import contextlib
import hashlib
import json
import gc
import math
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

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
from plot_task1_1 import PUBLISHED, plot_tpgt_comparison
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
# Stage i of K gets physics weight PHYSICS_STAGE_DECAY**(K-i), as in GENCO
# (HGNS_PF_pfdelta_bs64_seed1.yaml): last stage 1, each earlier stage halved.
PHYSICS_STAGE_DECAY = 0.5
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
TOPOLOGIES = ("n", "n-1", "n-2")
CONDITIONS = ("feasible", "nose", "around_nose")
PARQUET_FILES = ("bus_data.parquet", "branch_data.parquet", "gen_data.parquet", "y_bus_data.parquet")
# Train/valid splits with at least this many topologies keep non-dominant RPEs in float16 on CPU.
COMPACT_MIN_NETWORKS = 512
# Device copies of a split's networks stay resident when they fit min(fraction of free GPU memory,
# cap); otherwise an LRU keeps as many as the budget holds. CPU runs use a fixed budget.
DEVICE_CACHE_FREE_FRACTION = 0.35
DEVICE_CACHE_MAX_BYTES = 6 * 1024**3
DEVICE_CACHE_CPU_BYTES = 2 * 1024**3
# Steps mixing bus-type patterns: False runs every case in one per-case-type pass; True runs the
# largest pattern block-wise and the remaining cases in one per-case-type pass.
SPLIT_DOMINANT_PATTERN = False
# Bus error maps at checkpoint epochs: validation cases on the dominant topology.
BUS_MAP_VAL_CASES = 256
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


@dataclass(frozen=True)
class TaskSpec:
    """A PFΔ IEEE 118 task: ``data_id`` names the HF dataset, ``task_id`` the run folder and tables."""

    task_id: str
    data_id: str
    label: str
    description: str
    folders: tuple[tuple[str, str], ...]

    @property
    def is_task11(self) -> bool:
        return self.task_id == "1.1"

    @property
    def run_dir_name(self) -> str:
        return RUN_DIR_NAME if self.is_task11 else f"tpgt_task{self.task_id}_{RUN_TAG}"

    @property
    def file_tag(self) -> str:
        return self.task_id.replace(".", "_")

    @property
    def dataset_repo(self) -> str:
        return f"gridfm/pfdelta_task{self.data_id}"

    @property
    def dataset_env(self) -> str:
        return "GENCO_PFDELTA_TASK" + self.data_id.replace(".", "")


def _folders(topologies: tuple[str, ...], conditions: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((topology, condition) for topology in topologies for condition in conditions)


# Train/valid folders per task as published on Hugging Face (gridfm/pfdelta_task<id>, case118).
TASKS = {
    "1.1": TaskSpec("1.1", "1.1", "1.1", "train N", _folders(("n",), ("feasible",))),
    "1.2": TaskSpec("1.2", "1.2", "1.2", "train N + N-1", _folders(("n", "n-1"), ("feasible",))),
    "1.3": TaskSpec("1.3", "1.3", "1.3/2.1", "train N + N-1 + N-2", _folders(TOPOLOGIES, ("feasible",))),
    "2.3": TaskSpec(
        "2.3", "2.3", "2.3", "train N + N-1 + N-2 with 3x less data", _folders(TOPOLOGIES, ("feasible",))
    ),
    "4.1": TaskSpec("4.1", "4.1", "4.1", "90% feasible + 10% C2I", _folders(TOPOLOGIES, ("feasible", "nose"))),
    "4.2": TaskSpec(
        "4.2", "4.2", "4.2", "50% feasible + 10% C2I + 40% approaching infeasibility", _folders(TOPOLOGIES, CONDITIONS)
    ),
    "4.3": TaskSpec(
        "4.3", "4.3", "4.3", "20% C2I + 80% approaching infeasibility", _folders(TOPOLOGIES, ("nose", "around_nose"))
    ),
}
TASK_ALIASES = {"2.1": "1.3", "1.3/2.1": "1.3"}


def task_spec(name: str) -> TaskSpec:
    key = TASK_ALIASES.get(name.strip(), name.strip())
    if key not in TASKS:
        allowed = ", ".join([*TASKS, *TASK_ALIASES])
        raise ValueError(f"unknown PFΔ task {name!r}; choose one of {allowed}")
    return TASKS[key]


def parse_task() -> TaskSpec:
    for index, arg in enumerate(sys.argv):
        if arg == "--task" and index + 1 < len(sys.argv):
            return task_spec(sys.argv[index + 1])
        if arg.startswith("--task="):
            return task_spec(arg.split("=", 1)[1])
    return task_spec(os.environ.get("GENCO_TPGT_TASK", "").strip() or "1.1")


def ensure_runtime_packages() -> None:
    missing = []
    for name in ("pyarrow", "huggingface_hub", "matplotlib", "scipy", "networkx"):
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
    """``mask`` is a boolean bus mask or a bus index tensor."""
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
    """Receivers of one type and senders of one type; ``mask`` keeps same-component pairs only.

    ``mask`` is (receivers, senders) when bus types are shared by the batch, or (batch, receivers,
    senders) when they differ per case; then the buses are those of the type in any case and the
    mask also drops pairs whose types in that case are not this block's.
    """

    pair_id: int
    receivers: torch.Tensor
    senders: torch.Tensor
    mask: torch.Tensor


@dataclass
class TypeLayout:
    """Bus types of one forward pass, shared by the batch or per case.

    Shared: ``types`` is (N,) and ``index[t]`` lists the buses of type t. Per case: ``types`` is
    (B, N), ``index[t]`` lists the buses of type t in at least one case, ``selects[t]`` (B, len(index[t]))
    marks where they have it, and ``masks`` holds (PQ, PV, slack, PQ or PV) per case. ``present`` and the
    indices are known on the CPU, so a forward needs no device sync to find the types or blocks.
    """

    types: torch.Tensor
    index: tuple[torch.Tensor, ...]
    present: tuple[bool, ...]
    blocks: list[TypePairBlock]
    angle_index: torch.Tensor | None = None
    selects: tuple[torch.Tensor, ...] | None = None
    masks: tuple[torch.Tensor, ...] | None = None
    types_np: np.ndarray | None = field(default=None, repr=False)

    @property
    def per_case(self) -> bool:
        return self.selects is not None

    def take(self, select: torch.Tensor) -> TypeLayout:
        """Per-case layout of rows ``select`` of this per-case layout."""
        return TypeLayout(
            types=self.types[select],
            index=self.index,
            present=self.present,
            blocks=[TypePairBlock(b.pair_id, b.receivers, b.senders, b.mask[select]) for b in self.blocks],
            selects=tuple(values[select] for values in self.selects),
            masks=tuple(values[select] for values in self.masks),
        )


def host_to_device(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy without a stream sync: a pageable host-to-GPU copy can synchronize, a pinned one does not."""
    if device.type != "cuda" or values.device.type != "cpu":
        return values.to(device)
    return values.pin_memory().to(device, non_blocking=True)


def _bus_index(buses: np.ndarray, device: torch.device) -> torch.Tensor:
    return host_to_device(torch.as_tensor(np.asarray(buses, dtype=np.int64)), device)


def _layout_blocks(
    index: tuple[torch.Tensor, ...],
    present: tuple[bool, ...],
    block_mask: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
) -> list[TypePairBlock]:
    blocks = []
    for pair_id in range(9):
        receiver, sender = divmod(pair_id, 3)
        if not (present[receiver] and present[sender]):
            continue
        receivers, senders = index[receiver], index[sender]
        blocks.append(TypePairBlock(pair_id, receivers, senders, block_mask(pair_id, receivers, senders)))
    return blocks


def shared_layout(
    types: torch.Tensor,
    pair_type_masks: torch.Tensor | None = None,
    types_np: np.ndarray | None = None,
) -> TypeLayout:
    """Layout of (N,) ``types``; blocks only when ``pair_type_masks`` (9, N, N) is given."""
    if types_np is None:
        types_np = types.detach().cpu().numpy()
    index = tuple(_bus_index(np.flatnonzero(types_np == type_id), types.device) for type_id in range(3))
    present = tuple(bool(np.any(types_np == type_id)) for type_id in range(3))
    blocks = []
    if pair_type_masks is not None:
        blocks = _layout_blocks(
            index, present, lambda p, receivers, senders: pair_type_masks[p][receivers.view(-1, 1), senders.view(1, -1)]
        )
    angle_index = _bus_index(np.flatnonzero((types_np == 0) | (types_np == 1)), types.device)
    return TypeLayout(types, index, present, blocks, angle_index=angle_index, types_np=types_np)


def per_case_layout(
    types: torch.Tensor,
    pair_type_masks: torch.Tensor | None = None,
    types_np: np.ndarray | None = None,
) -> TypeLayout:
    """Layout of (B, N) ``types``; blocks only when per-case ``pair_type_masks`` (B, 9, N, N) is given."""
    if types_np is None:
        types_np = types.detach().cpu().numpy()
    union = [np.any(types_np == type_id, axis=0) for type_id in range(3)]
    index = tuple(_bus_index(np.flatnonzero(buses), types.device) for buses in union)
    present = tuple(bool(buses.any()) for buses in union)
    blocks = []
    if pair_type_masks is not None:
        blocks = _layout_blocks(
            index,
            present,
            lambda p, receivers, senders: pair_type_masks[:, p][:, receivers.view(-1, 1), senders.view(1, -1)],
        )
    pq, pv, slack = (types == type_id for type_id in range(3))
    return TypeLayout(
        types,
        index,
        present,
        blocks,
        selects=tuple(types[:, buses] == type_id for type_id, buses in enumerate(index)),
        masks=(pq, pv, slack, pq | pv),
    )


def type_layout(types: torch.Tensor | TypeLayout, pair_type_masks: torch.Tensor | None = None) -> TypeLayout:
    """``types`` as a layout: (N,) shared or (B, N) per case (builds indices with a device sync)."""
    if isinstance(types, TypeLayout):
        return types
    if types.dim() == 1:
        return shared_layout(types, pair_type_masks)
    return per_case_layout(types, pair_type_masks)


def type_pair_blocks(types: torch.Tensor, pair_type_masks: torch.Tensor) -> list[TypePairBlock]:
    """Non-empty (receiver type, sender type) blocks of the bus-pair matrix, one per type pair."""
    return type_layout(types, pair_type_masks).blocks


def gather_type(values: torch.Tensor, layout: TypeLayout, type_id: int) -> torch.Tensor:
    return values[:, layout.index[type_id]]


def write_type(base: torch.Tensor, layout: TypeLayout, type_id: int, values: torch.Tensor) -> torch.Tensor:
    """Write ``values`` (one row per ``layout.index[type_id]`` bus) where buses have that type; returns ``base``.

    Shared types write in place; per case the rows of other-type buses keep ``base``.
    """
    index = layout.index[type_id]
    if not layout.per_case:
        masked_write(base, index, values)
        return base
    keep = layout.selects[type_id].unsqueeze(-1)
    return base.index_copy(1, index, torch.where(keep, values.to(dtype=base.dtype), base.index_select(1, index)))


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
        layout: TypeLayout,
        ybus: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_in = nodal_residual(specified, unknown, layout, ybus)
        message = self._messages(hidden, pair, layout.blocks)
        updated = torch.zeros_like(hidden)
        for type_id, update in enumerate(self.update):
            if not layout.present[type_id]:
                continue
            features = torch.cat(
                (
                    gather_type(hidden, layout, type_id),
                    gather_type(message, layout, type_id),
                    gather_type(residual_in, layout, type_id).to(hidden.dtype),
                ),
                dim=-1,
            )
            updated = write_type(updated, layout, type_id, update(features))
        hidden_next = self.norm(hidden + updated)
        delta = torch.zeros_like(unknown)
        for type_id, decoder in enumerate(self.decode):
            if not layout.present[type_id]:
                continue
            delta = write_type(delta, layout, type_id, decoder(gather_type(hidden_next, layout, type_id)))
        if layout.per_case:
            angle_step = torch.where(layout.masks[3], delta[..., 1] * ETA_ANGLE, delta[..., 1])
            delta = torch.stack((delta[..., 0], angle_step), dim=-1)
        else:
            for type_id in (0, 1):
                if layout.present[type_id]:
                    buses = layout.index[type_id]
                    delta[:, buses, 1] = delta[:, buses, 1] * ETA_ANGLE
        unknown_next = constrain_unknown(unknown + ETA * delta, layout)
        residual_out = nodal_residual(specified, unknown_next, layout, ybus)
        return hidden_next, unknown_next, residual_out

    def _messages(
        self,
        hidden: torch.Tensor,
        pair: torch.Tensor,
        blocks: list[TypePairBlock],
    ) -> torch.Tensor:
        """Each type pair's weights see only its own receiver-type x sender-type block.

        Block scores are scattered into one (batch, heads, N, N) score matrix so the softmax
        runs over all senders of a receiver; cross-component entries stay -inf. ``pair`` is either
        shared by the batch (N, N, E) or per case (batch, N, N, E). Blocks with per-case masks
        (bus types differing per case) may overlap; each entry of a case is set by the one block of
        its types and weighted only there.
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
            if pair.dim() == 3:
                edge = pair[receivers, senders]
                projected = torch.einsum("nme,hae->hnma", edge, self.w_ew[p]).unsqueeze(0)
                bias = torch.einsum("nme,hae->hnma", edge, self.w_eb[p]).unsqueeze(0)
            else:
                edge = pair[:, receivers, senders]
                projected = torch.einsum("bnme,hae->bhnma", edge, self.w_ew[p])
                bias = torch.einsum("bnme,hae->bhnma", edge, self.w_eb[p])
            interaction = torch.relu(mixed * projected + bias)
            score = torch.einsum("bhnma,ha->bhnm", interaction, self.w_a[p, :, 0])
            if scores is None:
                scores = score.new_full((batch, HEADS, buses, buses), float("-inf"))
            if block.mask.dim() == 2:
                score = score.masked_fill(~block.mask, float("-inf"))
                scores[:, :, receivers, senders] = score.to(scores.dtype)
            else:
                own = block.mask.unsqueeze(1)
                scores[:, :, receivers, senders] = torch.where(
                    own, score.to(scores.dtype), scores[:, :, receivers, senders]
                )
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
            if block.mask.dim() == 3:
                block_weights = block_weights.masked_fill(~block.mask.unsqueeze(1), 0.0)
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
        layout: TypeLayout | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``types`` (N,) with ``pair_type_masks`` (9, N, N), or per case (B, N) with (B, 9, N, N).

        ``layout`` (precomputed by ``move_to`` / ``PassBatch``) replaces ``types`` and the masks.
        """
        if pair is None:
            pair = self.pair_encoder(rpe)
        if layout is None:
            if pair_type_masks is None:
                raise ValueError("TPGT forward needs pair_type_masks or a layout")
            layout = type_layout(types, pair_type_masks)
        act_dtype = autocast_activation_dtype(specified)
        hidden = torch.zeros(
            specified.shape[0],
            specified.shape[1],
            D,
            device=specified.device,
            dtype=act_dtype,
        )
        unknown = flat_start(specified, layout)
        for type_id, encoder in enumerate(self.encoders):
            if layout.present[type_id]:
                hidden = write_type(hidden, layout, type_id, encoder(gather_type(specified, layout, type_id)))
        residuals = []
        for stage in self.stages:
            hidden, unknown, residual = stage(hidden, unknown, specified, pair, layout, ybus)
            residuals.append(residual)
        return unknown, residuals


def constrain_unknown(unknown: torch.Tensor, types: torch.Tensor | TypeLayout) -> torch.Tensor:
    """Keep states in a range where the AC equations are meaningful.

    ``types`` here and in the helpers below is (N,), per case (B, N), or a ``TypeLayout``.
    """
    layout = type_layout(types)
    if layout.per_case:
        pq, _pv, _slack, angle_buses = layout.masks
        voltage = torch.where(pq, unknown[..., 0].clamp(min=V_MIN_PU, max=V_MAX_PU), unknown[..., 0])
        angle = torch.where(angle_buses, wrap_angle(unknown[..., 1]), unknown[..., 1])
        return torch.stack((voltage, angle), dim=-1)
    out = unknown.clone()
    if layout.present[0]:
        pq = layout.index[0]
        out[:, pq, 0] = unknown[:, pq, 0].clamp(min=V_MIN_PU, max=V_MAX_PU)
    if layout.present[0] or layout.present[1]:
        angle_buses = layout.angle_index
        out[:, angle_buses, 1] = wrap_angle(unknown[:, angle_buses, 1])
    return out


def flat_start(specified: torch.Tensor, types: torch.Tensor | TypeLayout) -> torch.Tensor:
    layout = type_layout(types)
    unknown = torch.zeros_like(specified)
    if layout.per_case:
        unknown[..., 0] = layout.masks[0].to(unknown.dtype)
    else:
        unknown[:, layout.index[0], 0] = 1.0
    return unknown


def pack_case(
    power_p: torch.Tensor,
    power_q: torch.Tensor,
    voltage: torch.Tensor,
    angle: torch.Tensor,
    types: torch.Tensor | TypeLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Specified quantities stay fixed. The target is the unknown part of the solution."""
    layout = type_layout(types)
    if layout.per_case:
        pq, pv, slack, _angle = layout.masks
        specified = torch.stack(
            (torch.where(slack, voltage, power_p), torch.where(pq, power_q, torch.where(pv, voltage, angle))), dim=-1
        )
        target = torch.stack(
            (torch.where(pq, voltage, torch.where(pv, power_q, power_p)), torch.where(slack, power_q, angle)), dim=-1
        )
        return specified, target
    specified = torch.zeros(*power_p.shape, 2, device=power_p.device, dtype=power_p.dtype)
    target = torch.zeros_like(specified)
    pq, pv, slack = layout.index
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
    types: torch.Tensor | TypeLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layout = type_layout(types)
    if layout.per_case:
        pq, pv, slack, _angle = layout.masks
        given_0, given_1 = specified[..., 0], specified[..., 1]
        solved_0, solved_1 = unknown[..., 0].to(specified.dtype), unknown[..., 1].to(specified.dtype)
        return (
            torch.where(slack, solved_0, given_0),
            torch.where(pq, given_1, torch.where(pv, solved_0, solved_1)),
            torch.where(pq, solved_0, torch.where(pv, given_1, given_0)),
            torch.where(slack, given_1, solved_1),
        )
    power_p = torch.empty(specified.shape[:2], device=specified.device, dtype=specified.dtype)
    power_q = torch.empty_like(power_p)
    voltage = torch.empty_like(power_p)
    angle = torch.empty_like(power_p)
    pq, pv, slack = layout.index
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
    types: torch.Tensor | TypeLayout,
    ybus: torch.Tensor,
) -> torch.Tensor:
    """AC mismatch in per unit. Ybus is already per unit, so this does not divide by baseMVA again.

    ``ybus`` is shared (N, N) or per case (batch, N, N). Returned in float32 (float64 when ``specified``
    is float64, so float64 models can be checked to round-off).
    Evaluated in complex128: with TF32 enabled, a complex64 Ybus matmul on CUDA has a ~0.02 p.u. mean
    error floor on case118 (|Y| up to ~390 with heavy cancellation), 10^4 x the label accuracy.
    """
    layout = type_layout(types)
    physical = constrain_unknown(unknown, layout)
    power_p, power_q, voltage, angle = assemble(specified, physical, layout)
    voltage_complex = voltage.double() * torch.exp(1j * angle.double())
    if ybus.dim() == 2:
        current = torch.matmul(ybus.to(torch.complex128), voltage_complex.transpose(0, 1)).transpose(0, 1)
    else:
        current = torch.matmul(ybus.to(torch.complex128), voltage_complex.unsqueeze(-1)).squeeze(-1)
    flow = voltage_complex * torch.conj(current)
    residual = torch.stack((power_p.double() - flow.real, power_q.double() - flow.imag), dim=-1)
    return residual.to(torch.promote_types(specified.dtype, torch.float32))


@dataclass
class TaskLossBreakdown:
    state_mse: torch.Tensor
    physics_sum: torch.Tensor
    physics_stages: tuple[torch.Tensor, ...]
    total: torch.Tensor


def wrap_angle_errors(error: torch.Tensor, layout: TypeLayout) -> torch.Tensor:
    """Wrap the angle slot of PQ and PV unknowns (in place for shared types)."""
    if layout.per_case:
        angle = torch.where(layout.masks[3], wrap_angle(error[..., 1]), error[..., 1])
        return torch.stack((error[..., 0], angle), dim=-1)
    error[:, layout.angle_index, 1] = wrap_angle(error[:, layout.angle_index, 1])
    return error


def physics_stage_weights(depth: int) -> tuple[float, ...]:
    return tuple(PHYSICS_STAGE_DECAY ** (depth - i) for i in range(1, depth + 1))


def task_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    residuals: list[torch.Tensor],
    types: torch.Tensor | TypeLayout,
) -> TaskLossBreakdown:
    error = prediction - target
    error = wrap_angle_errors(error, type_layout(types))
    state_mse = torch.mean(torch.sum(error * error, dim=-1))
    physics_stages = tuple(
        torch.mean(torch.sum(residual * residual, dim=-1)) for residual in residuals
    )
    weights = physics_stage_weights(len(physics_stages))
    physics_sum = sum(w * stage for w, stage in zip(weights, physics_stages))
    total = LAMBDA_STATE * state_mse + LAMBDA_PHYS * physics_sum
    return TaskLossBreakdown(state_mse, physics_sum, physics_stages, total)


def mean_pbl(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2))


def case_pbl(residual: torch.Tensor) -> torch.Tensor:
    """Mean PBL over buses for each case; residual shape (batch, buses, 2)."""
    return torch.mean(torch.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2), dim=-1)


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
    types: torch.Tensor | TypeLayout,
) -> dict[str, float]:
    """RMSE on unknown slots (same packing as pack_case)."""
    layout = type_layout(types)
    error = wrap_angle_errors((prediction - target).clone(), layout)
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
        if not layout.present[type_id]:
            continue
        if layout.per_case:
            diff = error[..., slot][layout.masks[type_id]]
        else:
            diff = error[:, layout.index[type_id], slot]
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
class BottleneckDiagnostics:
    """Where residual / state error lives - use this to decide architecture changes."""

    pbl_share_by_type: dict[str, float]
    mean_pbl_by_type: dict[str, float]
    p_vs_q_share: dict[str, float]
    genco_style_pbl: dict[str, float]
    genco_style_cut_frac: float
    stage_reduction: list[dict[str, float]]
    high_y_vs_rest: dict[str, float]
    hops_to_slack: list[dict[str, float]]
    worst_buses: list[dict[str, float | int | str]]
    angle_rmse_pq_pv: float
    voltage_rmse_pq: float


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
    bottleneck: BottleneckDiagnostics


def _hops_to_sources(adjacency: np.ndarray, sources: list[int]) -> np.ndarray:
    """Unweighted BFS hop distances; unreachable buses stay +inf."""
    buses = adjacency.shape[0]
    dist = np.full(buses, np.inf, dtype=np.float64)
    if not sources:
        return dist
    queue: deque[int] = deque()
    for source in sources:
        if 0 <= source < buses:
            dist[source] = 0.0
            queue.append(source)
    while queue:
        node = queue.popleft()
        neighbors = np.flatnonzero(adjacency[node])
        for nxt in neighbors:
            if dist[nxt] > dist[node] + 1.0:
                dist[nxt] = dist[node] + 1.0
                queue.append(int(nxt))
    return dist


def genco_style_residual(residual: torch.Tensor, types: torch.Tensor) -> torch.Tensor:
    """Zero PV-Q and slack P/Q mismatches the way GENCO-style reporting often does."""
    out = residual.clone()
    layout = type_layout(types)
    if layout.per_case:
        pq, pv, slack, _angle = layout.masks
        out[..., 1] = torch.where(pv, torch.zeros_like(out[..., 1]), out[..., 1])
        out[..., 0] = torch.where(slack, torch.zeros_like(out[..., 0]), out[..., 0])
        out[..., 1] = torch.where(slack, torch.zeros_like(out[..., 1]), out[..., 1])
        return out
    _pq, pv, slack = layout.index
    if layout.present[1]:
        out[:, pv, 1] = 0.0
    if layout.present[2]:
        out[:, slack, :] = 0.0
    return out


def collect_bottleneck_diagnostics(
    residual: torch.Tensor,
    stage_residuals: list[torch.Tensor],
    prediction: torch.Tensor,
    target: torch.Tensor,
    types: torch.Tensor,
    ybus: torch.Tensor,
    *,
    top_k: int = 10,
) -> BottleneckDiagnostics:
    """Break down final residual so architecture work targets the real mass, not one RMSE slot."""
    layout = type_layout(types)
    types_cpu = types.detach().cpu()
    if types_cpu.dim() > 1:
        types_np = types_cpu[0].numpy().astype(np.int64)
    else:
        types_np = types_cpu.numpy().astype(np.int64)
    y = ybus.detach()
    if y.dim() == 3:
        y = y[0]
    y_abs_diag = torch.abs(torch.diagonal(y)).detach().cpu().numpy().astype(np.float64)
    adj = (torch.abs(y).detach().cpu().numpy() > 1e-6) & ~np.eye(y.shape[0], dtype=bool)

    per_bus = torch.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2)
    mean_bus = per_bus.mean(dim=0).detach().cpu().numpy()
    abs_dp = residual[..., 0].abs().mean(dim=0).detach().cpu().numpy()
    abs_dq = residual[..., 1].abs().mean(dim=0).detach().cpu().numpy()
    total_mass = float(mean_bus.sum()) + 1e-12

    share_by_type: dict[str, float] = {}
    mean_by_type: dict[str, float] = {}
    for type_id, name in enumerate(TYPE_NAMES):
        mask = types_np == type_id
        if not mask.any():
            continue
        share_by_type[name] = float(mean_bus[mask].sum() / total_mass)
        mean_by_type[name] = float(mean_bus[mask].mean())

    dp_mass = float(np.abs(abs_dp).sum())
    dq_mass = float(np.abs(abs_dq).sum())
    pq_mass = dp_mass + dq_mass + 1e-12
    p_vs_q = {"P_share": dp_mass / pq_mass, "Q_share": dq_mass / pq_mass}

    genco_res = genco_style_residual(residual, types)
    genco_pbl = pbl_stats(genco_res)
    raw_mean = float(per_bus.mean())
    genco_cut = 0.0 if raw_mean <= 0 else max(0.0, 1.0 - genco_pbl["mean"] / raw_mean)

    stage_rows: list[dict[str, float]] = []
    prev = None
    first = None
    for stage_idx, stage_res in enumerate(stage_residuals, start=1):
        stats = pbl_stats(stage_res)
        mean = stats["mean"]
        if first is None:
            first = mean
        reduction_from_prev = 0.0 if prev is None or prev <= 0 else (prev - mean) / prev
        reduction_from_first = 0.0 if first is None or first <= 0 else (first - mean) / first
        stage_rows.append(
            {
                "stage": float(stage_idx),
                "mean": mean,
                "reduction_from_prev": reduction_from_prev,
                "reduction_from_first": reduction_from_first,
            }
        )
        prev = mean

    if mean_bus.size:
        threshold = float(np.quantile(y_abs_diag, 0.75))
        high = y_abs_diag >= threshold
        rest = ~high
        high_y = {
            "threshold_|Yii|": threshold,
            "high_y_buses": float(int(high.sum())),
            "high_y_mean_pbl": float(mean_bus[high].mean()) if high.any() else 0.0,
            "rest_mean_pbl": float(mean_bus[rest].mean()) if rest.any() else 0.0,
            "high_y_share": float(mean_bus[high].sum() / total_mass) if high.any() else 0.0,
        }
    else:
        high_y = {
            "threshold_|Yii|": 0.0,
            "high_y_buses": 0.0,
            "high_y_mean_pbl": 0.0,
            "rest_mean_pbl": 0.0,
            "high_y_share": 0.0,
        }

    slack_idx = [int(i) for i in np.flatnonzero(types_np == 2)]
    hops = _hops_to_sources(adj, slack_idx)
    hop_rows: list[dict[str, float]] = []
    finite = np.isfinite(hops)
    if finite.any():
        max_hop = int(np.nanmax(hops[finite]))
        for hop in range(0, min(max_hop, 8) + 1):
            mask = hops == hop
            if not mask.any():
                continue
            hop_rows.append(
                {
                    "hops": float(hop),
                    "buses": float(int(mask.sum())),
                    "mean_pbl": float(mean_bus[mask].mean()),
                    "share": float(mean_bus[mask].sum() / total_mass),
                }
            )
        far = hops > 8
        if far.any():
            hop_rows.append(
                {
                    "hops": 9.0,
                    "buses": float(int(far.sum())),
                    "mean_pbl": float(mean_bus[far].mean()),
                    "share": float(mean_bus[far].sum() / total_mass),
                }
            )

    order = np.argsort(-mean_bus)[: min(top_k, mean_bus.size)]
    worst: list[dict[str, float | int | str]] = []
    for bus in order:
        type_id = int(types_np[bus])
        worst.append(
            {
                "bus_1based": int(bus) + 1,
                "type": TYPE_NAMES[type_id] if 0 <= type_id < len(TYPE_NAMES) else str(type_id),
                "mean_pbl": float(mean_bus[bus]),
                "mean_|dP|": float(abs_dp[bus]),
                "mean_|dQ|": float(abs_dq[bus]),
                "|Yii|": float(y_abs_diag[bus]),
                "hops_to_slack": float(hops[bus]) if np.isfinite(hops[bus]) else -1.0,
            }
        )

    error = wrap_angle_errors((prediction - target).clone(), layout)
    if layout.per_case:
        pq_mask, pv_mask, _slack_mask, angle_mask = layout.masks
        angle_err = error[..., 1][angle_mask]
        volt_err = error[..., 0][pq_mask]
    else:
        pq, pv, _slack = layout.index
        angle_idx = layout.angle_index
        angle_err = error[:, angle_idx, 1].reshape(-1)
        volt_err = error[:, pq, 0].reshape(-1) if layout.present[0] else error.new_zeros(0)
    angle_rmse = float(torch.sqrt(torch.mean(angle_err * angle_err))) if angle_err.numel() else 0.0
    volt_rmse = float(torch.sqrt(torch.mean(volt_err * volt_err))) if volt_err.numel() else 0.0

    return BottleneckDiagnostics(
        pbl_share_by_type=share_by_type,
        mean_pbl_by_type=mean_by_type,
        p_vs_q_share=p_vs_q,
        genco_style_pbl=genco_pbl,
        genco_style_cut_frac=genco_cut,
        stage_reduction=stage_rows,
        high_y_vs_rest=high_y,
        hops_to_slack=hop_rows,
        worst_buses=worst,
        angle_rmse_pq_pv=angle_rmse,
        voltage_rmse_pq=volt_rmse,
    )


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
        final_residual = residuals[-1] if residuals else nodal_residual(specified, prediction, types, ybus)
        bottleneck = collect_bottleneck_diagnostics(
            final_residual, list(residuals), prediction, target, types, ybus
        )
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
        bottleneck=bottleneck,
    )


def format_bottleneck_diagnostics(bottleneck: BottleneckDiagnostics) -> str:
    lines = ["Bottleneck (architecture targeting - read this before changing depth/width):"]
    share = bottleneck.pbl_share_by_type
    mean = bottleneck.mean_pbl_by_type
    share_txt = "  ".join(
        f"{name} share {share[name]:.1%} mean {mean[name]:.4g}" for name in TYPE_NAMES if name in share
    )
    lines.append(f"  PBL mass by bus type: {share_txt}")
    slack_share = share.get("Slack", 0.0)
    if slack_share < 0.15:
        lines.append(
            f"  => Slack is NOT the main error mass ({slack_share:.1%} of sum PBL). "
            "Large slack_P RMSE is one bus's unknown P; mean PBL is dominated by PQ/PV angles."
        )
    else:
        lines.append(
            f"  => Slack holds {slack_share:.1%} of sum PBL - unusual; check slack packing / labels."
        )
    pq = bottleneck.p_vs_q_share
    lines.append(
        f"  Residual axis mass: |dP| {pq['P_share']:.1%}  |dQ| {pq['Q_share']:.1%}  "
        f"(GENCO-style zero PV-Q+slack -> PBL mean {bottleneck.genco_style_pbl['mean']:.4g}, "
        f"cuts {bottleneck.genco_style_cut_frac:.1%})"
    )
    lines.append(
        f"  State RMSE drivers: angle(PQ+PV) {bottleneck.angle_rmse_pq_pv:.4g} rad  "
        f"voltage(PQ) {bottleneck.voltage_rmse_pq:.4g} pu"
    )
    hy = bottleneck.high_y_vs_rest
    lines.append(
        f"  High-|Yii| top25% ({int(hy['high_y_buses'])} buses, thr {hy['threshold_|Yii|']:.4g}): "
        f"mean PBL {hy['high_y_mean_pbl']:.4g} vs rest {hy['rest_mean_pbl']:.4g} "
        f"(share {hy['high_y_share']:.1%})"
    )
    if bottleneck.stage_reduction:
        lines.append("  Stage residual reduction (stall => need more capacity/depth or better init):")
        for row in bottleneck.stage_reduction:
            lines.append(
                f"    stage {int(row['stage'])}: mean {row['mean']:.4g}  "
                f"d_prev {row['reduction_from_prev']:+.1%}  "
                f"vs stage1 {row['reduction_from_first']:+.1%}"
            )
    if bottleneck.hops_to_slack:
        lines.append("  Mean PBL by hops to slack:")
        for row in bottleneck.hops_to_slack:
            hop_label = ">=9" if row["hops"] >= 9 else str(int(row["hops"]))
            lines.append(
                f"    {hop_label:>3s} hops  {int(row['buses']):3d} buses  "
                f"mean {row['mean_pbl']:.4g}  share {row['share']:.1%}"
            )
    if bottleneck.worst_buses:
        lines.append("  Worst buses (1-based):")
        for row in bottleneck.worst_buses:
            lines.append(
                f"    bus {row['bus_1based']:3d} ({row['type']:5s}) PBL {row['mean_pbl']:.4g}  "
                f"|dP| {row['mean_|dP|']:.4g} |dQ| {row['mean_|dQ|']:.4g}  "
                f"|Yii| {row['|Yii|']:.4g}  hops {row['hops_to_slack']}"
            )
    lines.append(
        "  If stage d_prev ~0 late and high-|Y| share is large: try more stages / wider D before "
        "blaming slack. If P_share >> Q_share: prioritize angle accuracy."
    )
    return "\n".join(lines)


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
    lines.append(format_bottleneck_diagnostics(diagnostics.bottleneck))
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
    networks: list[Network] | DeviceNetworks,
    device: torch.device,
    reference: Network,
    per_network_pbl: dict[int, list[float]],
    *,
    networks_per_class: int = 3,
    cases_per_network: int = 8,
) -> None:
    """PBL by contingency class, plus RPE/residual audits and RPE ablation on the largest groups of each class."""
    lines = [f"=== Contingency diagnostics {label} ({split.count} cases, {len(networks)} topologies) ==="]
    classes = [contingency_class(network, reference) for network in split.networks]
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
    type_key: bytes | None = field(default=None, repr=False, compare=False)
    layout: TypeLayout | None = field(default=None, repr=False, compare=False)


def network_masks(network: Network) -> torch.Tensor:
    masks = network.pair_type_masks
    return type_pair_masks(network.types, network.component) if masks is None else masks


def network_layout(network: Network) -> TypeLayout:
    """Shared-type layout with attention blocks; precomputed by ``move_to``, else built on the fly."""
    layout = getattr(network, "layout", None)
    return shared_layout(network.types, network_masks(network)) if layout is None else layout


def network_type_key(network: Network) -> bytes:
    """Bus types and component labels as bytes; networks with equal keys share attention blocks and masks."""
    key = getattr(network, "type_key", None)
    if key is None:
        types = network.types.detach().cpu().numpy().astype(np.int64)
        component = network.component.detach().cpu().numpy().astype(np.int64)
        key = np.int64(types.size).tobytes() + types.tobytes() + component.tobytes()
        network.type_key = key
    return key


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

    Components are numbered in order of their lowest bus index; buses with no
    in-service branch form singleton components with zero hop and resistance.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, shortest_path

    y = ybus.astype(np.complex128)
    y_abs = np.abs(y)
    weight = np.zeros((bus_count, bus_count), dtype=np.float64)
    adjacency = np.zeros((bus_count, bus_count), dtype=bool)
    table = np.asarray(branches, dtype=np.float64).reshape(-1, 4)
    magnitude = np.hypot(table[:, 2], table[:, 3])
    table = table[magnitude > 0]
    if table.shape[0]:
        origin = table[:, 0].astype(np.int64)
        destination = table[:, 1].astype(np.int64)
        branch_weight = 1.0 / magnitude[magnitude > 0]
        rows = np.stack((origin, destination), axis=1).ravel()
        columns = np.stack((destination, origin), axis=1).ravel()
        np.add.at(weight, (rows, columns), np.repeat(branch_weight, 2))
        adjacency[rows, columns] = True

    graph = csr_matrix(adjacency.astype(np.float64))
    _count, labels = connected_components(graph, directed=False)
    _unique, first_bus, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first_bus.size, dtype=np.int64)
    rank[np.argsort(first_bus, kind="stable")] = np.arange(first_bus.size)
    component = rank[inverse.reshape(-1)].astype(np.int64)
    component_id = int(first_bus.size)

    hop = shortest_path(graph, method="D", directed=False, unweighted=True)
    hop[~np.isfinite(hop)] = 0.0

    scale_y = np.ones(component_id, dtype=np.float64)
    hop_scale = np.ones(component_id, dtype=np.float64)
    resistance_scale = np.ones(component_id, dtype=np.float64)
    effective = np.zeros((bus_count, bus_count), dtype=np.float64)
    for comp in range(component_id):
        buses = np.flatnonzero(component == comp)
        scale_y[comp] = component_admittance_scale(y_abs, component, comp)
        max_hop = float(hop[np.ix_(buses, buses)].max())
        hop_scale[comp] = max_hop if max_hop > 0 else 1.0
        if buses.size == 1:
            continue
        sub_weight = weight[np.ix_(buses, buses)]
        sub_laplacian = np.diag(np.sum(sub_weight, axis=1)) - sub_weight
        pseudoinverse = np.linalg.pinv(sub_laplacian)
        diagonal = np.diag(pseudoinverse)
        delta = (diagonal[:, None] + diagonal[None, :]) - 2.0 * pseudoinverse
        effective[np.ix_(buses, buses)] = np.maximum(delta, 0.0)
        max_resistance = float(effective[np.ix_(buses, buses)].max())
        resistance_scale[comp] = max_resistance if max_resistance > 0 else 1.0

    same = component[:, None] == component[None, :]
    off_diagonal = same & ~np.eye(bus_count, dtype=bool)
    features = np.zeros((bus_count, bus_count, RPE_DIM), dtype=np.float32)
    y_scale = scale_y[component][:, None]
    features[..., 0] = np.where(same, y.real / y_scale, 0.0)
    features[..., 1] = np.where(same, y.imag / y_scale, 0.0)
    features[..., 2] = np.where(same, adjacency, False)
    features[..., 3] = np.where(off_diagonal, hop / hop_scale[component][:, None], 0.0)
    features[..., 4] = np.where(off_diagonal, effective / resistance_scale[component][:, None], 0.0)

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


def parse_batch_size() -> int:
    for index, arg in enumerate(sys.argv):
        if arg == "--batch" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--batch="):
            return int(arg.split("=", 1)[1])
    env = os.environ.get("GENCO_TPGT_BATCH", "").strip()
    if env:
        return int(env)
    return BATCH_SIZE


def parse_micro_batch_size(*, default: int | None = None) -> int:
    for index, arg in enumerate(sys.argv):
        if arg == "--micro-batch" and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith("--micro-batch="):
            return int(arg.split("=", 1)[1])
    env = os.environ.get("GENCO_TPGT_MICRO_BATCH", "").strip()
    if env:
        return int(env)
    return MICRO_BATCH_SIZE if default is None else default


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


def network_key(network: Network) -> tuple[bytes, bytes]:
    """Hash key of (bus types, Ybus rounded to 1e-6); equal keys are confirmed with ``networks_match``."""
    types = network.types.detach().cpu().numpy().astype(np.int64)
    ybus = network.ybus.detach().cpu().numpy()
    packed = np.round(np.stack((ybus.real, ybus.imag)).astype(np.float64), 6) + 0.0
    return types.tobytes(), hashlib.blake2b(packed.tobytes(), digest_size=16).digest()


# Below this many networks a hash miss falls back to the pairwise allclose scan.
NETWORK_SCAN_LIMIT = 64


def merge_many(splits: list[Split], name: str | None = None) -> Split:
    """Concatenate ``splits`` in order; later topologies reuse a matching earlier ``Network``.

    The first split's networks are kept as they are, so its network ids do not change.
    """
    if not splits:
        raise ValueError("merge_many needs at least one split")
    merged_networks: list[Network] = []
    by_key: dict[tuple[bytes, bytes], list[int]] = {}

    def place(network: Network, reuse: bool) -> int:
        key = network_key(network)
        if reuse:
            for index in by_key.get(key, ()):
                if networks_match(merged_networks[index], network):
                    return index
            if len(merged_networks) <= NETWORK_SCAN_LIMIT:
                for index, existing in enumerate(merged_networks):
                    if networks_match(existing, network):
                        return index
        merged_networks.append(network)
        by_key.setdefault(key, []).append(len(merged_networks) - 1)
        return len(merged_networks) - 1

    mapped_index = []
    for position, split in enumerate(splits):
        network_map = np.array([place(network, position > 0) for network in split.networks], dtype=np.int64)
        mapped_index.append(torch.from_numpy(network_map[split.network_index.numpy()]))
    return Split(
        name=splits[0].name if name is None else name,
        count=sum(split.count for split in splits),
        power_p=torch.cat([split.power_p for split in splits]),
        power_q=torch.cat([split.power_q for split in splits]),
        voltage=torch.cat([split.voltage for split in splits]),
        angle=torch.cat([split.angle for split in splits]),
        network_index=torch.cat(mapped_index),
        networks=merged_networks,
    )


def merge_splits(base: Split, extra: Split) -> Split:
    """Append ``extra`` scenarios after ``base``; reuse matching ``Network`` objects."""
    return merge_many([base, extra])


def compact_networks(split: Split, min_networks: int | None = None) -> int:
    """Store RPEs of all but the dominant topology as float16 when ``split`` has many topologies.

    Under CUDA autocast the pair encoder sees float16 inputs anyway; ``move_to`` casts back to float32.
    Returns the number of networks converted.
    """
    if len(split.networks) < (COMPACT_MIN_NETWORKS if min_networks is None else min_networks):
        return 0
    keep = dominant_network_id(split)
    converted = 0
    for index, network in enumerate(split.networks):
        if index != keep and network.rpe.dtype != torch.float16:
            network.rpe = network.rpe.to(torch.float16)
            converted += 1
    return converted


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

    _ids, first_local = np.unique(network_ids, return_index=True)
    representatives = [int(local) for local in first_local]
    with blas_single_thread():
        networks = _representative_networks(
            name, representatives, scenario_ids, type_code, y_by_scenario, status, origin, destination,
            resistance, reactance, branch_count,
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


def blas_single_thread():
    """Single-threaded BLAS while building many 118x118 RPEs (thread start-up dominates such small SVDs)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def _representative_networks(
    name: str,
    representatives: list[int],
    scenario_ids: list[int],
    type_code: np.ndarray,
    y_by_scenario: dict[int, list[YBusEntry]],
    status: np.ndarray,
    origin: np.ndarray,
    destination: np.ndarray,
    resistance: np.ndarray,
    reactance: np.ndarray,
    branch_count: int,
) -> list[Network]:
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
    return networks


def log_counts(data_root: Path, results: Path, spec: TaskSpec | None = None) -> dict:
    spec = spec or TASKS["1.1"]
    report = {}
    if spec.is_task11:
        folders = [("train", "n", "feasible"), ("valid", "n", "feasible")]
    else:
        folders = [("train", *folder) for folder in task_folders(data_root, "train")]
        folders.extend(("valid", *folder) for folder in task_folders(data_root, "valid"))
    folders.extend(("test", topology, condition) for _label, topology, condition in TEST_PARTS)
    for split, topology, condition in folders:
        raw = data_root / "case118" / split / "case118" / topology / condition / "raw"
        count = scenario_count(raw)
        key = f"{split}/{topology}/{condition}"
        if not spec.is_task11:
            report[key] = {"loaded": count}
            print(f"count {key}: {count}", flush=True)
            continue
        expected = EXPECTED_COUNTS[(split, topology, condition)]
        report[key] = {"loaded": count, "expected": expected}
        print(f"count {key}: {count} (expected {expected})", flush=True)
        if count != expected:
            print(f"  count differs from the PFΔ Task 1.1 split. Using the {count} loaded scenarios.", flush=True)
    path = results / f"task{spec.file_tag}_tpgt_counts.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def raw_dir(data_root: Path, split: str, topology: str, condition: str) -> Path:
    return data_root / "case118" / split / "case118" / topology / condition / "raw"


def task_folders(data_root: Path, split: str) -> list[tuple[str, str]]:
    """(topology, condition) folders of ``split`` that hold a bus_data parquet, in N, N-1, N-2 order."""
    return [
        (topology, condition)
        for topology in TOPOLOGIES
        for condition in CONDITIONS
        if (raw_dir(data_root, split, topology, condition) / "bus_data.parquet").is_file()
    ]


def folder_mix_text(parts: list[tuple[tuple[str, str], int]]) -> str:
    """Per-folder counts with shares, then shares by condition and by topology."""
    total = max(sum(count for _folder, count in parts), 1)
    by_folder = ", ".join(f"{t}/{c} {count} ({100 * count / total:.1f}%)" for (t, c), count in parts)
    by_condition = {}
    by_topology = {}
    for (topology, condition), count in parts:
        by_condition[condition] = by_condition.get(condition, 0) + count
        by_topology[topology] = by_topology.get(topology, 0) + count
    conditions = " ".join(f"{name} {100 * count / total:.1f}%" for name, count in by_condition.items())
    topologies = " ".join(f"{name.upper()} {100 * count / total:.1f}%" for name, count in by_topology.items())
    return f"{by_folder} | {conditions} | {topologies}"


def split_cache_dirs(checkpoints: Path, spec: TaskSpec | None = None) -> tuple[Path, Path]:
    """Return (fast_cache, persistent_cache). On Colab, fast_cache is local SSD."""
    persistent = checkpoints / "cache"
    if in_colab():
        if spec is None or spec.is_task11:
            return COLAB_LOCAL_CACHE, persistent
        return Path(f"/content/{spec.run_dir_name}_cache"), persistent
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


def _compact_if(split: Split, name: str, compact: bool) -> Split:
    if compact:
        converted = compact_networks(split)
        if converted:
            print(f"  {name}: {converted} of {len(split.networks)} topology RPEs stored as float16", flush=True)
    return split


def cached_split(
    active: Path,
    persistent: Path,
    raw: Path,
    name: str,
    limit: int | None = None,
    *,
    compact: bool = False,
) -> Split:
    """Load ``name`` from the split cache or build it from parquet.

    ``compact`` (train/valid folders of multi-topology tasks) stores RPEs as float16 once the
    folder has ``COMPACT_MIN_NETWORKS`` topologies; small folders are unaffected.
    """
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
            return _compact_if(payload, name, compact)
        if version_ok and limit > old_limit:
            existing = torch.load(tensor_path, map_location="cpu", weights_only=False)
            if isinstance(existing, Split) and existing.count == old_limit:
                build_started = time.perf_counter()
                split = extend_split_prefix(raw, name, existing, limit)
                print(
                    f"  extended {name} split in {time.perf_counter() - build_started:.1f}s",
                    flush=True,
                )
                _compact_if(split, name, compact)
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
    _compact_if(split, name, compact)
    save_started = time.perf_counter()
    torch.save(split, tensor_path)
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    print(f"  saved {name} cache in {time.perf_counter() - save_started:.1f}s", flush=True)
    sync_split_cache_to_drive(active, persistent)
    return split


def load_task_split(data_root: Path, split_name: str, active: Path, persistent: Path) -> Split:
    """Union of every ``split_name`` folder of the task (each cached on its own, DATA_FRACTION per folder).

    A task with a single folder (Task 1.1) returns that folder's split unchanged.
    """
    folders = task_folders(data_root, split_name)
    if not folders:
        raise FileNotFoundError(f"No {split_name} folders under {data_root / 'case118' / split_name}")
    multi = len(folders) > 1
    parts = [
        cached_split(
            active,
            persistent,
            raw_dir(data_root, split_name, topology, condition),
            f"{split_name}_{topology}_{condition}",
            compact=multi,
        )
        for topology, condition in folders
    ]
    print(
        f"{split_name} mix: " + folder_mix_text([(folder, part.count) for folder, part in zip(folders, parts)]),
        flush=True,
    )
    if not multi:
        return parts[0]
    merge_started = time.perf_counter()
    merged = merge_many(parts, name=split_name)
    print(
        f"{split_name}: merged {len(parts)} folders into {merged.count} cases, {len(merged.networks)} topologies "
        f"in {time.perf_counter() - merge_started:.1f}s",
        flush=True,
    )
    return merged


DATASET_RELATIVE = Path("Foundation Model") / "vendor" / "pfdelta_task1.1"
WINDOWS_DRIVE = Path(r"K:\My Drive")
COLAB_DRIVE = Path("/content/drive/MyDrive")
# Fast split-cache mirror on Colab runtime disk (not synced across disconnects).
COLAB_LOCAL_CACHE = Path(f"/content/{RUN_DIR_NAME}_cache")


def dataset_relative(spec: TaskSpec | None = None) -> Path:
    spec = spec or TASKS["1.1"]
    if spec.is_task11:
        return DATASET_RELATIVE
    return Path("Foundation Model") / "vendor" / f"pfdelta_task{spec.data_id}"


def dataset_request(spec: TaskSpec | None = None) -> Path:
    spec = spec or TASKS["1.1"]
    return Path(os.environ.get(spec.dataset_env, str(dataset_relative(spec))))


def google_drive_root() -> Path | None:
    """Colab MyDrive after mount, or the Windows Google Drive client root."""
    if in_colab():
        return mount_drive()
    if WINDOWS_DRIVE.is_dir():
        return WINDOWS_DRIVE
    return None


def dataset_candidates(drive: Path | None, spec: TaskSpec | None = None) -> list[Path]:
    spec = spec or TASKS["1.1"]
    requested = dataset_request(spec)
    relative = dataset_relative(spec)
    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    elif drive is not None:
        candidates.append(drive / requested)
    for extra in (WINDOWS_DRIVE / relative, COLAB_DRIVE / relative):
        if extra not in candidates:
            candidates.append(extra)
    return candidates


def drive_dataset(drive: Path | None, spec: TaskSpec | None = None) -> Path | None:
    """Return the saved dataset root of ``spec`` (default Task 1.1) when case118 is already on Drive.

    This only checks directory names. It does not copy the dataset.
    """
    for candidate in dataset_candidates(drive, spec):
        if (candidate / "case118").is_dir():
            return candidate
    return None


def dataset_missing_files(data_root: Path, spec: TaskSpec) -> list[Path]:
    """Parquet files of ``spec``'s published train/valid/test folders that are absent under ``data_root``."""
    folders = [("test", topology, condition) for _label, topology, condition in TEST_PARTS]
    for split in ("train", "valid"):
        folders.extend((split, *folder) for folder in spec.folders)
    return [
        raw_dir(data_root, *folder) / name
        for folder in folders
        for name in PARQUET_FILES
        if not (raw_dir(data_root, *folder) / name).is_file()
    ]


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


def task_results_dir(checkpoints: Path, results_root: Path, spec: TaskSpec | None = None) -> Path:
    """Task 1.1 keeps ``<results_root>/<RUN_DIR_NAME>``; other tasks write ``<task folder>/results``."""
    spec = spec or TASKS["1.1"]
    if spec.is_task11:
        return results_root / RUN_DIR_NAME
    return checkpoints / "results"


def drive_output_dirs(drive: Path | None, spec: TaskSpec | None = None) -> tuple[Path, Path] | None:
    """Checkpoint directory and TPGT result directory on Drive, both versioned by ``RUN_TAG``.

    Colab writes under MyDrive/Foundation Model once Drive is mounted.
    Windows writes under K:\\My Drive\\Foundation Model when that folder exists.
    Task 1.1: ``tpgt_task11_<RUN_TAG>`` and ``results/tpgt_task11_<RUN_TAG>``; other tasks:
    ``tpgt_task<id>_<RUN_TAG>`` with ``results`` inside it.
    """
    spec = spec or TASKS["1.1"]
    if drive is None:
        return None
    foundation = drive / "Foundation Model"
    if not in_colab() and not foundation.is_dir():
        return None
    checkpoints = foundation / spec.run_dir_name
    results = task_results_dir(checkpoints, foundation / "results", spec)
    checkpoints.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return checkpoints, results


def ensure_dataset(work: Path, drive: Path | None = None, spec: TaskSpec | None = None) -> Path:
    spec = spec or TASKS["1.1"]
    if not spec.is_task11:
        return ensure_task_dataset(work, drive, spec)
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


def ensure_task_dataset(work: Path, drive: Path | None, spec: TaskSpec) -> Path:
    """Dataset root of ``spec``: the Drive copy when complete, else (resume) download into it."""
    saved = drive_dataset(drive, spec)
    if saved is not None and not dataset_missing_files(saved, spec):
        print(f"Dataset already present: {saved}", flush=True)
        return saved
    if saved is not None:
        data = saved
    else:
        requested = dataset_request(spec)
        if requested.is_absolute():
            data = requested
        elif drive is not None and (in_colab() or (drive / "Foundation Model").is_dir()):
            data = drive / requested
        else:
            data = work / f"pfdelta_task{spec.data_id}"
    if data.joinpath("case118").is_dir() and not dataset_missing_files(data, spec):
        print(f"Dataset already present: {data}", flush=True)
        return data
    from huggingface_hub import snapshot_download

    print(f"Downloading {spec.dataset_repo} to {data}", flush=True)
    data.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec.dataset_repo, repo_type="dataset", local_dir=str(data))
    missing = dataset_missing_files(data, spec)
    if missing:
        raise FileNotFoundError(f"Download of {spec.dataset_repo} finished without {missing[:3]}")
    return data


def move_to(network: Network, device: torch.device) -> Network:
    """Device copy with pair-type masks, attention blocks and type key precomputed (``Network.layout``)."""
    types = host_to_device(network.types, device)
    component = host_to_device(network.component, device)
    pair_embed = network.pair_embed.to(device) if network.pair_embed is not None else None
    masks = network.pair_type_masks
    if masks is None:
        masks = type_pair_masks(types, component)
    else:
        masks = masks.to(device)
    return Network(
        ybus=host_to_device(network.ybus, device),
        rpe=host_to_device(network.rpe, device).to(dtype=torch.float32),
        component=component,
        types=types,
        pair_type_masks=masks,
        pair_embed=pair_embed,
        branches=getattr(network, "branches", None),
        type_key=network_type_key(network),
        layout=shared_layout(types, masks, types_np=network.types.detach().cpu().numpy()),
    )


def network_device_bytes(buses: int) -> int:
    """Device bytes of one moved network: float32 RPE, complex64 Ybus, pair-type and block masks, indices."""
    pairs = buses * buses
    return pairs * (RPE_DIM * 4 + 8 + 9 + 1) + buses * 8 * 6


def device_cache_budget(device: torch.device) -> int:
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(min(DEVICE_CACHE_FREE_FRACTION * free, DEVICE_CACHE_MAX_BYTES))
    return DEVICE_CACHE_CPU_BYTES


class DeviceNetworks:
    """Device copies of ``networks`` made on first use; the ``capacity`` most recently used are kept.

    Without an explicit ``capacity`` every network stays resident when all of them fit ``budget_bytes``
    (default ``device_cache_budget``); otherwise the LRU holds as many as the budget allows.
    """

    def __init__(
        self,
        networks: list[Network],
        device: torch.device,
        capacity: int | None = None,
        budget_bytes: int | None = None,
    ) -> None:
        self.networks = networks
        self.device = device
        buses = int(networks[0].types.numel()) if networks else 0
        self.bytes_per_network = network_device_bytes(buses)
        if capacity is None:
            self.budget_bytes = device_cache_budget(device) if budget_bytes is None else int(budget_bytes)
            if self.bytes_per_network * len(networks) <= self.budget_bytes:
                capacity = len(networks)
            else:
                capacity = self.budget_bytes // max(self.bytes_per_network, 1)
        else:
            self.budget_bytes = int(capacity) * self.bytes_per_network
        self.capacity = max(1, int(capacity))
        self._moved: OrderedDict[int, Network] = OrderedDict()

    @property
    def all_resident(self) -> bool:
        return self.capacity >= len(self.networks)

    def describe(self) -> str:
        if self.all_resident:
            total = self.bytes_per_network * len(self.networks)
            return f"all {len(self.networks)} networks resident ({total / 1024**2:.1f} MB)"
        return (
            f"LRU capacity {self.capacity} of {len(self.networks)} networks "
            f"({self.bytes_per_network / 1024:.0f} KB each, budget {self.budget_bytes / 1024**2:.0f} MB)"
        )

    def __len__(self) -> int:
        return len(self.networks)

    def __getitem__(self, network_id: int) -> Network:
        network_id = int(network_id)
        moved = self._moved.get(network_id)
        if moved is None:
            moved = move_to(self.networks[network_id], self.device)
            self._moved[network_id] = moved
            while len(self._moved) > self.capacity:
                self._moved.popitem(last=False)
        else:
            self._moved.move_to_end(network_id)
        return moved

    def resident(self) -> list[Network]:
        return list(self._moved.values())


def clear_pair_embed_cache(networks: list[Network] | DeviceNetworks) -> None:
    for network in networks.resident() if isinstance(networks, DeviceNetworks) else networks:
        network.pair_embed = None


def pair_features(model: TPGT, network: Network) -> torch.Tensor:
    """Edge embeddings for this topology; cached in eval on that Network object only."""
    if not model.training and network.pair_embed is not None:
        return network.pair_embed
    encoded = model.pair_encoder(network.rpe)
    if not model.training:
        network.pair_embed = encoded
    return encoded


def forward_cases(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    pair: torch.Tensor,
    ybus: torch.Tensor | None = None,
    layout: TypeLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Prediction, target and stage residuals of ``index`` on ``network``'s bus types.

    ``pair`` and ``ybus`` default to the shared (N, N, ...) tensors of ``network``; per case they carry
    a leading batch dimension (cases of several topologies). A per-case ``layout`` replaces
    ``network``'s bus types (topologies with different bus types, see ``PassBatch``).
    """
    if layout is None:
        layout = network_layout(network)
    chosen = torch.as_tensor(index, dtype=torch.long)
    if device.type == "cuda":
        rows = torch.stack((split.power_p[chosen], split.power_q[chosen], split.voltage[chosen], split.angle[chosen]))
        power_p, power_q, voltage, angle = host_to_device(rows, device).unbind(0)
    else:
        power_p = split.power_p[chosen].to(device, non_blocking=True)
        power_q = split.power_q[chosen].to(device, non_blocking=True)
        voltage = split.voltage[chosen].to(device, non_blocking=True)
        angle = split.angle[chosen].to(device, non_blocking=True)
    specified, target = pack_case(power_p, power_q, voltage, angle, layout)
    prediction, residuals = model(
        specified,
        layout.types,
        network.rpe,
        network.pair_type_masks,
        network.ybus if ybus is None else ybus,
        pair=pair,
        layout=layout,
    )
    return prediction, target, residuals


def run_batch(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    pair: torch.Tensor | None = None,
    ybus: torch.Tensor | None = None,
    layout: TypeLayout | None = None,
) -> tuple[TaskLossBreakdown, torch.Tensor]:
    breakdown, residuals = run_batch_residuals(model, split, network, index, device, pair, ybus, layout)
    return breakdown, mean_pbl(residuals[-1])


def run_batch_residuals(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    pair: torch.Tensor | None = None,
    ybus: torch.Tensor | None = None,
    layout: TypeLayout | None = None,
) -> tuple[TaskLossBreakdown, list[torch.Tensor]]:
    if pair is None:
        pair = pair_features(model, network)
    if layout is None:
        layout = network_layout(network)
    prediction, target, residuals = forward_cases(model, split, network, index, device, pair, ybus, layout)
    return task_loss(prediction, target, residuals, layout), residuals


def group_by_type_pattern(
    groups: list[tuple[Network, np.ndarray]],
) -> list[list[tuple[Network, np.ndarray]]]:
    """(network, case index) pairs split by ``network_type_key``, patterns and members in order of appearance."""
    by_key: dict[bytes, list[tuple[Network, np.ndarray]]] = {}
    for network, index in groups:
        by_key.setdefault(network_type_key(network), []).append((network, np.asarray(index)))
    return list(by_key.values())


def stacked_pair_and_ybus(
    model: TPGT,
    networks: list[Network],
    keep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pair embeddings (U, N, N, D_E) from one pair-encoder call and complex128 Ybus (U, N, N)."""
    rpe = torch.stack([network.rpe for network in networks])
    rpe = rpe_channel_dropout(rpe, RPE_CHANNEL_DROPOUT, model.training, keep=keep)
    pair = model.pair_encoder(rpe)
    ybus = torch.stack([network.ybus for network in networks]).to(torch.complex128)
    return pair, ybus


class PassBatch:
    """Cases of one forward pass over ``networks``; ``owner[k]`` is the network position of ``cases[k]``.

    ``kind`` is "single" (one topology: shared pair embedding, Ybus and blocks), "shared" (topologies with
    equal ``network_type_key``: shared blocks, per-case pair embeddings and Ybus) or "mixed" (bus types
    differ: per-case ``TypeLayout`` as well).
    """

    def __init__(self, networks: list[Network], cases: np.ndarray, owner: np.ndarray | None = None) -> None:
        self.networks = networks
        self.cases = np.asarray(cases)
        self.owner = np.zeros(len(self.cases), dtype=np.int64) if owner is None else np.asarray(owner)
        self.ybus: torch.Tensor | None = None
        self.select: torch.Tensor | None = None
        self.member_layout: TypeLayout | None = None
        if len(networks) == 1:
            self.kind = "single"
            return
        keys = {network_type_key(network) for network in networks}
        self.kind = "shared" if len(keys) == 1 else "mixed"
        self.ybus = torch.stack([network.ybus for network in networks]).to(torch.complex128)
        self.select = host_to_device(torch.as_tensor(self.owner, dtype=torch.long), self.ybus.device)
        if self.kind == "mixed":
            layouts = [network_layout(network) for network in networks]
            self.member_layout = per_case_layout(
                torch.stack([layout.types for layout in layouts]),
                torch.stack([network_masks(network) for network in networks]),
                types_np=np.stack([layout.types_np for layout in layouts]),
            )

    @classmethod
    def from_groups(cls, groups: list[tuple[Network, np.ndarray]]) -> PassBatch:
        networks = [network for network, _index in groups]
        if len(groups) == 1:
            return cls(networks, np.asarray(groups[0][1]))
        cases = np.concatenate([np.asarray(index) for _network, index in groups])
        owner = np.repeat(np.arange(len(groups)), [len(index) for _network, index in groups])
        return cls(networks, cases, owner)

    def __len__(self) -> int:
        return len(self.cases)

    def encode(self, model: TPGT, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Pair embeddings: (N, N, D_E) for one topology (cached in eval), else (U, N, N, D_E) in one call."""
        if self.kind == "single":
            network = self.networks[0]
            if not model.training:
                return pair_features(model, network)
            rpe = rpe_channel_dropout(network.rpe, RPE_CHANNEL_DROPOUT, model.training, keep=keep)
            return model.pair_encoder(rpe)
        rpe = torch.stack([network.rpe for network in self.networks])
        rpe = rpe_channel_dropout(rpe, RPE_CHANNEL_DROPOUT, model.training, keep=keep)
        return model.pair_encoder(rpe)

    def run(
        self,
        model: TPGT,
        split: Split,
        pair: torch.Tensor,
        device: torch.device,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[TaskLossBreakdown, list[torch.Tensor]]:
        """Loss and stage residuals of ``cases[start:stop]`` with ``pair`` (one row per topology) from ``encode``."""
        chunk = self.cases[start:stop]
        if self.kind == "single":
            return run_batch_residuals(model, split, self.networks[0], chunk, device, pair=pair)
        select = self.select[start:stop]
        layout = None if self.member_layout is None else self.member_layout.take(select)
        return run_batch_residuals(
            model, split, self.networks[0], chunk, device, pair=pair[select], ybus=self.ybus[select], layout=layout
        )


def step_passes(groups: list[tuple[Network, np.ndarray]]) -> list[list[tuple[Network, np.ndarray]]]:
    """Forward passes of one step: all groups in one pass, or (``SPLIT_DOMINANT_PATTERN``) the largest
    bus-type pattern in its own pass and every other case in a second one."""
    if len(groups) == 1 or not SPLIT_DOMINANT_PATTERN:
        return [groups]
    patterns = group_by_type_pattern(groups)
    if len(patterns) == 1:
        return [groups]
    dominant = max(patterns, key=lambda members: sum(len(index) for _network, index in members))
    rest = [member for members in patterns if members is not dominant for member in members]
    return [dominant, rest]


STEP_PHASES = ("move", "pair", "fwd", "bwd", "opt")


class StepTimer:
    """Time per training-step phase (network moves/prep, pair encoder, forward, backward, optimizer).

    On CUDA each phase records two CUDA events, read with one synchronize only when ``summary`` is called,
    so timing adds no sync to the training loop. Summaries report per-step means since the last
    ``summary`` (``scope="window"``) or since the timer was made (``scope="epoch"``).
    """

    def __init__(self, device: torch.device | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.cuda = enabled and device is not None and device.type == "cuda"
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._scopes = {"window": self._empty(), "epoch": self._empty()}

    @staticmethod
    def _empty() -> dict[str, float]:
        return {"steps": 0, "seconds": 0.0, "passes": 0, "topologies": 0, **dict.fromkeys(STEP_PHASES, 0.0)}

    def _add(self, key: str, value: float) -> None:
        for totals in self._scopes.values():
            totals[key] += value

    @contextlib.contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                self._pending.append((name, start, end))
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self._add(name, time.perf_counter() - started)

    def add_pass(self) -> None:
        if self.enabled:
            self._add("passes", 1)

    def end_step(self, topologies: int, seconds: float) -> None:
        if self.enabled:
            self._add("steps", 1)
            self._add("topologies", topologies)
            self._add("seconds", seconds)

    def _resolve(self) -> None:
        if not self._pending:
            return
        self._pending[-1][2].synchronize()
        for name, start, end in self._pending:
            self._add(name, start.elapsed_time(end) / 1000.0)
        self._pending.clear()

    def summary(self, scope: str = "window") -> str:
        """``t/step 3.472s (move 1.200 pair 0.300 fwd 0.900 bwd 1.000 opt 0.050) passes 5.1 topo 18.0``."""
        self._resolve()
        totals = self._scopes[scope]
        steps = max(int(totals["steps"]), 1)
        phases = " ".join(f"{name} {totals[name] / steps:.3f}" for name in STEP_PHASES)
        text = (
            f"t/step {totals['seconds'] / steps:.3f}s ({phases}) "
            f"passes {totals['passes'] / steps:.1f} topo {totals['topologies'] / steps:.1f}"
        )
        if scope == "window":
            self._scopes["window"] = self._empty()
        return text


NO_TIMER = StepTimer(enabled=False)


def _weighted_loss_breakdown(
    parts: list[tuple[TaskLossBreakdown, float]],
) -> TaskLossBreakdown:
    """Case-weighted mean of the parts' (detached) losses; stays on the device so logging needs no sync."""
    weight = sum(weight for _breakdown, weight in parts)
    if weight <= 0:
        raise RuntimeError("empty training batch")

    def mean(values: list[torch.Tensor]) -> torch.Tensor:
        return (sum(value.double() * w for value, (_b, w) in zip(values, parts)) / weight).float()

    return TaskLossBreakdown(
        mean([b.state_mse for b, _w in parts]),
        mean([b.physics_sum for b, _w in parts]),
        tuple(mean([b.physics_stages[stage_idx] for b, _w in parts]) for stage_idx in range(DEPTH)),
        mean([b.total for b, _w in parts]),
    )


def rpe_channel_keep(channels: int, probability: float, training: bool, device: torch.device) -> torch.Tensor | None:
    """Per-step channel keep mask (one draw shared by every topology in the step); None when inactive."""
    if not training or probability <= 0.0:
        return None
    return torch.rand(channels, device=device) >= probability


def rpe_channel_dropout(
    rpe: torch.Tensor,
    probability: float,
    training: bool,
    keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Zero whole RPE channels independently with ``probability`` (no rescaling); identity unless training."""
    if not training or probability <= 0.0:
        return rpe
    if keep is None:
        keep = rpe_channel_keep(rpe.shape[-1], probability, training, rpe.device)
    return rpe * keep.to(rpe.dtype)


def train_step(
    model: TPGT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    split: Split,
    groups: list[tuple[Network, np.ndarray]],
    device: torch.device,
    timer: StepTimer | None = None,
) -> TaskLossBreakdown:
    """One optimizer step over all ``groups`` (topology, case index).

    The cases run as one forward/backward (``step_passes``, ``PassBatch``): a single topology with its
    shared pair embedding and Ybus, several with per-case pair embeddings and Ybus from one pair-encoder
    call, and per-case bus types when their types differ. Each micro-batch loss is weighted by its share
    of the step's cases, so the gradient is the mean over every case of the step, exactly as for a
    single-topology batch.
    """
    timer = NO_TIMER if timer is None else timer
    n = sum(len(index) for _network, index in groups)
    micro = min(MICRO_BATCH_SIZE, n)
    use_amp = device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)
    logged_parts: list[tuple[TaskLossBreakdown, float]] = []
    keep = rpe_channel_keep(RPE_DIM, RPE_CHANNEL_DROPOUT, model.training, groups[0][0].rpe.device)
    for members in step_passes(groups):
        with timer.phase("move"):
            batch = PassBatch.from_groups(members)
        with timer.phase("pair"), torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            pair_embed = batch.encode(model, keep)
        for start in range(0, len(batch), micro):
            weight = len(batch.cases[start : start + micro]) / n
            retain = start + micro < len(batch)
            with timer.phase("fwd"), torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                breakdown, _residuals = batch.run(model, split, pair_embed, device, start, start + micro)
            with timer.phase("bwd"):
                scaled = breakdown.total * weight
                if scaler is not None:
                    scaler.scale(scaled).backward(retain_graph=retain)
                else:
                    scaled.backward(retain_graph=retain)
            timer.add_pass()
            logged_parts.append((detached_breakdown(breakdown), weight))
            del breakdown, scaled, _residuals
    with timer.phase("opt"):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    return _weighted_loss_breakdown(logged_parts)


def detached_breakdown(breakdown: TaskLossBreakdown) -> TaskLossBreakdown:
    return TaskLossBreakdown(
        breakdown.state_mse.detach(),
        breakdown.physics_sum.detach(),
        tuple(stage.detach() for stage in breakdown.physics_stages),
        breakdown.total.detach(),
    )


def train_batch(
    model: TPGT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
) -> TaskLossBreakdown:
    """One optimizer step over ``index`` (up to BATCH_SIZE cases) on one topology."""
    return train_step(model, optimizer, scaler, split, [(network, index)], device)


def iterate_steps(
    split: Split,
    seed: int,
    epoch: int,
    batch_size: int | None = None,
):
    """Training steps: one global shuffle per epoch, ``batch_size`` cases per step, split by topology.

    Yields lists of (network_id, case index) in order of first appearance within the step. With one
    topology this is the same permutation and the same batches as shuffling that topology alone.
    ``batch_size`` defaults to the current global ``BATCH_SIZE`` (so ``--batch`` / env overrides apply).
    """
    if batch_size is None:
        batch_size = BATCH_SIZE
    generator = torch.Generator()
    generator.manual_seed(seed * 1000 + epoch)
    if split.count > 1:
        order = torch.randperm(split.count, generator=generator).numpy()
    else:
        order = np.arange(split.count)
    network_index = split.network_index.numpy()
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        owners = network_index[chunk]
        _ids, first = np.unique(owners, return_index=True)
        yield [(int(owners[position]), chunk[owners == owners[position]]) for position in np.sort(first)]


def iterate_indexes(
    split: Split,
    seed: int,
    epoch: int,
    shuffle: bool,
    batch_size: int | None = None,
):
    """Batches that never mix topologies: networks in id order, cases in split order unless ``shuffle``."""
    if batch_size is None:
        batch_size = BATCH_SIZE
    generator = torch.Generator()
    generator.manual_seed(seed * 1000 + epoch)
    network_index = split.network_index.numpy()
    order = np.argsort(network_index, kind="stable")
    bounds = np.searchsorted(network_index[order], np.arange(len(split.networks) + 1))
    for network_id in range(len(split.networks)):
        index = order[bounds[network_id] : bounds[network_id + 1]]
        if shuffle and index.size > 1:
            index = index[torch.randperm(index.size, generator=generator).numpy()]
        for start in range(0, len(index), batch_size):
            yield network_id, index[start : start + batch_size]


def iterate_type_batches(split: Split, batch_size: int = EVAL_BATCH_SIZE, mix_patterns: bool = False):
    """Batches of up to ``batch_size`` cases whose topologies share ``network_type_key``.

    Patterns in order of their first network id; within a pattern, networks in id order and cases in
    split order. Yields (network id per case, case index). With one topology this matches
    ``iterate_indexes`` without shuffling. ``mix_patterns`` keeps the order but fills every batch,
    so a batch may span bus-type patterns.
    """
    keys: dict[bytes, int] = {}
    pattern = np.array(
        [keys.setdefault(network_type_key(network), len(keys)) for network in split.networks], dtype=np.int64
    )
    network_index = split.network_index.numpy()
    case_pattern = pattern[network_index]
    order = np.lexsort((np.arange(split.count), network_index, case_pattern))
    bounds = [] if mix_patterns else np.flatnonzero(np.diff(case_pattern[order])) + 1
    for run in np.split(order, bounds):
        for start in range(0, len(run), batch_size):
            index = run[start : start + batch_size]
            yield network_index[index], index


def evaluate_pbl(
    model: TPGT,
    split: Split,
    networks: list[Network],
    device: torch.device,
    batch_size: int = EVAL_BATCH_SIZE,
    per_network_out: dict[int, list[float]] | None = None,
) -> float:
    """Mean final-stage PBL over ``split`` in full batches (several topologies and bus types per batch).

    A single-topology batch reuses that topology's cached pair embedding, dropped once its cases are done.
    """
    model.eval()
    clear_pair_embed_cache(networks)
    total = 0.0
    seen = 0
    cached: dict[int, Network] = {}
    with torch.no_grad():
        for owners, index in iterate_type_batches(split, batch_size=batch_size, mix_patterns=True):
            ids, inverse = np.unique(owners, return_inverse=True)
            inverse = inverse.reshape(-1)
            for network_id in [key for key in cached if key not in set(ids.tolist())]:
                cached.pop(network_id).pair_embed = None
            if len(ids) == 1:
                network = networks[int(ids[0])]
                _breakdown, score = run_batch(model, split, network, index, device)
                cached[int(ids[0])] = network
                sums = [float(score) * len(index)]
                counts = [len(index)]
            else:
                batch = PassBatch([networks[int(network_id)] for network_id in ids], index, inverse)
                _breakdown, residuals = batch.run(model, split, batch.encode(model), device)
                del batch
                per_case = case_pbl(residuals[-1]).double().cpu().numpy()
                sums = np.bincount(inverse, weights=per_case, minlength=len(ids)).tolist()
                counts = np.bincount(inverse, minlength=len(ids)).tolist()
            for network_id, weighted, cases in zip(ids, sums, counts):
                total += weighted
                seen += cases
                if per_network_out is not None:
                    bucket = per_network_out.setdefault(int(network_id), [0.0, 0.0])
                    bucket[0] += weighted
                    bucket[1] += cases
    for network in cached.values():
        network.pair_embed = None
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
    timing: str | None = None,
) -> str:
    n = max(batches, 1)
    stage_text = "  ".join(f"phys_s{i + 1} {value / n:.4f}" for i, value in enumerate(physics_stages))
    lr_text = "" if lr is None else f"  lr {lr:.1e}"
    timing_text = "" if timing is None else f"  {timing}"
    return (
        f"seed {seed} epoch {epoch_number}/{EPOCHS}  "
        f"state {state_mse / n:.4f}  phys_w {physics_sum / n:.4f}  {stage_text}  "
        f"train {train_total / n:.4f}  val_pbl {val_display}{lr_text}{timing_text}"
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
    cache_budget = device_cache_budget(device)
    train_share = len(train_split.networks) / max(len(train_split.networks) + len(val_split.networks), 1)
    train_networks = DeviceNetworks(train_split.networks, device, budget_bytes=int(cache_budget * train_share))
    val_networks = DeviceNetworks(
        val_split.networks, device, budget_bytes=cache_budget - int(cache_budget * train_share)
    )
    print(
        f"device network cache (budget {cache_budget / 1024**2:.0f} MB): "
        f"train {train_networks.describe()}; valid {val_networks.describe()}",
        flush=True,
    )
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
            # Loss sums (total, state, physics, stages) stay on the device in float64 until printed.
            running = torch.zeros(3 + DEPTH, dtype=torch.float64, device=device)
            batches = 0
            logged_first_batch = False
            batch_durations: list[float] = []
            timer = StepTimer(device)
            for step_groups in iterate_steps(train_split, seed, epoch):
                if not logged_first_batch:
                    first_batch_started = time.perf_counter()
                step_started = time.perf_counter()
                with timer.phase("move"):
                    moved = [(train_networks[network_id], index) for network_id, index in step_groups]
                breakdown = train_step(
                    model,
                    optimizer,
                    grad_scaler if device.type == "cuda" else None,
                    train_split,
                    moved,
                    device,
                    timer=timer,
                )
                del moved
                if not logged_first_batch:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - first_batch_started
                    cases = sum(len(index) for _network_id, index in step_groups)
                    topologies = "" if len(step_groups) == 1 else f", {len(step_groups)} topologies"
                    print(
                        f"  seed {seed} epoch {epoch + 1} first batch ({cases} cases{topologies}) "
                        f"forward+backward {elapsed:.2f}s",
                        flush=True,
                    )
                    logged_first_batch = True
                batch_durations.append(time.perf_counter() - step_started)
                timer.end_step(len(step_groups), batch_durations[-1])
                losses = (breakdown.total, breakdown.state_mse, breakdown.physics_sum, *breakdown.physics_stages)
                running += torch.stack([value.detach().to(device) for value in losses]).double()
                batches += 1
                if batches % 250 == 0:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    window = batch_durations[-250:]
                    avg_sec = sum(window) / len(window)
                    print(
                        f"  seed {seed} epoch {epoch + 1} batch {batches} train {float(running[0]) / batches:.4f} "
                        f"avg {avg_sec:.3f}s/batch (last {len(window)})  {timer.summary()}",
                        flush=True,
                    )
            running_total, running_state, running_phys, *running_phys_stages = running.tolist()
            epoch_timing = timer.summary("epoch")
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
                    timing=epoch_timing,
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
    spec: TaskSpec | None = None,
) -> None:
    spec = spec or TASKS["1.1"]
    c2i_counts = {"C2I-N": counts["C2I-N"], "C2I-N-1": counts["C2I-N-1"], "C2I-N-2": counts["C2I-N-2"]}
    scores = dict(scores)
    scores["C2I"] = aggregate_c2i(scores, c2i_counts)
    columns = [*SEED_COLUMNS, "epoch"]
    rows = []
    for category in (*CATEGORIES,):
        rows.append(
            {
                "model": MODEL_NAME,
                "task": spec.task_id,
                "test_category": category,
                "seed": str(seed),
                "mean_pbl_pu": f"{scores[category]:.8g}",
                "checkpoint_path": str(checkpoint),
                "source": "independent_training",
                "epoch": str(epoch),
            }
        )
    upsert_csv(
        results / f"task{spec.file_tag}_tpgt_seed_metrics.csv",
        columns,
        rows,
        ["model", "task", "test_category", "seed", "epoch"],
    )


def write_summary(results: Path, spec: TaskSpec | None = None) -> list[dict]:
    spec = spec or TASKS["1.1"]
    latest: dict[tuple[str, str], tuple[int, float]] = {}
    for row in read_csv(results / f"task{spec.file_tag}_tpgt_seed_metrics.csv"):
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
    path = results / f"task{spec.file_tag}_tpgt_summary.csv"
    if summary:
        upsert_csv(path, SUMMARY_COLUMNS, summary, ["model", "test_category"])
    return summary


def print_comparison(summary: list[dict], spec: TaskSpec | None = None) -> None:
    spec = spec or TASKS["1.1"]
    print(f"\nTask {spec.label} mean power-balance loss (per unit)", flush=True)
    print("PUBLISHED", flush=True)
    for model, scores in PUBLISHED.get(spec.task_id, {}).items():
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


def comparison_figure_path(results: Path, spec: TaskSpec | None = None) -> Path:
    spec = spec or TASKS["1.1"]
    return results / f"figure4_task{spec.task_id}_tpgt.svg"


def plot_comparison(results: Path, summary: list[dict], spec: TaskSpec | None = None) -> None:
    """Bar plot (SVG + PNG) of TPGT mean ± std over the seeds scored so far against the published models."""
    spec = spec or TASKS["1.1"]
    by_category = {row["test_category"]: row for row in summary}
    if any(category not in by_category for category in CATEGORIES):
        return
    if any(int(by_category[category]["n_seeds"]) < 1 for category in CATEGORIES):
        return
    tpgt = {
        category: (float(by_category[category]["mean_across_seeds"]), float(by_category[category]["std_across_seeds"]))
        for category in CATEGORIES
    }
    seeds = min(int(by_category[category]["n_seeds"]) for category in CATEGORIES)
    output = comparison_figure_path(results, spec)
    plot_tpgt_comparison(spec.task_id, spec.label, tpgt, output, tpgt_label=f"TPGT (this run, {seeds} seed(s))")
    try:
        from IPython.display import SVG, display

        display(SVG(filename=str(output)))
    except Exception:
        pass


def bus_map_api() -> SimpleNamespace:
    """Functions ``plot_bus_error_map`` needs from this module (works when run as a script or via %run)."""
    return SimpleNamespace(
        move_to=move_to,
        pair_features=pair_features,
        pack_case=pack_case,
        dominant_network_id=dominant_network_id,
        network_case_counts=network_case_counts,
        contingency_class=contingency_class,
    )


def write_checkpoint_bus_maps(
    model: TPGT,
    val_split: Split,
    n1_split: Split,
    device: torch.device,
    out_dir: Path,
    epoch_number: int,
    checkpoint: Path,
) -> None:
    """IEEE 118 per-bus PBL maps for this checkpoint; any failure is logged and training continues."""
    started = time.perf_counter()
    try:
        from plot_bus_error_map import write_bus_maps

        with torch.no_grad():
            written = write_bus_maps(
                bus_map_api(),
                model,
                val_split,
                n1_split,
                device,
                out_dir,
                epoch=epoch_number,
                checkpoint_name=checkpoint.name,
                val_cases=BUS_MAP_VAL_CASES,
                batch=EVAL_BATCH_SIZE,
            )
        print(
            f"Bus error maps ({len(written)} PNG) written to {out_dir} in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    except Exception as error:
        print(f"Bus error maps skipped for epoch {epoch_number}: {type(error).__name__}: {error}", flush=True)


def test_model(
    model: TPGT,
    data_root: Path,
    device: torch.device,
    active_cache: Path,
    persistent_cache: Path,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    after_split: Callable[[str, Split, DeviceNetworks, dict[int, list[float]]], None] | None = None,
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
        networks = DeviceNetworks(split.networks, device)
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
    global BATCH_SIZE, DATA_FRACTION, MICRO_BATCH_SIZE, EVAL_BATCH_SIZE, VAL_EVERY
    ensure_runtime_packages()
    DATA_FRACTION = parse_data_fraction()
    BATCH_SIZE = parse_batch_size()
    MICRO_BATCH_SIZE = parse_micro_batch_size(default=BATCH_SIZE)
    EVAL_BATCH_SIZE = parse_eval_batch_size()
    VAL_EVERY = parse_val_every()
    if BATCH_SIZE < 1:
        raise ValueError(f"batch must be >= 1, got {BATCH_SIZE}")
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
    spec = parse_task()
    print(f"PFΔ Task {spec.label}: {spec.description} (dataset {spec.dataset_repo})", flush=True)
    drive = google_drive_root()
    outputs = drive_output_dirs(drive, spec)
    work, legacy_results = legacy_locations(drive)
    if outputs is not None:
        checkpoints, results = outputs
    else:
        work.mkdir(parents=True, exist_ok=True)
        checkpoints = work / spec.run_dir_name
        checkpoints.mkdir(parents=True, exist_ok=True)
        results = task_results_dir(checkpoints, legacy_results, spec)
        results.mkdir(parents=True, exist_ok=True)
    os.environ["GENCO_RESULTS_DIR"] = str(results)
    os.environ["GENCO_WORK_ROOT"] = str(checkpoints.parent)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    from common import use_selected_results

    use_selected_results()
    ensure_dirs()
    results.mkdir(parents=True, exist_ok=True)
    data = ensure_dataset(work, drive, spec)
    print(f"Dataset: {data}", flush=True)
    print(f"Run tag {RUN_TAG} (checkpoint_version {CHECKPOINT_VERSION}, {DEPTH} stages, {RPE_DIM}-channel RPE)", flush=True)
    stage_weight_text = " ".join(f"s{i + 1} {w:g}" for i, w in enumerate(physics_stage_weights(DEPTH)))
    print(f"Physics stage weights (GENCO-style): {stage_weight_text}", flush=True)
    print(f"Checkpoints: {checkpoints}", flush=True)
    print(f"Results directory: {results}", flush=True)
    for split_name in ("train", "valid"):
        found = tuple(task_folders(data, split_name))
        if found != spec.folders:
            print(
                f"Note: {split_name} folders {found} differ from the published Task {spec.label} layout {spec.folders}; "
                "using the folders found.",
                flush=True,
            )
    log_counts(data, results, spec)
    active_cache, persistent_cache = split_cache_dirs(checkpoints, spec)
    print(f"Split cache (active): {active_cache}", flush=True)
    if persistent_cache != active_cache:
        print(f"Split cache (Drive): {persistent_cache}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    load_started = time.perf_counter()
    train_split = load_task_split(data, "train", active_cache, persistent_cache)
    val_split = load_task_split(data, "valid", active_cache, persistent_cache)
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
    held_splits: dict[str, Split] = {}
    for seed in SEEDS:
        seed_dir = checkpoints / f"seed_{seed}"
        if (seed_dir / "complete.json").exists() and read_completed_epochs(seed_dir / "complete.json") >= EPOCHS:
            print(f"Seed {seed} already completed {EPOCHS} epochs. Skipping.", flush=True)
            continue

        def on_epoch_end(model: TPGT, epoch_number: int, seed: int = seed) -> tuple[dict[str, float], dict[str, int]]:
            def after_test_split(
                label: str,
                split: Split,
                networks: DeviceNetworks,
                per_network: dict[int, list[float]],
            ) -> None:
                if label == "N-1":
                    held_splits["N-1"] = split
                if debug_diagnostics and label in CONTINGENCY_DIAGNOSTIC_LABELS:
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
                after_split=after_test_split,
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
            # record_seed / summary CSV / plots only on Drive checkpoint epochs (not every epoch).
            record_seed(results, seed, scores, counts, checkpoint, epoch_number, spec)
            summary = write_summary(results, spec)
            print_comparison(summary, spec)
            try:
                plot_comparison(results, summary, spec)
            except Exception as error:
                print(f"Bar plot skipped for epoch {epoch_number}: {type(error).__name__}: {error}", flush=True)
            n1_split = held_splits.pop("N-1", None)
            if n1_split is not None:
                write_checkpoint_bus_maps(
                    model,
                    val_split,
                    n1_split,
                    device,
                    checkpoints / "bus_error_maps" / f"seed_{seed}" / f"epoch_{epoch_number}",
                    epoch_number,
                    checkpoint,
                )

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
    spec = parse_task()
    print(f"Diagnose PFΔ Task {spec.label}", flush=True)
    drive = google_drive_root()
    outputs = drive_output_dirs(drive, spec)
    work, legacy_results = legacy_locations(drive)
    if outputs is not None:
        checkpoints, results = outputs
    else:
        checkpoints = work / spec.run_dir_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_cuda_training(device)
    data = ensure_dataset(work, drive, spec)
    active_cache, persistent_cache = split_cache_dirs(checkpoints, spec)
    val_split = load_task_split(data, "valid", active_cache, persistent_cache)
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
        networks = DeviceNetworks(split.networks, device)
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
            elif args[index] in ("--data-fraction", "--task") and index + 1 < len(args):
                index += 2
            elif args[index].startswith(("--data-fraction=", "--task=")) or args[index] in (
                "--rpe-ablation",
                "--no-rpe-ablation",
            ):
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
