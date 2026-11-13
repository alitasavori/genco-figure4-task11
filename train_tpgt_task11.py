"""Train the unshared type-pair graph Transformer on IEEE 118-bus PFΔ tasks (default Task 1.1).

Option II from the draft: each correction stage has its own weights.

Tasks (``--task`` or ``GENCO_TPGT_TASK``; see ``TASKS``)
----------------------------------------------------
1.1 (default), 1.2, 1.3 (alias 2.1), 2.3, 3.1, 4.1, 4.2, 4.3. Data: ``Foundation Model/vendor/pfdelta_task<id>``
(override ``GENCO_PFDELTA_TASK<id without dot>``, e.g. ``GENCO_PFDELTA_TASK42``), downloaded from
``gridfm/pfdelta_task<id>`` when missing. ``--case case57`` / ``GENCO_TPGT_CASE=case57`` runs Task 1.1
train+test on IEEE 57 (``tpgt_task11_case57_<RUN_TAG>``, vendor ``pfdelta_task1.1_case57`` from MIT PFΔ JSON).
Training and validation use the union of the task's
train/valid folders (each cached and cut to ``DATA_FRACTION`` on its own, then merged); every task
is tested on N, N-1, N-2 and C2I. Task 1.1 keeps ``tpgt_task11_<RUN_TAG>``; other tasks write to
``tpgt_task<id>_<RUN_TAG>`` (results inside it). Each optimizer step takes the next 32 cases of
one global shuffle and runs them in one forward/backward: a single topology uses the shared
block-wise path, topologies with equal bus types (line outages) share it with per-case pair
embeddings and Ybus, and topologies with different bus types (generator outages) run per-case
type masks (``TypeLayout``). Device copies of the networks (``DeviceNetworks``) stay resident when
they fit the memory budget, otherwise in an LRU. When the train split has more topologies than
``MIX_TOPOLOGIES_THRESHOLD``, steps do not globally shuffle (that OOMs on N-2); same-bus-type
grids are packed up to ``PACK_MAX_TOPOLOGIES`` / ``PACK_MAX_CASES`` instead, and a chunk that
already has ``PACK_MAX_CASES`` cases stays one topology. Validation and test batch the same way.
A progress line (about once a minute, and every 50 steps) and the epoch line break the step time
into network moves, pair encoder, forward, backward and optimizer (``StepTimer``). Checkpoint
epochs also write a bar plot against the
published numbers (latest under ``results/``, plus a per-epoch copy under
``bar_plots/seed_*/epoch_*/``), IEEE 118 bus error maps, and every console
report (epoch/step lines, test PBL, forward/contingency diagnostics, comparison)
under ``seed_*/training_log.txt``, ``seed_*/metrics.jsonl``, and
``seed_*/reports/epoch_*/`` on Drive.
The hidden sizes, depth, batch, and epoch cap below are implementation
choices so a Colab GPU can finish a run. They are not paper hyperparameters.

IEEE 118 is meshed, so pair features use the full in-service topology:
complex effective impedance Re/Im (2 channels). See `meshed_rpe`.

Architecture (default ``v7``: checkpoint_version ``meshed-rpe-2-d4-v7``)
------------------------------------------------------------------------
* **Correction stages** (``DEPTH``), each with its own weights; the physics
  loss sums stage residuals with GENCO weights ``0.5**(K-i)`` (last stage 1).
  Default ``DEPTH=4``; preset ``v8`` / ``v9`` / ``v10`` use ``DEPTH=8`` with wider
  ``D/D_E/D_A/HEADS``.
* **Attention**: ``grit`` (default; v7–v9) uses pair-modulated GRIT scores on
  ``pair_encoder(RPE)``;   ``v10`` uses Approach C joint MLP scores on **raw** RPE
  ``r_ij`` (no pair encoder); ``v11`` uses Park et al. GRPE (shared hop/edge
  ``P/E`` in score and values); ``v11e`` adds stage ``U_Q/U_K/U_V`` on Zeff ``r_ij``;
  ``v12`` keeps the v10 MLP on an Exphormer phys∪expander∪virtual graph (15 pairs;
  dense ``-inf`` mask default; sparse via ``GENCO_TPGT_EXPHORMER_DENSE=0``).
  Q/K/V are type-pair-specific:
  ``q_{i|t_j}^{(ℓ,h)}=W_Q^{(ℓ,h,t_i←t_j)} h_i^{(ℓ-1)}``,
  ``k_{j|t_i}^{(ℓ,h)}=W_K^{(ℓ,h,t_i←t_j)} h_j^{(ℓ-1)}``.
  v10: ``s=MLP([q‖k‖r])``; v11: ``s=(q·k + q·(P^Q+E^Q) + k·(P^K+E^K))/√D_A`` with
  ``o=Σ α (v+P^V+E^V)``. Softmax and type-pair blocks are unchanged. Angle step
  uses ``η_angle=1``.
* **2-channel RPE**: ``Zeff_Re_norm``, ``Zeff_Im_norm`` (complex Laplacian effective impedance).
* **RPE channel dropout**: during training each channel is zeroed with
  probability ``RPE_CHANNEL_DROPOUT`` per optimizer step; never at eval.
* **Versioned run folder**: checkpoints go to ``Foundation Model/tpgt_task11_<RUN_TAG>``
  and tables/plot to ``Foundation Model/results/tpgt_task11_<RUN_TAG>``, so runs
  of older architectures (``tpgt_task11/``, ``tpgt_task11_v7/``) are left untouched.
  Select with ``GENCO_TPGT_ARCH=v8`` / ``--arch v8`` (same packing, residual, loss).

Performance notes (Colab / large train split)
---------------------------------------------
* **Split cache**: Stamps record ``DATA_FRACTION``, scenario limit, and train RPE scales.
  On Colab the Drive ``checkpoints/cache`` tree is mirrored to ``/content/*_cache`` once at
  startup (and again after a purge); split ``.pt`` files rebuild only when stamps mismatch
  (arch / ``checkpoint_version``, ``DATA_FRACTION``, dataset bytes, or ``CACHE_VERSION`` bump).
  Legacy stamps missing only ``data_fraction`` or per-split RPE fields are patched in place.
  Set ``GENCO_TPGT_REBUILD_CACHE=1`` to force a full rebuild. Loaded caches refresh
  ``network.rpe`` from branches + scales before use; batch size and optimizer settings do not
  affect split caches.
* **Drive I/O**: The 48600-case train/valid tensors live under the checkpoint
  tree on Google Drive. ``torch.load`` from Drive is often minutes slower than
  from ``/content`` local disk. On Colab, ``split_cache_dirs`` mirrors ``.pt``
  caches to ``/content/tpgt_task11_<RUN_TAG>_cache`` after the first Drive read or build.
* **``meshed_rpe``**: Computed only when building or refreshing the split cache,
  once per *distinct* in-service topology (see ``read_split`` representatives),
  not per scenario and not per training batch.
* **``pair_encoder``**: For GRIT arches, runs once per forward on the static
  ``118×118`` RPEs (stacked when several share bus types). For ``v10`` / ``v11`` /
  ``v11e`` / ``v12`` the encoder is ``Identity`` (raw ``r_ij`` path; GRPE also uses
  ``hop_id``/``edge_id``; Exphormer uses ``exph_allow``/edge lists). In ``eval`` mode a
  single-topology batch may cache pair features on ``Network.pair_embed``.
* **Type-pair attention**: ``CorrectionStage._messages`` evaluates each of the
  nine (or fifteen for Exphormer) type-pair weight sets only on its own
  receiver-type × sender-type block (both heads batched), then takes one softmax over
  allowed senders per receiver. With per-case bus types a block spans the buses of that
  type in any case and per-case masks keep each case's own pairs (the dense masked
  formulation restricted to those buses).
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
* **LR / grads**: Adam at ``LEARNING_RATE`` with a **cosine ceiling** over
  ``EPOCHS`` (lr is never above the cosine curve) and **ReduceLROnPlateau** on
  ``val_pbl`` when validation runs (halve lr after ``LR_PLATEAU_PATIENCE``
  checks without improvement; disable with ``GENCO_TPGT_LR_PLATEAU=0``).
  ``GRAD_CLIP_NORM`` on every optimizer step. Early stopping uses separate val
  patience. The epoch line shows ``lr``; scheduler state is checkpointed.
* **Debug (on by default)**: every validation pass also scores a fixed training
  subset (``train_pbl`` and ``gap`` in the epoch line). On checkpoint epochs the
  run prints the forward report, RPE zero-ablation, RPE and residual correctness
  audits on the validation topology, and contingency diagnostics on test N-1/N-2
  (per-topology PBL spread, audits and ablation on the largest outage topologies,
  RPE shift from the training topology). Disable with ``GENCO_TPGT_DEBUG=0`` or
  ``--no-debug``; skip RPE ablation with ``GENCO_TPGT_RPE_ABLATION=0``, bus-shunt
  (``busBS``) node ablation with ``GENCO_TPGT_SHUNT_ABLATION=0`` (v9 only).
  ``--eval-only --task 3.1`` scores Table 4 test splits without training: same
  split-cache prelude as ``main()`` (``reconcile_split_caches``, then ``load_task_split``
  train on ``pfdelta_task3.1``). When ``--checkpoint-root`` / ``GENCO_TPGT_CHECKPOINT_ROOT``
  points at another run (e.g. Task 1.1 weights), train RPE scales are taken from that
  run's cache instead of rebuilding Task 3.1 train caches. ``--diagnose`` runs the same on a saved checkpoint (``--no-contingency``,
  ``--no-rpe-ablation`` / ``--no-shunt-ablation`` to trim). ``--sanity-rpe`` audits one train case.
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
import plot_task1_1 as plot_task1_1_module
from plot_task1_1 import PUBLISHED, categories_for, plot_tpgt_comparison

write_task31_table = getattr(plot_task1_1_module, "write_task31_table", None)
display_task31_table = getattr(plot_task1_1_module, "display_task31_table", None)
from storage import ROOT, in_colab, mount_drive
import exphormer_tpgt as exph

# Implementation choices, not values from the draft.
D = 48
D_E = 24
D_A = 24
HEADS = 2
DEPTH = 4
ETA = 1.0
BATCH_SIZE = 32
# Adam start LR matches GENCO Task 1.1; cosine anneals it so late epochs take tinier steps.
# Grad clip caps single-batch explosions (epoch-7 style spikes). Early stop still on val PBL.
LEARNING_RATE = 5e-4
LR_COSINE_MIN_RATIO = 0.05  # final lr = LEARNING_RATE * this (2.5e-5 at defaults)
LR_PLATEAU_PATIENCE = 5  # validation passes without val_pbl improvement before halving lr
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_THRESHOLD = 1e-6  # min val_pbl improvement to reset plateau patience
GRAD_CLIP_NORM = 1.0  # 0 disables; clip after GradScaler.unscale_
EARLY_STOP_PATIENCE = 100
EPOCHS = 100
# Drive checkpoints (and results CSV / comparison plot) every 3rd epoch and at the final epoch.
# The four held-out test PBL scores run on checkpoint epochs (3, 6, …) and the final epoch.
CHECKPOINT_EVERY = 3
SEEDS = (0, 1, 2)
BASE_MVA = 100.0
LAMBDA_STATE = 0.5
LAMBDA_PHYS = 0.5
# Stage i of K gets physics weight PHYSICS_STAGE_DECAY**(K-i), as in GENCO
# (HGNS_PF_pfdelta_bs64_seed1.yaml): last stage 1, each earlier stage halved.
PHYSICS_STAGE_DECAY = 0.5
MODEL_NAME = "TPGT"
TYPE_NAMES = ("PQ", "PV", "Slack")
CACHE_VERSION = "tpgt-cache-12"
RPE_FEATURE_NAMES = ("Zeff_Re_norm", "Zeff_Im_norm")
RPE_DIM = len(RPE_FEATURE_NAMES)
# Per optimizer step, each RPE channel is zeroed (whole 118x118 plane, no rescaling) with this
# probability. Training only; validation, test and diagnostics always see the full RPE.
RPE_CHANNEL_DROPOUT = 0.1
# Bumped when pair-encoder input size or architecture changes; old Drive checkpoints are ignored.
CHECKPOINT_VERSION = "meshed-rpe-2-d4-v7"
# Checkpoints, split caches and result tables live in tpgt_task11_<RUN_TAG> so runs of an
# older architecture (e.g. the 2-stage, 10-channel run in tpgt_task11/) are never overwritten.
# Non-118 cases use ``tpgt_task11_<case>_<RUN_TAG>`` (e.g. ``tpgt_task11_case57_v9``).
RUN_TAG = "v7"
CASE_NAME = "case118"
RUN_DIR_NAME = f"tpgt_task11_{RUN_TAG}"
# Residual/state packing stays 2 known slots per type. v9 node encoders see all four PF
# quantities (P, Q, V, θ) with unknowns zeroed, plus bus-shunt B from parquet ``BS`` (device only).
NODE_SPECIFIED_DIM = 2
NODE_PQVT_DIM = 4
NODE_SHUNT_B_DIM = 1
NODE_FEATURE_DIM = NODE_PQVT_DIM + NODE_SHUNT_B_DIM  # 5 = (P,Q,V,θ,Bs)
NODE_ENCODER_FEATURE_NAMES = ("P", "Q", "V", "theta", "busBS")
USE_NODE_SHUNT = False
# ``grit``: ReLU(ρ(W_Q^{t_i←t_j} h_i + W_K^{t_i←t_j} h_j) ⊙ W_Ew e_ij + W_Eb e_ij) then linear score.
# ``joint_mlp``: q_{i|t_j}=W_Q^{t_i←t_j} h_i, k_{j|t_i}=W_K^{t_i←t_j} h_j;
#   s_ij = MLP_{t_i←t_j}^{ℓ,h}([q_{i|t_j} ‖ k_{j|t_i} ‖ r_ij]) on raw RPE (Approach C; v10).
# ``grpe``: Park et al. GRPE — shared hop/edge embeddings in score and values (v11).
# ``grpe_elec``: GRPE + stage-specific Zeff bilinear U_Q/U_K/U_V (v11e).
# ``exphormer``: v10 joint MLP on Exphormer phys∪expander∪virtual sparse graph (v12).
ATTENTION_KIND = "grit"
# Attention kinds that use Identity pair encoder (raw RPE still stored on Network for Zeff / elec).
RAW_RPE_ATTENTION_KINDS = frozenset({"joint_mlp", "grpe", "grpe_elec", "exphormer"})
GRPE_ATTENTION_KINDS = frozenset({"grpe", "grpe_elec"})
# Shortest-path hop bins: 0..GRPE_MAX_HOP plus one overflow bin (hop > max or unreachable).
GRPE_MAX_HOP = 8
GRPE_EDGE_SELF = 0
GRPE_EDGE_BRANCH = 1
GRPE_EDGE_NONADJ = 2
GRPE_NUM_EDGE_TYPES = 3
# Exphormer (v12): expander degree and dense-vs-sparse kernel.
# Dense -inf mask is the default for IEEE-118 speed (batched GEMMs). Sparse remains for large N.
EXPHORMER_D_EXP = 4
EXPHORMER_DENSE = True
EXPHORMER_NUM_PAIRS = 15


def grpe_num_hop_bins(max_hop: int | None = None) -> int:
    """Bins ``0..max_hop`` plus one overflow bin (``max_hop+1`` for hop>max or unreachable)."""
    limit = GRPE_MAX_HOP if max_hop is None else int(max_hop)
    return limit + 2


# Joint score MLP hidden widths (per type-pair, per head, per stage).
# Input is 2*D_A + RPE_DIM (=98 for D_A=48); output is a scalar. v10: 98 → 128 → 64 → 1.
ATTENTION_MLP_HIDDENS: tuple[int, ...] = (128, 64)
# Named width/depth presets (same math/loss/packing). ``GENCO_TPGT_ARCH`` / ``--arch`` selects one;
# optional ``GENCO_TPGT_D`` / ``DEPTH`` / ``D_E`` / ``D_A`` / ``HEADS`` / ``RUN_TAG`` override fields.
ARCH_PRESETS: dict[str, dict[str, int | str | bool | tuple[int, ...]]] = {
    "v7": {
        "D": 48,
        "D_E": 24,
        "D_A": 24,
        "HEADS": 2,
        "DEPTH": 4,
        "RUN_TAG": "v7",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d4-v7",
        "USE_NODE_SHUNT": False,
        "ATTENTION_KIND": "grit",
    },
    # Deeper iterative refine + wider type-pair attention; new Drive folder ``tpgt_task11_v8``.
    "v8": {
        "D": 96,
        "D_E": 48,
        "D_A": 48,
        "HEADS": 4,
        "DEPTH": 8,
        "RUN_TAG": "v8",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d8-w96-h4-v8",
        "USE_NODE_SHUNT": False,
        "ATTENTION_KIND": "grit",
    },
    # v8 width/depth + (P,Q,V,θ,Bs) node features; Zeff Re/Im RPE; folder ``tpgt_task11_v9``.
    "v9": {
        "D": 96,
        "D_E": 48,
        "D_A": 48,
        "HEADS": 4,
        "DEPTH": 8,
        "RUN_TAG": "v9",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d8-w96-h4-pqvt-busBS-v9",
        "USE_NODE_SHUNT": True,
        "ATTENTION_KIND": "grit",
    },
    # v9 node/RPE features with v7 depth (4 stages); separate ``RUN_TAG`` / checkpoint version.
    "v9d4": {
        "D": 96,
        "D_E": 48,
        "D_A": 48,
        "HEADS": 4,
        "DEPTH": 4,
        "RUN_TAG": "v9d4",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d4-w96-h4-pqvt-busBS-v9d4",
        "USE_NODE_SHUNT": True,
        "ATTENTION_KIND": "grit",
    },
    # v9 capacity/features with joint MLP pair attention on raw RPE (Approach C; no pair encoder).
    "v10": {
        "D": 96,
        "D_E": 48,  # unused by attention (kept for preset symmetry); pair_encoder is Identity.
        "D_A": 48,
        "HEADS": 2,
        "DEPTH": 8,
        "RUN_TAG": "v10",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d8-w96-h2-pqvt-busBS-mlpattn-128-64-rawrpe-v10",
        "USE_NODE_SHUNT": True,
        "ATTENTION_KIND": "joint_mlp",
        "ATTENTION_MLP_HIDDENS": (128, 64),
    },
    # Park et al. GRPE (shared hop/edge embeddings + graph-encoded values); HEADS=2.
    "v11": {
        "D": 96,
        "D_E": 48,  # unused by attention; pair_encoder is Identity.
        "D_A": 48,
        "HEADS": 2,
        "DEPTH": 8,
        "RUN_TAG": "v11",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d8-w96-h2-pqvt-busBS-grpe-full-v11",
        "USE_NODE_SHUNT": True,
        "ATTENTION_KIND": "grpe",
    },
    # v11 GRPE + electrical Zeff bilinear extension (stage-specific U_Q/U_K/U_V).
    "v11e": {
        "D": 96,
        "D_E": 48,
        "D_A": 48,
        "HEADS": 2,
        "DEPTH": 8,
        "RUN_TAG": "v11e",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d8-w96-h2-pqvt-busBS-grpe-elec-v11e",
        "USE_NODE_SHUNT": True,
        "ATTENTION_KIND": "grpe_elec",
    },
    # v10 joint MLP + Exphormer phys∪expander∪virtual attention graph (dense mask default).
    "v12": {
        "D": 96,
        "D_E": 48,
        "D_A": 48,
        "HEADS": 2,
        "DEPTH": 8,
        "RUN_TAG": "v12",
        "CHECKPOINT_VERSION": "meshed-rpe-2-d8-w96-h2-pqvt-busBS-exphormer-dense-v12",
        "USE_NODE_SHUNT": True,
        "ATTENTION_KIND": "exphormer",
        "ATTENTION_MLP_HIDDENS": (128, 64),
    },
}
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
ETA_ANGLE = 1.0

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
# When train has more topologies than this, steps do not use a global shuffle (that put up to
# BATCH_SIZE distinct N-2 grids in one backward and OOM'd). Same-bus-type grids are still packed,
# capped below, so a one-case N-2 topology is not its own optimizer step.
MIX_TOPOLOGIES_THRESHOLD = 64
# Caps for that pack. A chunk that already has PACK_MAX_CASES cases stays one topology (shared
# pair embedding, the path that fits at BATCH_SIZE). GENCO_TPGT_PACK_TOPOLOGIES=1 disables packing.
PACK_MAX_TOPOLOGIES = 8
PACK_MAX_CASES = 16
# Heartbeat during an epoch. The step timer's window resets on each line.
PROGRESS_EVERY_STEPS = 50
PROGRESS_EVERY_SECONDS = 60.0
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
    "3.1": TaskSpec(
        "3.1",
        "3.1",
        "3.1",
        "train IEEE 118 (N + N-1 + N-2); test IEEE 57, IEEE 118, and GOC 500",
        _folders(TOPOLOGIES, ("feasible",)),
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
# Task 3.1 keeps the IEEE 118 train/valid folders and adds these test grids under case118/test/.
TASK31_TEST_CASES = ("case57", "case118", "case500")


def eval_case_names(spec: TaskSpec | None = None) -> tuple[str, ...]:
    """Parquet case folder used at test time. Task 3.1 is multi-grid; others use ``CASE_NAME``."""
    spec = spec or TASKS["1.1"]
    if spec.task_id == "3.1":
        return TASK31_TEST_CASES
    return (CASE_NAME,)


def parse_case_name() -> str:
    """``--case case57`` / ``GENCO_TPGT_CASE``; default IEEE 118."""
    for index, arg in enumerate(sys.argv):
        if arg == "--case" and index + 1 < len(sys.argv):
            return sys.argv[index + 1].strip()
        if arg.startswith("--case="):
            return arg.split("=", 1)[1].strip()
    return os.environ.get("GENCO_TPGT_CASE", "").strip() or "case118"


def apply_case_config() -> str:
    """Set ``CASE_NAME`` and fold non-118 cases into ``RUN_DIR_NAME`` (e.g. ``tpgt_task11_case57_v9``)."""
    global CASE_NAME, RUN_DIR_NAME
    name = parse_case_name()
    if not name.startswith("case"):
        raise ValueError(f"case name must look like case57/case118, got {name!r}")
    CASE_NAME = name
    if CASE_NAME == "case118":
        RUN_DIR_NAME = f"tpgt_task11_{RUN_TAG}"
    else:
        RUN_DIR_NAME = f"tpgt_task11_{CASE_NAME}_{RUN_TAG}"
    return CASE_NAME


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


def wants_eval_only() -> bool:
    if "--eval-only" in sys.argv:
        return True
    return os.environ.get("GENCO_TPGT_EVAL_ONLY", "").strip().lower() in ("1", "true", "yes")


def parse_checkpoint_root(drive: Path | None) -> Path | None:
    """Weight directory (contains ``seed_*/last.pt``). Default: Task 1.1 folder for current ``RUN_TAG``."""
    for index, arg in enumerate(sys.argv):
        if arg == "--checkpoint-root" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
        if arg.startswith("--checkpoint-root="):
            return Path(arg.split("=", 1)[1])
    env = os.environ.get("GENCO_TPGT_CHECKPOINT_ROOT", "").strip()
    if env:
        return Path(env)
    if drive is not None:
        return drive / "Foundation Model" / RUN_DIR_NAME
    return None


def eval_checkpoint_payload(seed_dir: Path, device: torch.device) -> tuple[dict, int, Path] | None:
    """``best_state`` from ``last.pt`` when present, else ``model``."""
    last_path = seed_dir / "last.pt"
    payload = try_load_checkpoint_file(last_path, device)
    if payload is None:
        return None
    reason = checkpoint_version_reason(payload)
    if reason is not None:
        raise RuntimeError(f"{last_path}: {reason}")
    pair_mismatch = pair_encoder_input_mismatch(payload)
    if pair_mismatch is not None:
        raise RuntimeError(f"{last_path}: {pair_mismatch}")
    state = payload.get("best_state")
    if not isinstance(state, dict):
        state = payload.get("model")
    if not isinstance(state, dict):
        return None
    cpu_state = {key: value.cpu() if torch.is_tensor(value) else value for key, value in state.items()}
    return cpu_state, int(payload.get("epoch", 0)), last_path


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


@dataclass
class GrpePairFeatures:
    """Looked-up shared GRPE embeddings for one forward.

    Each tensor is ``(HEADS, N, N, D_A)`` (shared topology) or ``(B, HEADS, N, N, D_A)``.
    """

    p_q: torch.Tensor
    p_k: torch.Tensor
    p_v: torch.Tensor
    e_q: torch.Tensor
    e_k: torch.Tensor
    e_v: torch.Tensor


class GrpeEncodings(nn.Module):
    """Shared hop / edge-relation embeddings ``P^{Q,K,V}`` and ``E^{Q,K,V}`` (Park et al. GRPE)."""

    def __init__(self) -> None:
        super().__init__()
        hop_bins = grpe_num_hop_bins()
        self.p_q = nn.Parameter(torch.empty(hop_bins, HEADS, D_A))
        self.p_k = nn.Parameter(torch.empty(hop_bins, HEADS, D_A))
        self.p_v = nn.Parameter(torch.empty(hop_bins, HEADS, D_A))
        self.e_q = nn.Parameter(torch.empty(GRPE_NUM_EDGE_TYPES, HEADS, D_A))
        self.e_k = nn.Parameter(torch.empty(GRPE_NUM_EDGE_TYPES, HEADS, D_A))
        self.e_v = nn.Parameter(torch.empty(GRPE_NUM_EDGE_TYPES, HEADS, D_A))
        for parameter in (self.p_q, self.p_k, self.p_v, self.e_q, self.e_k, self.e_v):
            _init_matrices(parameter)

    @staticmethod
    def _lookup(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """``table[index]`` → ``(H,N,N,D_A)`` or ``(B,H,N,N,D_A)``."""
        emb = table[index]
        if emb.dim() == 4:
            return emb.permute(2, 0, 1, 3).contiguous()
        return emb.permute(0, 3, 1, 2, 4).contiguous()

    def forward(self, hop_id: torch.Tensor, edge_id: torch.Tensor) -> GrpePairFeatures:
        return GrpePairFeatures(
            p_q=self._lookup(self.p_q, hop_id),
            p_k=self._lookup(self.p_k, hop_id),
            p_v=self._lookup(self.p_v, hop_id),
            e_q=self._lookup(self.e_q, edge_id),
            e_k=self._lookup(self.e_k, edge_id),
            e_v=self._lookup(self.e_v, edge_id),
        )


def _gather_grpe_block(emb: torch.Tensor, receivers: torch.Tensor, senders: torch.Tensor) -> torch.Tensor:
    """Slice pair embeddings to a type-pair block: ``(H,Nr,Ns,D_A)`` or ``(B,H,Nr,Ns,D_A)``."""
    if emb.dim() == 4:
        return emb[:, receivers][:, :, senders]
    return emb[:, :, receivers][:, :, :, senders]


class CorrectionStage(nn.Module):
    """One unshared correction stage. Type-pair Q/K/V (9 or 15) with kind-specific scores.

    Attention kind is selected by ``ATTENTION_KIND`` at construction:
    * ``grit`` (default): pair-conditioned GRIT scores and edge-augmented values.
    * ``joint_mlp`` (v10): nine type-pair × head score MLPs on raw RPE.
    * ``grpe`` (v11): Park GRPE with shared hop/edge embeddings in score and values.
    * ``grpe_elec`` (v11e): GRPE plus stage-specific Zeff bilinear ``U_Q/U_K/U_V``.
    * ``exphormer`` (v12): 15 type-pairs on phys∪expander∪virtual graph (v10 MLP scores).
    """

    def __init__(self) -> None:
        super().__init__()
        self.attention_kind = ATTENTION_KIND
        n_pairs = EXPHORMER_NUM_PAIRS if self.attention_kind == "exphormer" else 9
        # Directed type-pairs × HEADS: each (pair, head) has its own Q/K/V (and score params).
        shape = (n_pairs, HEADS)
        self.w_q = nn.Parameter(torch.empty(*shape, D_A, D))
        self.w_k = nn.Parameter(torch.empty(*shape, D_A, D))
        self.w_v = nn.Parameter(torch.empty(*shape, D_A, D))
        self.w_o = nn.Parameter(torch.empty(HEADS, D, D_A))
        grit_params: list[nn.Parameter] = []
        mlp_params: list[nn.Parameter] = []
        grpe_params: list[nn.Parameter] = []
        # Defaults: absent unless the active attention kind needs them.
        self.score_w1 = self.score_b1 = self.score_w2 = self.score_b2 = None
        self.score_w3 = self.score_b3 = None
        self.u_q = self.u_k = self.u_v = None
        self.w_ew = self.w_eb = self.w_a = self.w_ev = None
        self.edge_bias = None
        self.update_virt = None
        self.norm_virt = None
        if self.attention_kind in ("joint_mlp", "exphormer"):
            h1, h2 = (int(ATTENTION_MLP_HIDDENS[0]), int(ATTENTION_MLP_HIDDENS[1]))
            # Raw RPE channels (Zeff Re/Im), not pair_encoder(D_E) embeddings.
            # Score MLP: (2*D_A+RPE_DIM) → h1 → h2 → 1  (v10 default: 98 → 128 → 64 → 1).
            score_in = 2 * D_A + RPE_DIM
            self.score_w1 = nn.Parameter(torch.empty(*shape, h1, score_in))
            self.score_b1 = nn.Parameter(torch.zeros(*shape, h1))
            self.score_w2 = nn.Parameter(torch.empty(*shape, h2, h1))
            self.score_b2 = nn.Parameter(torch.zeros(*shape, h2))
            self.score_w3 = nn.Parameter(torch.empty(*shape, 1, h2))
            self.score_b3 = nn.Parameter(torch.zeros(*shape, 1))
            mlp_params = [self.score_w1, self.score_w2, self.score_w3]
            if self.attention_kind == "exphormer":
                self.edge_bias = nn.Parameter(torch.zeros(exph.NUM_EDGE_TAU, HEADS))
                self.update_virt = MLP(D + D, D)
                self.norm_virt = nn.LayerNorm(D)
        elif self.attention_kind == "grpe":
            # Shared P/E live on TPGT.grpe_encodings; stage keeps type-pair Q/K/V only.
            pass
        elif self.attention_kind == "grpe_elec":
            # Stage-specific electrical extension on raw Zeff RPE.
            self.u_q = nn.Parameter(torch.empty(*shape, D_A, RPE_DIM))
            self.u_k = nn.Parameter(torch.empty(*shape, D_A, RPE_DIM))
            self.u_v = nn.Parameter(torch.empty(*shape, D_A, RPE_DIM))
            grpe_params = [self.u_q, self.u_k, self.u_v]
        else:
            self.w_ew = nn.Parameter(torch.empty(*shape, D_A, D_E))
            self.w_eb = nn.Parameter(torch.empty(*shape, D_A, D_E))
            self.w_a = nn.Parameter(torch.empty(*shape, 1, D_A))
            self.w_ev = nn.Parameter(torch.empty(*shape, D_A, D_A))
            grit_params = [self.w_ew, self.w_eb, self.w_a, self.w_ev]
        for parameter in (self.w_q, self.w_k, self.w_v, self.w_o, *grit_params, *mlp_params, *grpe_params):
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
        grpe: GrpePairFeatures | None = None,
        z_virt: torch.Tensor | None = None,
        exph_allow: torch.Tensor | None = None,
        exph_tau: torch.Tensor | None = None,
        virt_rpe: torch.Tensor | None = None,
        types: torch.Tensor | None = None,
        exph_recv: torch.Tensor | None = None,
        exph_send: torch.Tensor | None = None,
        exph_edge_tau: torch.Tensor | None = None,
        exph_pair_id: torch.Tensor | None = None,
        use_dense_exph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_in = nodal_residual(specified, unknown, layout, ybus)
        z_next: torch.Tensor | None = None
        if self.attention_kind == "exphormer":
            if z_virt is None or exph_allow is None or exph_tau is None or virt_rpe is None or types is None:
                raise ValueError("exphormer stage requires z_virt, allow/tau, virt_rpe, and types")
            message, z_next = self._messages_exphormer(
                hidden,
                z_virt,
                pair,
                types,
                exph_allow,
                exph_tau,
                virt_rpe,
                exph_recv=exph_recv,
                exph_send=exph_send,
                exph_edge_tau=exph_edge_tau,
                exph_pair_id=exph_pair_id,
                use_dense=use_dense_exph,
            )
        else:
            message = self._messages(hidden, pair, layout.blocks, grpe=grpe)
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
        if z_next is not None:
            return hidden_next, unknown_next, residual_out, z_next
        return hidden_next, unknown_next, residual_out

    def _messages_exphormer(
        self,
        hidden: torch.Tensor,
        z_virt: torch.Tensor,
        pair: torch.Tensor,
        types: torch.Tensor,
        allow: torch.Tensor,
        tau: torch.Tensor,
        virt_rpe: torch.Tensor,
        *,
        exph_recv: torch.Tensor | None,
        exph_send: torch.Tensor | None,
        exph_edge_tau: torch.Tensor | None,
        exph_pair_id: torch.Tensor | None,
        use_dense: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.score_w1 is not None and self.edge_bias is not None
        assert self.update_virt is not None and self.norm_virt is not None
        n_virt = int(z_virt.shape[1])
        types_aug = exph.augmented_types(types, n_virt)
        pair_aug = exph.build_exphormer_pair_features(pair, allow, tau, types_aug, virt_rpe)
        kwargs = dict(
            w_q=self.w_q,
            w_k=self.w_k,
            w_v=self.w_v,
            w_o=self.w_o,
            score_w1=self.score_w1,
            score_b1=self.score_b1,
            score_w2=self.score_w2,
            score_b2=self.score_b2,
            score_w3=self.score_w3,
            score_b3=self.score_b3,
            edge_bias=self.edge_bias,
        )
        if use_dense or exph_recv is None:
            msg_r, msg_v = exph.dense_exphormer_messages(
                hidden, z_virt, pair_aug, types_aug, allow, tau, **kwargs
            )
        else:
            msg_r, msg_v = exph.sparse_exphormer_messages(
                hidden,
                z_virt,
                pair_aug,
                types_aug,
                exph_recv,
                exph_send,
                exph_edge_tau,
                exph_pair_id,
                **kwargs,
            )
        # Virt update MLP stays under ambient autocast; match message dtype to z_virt.
        msg_v = msg_v.to(dtype=z_virt.dtype)
        msg_r = msg_r.to(dtype=hidden.dtype)
        z_next = self.norm_virt(z_virt + self.update_virt(torch.cat((z_virt, msg_v), dim=-1)))
        return msg_r, z_next


    def _messages(
        self,
        hidden: torch.Tensor,
        pair: torch.Tensor,
        blocks: list[TypePairBlock],
        grpe: GrpePairFeatures | None = None,
    ) -> torch.Tensor:
        """Each type pair's weights see only its own receiver-type x sender-type block.

        Block scores are scattered into one (batch, heads, N, N) score matrix so the softmax
        runs over all senders of a receiver; cross-component entries stay -inf. ``pair`` is either
        shared by the batch (N, N, E) or per case (batch, N, N, E). Blocks with per-case masks
        (bus types differing per case) may overlap; each entry of a case is set by the one block of
        its types and weighted only there.
        """
        if self.attention_kind == "joint_mlp":
            return self._messages_joint_mlp(hidden, pair, blocks)
        if self.attention_kind in GRPE_ATTENTION_KINDS:
            if grpe is None:
                raise ValueError(f"{self.attention_kind} attention requires shared GrpePairFeatures")
            return self._messages_grpe(hidden, pair, blocks, grpe)
        return self._messages_grit(hidden, pair, blocks)

    def _scatter_block_scores(
        self,
        scores: torch.Tensor | None,
        score: torch.Tensor,
        block: TypePairBlock,
        batch: int,
        buses: int,
    ) -> torch.Tensor:
        receivers = block.receivers.view(-1, 1)
        senders = block.senders.view(1, -1)
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
        return scores

    def _messages_grit(
        self,
        hidden: torch.Tensor,
        pair: torch.Tensor,
        blocks: list[TypePairBlock],
    ) -> torch.Tensor:
        batch, buses, _ = hidden.shape
        scores: torch.Tensor | None = None
        interactions = []
        assert self.w_ew is not None and self.w_eb is not None
        assert self.w_a is not None and self.w_ev is not None
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
            scores = self._scatter_block_scores(scores, score, block, batch, buses)
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

    def _messages_joint_mlp(
        self,
        hidden: torch.Tensor,
        pair: torch.Tensor,
        blocks: list[TypePairBlock],
    ) -> torch.Tensor:
        """Approach C on raw RPE ``pair``.

        Type-pair Q/K (receiver ``i``, sender ``j``):
        ``q_{i|t_j}^{h}=W_Q^{h,t_i←t_j} h_i``, ``k_{j|t_i}^{h}=W_K^{h,t_i←t_j} h_j``,
        then ``s_ij^{h}=MLP_{t_i←t_j}^{h}([q_{i|t_j} ‖ k_{j|t_i} ‖ r_ij])``
        with widths ``ATTENTION_MLP_HIDDENS`` (default 98→128→64→1).
        """
        batch, buses, _ = hidden.shape
        scores: torch.Tensor | None = None
        assert self.score_w1 is not None and self.score_b1 is not None
        assert self.score_w2 is not None and self.score_b2 is not None
        assert self.score_w3 is not None and self.score_b3 is not None
        if pair.shape[-1] != RPE_DIM:
            raise ValueError(
                f"joint MLP attention expects raw RPE last-dim {RPE_DIM}, got {tuple(pair.shape)}"
            )
        for block in blocks:
            p = block.pair_id
            receivers = block.receivers.view(-1, 1)
            senders = block.senders.view(1, -1)
            query = torch.einsum("bnd,had->bhna", hidden[:, block.receivers], self.w_q[p])
            key = torch.einsum("bnd,had->bhna", hidden[:, block.senders], self.w_k[p])
            if pair.dim() == 3:
                edge = pair[receivers, senders].unsqueeze(0).unsqueeze(1)
                edge = edge.expand(batch, HEADS, -1, -1, -1)
            else:
                edge = pair[:, receivers, senders].unsqueeze(1).expand(-1, HEADS, -1, -1, -1)
            # (B,H,Nr,Ns,2*D_A+RPE_DIM): one score MLP per type-pair (p) and head.
            joint = torch.cat(
                (
                    query.unsqueeze(-2).expand(-1, -1, -1, key.shape[-2], -1),
                    key.unsqueeze(-3).expand(-1, -1, query.shape[-2], -1, -1),
                    edge,
                ),
                dim=-1,
            )
            h1 = torch.relu(
                torch.einsum("bhnms,has->bhnma", joint, self.score_w1[p])
                + self.score_b1[p].view(1, HEADS, 1, 1, -1)
            )
            # score_w2[p]: (HEADS, h2, h1); avoid reusing subscript ``b`` (batch).
            h2 = torch.relu(
                torch.einsum("bhnma,hca->bhnmc", h1, self.score_w2[p])
                + self.score_b2[p].view(1, HEADS, 1, 1, -1)
            )
            score = torch.einsum("bhnmc,hc->bhnm", h2, self.score_w3[p, :, 0])
            score = score + self.score_b3[p].view(1, HEADS, 1, 1)
            scores = self._scatter_block_scores(scores, score, block, batch, buses)
        if scores is None:
            return hidden.new_zeros(hidden.shape)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        row_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)
        mixed_heads = weights.new_zeros((batch, HEADS, buses, D_A))
        for block in blocks:
            p = block.pair_id
            block_weights = weights[:, :, block.receivers.view(-1, 1), block.senders.view(1, -1)]
            if block.mask.dim() == 3:
                block_weights = block_weights.masked_fill(~block.mask.unsqueeze(1), 0.0)
            value = torch.einsum("bnd,had->bhna", hidden[:, block.senders], self.w_v[p])
            gathered = torch.einsum("bhnm,bhma->bhna", block_weights, value)
            mixed_heads = mixed_heads.index_add(2, block.receivers, gathered.to(mixed_heads.dtype))
        return torch.einsum("bhna,hda->bnd", mixed_heads, self.w_o)

    def _messages_grpe(
        self,
        hidden: torch.Tensor,
        pair: torch.Tensor,
        blocks: list[TypePairBlock],
        grpe: GrpePairFeatures,
    ) -> torch.Tensor:
        """Park et al. GRPE score/values; optional electrical Zeff terms when ``grpe_elec``.

        Shared topology/edge embeddings:
        ``s_ij = (q·k + q·(P^Q+E^Q) + k·(P^K+E^K)) / √D_A``,
        ``o_i = Σ_j α_ij (v_j + P^V + E^V)``.

        ``grpe_elec`` adds stage ``U_Q/U_K/U_V`` bilinear terms on raw RPE ``pair``.
        """
        batch, buses, _ = hidden.shape
        scores: torch.Tensor | None = None
        electrical = self.attention_kind == "grpe_elec"
        if electrical:
            assert self.u_q is not None and self.u_k is not None and self.u_v is not None
            if pair.shape[-1] != RPE_DIM:
                raise ValueError(
                    f"grpe_elec expects raw RPE last-dim {RPE_DIM}, got {tuple(pair.shape)}"
                )
        scale = float(D_A) ** 0.5
        for block in blocks:
            p = block.pair_id
            receivers = block.receivers.view(-1, 1)
            senders = block.senders.view(1, -1)
            query = torch.einsum("bnd,had->bhna", hidden[:, block.receivers], self.w_q[p])
            key = torch.einsum("bnd,had->bhna", hidden[:, block.senders], self.w_k[p])
            pq = _gather_grpe_block(grpe.p_q, block.receivers, block.senders)
            pk = _gather_grpe_block(grpe.p_k, block.receivers, block.senders)
            eq = _gather_grpe_block(grpe.e_q, block.receivers, block.senders)
            ek = _gather_grpe_block(grpe.e_k, block.receivers, block.senders)
            if pq.dim() == 4:
                pq = pq.unsqueeze(0).expand(batch, -1, -1, -1, -1)
                pk = pk.unsqueeze(0).expand(batch, -1, -1, -1, -1)
                eq = eq.unsqueeze(0).expand(batch, -1, -1, -1, -1)
                ek = ek.unsqueeze(0).expand(batch, -1, -1, -1, -1)
            topo_q = pq + eq
            topo_k = pk + ek
            node_dot = torch.einsum("bhna,bhma->bhnm", query, key)
            q_topo = torch.einsum("bhra,bhrsa->bhrs", query, topo_q)
            k_topo = torch.einsum("bhsa,bhrsa->bhrs", key, topo_k)
            score = (node_dot + q_topo + k_topo) / scale
            if electrical:
                if pair.dim() == 3:
                    edge = pair[receivers, senders].unsqueeze(0).unsqueeze(1)
                    edge = edge.expand(batch, HEADS, -1, -1, -1)
                else:
                    edge = pair[:, receivers, senders].unsqueeze(1).expand(-1, HEADS, -1, -1, -1)
                uk_e = torch.einsum("bhnme,hae->bhnma", edge, self.u_k[p])
                uq_e = torch.einsum("bhnme,hae->bhnma", edge, self.u_q[p])
                q_edge = torch.einsum("bhna,bhnma->bhnm", query, uk_e)
                k_edge = torch.einsum("bhma,bhnma->bhnm", key, uq_e)
                score = score + (q_edge + k_edge) / scale
            scores = self._scatter_block_scores(scores, score, block, batch, buses)
        if scores is None:
            return hidden.new_zeros(hidden.shape)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        row_sum = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)
        mixed_heads = weights.new_zeros((batch, HEADS, buses, D_A))
        for block in blocks:
            p = block.pair_id
            receivers = block.receivers.view(-1, 1)
            senders = block.senders.view(1, -1)
            block_weights = weights[:, :, receivers, senders]
            if block.mask.dim() == 3:
                block_weights = block_weights.masked_fill(~block.mask.unsqueeze(1), 0.0)
            value = torch.einsum("bnd,had->bhna", hidden[:, block.senders], self.w_v[p])
            pv = _gather_grpe_block(grpe.p_v, block.receivers, block.senders)
            ev = _gather_grpe_block(grpe.e_v, block.receivers, block.senders)
            if pv.dim() == 4:
                pv = pv.unsqueeze(0).expand(batch, -1, -1, -1, -1)
                ev = ev.unsqueeze(0).expand(batch, -1, -1, -1, -1)
            # (B,H,Nr,Ns,D_A): v_j + P^V_ij + E^V_ij (+ U_V r for grpe_elec).
            value_ij = value.unsqueeze(-3) + pv + ev
            if electrical:
                if pair.dim() == 3:
                    edge = pair[receivers, senders].unsqueeze(0).unsqueeze(1)
                    edge = edge.expand(batch, HEADS, -1, -1, -1)
                else:
                    edge = pair[:, receivers, senders].unsqueeze(1).expand(-1, HEADS, -1, -1, -1)
                uv_e = torch.einsum("bhnme,hae->bhnma", edge, self.u_v[p])
                value_ij = value_ij + uv_e
            gathered = torch.einsum("bhnm,bhnma->bhna", block_weights, value_ij)
            mixed_heads = mixed_heads.index_add(2, block.receivers, gathered.to(mixed_heads.dtype))
        return torch.einsum("bhna,hda->bnd", mixed_heads, self.w_o)


def bus_shunt_gb(ybus: torch.Tensor) -> torch.Tensor:
    """Per-bus shunt+charging ``(G, B)`` from the Ybus row-sum (series branch terms cancel).

    ``ybus`` is shared ``(N, N)`` or per case ``(B, N, N)`` → ``(N, 2)`` or ``(B, N, 2)``.
    """
    row = ybus.sum(dim=-1)
    return torch.stack((row.real, row.imag), dim=-1)


def bus_shunt_b(ybus: torch.Tensor) -> torch.Tensor:
    """``Im(∑_j Y_ij)``: all to-ground susceptance in Y (bus shunt **and** line charging).

    Prefer parquet ``BS`` for the node feature (device-only). This helper is for diagnostics.
    """
    return ybus.sum(dim=-1).imag


def network_shunt_b(network: Network) -> torch.Tensor:
    """Bus-shunt susceptance ``BS`` (p.u.); zeros if the network was built without it."""
    shunt = getattr(network, "shunt_b", None)
    if shunt is None:
        return torch.zeros(int(network.types.shape[-1]), dtype=torch.float32, device=network.types.device)
    return shunt


def node_encoder_features(
    specified: torch.Tensor,
    layout: TypeLayout,
    shunt_b: torch.Tensor,
) -> torch.Tensor:
    """``(P, Q, V, θ, B_s)`` per bus for type encoders.

    Unknown slots use the same ``flat_start`` as the physics iterate (PQ ``V=1`` pu, other
  free variables 0) so attention does not see ``V=0`` on PQ while ``nodal_residual`` starts at 1 pu.
    ``B_s`` is bus shunt device only (parquet ``BS``).
    """
    power_p, power_q, voltage, angle = assemble(specified, flat_start(specified, layout), layout)
    bs = shunt_b.to(device=specified.device, dtype=specified.dtype)
    if bs.dim() == 1:
        bs = bs.unsqueeze(0).expand(specified.shape[0], -1)
    return torch.stack((power_p, power_q, voltage, angle, bs), dim=-1)


class TPGT(nn.Module):
    """Fixed-depth type-pair graph Transformer with independently learned stages."""

    def __init__(self) -> None:
        super().__init__()
        node_in = NODE_FEATURE_DIM if USE_NODE_SHUNT else NODE_SPECIFIED_DIM
        # v9: (P,Q,V,θ,Bs) for all types. Else: packed specified (2). No ground node.
        self.encoders = nn.ModuleList(MLP(node_in, D) for _ in range(3))
        self.use_node_shunt = USE_NODE_SHUNT
        self.attention_kind = ATTENTION_KIND
        # v10/v11/v11e/v12 consume raw RPE path; GRIT still uses a learned pair encoder.
        self.use_raw_rpe = ATTENTION_KIND in RAW_RPE_ATTENTION_KINDS
        self.pair_encoder: nn.Module = nn.Identity() if self.use_raw_rpe else MLP(RPE_DIM, D_E)
        self.grpe_encodings: GrpeEncodings | None = (
            GrpeEncodings() if ATTENTION_KIND in GRPE_ATTENTION_KINDS else None
        )
        self.theta_u: nn.Parameter | None = None
        self.virt_rpe: nn.Parameter | None = None
        self.use_dense_exph = EXPHORMER_DENSE
        if ATTENTION_KIND == "exphormer":
            self.theta_u = nn.Parameter(torch.empty(D))
            self.virt_rpe = nn.Parameter(torch.empty(6, RPE_DIM))
            nn.init.xavier_uniform_(self.theta_u.view(1, -1))
            nn.init.xavier_uniform_(self.virt_rpe)
        self.stages = nn.ModuleList(CorrectionStage() for _ in range(DEPTH))

    def encode_pairs(self, rpe: torch.Tensor) -> torch.Tensor:
        """Features for correction-stage attention (raw RPE for v10/v11/v12, else ``pair_encoder``)."""
        return rpe if self.use_raw_rpe else self.pair_encoder(rpe)

    def encode_grpe(
        self,
        hop_id: torch.Tensor,
        edge_id: torch.Tensor,
    ) -> GrpePairFeatures | None:
        """Lookup shared hop/edge embeddings when attention is GRPE."""
        if self.grpe_encodings is None:
            return None
        return self.grpe_encodings(hop_id, edge_id)

    def forward(
        self,
        specified: torch.Tensor,
        types: torch.Tensor,
        rpe: torch.Tensor,
        pair_type_masks: torch.Tensor,
        ybus: torch.Tensor,
        pair: torch.Tensor | None = None,
        layout: TypeLayout | None = None,
        shunt_b: torch.Tensor | None = None,
        hop_id: torch.Tensor | None = None,
        edge_id: torch.Tensor | None = None,
        grpe: GrpePairFeatures | None = None,
        exph_allow: torch.Tensor | None = None,
        exph_tau: torch.Tensor | None = None,
        exph_n_virt: int | None = None,
        exph_recv: torch.Tensor | None = None,
        exph_send: torch.Tensor | None = None,
        exph_edge_tau: torch.Tensor | None = None,
        exph_pair_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``types`` (N,) with ``pair_type_masks`` (9, N, N), or per case (B, N) with (B, 9, N, N).

        ``layout`` (precomputed by ``move_to`` / ``PassBatch``) replaces ``types`` and the masks.
        ``shunt_b`` is bus-shunt ``BS`` (N,) or (B, N); required for v9 node features.
        ``hop_id`` / ``edge_id`` (or precomputed ``grpe``) required for ``grpe`` / ``grpe_elec``.
        Exphormer needs ``exph_allow`` / ``exph_tau`` / ``exph_n_virt``.
        """
        if pair is None:
            pair = self.encode_pairs(rpe)
        if layout is None:
            if pair_type_masks is None:
                raise ValueError("TPGT forward needs pair_type_masks or a layout")
            layout = type_layout(types, pair_type_masks)
        if grpe is None and self.grpe_encodings is not None:
            if hop_id is None or edge_id is None:
                raise ValueError("GRPE attention requires hop_id and edge_id (or grpe=...)")
            grpe = self.encode_grpe(hop_id, edge_id)
        # Exphormer attention is float32-only; keep activations in float32 even if AMP is on.
        act_dtype = (
            torch.float32
            if self.attention_kind == "exphormer"
            else autocast_activation_dtype(specified)
        )
        hidden = torch.zeros(
            specified.shape[0],
            specified.shape[1],
            D,
            device=specified.device,
            dtype=act_dtype,
        )
        unknown = flat_start(specified, layout)
        if self.use_node_shunt:
            if shunt_b is None:
                shunt_b = specified.new_zeros(specified.shape[:2])
            node_feats = node_encoder_features(specified, layout, shunt_b).to(dtype=act_dtype)
        else:
            node_feats = specified
        for type_id, encoder in enumerate(self.encoders):
            if not layout.present[type_id]:
                continue
            hidden = write_type(hidden, layout, type_id, encoder(gather_type(node_feats, layout, type_id)))
        z_virt: torch.Tensor | None = None
        if self.attention_kind == "exphormer":
            if exph_allow is None or exph_tau is None or exph_n_virt is None or exph_n_virt <= 0:
                raise ValueError("exphormer attention requires exph_allow, exph_tau, and exph_n_virt")
            assert self.theta_u is not None and self.virt_rpe is not None
            z_virt = self.theta_u.to(dtype=act_dtype).view(1, 1, -1).expand(
                specified.shape[0], int(exph_n_virt), -1
            ).clone()
        residuals = []
        shared_types = types if types.dim() == 1 else layout.types[0]
        for stage in self.stages:
            if self.attention_kind == "exphormer":
                hidden, unknown, residual, z_virt = stage(
                    hidden,
                    unknown,
                    specified,
                    pair,
                    layout,
                    ybus,
                    z_virt=z_virt,
                    exph_allow=exph_allow,
                    exph_tau=exph_tau,
                    virt_rpe=self.virt_rpe,
                    types=shared_types,
                    exph_recv=exph_recv,
                    exph_send=exph_send,
                    exph_edge_tau=exph_edge_tau,
                    exph_pair_id=exph_pair_id,
                    use_dense_exph=self.use_dense_exph,
                )
            else:
                hidden, unknown, residual = stage(
                    hidden, unknown, specified, pair, layout, ybus, grpe=grpe
                )
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


def state_mae_by_type(
    prediction: torch.Tensor,
    target: torch.Tensor,
    types: torch.Tensor | TypeLayout,
) -> dict[str, float]:
    """Mean absolute error on unknown slots (same packing as pack_case)."""
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
        out[name] = float(torch.mean(diff.abs()))
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
    if getattr(model, "use_raw_rpe", False) or isinstance(model.pair_encoder, nn.Identity):
        return [{"name": name, "weight_l2": 0.0} for name in RPE_FEATURE_NAMES]
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
    *,
    shunt_b: torch.Tensor | None = None,
) -> dict[str, float]:
    """Loss and final-stage PBL on a fixed batch with precomputed pair embeddings."""
    model.eval()
    chosen = np.asarray(index, dtype=np.int64)
    power_p = split.power_p[chosen].to(device, non_blocking=True)
    power_q = split.power_q[chosen].to(device, non_blocking=True)
    voltage = split.voltage[chosen].to(device, non_blocking=True)
    angle = split.angle[chosen].to(device, non_blocking=True)
    specified, target = pack_case(power_p, power_q, voltage, angle, network.types)
    if shunt_b is None:
        shunt_b = network_shunt_b(network)
    hop_id, edge_id = ensure_network_grpe_ids(network)
    with torch.no_grad():
        prediction, residuals = model(
            specified,
            network.types,
            network.rpe,
            network.pair_type_masks,
            network.ybus,
            pair=pair,
            shunt_b=shunt_b,
            hop_id=hop_id,
            edge_id=edge_id,
            **exphormer_forward_kwargs(network),
        )
        breakdown = task_loss(prediction, target, residuals, network.types)
        final_pbl = pbl_stats(residuals[-1])
    return {
        "pbl_mean": float(final_pbl["mean"]),
        "state_mse": float(breakdown.state_mse),
        "total_loss": float(breakdown.total),
    }


def type_encoder_input_weight_norms(model: TPGT) -> list[dict[str, float]]:
    """L2 norm of each node-feature column on the first Linear layer (PQ/PV/slack encoders)."""
    if not model.use_node_shunt:
        return []
    names = NODE_ENCODER_FEATURE_NAMES
    rows: list[dict[str, float]] = []
    for type_id, encoder in enumerate(model.encoders):
        first = encoder.net[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("type encoder first layer must be Linear")
        norms = torch.linalg.vector_norm(first.weight.detach(), dim=0)
        type_name = TYPE_NAMES[type_id] if type_id < len(TYPE_NAMES) else str(type_id)
        for channel in range(min(int(norms.numel()), len(names))):
            rows.append(
                {
                    "type": type_name,
                    "name": names[channel],
                    "weight_l2": float(norms[channel]),
                }
            )
    return rows


def collect_bus_shunt_ablation(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float] | None, list[dict[str, float]] | None]:
    """Zero parquet device-shunt ``busBS`` in node encoders; compare batch metrics to the baseline."""
    if not model.use_node_shunt:
        return None, None
    model.eval()
    clear_pair_embed_cache([network])
    pair = pair_features(model, network)
    shunt_b = network_shunt_b(network).to(device)
    baseline = eval_batch_with_pair(model, split, network, index, device, pair, shunt_b=shunt_b)
    zero_shunt = torch.zeros_like(shunt_b)
    metrics = eval_batch_with_pair(model, split, network, index, device, pair, shunt_b=zero_shunt)
    row = {
        "name": "busBS",
        "delta_pbl_mean": metrics["pbl_mean"] - baseline["pbl_mean"],
        "delta_state_mse": metrics["state_mse"] - baseline["state_mse"],
        "delta_total_loss": metrics["total_loss"] - baseline["total_loss"],
        "mean_abs_bs": float(shunt_b.abs().mean()),
    }
    clear_pair_embed_cache([network])
    return baseline, [row]


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
        ablated_pair = model.encode_pairs(rpe_ablated)
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
    if ATTENTION_KIND in GRPE_ATTENTION_KINDS:
        lines = [
            f"Pair-encoder weights: skipped ({ATTENTION_KIND}: Identity pair path; "
            "topology via shared hop/edge embeddings"
            + ("; Zeff r_ij via stage U_*" if ATTENTION_KIND == "grpe_elec" else "")
            + ")."
        ]
    elif ATTENTION_KIND in RAW_RPE_ATTENTION_KINDS:
        lines = [
            f"Pair-encoder weights: skipped ({ATTENTION_KIND} attention uses raw RPE r_ij; no pair encoder)."
        ]
    else:
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


def format_bus_shunt_ablation_diagnostics(
    weight_rows: list[dict[str, float]],
    baseline: dict[str, float] | None,
    ablation_rows: list[dict[str, float]] | None,
) -> str:
    if not weight_rows and baseline is None:
        return "Bus-shunt node feature (busBS): not used (arch without device BS channel)."
    lines = ["Type-encoder input weight L2 per node feature (structural; not causal):"]
    for row in sorted(weight_rows, key=lambda item: item["weight_l2"], reverse=True):
        lines.append(
            f"  {row['type']:5s} {row['name']:6s} weight_l2 {row['weight_l2']:.4g}"
        )
    if baseline is None or ablation_rows is None:
        lines.append(
            "Bus-shunt zero-ablation: not run (v9 debug; use --diagnose or GENCO_TPGT_SHUNT_ABLATION=1)."
        )
        return "\n".join(lines)
    lines.append(
        f"Bus-shunt (busBS) zero-ablation on same batch (baseline PBL mean {baseline['pbl_mean']:.6g}, "
        f"state_mse {baseline['state_mse']:.6g}):"
    )
    lines.append(
        "  Positive ΔPBL after zeroing busBS => channel helped; negative => zeroing improved metrics."
    )
    for row in ablation_rows:
        extra = (
            f"  mean |BS| {row['mean_abs_bs']:.4g}  "
            if "mean_abs_bs" in row
            else "  "
        )
        lines.append(
            f"  {row['name']:6s} {extra}ΔPBL {row['delta_pbl_mean']:+.4g}  "
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
    topk_pbl_share: float
    angle_rmse_high_y: float
    angle_rmse_rest: float
    angle_rmse_worst: float
    genco_n_gap: float


@dataclass
class ForwardDiagnostics:
    oracle_pbl: dict[str, float]
    flat_start_pbl: dict[str, float]
    model_pbl: dict[str, float]
    stage_pbl: list[dict[str, float]]
    state_rmse: dict[str, float]
    state_mae: dict[str, float]
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

    error = wrap_angle_errors((prediction - target).clone(), layout)
    per_bus_abs_v = error[..., 0].abs().mean(dim=0).detach().cpu().numpy()
    per_bus_abs_theta = error[..., 1].abs().mean(dim=0).detach().cpu().numpy()

    order = np.argsort(-mean_bus)[: min(top_k, mean_bus.size)]
    worst: list[dict[str, float | int | str]] = []
    for bus in order:
        type_id = int(types_np[bus])
        row: dict[str, float | int | str] = {
            "bus_1based": int(bus) + 1,
            "type": TYPE_NAMES[type_id] if 0 <= type_id < len(TYPE_NAMES) else str(type_id),
            "mean_pbl": float(mean_bus[bus]),
            "mean_|dP|": float(abs_dp[bus]),
            "mean_|dQ|": float(abs_dq[bus]),
            "|Yii|": float(y_abs_diag[bus]),
            "hops_to_slack": float(hops[bus]) if np.isfinite(hops[bus]) else -1.0,
        }
        if type_id == 0:
            row["mae_V_pu"] = float(per_bus_abs_v[bus])
            row["mae_theta_rad"] = float(per_bus_abs_theta[bus])
        elif type_id == 1:
            row["mae_theta_rad"] = float(per_bus_abs_theta[bus])
        else:
            row["mae_P_pu"] = float(per_bus_abs_v[bus])
            row["mae_Q_pu"] = float(per_bus_abs_theta[bus])
        worst.append(row)
    angle_bus_mask = (types_np == 0) | (types_np == 1)
    if layout.per_case:
        pq_mask, pv_mask, _slack_mask, angle_mask = layout.masks
        angle_err = error[..., 1][angle_mask]
        volt_err = error[..., 0][pq_mask]
        # Per-bus mean |angle error| over the batch for PQ/PV buses (shared topology).
        per_bus_angle = error[..., 1].abs().mean(dim=0).detach().cpu().numpy()
    else:
        pq, pv, _slack = layout.index
        angle_idx = layout.angle_index
        angle_err = error[:, angle_idx, 1].reshape(-1)
        volt_err = error[:, pq, 0].reshape(-1) if layout.present[0] else error.new_zeros(0)
        per_bus_angle = error[..., 1].abs().mean(dim=0).detach().cpu().numpy()
    angle_rmse = float(torch.sqrt(torch.mean(angle_err * angle_err))) if angle_err.numel() else 0.0
    volt_rmse = float(torch.sqrt(torch.mean(volt_err * volt_err))) if volt_err.numel() else 0.0

    def _rmse_on(mask: np.ndarray) -> float:
        take = mask & angle_bus_mask
        if not take.any():
            return 0.0
        vals = per_bus_angle[take]
        return float(np.sqrt(np.mean(vals * vals)))

    high_mask = y_abs_diag >= high_y["threshold_|Yii|"] if mean_bus.size else np.zeros(0, dtype=bool)
    angle_high = _rmse_on(high_mask) if mean_bus.size else 0.0
    angle_rest = _rmse_on(~high_mask) if mean_bus.size else 0.0
    worst_mask = np.zeros(mean_bus.size, dtype=bool)
    if mean_bus.size:
        worst_mask[order] = True
    angle_worst = _rmse_on(worst_mask) if mean_bus.size else 0.0
    topk_share = float(mean_bus[order].sum() / total_mass) if mean_bus.size else 0.0
    # Published GENCO Base Task 1.1 N mean PBL (plot_task1_1.PUBLISHED).
    genco_n = 1.4e-3
    genco_gap = raw_mean / genco_n if genco_n > 0 else float("inf")

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
        topk_pbl_share=topk_share,
        angle_rmse_high_y=angle_high,
        angle_rmse_rest=angle_rest,
        angle_rmse_worst=angle_worst,
        genco_n_gap=genco_gap,
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
        hop_id, edge_id = ensure_network_grpe_ids(network)
        layout = network_layout(network)
        prediction, residuals = model(
            specified,
            types,
            network.rpe,
            network_masks(network),
            ybus,
            pair=pair,
            layout=layout,
            shunt_b=network_shunt_b(network),
            hop_id=hop_id,
            edge_id=edge_id,
            **exphormer_forward_kwargs(network),
        )
        breakdown = task_loss(prediction, target, residuals, types)
        stage_pbl = [pbl_stats(r) for r in residuals]
        model_pbl = stage_pbl[-1] if stage_pbl else pbl_from_unknown(specified, prediction, types, ybus)
        state_rmse = state_rmse_by_type(prediction, target, types)
        state_mae = state_mae_by_type(prediction, target, types)
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
        state_mae=state_mae,
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
        f"voltage(PQ) {bottleneck.voltage_rmse_pq:.4g} pu  "
        f"(~{bottleneck.genco_n_gap:.0f}x GENCO published N PBL)"
    )
    lines.append(
        f"  Angle |err| RMSE: high-|Yii| {bottleneck.angle_rmse_high_y:.4g} rad  "
        f"rest {bottleneck.angle_rmse_rest:.4g}  "
        f"worst-{len(bottleneck.worst_buses)} buses {bottleneck.angle_rmse_worst:.4g}  "
        f"(top-{len(bottleneck.worst_buses)} PBL mass {bottleneck.topk_pbl_share:.1%})"
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
        last = bottleneck.stage_reduction[-1]["reduction_from_prev"]
        if last >= 0.15:
            lines.append(
                "  => Late stages still cut a lot: more DEPTH (iterative angle refine) should help "
                "before/with width."
            )
        elif last < 0.05:
            lines.append(
                "  => Late stages nearly stall: prefer wider D / more heads (capacity), not only more stages."
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
        lines.append("  Worst buses (1-based bus index; same as PFΔ bus column + 1):")
        for row in bottleneck.worst_buses:
            state = ""
            if "mae_V_pu" in row:
                state = f"  |dV| {row['mae_V_pu']:.4g} pu  |dθ| {row['mae_theta_rad']:.4g} rad"
            elif "mae_theta_rad" in row:
                state = f"  |dθ| {row['mae_theta_rad']:.4g} rad  (V specified)"
            elif "mae_P_pu" in row:
                state = f"  |dP|state {row['mae_P_pu']:.4g} pu  |dQ|state {row['mae_Q_pu']:.4g} pu"
            lines.append(
                f"    bus {row['bus_1based']:3d} ({row['type']:5s}) PBL {row['mean_pbl']:.4g}  "
                f"|dP| {row['mean_|dP|']:.4g} |dQ| {row['mean_|dQ|']:.4g}  "
                f"|Yii| {row['|Yii|']:.4g}  hops {row['hops_to_slack']}{state}"
            )
    lines.append(
        "  Recipe toward GENCO early: (1) DATA_FRACTION=1.0 (0.1 cannot match full-data GENCO), "
        "(2) GENCO_TPGT_ARCH=v8 (DEPTH 8, D 96; same residual/loss logic), "
        "(3) BATCH 64 on L4, (4) do not chase slack_P RMSE."
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
    mae = diagnostics.state_mae
    lines.append(
        "State MAE vs labels (pu or rad): "
        + "  ".join(f"{key} {value:.6g}" for key, value in sorted(mae.items()))
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
        hop_id, edge_id = ensure_network_grpe_ids(network)
        prediction, residuals = model(
            specified,
            types,
            network.rpe,
            network_masks(network),
            ybus,
            pair=pair,
            layout=network_layout(network),
            shunt_b=network_shunt_b(network),
            hop_id=hop_id,
            edge_id=edge_id,
            **exphormer_forward_kwargs(network),
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


def build_forward_diagnostics_text(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    tag: str,
    *,
    rpe_ablation: bool = False,
    shunt_ablation: bool = False,
) -> str:
    report = collect_forward_diagnostics(model, split, network, index, device)
    text = format_forward_diagnostics(tag, report)
    weight_rows = pair_encoder_input_weight_norms(model)
    baseline: dict[str, float] | None = None
    ablation_rows: list[dict[str, float]] | None = None
    if rpe_ablation:
        baseline, ablation_rows = collect_rpe_channel_ablation(model, split, network, index, device)
    text = text + "\n" + format_rpe_effectiveness_diagnostics(weight_rows, baseline, ablation_rows)
    node_weight_rows = type_encoder_input_weight_norms(model)
    shunt_baseline: dict[str, float] | None = None
    shunt_rows: list[dict[str, float]] | None = None
    if shunt_ablation:
        shunt_baseline, shunt_rows = collect_bus_shunt_ablation(model, split, network, index, device)
    text = text + "\n" + format_bus_shunt_ablation_diagnostics(node_weight_rows, shunt_baseline, shunt_rows)
    text = text + "\n" + format_rpe_correctness_checks(audit_rpe_correctness(network))
    text = text + "\n" + format_correctness_checks(
        "Residual correctness audit (power-balance computation):",
        audit_residual_correctness(model, split, network, index, device),
        "Fix nodal_residual, units, or Ybus grouping before trusting PBL numbers.",
    )
    return text


def print_forward_diagnostics(
    model: TPGT,
    split: Split,
    network: Network,
    index: np.ndarray,
    device: torch.device,
    tag: str,
    *,
    rpe_ablation: bool = False,
    shunt_ablation: bool = False,
    seed_dir: Path | None = None,
    epoch: int | None = None,
) -> str:
    text = build_forward_diagnostics_text(
        model,
        split,
        network,
        index,
        device,
        tag,
        rpe_ablation=rpe_ablation,
        shunt_ablation=shunt_ablation,
    )
    if seed_dir is None:
        print(text, flush=True)
    else:
        append_seed_log(seed_dir, text)
        if epoch is not None and saves_checkpoint_epoch(int(epoch)):
            write_seed_report(seed_dir, int(epoch), "forward_diagnostics.txt", text)
    return text


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


def wants_shunt_ablation(*, default: bool) -> bool:
    if "--no-shunt-ablation" in sys.argv:
        return False
    if "--shunt-ablation" in sys.argv:
        return True
    flag = os.environ.get("GENCO_TPGT_SHUNT_ABLATION", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    if flag in ("0", "false", "no"):
        return False
    return default and USE_NODE_SHUNT


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
    seed_dir: Path | None = None,
    epoch: int | None = None,
) -> str:
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
    text = "\n".join(lines)
    if seed_dir is None:
        print(text, flush=True)
    else:
        append_seed_log(seed_dir, text)
        if epoch is not None and saves_checkpoint_epoch(int(epoch)):
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label.strip()) or "contingency"
            write_seed_report(seed_dir, int(epoch), f"contingency_{safe}.txt", text)
    return text


@dataclass
class Network:
    ybus: torch.Tensor
    rpe: torch.Tensor
    component: torch.Tensor
    types: torch.Tensor
    pair_type_masks: torch.Tensor | None = field(default=None, repr=False)
    pair_embed: torch.Tensor | None = field(default=None, repr=False)
    branches: tuple[tuple[int, int, float, float], ...] | None = field(default=None, repr=False)
    # Bus shunt susceptance from parquet ``BS`` (p.u.); device-only, not line charging.
    shunt_b: torch.Tensor | None = field(default=None, repr=False)
    # Global Zeff normalizers used when building ``rpe`` (shared across topologies in a split).
    rpe_re_scale: float | None = field(default=None, repr=False)
    rpe_im_scale: float | None = field(default=None, repr=False)
    # GRPE topology bins (Park et al.): hop distance id and discrete edge-relation id.
    hop_id: torch.Tensor | None = field(default=None, repr=False)
    edge_id: torch.Tensor | None = field(default=None, repr=False)
    # Exphormer (v12): dense allow/tau on augmented N+C nodes; optional COO edges for sparse kernel.
    exph_allow: torch.Tensor | None = field(default=None, repr=False)
    exph_tau: torch.Tensor | None = field(default=None, repr=False)
    exph_n_virt: int = field(default=0, repr=False)
    exph_recv: torch.Tensor | None = field(default=None, repr=False)
    exph_send: torch.Tensor | None = field(default=None, repr=False)
    exph_edge_tau: torch.Tensor | None = field(default=None, repr=False)
    exph_pair_id: torch.Tensor | None = field(default=None, repr=False)
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


def _effective_impedance_matrix(laplacian: np.ndarray) -> np.ndarray:
    """``Z_eff[i,j] = L^†_ii + L^†_jj - 2 L^†_ij`` for real or complex Laplacian ``L``."""
    pseudoinverse = np.linalg.pinv(laplacian)
    diagonal = np.diag(pseudoinverse)
    return (diagonal[:, None] + diagonal[None, :]) - 2.0 * pseudoinverse


def _meshed_zeff_and_component(
    bus_count: int,
    branches: list[tuple[int, int, float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Complex ``Z_eff`` and per-bus component ids from in-service branches only."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    admittance = np.zeros((bus_count, bus_count), dtype=np.complex128)
    adjacency = np.zeros((bus_count, bus_count), dtype=bool)
    table = np.asarray(branches, dtype=np.float64).reshape(-1, 4)
    for origin, destination, resistance, reactance in table:
        impedance = resistance + 1j * reactance
        if abs(impedance) <= 0.0:
            continue
        branch_y = 1.0 / impedance
        i = int(origin)
        j = int(destination)
        admittance[i, j] += branch_y
        admittance[j, i] += branch_y
        adjacency[i, j] = True
        adjacency[j, i] = True

    graph = csr_matrix(adjacency.astype(np.float64))
    _count, labels = connected_components(graph, directed=False)
    _unique, first_bus, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(first_bus.size, dtype=np.int64)
    rank[np.argsort(first_bus, kind="stable")] = np.arange(first_bus.size)
    component = rank[inverse.reshape(-1)].astype(np.int64)
    component_id = int(first_bus.size)

    z_eff = np.zeros((bus_count, bus_count), dtype=np.complex128)
    for comp in range(component_id):
        buses = np.flatnonzero(component == comp)
        if buses.size == 1:
            continue
        sub_y = admittance[np.ix_(buses, buses)]
        sub_laplacian = np.diag(np.sum(sub_y, axis=1)) - sub_y
        block = _effective_impedance_matrix(sub_laplacian)
        z_eff[np.ix_(buses, buses)] = block

    if np.max(np.abs(z_eff - z_eff.T)) > 1e-4:
        raise RuntimeError("Complex effective impedance is not symmetric.")
    return z_eff, component


def meshed_zeff_peak_abs(
    z_eff: np.ndarray,
    component: np.ndarray,
    bus_count: int,
) -> tuple[float, float]:
    """Max |Re| and |Im| over same-component off-diagonal ``Z_eff`` entries."""
    same = component[:, None] == component[None, :]
    off_diagonal = same & ~np.eye(bus_count, dtype=bool)
    if not np.any(off_diagonal):
        return 1.0, 1.0
    max_re = float(np.max(np.abs(z_eff.real[off_diagonal])))
    max_im = float(np.max(np.abs(z_eff.imag[off_diagonal])))
    return max(max_re, 1e-30), max(max_im, 1e-30)


def global_zeff_rpe_scales(
    branch_lists: list[list[tuple[int, int, float, float]]],
    bus_count: int = 118,
) -> tuple[float, float]:
    """``s_R``, ``s_X`` = max off-diagonal |Re| and |Im| of ``Z_eff`` over every topology (floored at 1)."""
    _s_r, _s_x, _raw_r, _raw_x = fit_global_zeff_rpe_scales_from_branch_lists(branch_lists, bus_count)
    return _s_r, _s_x


def fit_global_zeff_rpe_scales_from_branch_lists(
    branch_lists: list[list[tuple[int, int, float, float]]],
    bus_count: int,
) -> tuple[float, float, float, float]:
    max_re = 0.0
    max_im = 0.0
    for branches in branch_lists:
        if not branches:
            continue
        z_eff, component = _meshed_zeff_and_component(bus_count, branches)
        peak_re, peak_im = meshed_zeff_peak_abs(z_eff, component, bus_count)
        max_re = max(max_re, peak_re)
        max_im = max(max_im, peak_im)
    return max(max_re, 1.0), max(max_im, 1.0), max_re, max_im


def fit_global_zeff_rpe_scales_from_networks(
    networks: list[Network],
) -> tuple[float, float, float, float]:
    """Train-fit max |Z_eff| peaks across every ``Network`` (correct ``bus_count`` per grid)."""
    max_re = 0.0
    max_im = 0.0
    for network in networks:
        if not network.branches:
            continue
        bus_count = int(network.ybus.shape[0])
        z_eff, component = _meshed_zeff_and_component(bus_count, list(network.branches))
        peak_re, peak_im = meshed_zeff_peak_abs(z_eff, component, bus_count)
        max_re = max(max_re, peak_re)
        max_im = max(max_im, peak_im)
    if not any(network.branches for network in networks):
        print("  RPE scale fit: no branch lists on train networks; using s_R=s_X=1.0", flush=True)
        return 1.0, 1.0, 0.0, 0.0
    return max(max_re, 1.0), max(max_im, 1.0), max_re, max_im


def build_grpe_graph_ids(
    bus_count: int,
    branches: list[tuple[int, int, float, float]] | tuple[tuple[int, int, float, float], ...],
    *,
    max_hop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """All-pairs hop ids and discrete edge-relation ids for Park GRPE.

    * ``hop_id[i,j]`` ∈ ``{0..max_hop, max_hop+1}`` (overflow / unreachable).
    * ``edge_id[i,j]`` ∈ ``{self=0, in-service branch=1, non-adjacent=2}``.
    Cross-component attention is still blocked by existing component masks.
    """
    from scipy.sparse.csgraph import shortest_path

    hop_limit = GRPE_MAX_HOP if max_hop is None else int(max_hop)
    overflow = hop_limit + 1
    adjacency = np.zeros((bus_count, bus_count), dtype=bool)
    table = np.asarray(branches, dtype=np.float64).reshape(-1, 4) if branches else np.zeros((0, 4))
    for origin, destination, resistance, reactance in table:
        impedance = resistance + 1j * reactance
        if abs(impedance) <= 0.0:
            continue
        i = int(origin)
        j = int(destination)
        adjacency[i, j] = True
        adjacency[j, i] = True

    if bus_count == 0:
        empty = np.zeros((0, 0), dtype=np.int64)
        return empty, empty.copy()

    dist = shortest_path(adjacency.astype(np.float64), directed=False, unweighted=True)
    hop = np.full((bus_count, bus_count), overflow, dtype=np.int64)
    finite = np.isfinite(dist)
    hop[finite] = np.minimum(dist[finite].astype(np.int64), overflow)
    np.fill_diagonal(hop, 0)

    edge = np.full((bus_count, bus_count), GRPE_EDGE_NONADJ, dtype=np.int64)
    edge[adjacency] = GRPE_EDGE_BRANCH
    np.fill_diagonal(edge, GRPE_EDGE_SELF)
    return hop, edge


def attach_grpe_graph_ids(network: Network, *, max_hop: int | None = None) -> Network:
    """Set ``network.hop_id`` / ``edge_id`` from in-service ``branches`` (CPU long tensors)."""
    bus_count = int(network.types.shape[-1])
    branches = list(network.branches) if network.branches is not None else []
    hop, edge = build_grpe_graph_ids(bus_count, branches, max_hop=max_hop)
    network.hop_id = torch.from_numpy(hop)
    network.edge_id = torch.from_numpy(edge)
    return network


def ensure_network_grpe_ids(network: Network) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hop/edge ids, building them once if a cached Network lacks the fields.

    Lazily built ids are moved onto ``network.types``'s device so diagnostics on a
    GPU-resident Network do not pass CPU index tensors into GRPE attention.
    """
    hop = getattr(network, "hop_id", None)
    edge = getattr(network, "edge_id", None)
    if hop is None or edge is None:
        attach_grpe_graph_ids(network)
        hop = network.hop_id
        edge = network.edge_id
    assert hop is not None and edge is not None
    device = network.types.device
    if hop.device != device:
        hop = hop.to(device=device, non_blocking=device.type == "cuda")
        network.hop_id = hop
    if edge.device != device:
        edge = edge.to(device=device, non_blocking=device.type == "cuda")
        network.edge_id = edge
    return hop, edge


def attach_exphormer_graph(network: Network, *, d_exp: int | None = None, seed: int | None = None) -> Network:
    """Set Exphormer allow/tau (+ COO edges) from branches and component labels."""
    bus_count = int(network.types.shape[-1])
    branches = list(network.branches) if network.branches is not None else []
    component = network.component.detach().cpu().numpy().astype(np.int64)
    degree = EXPHORMER_D_EXP if d_exp is None else int(d_exp)
    allow, tau, n_virt = exph.build_exphormer_graph(
        bus_count, branches, component, d_exp=degree, seed=seed
    )
    types_aug = np.concatenate(
        [
            network.types.detach().cpu().numpy().astype(np.int64),
            np.full(n_virt, exph.TYPE_VIRTUAL, dtype=np.int64),
        ]
    )
    recv, send, edge_tau, pair_ids = exph.sparse_edges_from_dense(allow, tau, types_aug)
    network.exph_allow = torch.from_numpy(allow)
    network.exph_tau = torch.from_numpy(tau)
    network.exph_n_virt = int(n_virt)
    network.exph_recv = torch.from_numpy(recv)
    network.exph_send = torch.from_numpy(send)
    network.exph_edge_tau = torch.from_numpy(edge_tau)
    network.exph_pair_id = torch.from_numpy(pair_ids)
    return network


def ensure_network_exphormer(network: Network) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(allow, tau, n_virt)`` on ``network.types`` device, building if missing."""
    allow = getattr(network, "exph_allow", None)
    tau = getattr(network, "exph_tau", None)
    n_virt = int(getattr(network, "exph_n_virt", 0) or 0)
    if allow is None or tau is None or n_virt <= 0:
        attach_exphormer_graph(network)
        allow = network.exph_allow
        tau = network.exph_tau
        n_virt = int(network.exph_n_virt)
    assert allow is not None and tau is not None
    device = network.types.device
    if allow.device != device:
        allow = allow.to(device=device, non_blocking=device.type == "cuda")
        network.exph_allow = allow
    if tau.device != device:
        tau = tau.to(device=device, non_blocking=device.type == "cuda")
        network.exph_tau = tau
    for attr in ("exph_recv", "exph_send", "exph_edge_tau", "exph_pair_id"):
        tensor = getattr(network, attr, None)
        if tensor is not None and tensor.device != device:
            setattr(network, attr, tensor.to(device=device, non_blocking=device.type == "cuda"))
    return allow, tau, n_virt


def exphormer_forward_kwargs(network: Network) -> dict:
    """Keyword args for ``TPGT.forward`` when ``ATTENTION_KIND==exphormer``; else empty."""
    if ATTENTION_KIND != "exphormer":
        return {}
    allow, tau, n_virt = ensure_network_exphormer(network)
    return {
        "exph_allow": allow,
        "exph_tau": tau,
        "exph_n_virt": n_virt,
        "exph_recv": getattr(network, "exph_recv", None),
        "exph_send": getattr(network, "exph_send", None),
        "exph_edge_tau": getattr(network, "exph_edge_tau", None),
        "exph_pair_id": getattr(network, "exph_pair_id", None),
    }


def meshed_rpe(
    bus_count: int,
    branches: list[tuple[int, int, float, float]],
    ybus: np.ndarray,
    *,
    re_scale: float | None = None,
    im_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Two pair features: normalized Re/Im of complex effective impedance (Method B).

    Branch admittance ``y_ij = 1/(r_ij + j x_ij)`` on **off-diagonals only** (parallel
    branches add); diagonal entries stay zero until the Laplacian step. This is a
    weighted adjacency ``W``, not the assembled bus ``Y_bus`` (whose rows sum to zero).
    Complex Laplacian ``L = diag(W 1) - W``. Then
    ``Z_eff(i,j) = L^†_ii + L^†_jj - 2 L^†_ij``.

    Normalization: if ``re_scale`` / ``im_scale`` are given, divide all channels by
    those **global** positive scalars (same across topologies in a split). If omitted,
    use legacy per-connected-component maxima (tests only).

    ``ybus`` is unused (residual still uses full scenario Ybus).
    """
    del ybus
    z_eff, component = _meshed_zeff_and_component(bus_count, branches)
    component_id = int(component.max()) + 1 if component.size else 0

    if re_scale is None or im_scale is None:
        re_per = np.ones(component_id, dtype=np.float64)
        im_per = np.ones(component_id, dtype=np.float64)
        for comp in range(component_id):
            buses = np.flatnonzero(component == comp)
            if buses.size <= 1:
                continue
            block = z_eff[np.ix_(buses, buses)]
            off = block.copy()
            np.fill_diagonal(off, 0.0)
            max_re = float(np.max(np.abs(off.real))) if off.size else 0.0
            max_im = float(np.max(np.abs(off.imag))) if off.size else 0.0
            re_per[comp] = max_re if max_re > 0 else 1.0
            im_per[comp] = max_im if max_im > 0 else 1.0
        re_div = re_per[component][:, None]
        im_div = im_per[component][:, None]
    else:
        re_div = float(re_scale)
        im_div = float(im_scale)

    same = component[:, None] == component[None, :]
    off_diagonal = same & ~np.eye(bus_count, dtype=bool)
    features = np.zeros((bus_count, bus_count, RPE_DIM), dtype=np.float32)
    re = np.where(off_diagonal, z_eff.real / re_div, 0.0)
    im = np.where(off_diagonal, z_eff.imag / im_div, 0.0)
    features[..., 0] = re.astype(np.float32)
    features[..., 1] = im.astype(np.float32)

    if not np.isfinite(features).all():
        raise RuntimeError("Meshed RPE contains non-finite values.")
    return features, component


RPE_SCALES_FILE = "rpe_global_scales.json"


def rpe_scales_path(persistent: Path) -> Path:
    return persistent / RPE_SCALES_FILE


def save_rpe_global_scales(
    persistent: Path,
    re_scale: float,
    im_scale: float,
    *,
    raw_re: float | None = None,
    raw_im: float | None = None,
    source: str = "train_fit",
) -> None:
    """Persist train-fit ``s_R,s_X`` for valid/test cache builds (Drive ``checkpoints/cache``)."""
    persistent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "rpe_re_scale": re_scale,
        "rpe_im_scale": im_scale,
        "source": source,
    }
    if raw_re is not None:
        payload["rpe_re_peak_raw"] = raw_re
    if raw_im is not None:
        payload["rpe_im_peak_raw"] = raw_im
    rpe_scales_path(persistent).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_rpe_scales_payload(cache_dir: Path) -> dict | None:
    path = rpe_scales_path(cache_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_rpe_global_scales(persistent: Path) -> tuple[float, float] | None:
    payload = read_rpe_scales_payload(persistent)
    if payload is None:
        return None
    if payload.get("version") != CACHE_VERSION or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        return None
    return float(payload["rpe_re_scale"]), float(payload["rpe_im_scale"])


def log_rpe_scales_provenance(cache_dir: Path, *, context: str) -> None:
    """Explain where ``s_R,s_X`` came from (file vs missing vs version mismatch vs unit floor)."""
    path = rpe_scales_path(cache_dir)
    payload = read_rpe_scales_payload(cache_dir)
    if payload is None:
        if not path.is_file():
            print(f"  RPE scales ({context}): no {RPE_SCALES_FILE} under {cache_dir}", flush=True)
        else:
            print(
                f"  RPE scales ({context}): {path.name} present but unreadable or wrong "
                f"{CACHE_VERSION}/{CHECKPOINT_VERSION}",
                flush=True,
            )
        return
    if payload.get("version") != CACHE_VERSION or payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        print(
            f"  RPE scales ({context}): ignoring stale {path} "
            f"(want {CACHE_VERSION}+{CHECKPOINT_VERSION}, "
            f"got {payload.get('version')}+{payload.get('checkpoint_version')})",
            flush=True,
        )
        return
    re_scale = float(payload["rpe_re_scale"])
    im_scale = float(payload["rpe_im_scale"])
    raw_re = payload.get("rpe_re_peak_raw")
    raw_im = payload.get("rpe_im_peak_raw")
    source = payload.get("source", "unknown")
    note = f"s_R={re_scale:.6g} s_X={im_scale:.6g} source={source} file={path}"
    if raw_re is not None and raw_im is not None:
        note += f" (train peaks raw={float(raw_re):.6g}, {float(raw_im):.6g}"
        if re_scale == 1.0 and im_scale == 1.0 and (float(raw_re) < 1.0 or float(raw_im) < 1.0):
            note += "; applied max(peak,1) floor, not a default"
        note += ")"
    elif re_scale == 1.0 and im_scale == 1.0:
        note += " (unit scales: refit train or inspect peaks; not used when file missing — build would fail)"
    print(f"  RPE scales ({context}): {note}", flush=True)


def cross_grid_checkpoint_root(checkpoints: Path, ckpt_root: Path | None) -> bool:
    """True when eval loads weights from a different Drive run than this task's cache folder."""
    if ckpt_root is None:
        return False
    try:
        return ckpt_root.resolve() != checkpoints.resolve()
    except OSError:
        return ckpt_root != checkpoints


def copy_training_rpe_scales(persistent: Path, weight_run: Path) -> None:
    """Reuse ``s_R,s_X`` from another run (cross-grid Table 4 eval with foreign checkpoints)."""
    if load_rpe_global_scales(persistent) is not None:
        return
    candidates = [weight_run / "cache", Path(f"/content/{weight_run.name}_cache")]
    for folder in candidates:
        scales = load_rpe_global_scales(folder)
        if scales is not None:
            payload = read_rpe_scales_payload(folder)
            raw_re = None if payload is None else payload.get("rpe_re_peak_raw")
            raw_im = None if payload is None else payload.get("rpe_im_peak_raw")
            source = "copied_from_" + folder.name
            save_rpe_global_scales(
                persistent,
                scales[0],
                scales[1],
                raw_re=float(raw_re) if raw_re is not None else None,
                raw_im=float(raw_im) if raw_im is not None else None,
                source=source,
            )
            log_rpe_scales_provenance(folder, context="weight run")
            print(
                f"RPE scales for eval caches: copied from {folder} -> {rpe_scales_path(persistent)}",
                flush=True,
            )
            return
    raise RuntimeError(
        f"Missing RPE scales for eval caches (expected {rpe_scales_path(persistent)}). "
        f"Train on this arch once so {weight_run / 'cache' / RPE_SCALES_FILE} exists, "
        f"or copy that file into {persistent}."
    )


def warm_split_caches(
    data: Path,
    active: Path,
    persistent: Path,
    spec: TaskSpec,
    *,
    weight_run: Path | None = None,
    load_valid: bool = True,
) -> tuple[Split | None, Split | None]:
    """``main()``-style cache setup before train or Task 3.1 test-only eval."""
    print(f"Split cache (active): {active}", flush=True)
    if persistent != active:
        print(f"Split cache (Drive): {persistent}", flush=True)
        if persistent.is_dir():
            on_drive = len(list(persistent.glob("*.pt")))
            print(f"  reusing up to {on_drive} cached split .pt on Drive (mirroring to {active})", flush=True)
        sync_split_cache_from_drive(persistent, active)
    reconcile_split_caches(active, persistent, data)
    checkpoints = persistent.parent
    if cross_grid_checkpoint_root(checkpoints, weight_run):
        print(
            f"Cross-grid eval: RPE scales from weight run {weight_run} "
            f"(skip Task {spec.label} train cache; test caches still use {spec.dataset_repo}).",
            flush=True,
        )
        copy_training_rpe_scales(persistent, weight_run)
        log_rpe_scales_provenance(persistent, context="Task 3.1 eval cache")
        scales_path = rpe_scales_path(persistent)
        if persistent != active and scales_path.is_file():
            _mirror_cache_file(scales_path, active / RPE_SCALES_FILE)
        return None, None
    load_started = time.perf_counter()
    train_split = load_task_split(data, "train", active, persistent)
    val_split = load_task_split(data, "valid", active, persistent) if load_valid else None
    ready = f"{train_split.count} train"
    if val_split is not None:
        ready += f", {val_split.count} valid"
    print(
        f"Train split caches ready in {time.perf_counter() - load_started:.1f}s "
        f"({ready} cases; scales {rpe_scales_path(persistent)})",
        flush=True,
    )
    return train_split, val_split


# Back-compat alias (tests).
ensure_eval_rpe_scales = copy_training_rpe_scales


def split_rpe_scales(split: Split) -> tuple[float, float]:
    for network in split.networks:
        if network.rpe_re_scale is not None and network.rpe_im_scale is not None:
            return network.rpe_re_scale, network.rpe_im_scale
    branch_lists = [list(network.branches) for network in split.networks if network.branches]
    return global_zeff_rpe_scales(branch_lists)


def unify_network_rpe(
    networks: list[Network],
    rpe_scales: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Recompute ``rpe`` on each network with shared ``s_R``, ``s_X`` (train scales or fit from ``networks``)."""
    if rpe_scales is not None:
        global_re, global_im = rpe_scales
    else:
        if not any(network.branches for network in networks):
            print("  unify_network_rpe: no branches; using s_R=s_X=1.0", flush=True)
            return 1.0, 1.0
        global_re, global_im, _raw_r, _raw_x = fit_global_zeff_rpe_scales_from_networks(networks)
    for network in networks:
        if network.branches is None:
            continue
        ybus = network.ybus.detach().cpu().numpy()
        features, component = meshed_rpe(
            int(ybus.shape[0]),
            list(network.branches),
            ybus,
            re_scale=global_re,
            im_scale=global_im,
        )
        network.rpe = torch.from_numpy(features)
        network.component = torch.from_numpy(component.astype(np.int64))
        network.rpe_re_scale = global_re
        network.rpe_im_scale = global_im
        network.pair_embed = None
        attach_grpe_graph_ids(network)
        if ATTENTION_KIND == "exphormer":
            attach_exphormer_graph(network)
    return global_re, global_im


def unify_rpe_normalization(
    split: Split,
    rpe_scales: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Same as ``unify_network_rpe`` over ``split.networks``."""
    return unify_network_rpe(split.networks, rpe_scales)


@dataclass
class CorrectnessCheck:
    name: str
    passed: bool
    detail: str
    value: float | None = None


def audit_rpe_correctness(network: Network, *, recompute_atol: float = 1e-4) -> list[CorrectnessCheck]:
    """Structural checks plus optional full recompute from stored Ybus and in-service branches."""
    rpe = network.rpe.detach().cpu().numpy().astype(np.float32, copy=False)
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
    for channel, label in enumerate(("Re", "Im")):
        diag_ok = bool(np.allclose(rpe[diagonal, diagonal, channel], 0.0))
        checks.append(
            CorrectnessCheck(
                f"diag_zeff_{label.lower()}_zero",
                diag_ok,
                f"diagonal Zeff {label} is 0" if diag_ok else f"diagonal Zeff {label} must be 0",
            )
        )
        sym_ok = bool(np.allclose(rpe[:, :, channel], rpe[:, :, channel].T, atol=1e-4))
        checks.append(
            CorrectnessCheck(
                f"zeff_{label.lower()}_symmetric",
                sym_ok,
                f"normalized Zeff {label} symmetric" if sym_ok else f"Zeff {label} channel not symmetric",
            )
        )

    if getattr(network, "branches", None) is not None:
        branches = network.branches
        assert branches is not None
        re_s = getattr(network, "rpe_re_scale", None)
        im_s = getattr(network, "rpe_im_scale", None)
        expected, expected_component = meshed_rpe(
            bus_count,
            list(branches),
            ybus,
            re_scale=re_s,
            im_scale=im_s,
        )
        max_diff = float(np.max(np.abs(expected - rpe)))
        recompute_ok = max_diff <= recompute_atol and np.array_equal(expected_component, component)
        scale_note = (
            f"s_R={re_s} s_X={im_s}"
            if re_s is not None and im_s is not None
            else "scales missing (per-component fallback; check move_to preserves rpe_*_scale)"
        )
        checks.append(
            CorrectnessCheck(
                "recompute_from_branches",
                recompute_ok,
                f"max |recomputed - cached| = {max_diff:.3e} (tol {recompute_atol:.0e}; {scale_note})",
            )
        )
    else:
        checks.append(
            CorrectnessCheck(
                "recompute_from_branches",
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


def parquet_bus_count(raw: Path) -> int:
    """Buses per scenario: length of the first scenario block in ``bus_data.parquet``."""
    import pyarrow.parquet as pq

    scenario = pq.read_table(raw / "bus_data.parquet", columns=["scenario"]).column("scenario").to_numpy()
    if scenario.size == 0:
        raise RuntimeError(f"{raw} bus_data is empty.")
    change = np.flatnonzero(scenario != scenario[0])
    width = int(change[0]) if change.size else int(scenario.size)
    if width < 1 or scenario.size % width != 0:
        raise RuntimeError(f"{raw} bus rows {scenario.size} are not a multiple of {width}.")
    return width


def scenario_count(raw: Path) -> int:
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(raw / "bus_data.parquet").metadata.num_rows
    width = parquet_bus_count(raw)
    if rows % width != 0:
        raise RuntimeError(f"{raw} bus rows {rows} are not a multiple of {width}.")
    return rows // width


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


def parse_arch_name() -> str | None:
    for index, arg in enumerate(sys.argv):
        if arg == "--arch" and index + 1 < len(sys.argv):
            return sys.argv[index + 1].strip().lower()
        if arg.startswith("--arch="):
            return arg.split("=", 1)[1].strip().lower()
    env = os.environ.get("GENCO_TPGT_ARCH", "").strip().lower()
    return env or None


def _parse_mlp_hiddens(raw: object, default: tuple[int, ...] = (128, 64)) -> tuple[int, ...]:
    """Parse score-MLP hidden widths from preset/env (``128,64`` or a sequence)."""
    if raw is None:
        return tuple(int(x) for x in default)
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace("x", ",").split(",") if p.strip()]
        widths = tuple(int(p) for p in parts)
    elif isinstance(raw, (list, tuple)):
        widths = tuple(int(x) for x in raw)
    else:
        widths = (int(raw),)
    if len(widths) != 2 or any(w < 1 for w in widths):
        raise ValueError(
            f"ATTENTION_MLP_HIDDENS must be two positive ints (h1,h2), got {raw!r}"
        )
    return widths


def apply_architecture_config() -> str:
    """Apply ``ARCH_PRESETS`` / per-dim env overrides. Same residual, packing, and loss logic."""
    global D, D_E, D_A, HEADS, DEPTH, RUN_TAG, RUN_DIR_NAME, CHECKPOINT_VERSION
    global USE_NODE_SHUNT, ATTENTION_KIND, ATTENTION_MLP_HIDDENS, GRPE_MAX_HOP, EXPHORMER_DENSE, EXPHORMER_D_EXP
    name = parse_arch_name()
    if name:
        if name not in ARCH_PRESETS:
            allowed = ", ".join(sorted(ARCH_PRESETS))
            raise ValueError(f"unknown arch {name!r}; choose one of {allowed}")
        preset = ARCH_PRESETS[name]
        D = int(preset["D"])
        D_E = int(preset["D_E"])
        D_A = int(preset["D_A"])
        HEADS = int(preset["HEADS"])
        DEPTH = int(preset["DEPTH"])
        RUN_TAG = str(preset["RUN_TAG"])
        CHECKPOINT_VERSION = str(preset["CHECKPOINT_VERSION"])
        USE_NODE_SHUNT = bool(preset.get("USE_NODE_SHUNT", False))
        ATTENTION_KIND = str(preset.get("ATTENTION_KIND", "grit"))
        ATTENTION_MLP_HIDDENS = _parse_mlp_hiddens(
            preset.get("ATTENTION_MLP_HIDDENS", ATTENTION_MLP_HIDDENS)
        )
    else:
        name = RUN_TAG

    for env_key, global_name in (
        ("GENCO_TPGT_D", "D"),
        ("GENCO_TPGT_D_E", "D_E"),
        ("GENCO_TPGT_D_A", "D_A"),
        ("GENCO_TPGT_HEADS", "HEADS"),
        ("GENCO_TPGT_DEPTH", "DEPTH"),
    ):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            globals()[global_name] = int(raw)
    mlp_env = os.environ.get("GENCO_TPGT_ATTENTION_MLP_HIDDENS", "").strip()
    if not mlp_env:
        # Backward-compatible single-width env: interpret as (H, H//2) if set.
        legacy = os.environ.get("GENCO_TPGT_ATTENTION_MLP_HIDDEN", "").strip()
        if legacy:
            h = int(legacy)
            mlp_env = f"{h},{max(1, h // 2)}"
    if mlp_env:
        ATTENTION_MLP_HIDDENS = _parse_mlp_hiddens(mlp_env)
    run_override = os.environ.get("GENCO_TPGT_RUN_TAG", "").strip()
    if run_override:
        RUN_TAG = run_override
    shunt_override = os.environ.get("GENCO_TPGT_NODE_SHUNT", "").strip().lower()
    if shunt_override in ("1", "true", "yes"):
        USE_NODE_SHUNT = True
    elif shunt_override in ("0", "false", "no"):
        USE_NODE_SHUNT = False
    attn_override = os.environ.get("GENCO_TPGT_ATTENTION", "").strip().lower()
    if attn_override:
        ATTENTION_KIND = attn_override
    hop_env = os.environ.get("GENCO_TPGT_GRPE_MAX_HOP", "").strip()
    if hop_env:
        GRPE_MAX_HOP = int(hop_env)
    # Re-read after overrides (globals() assignments above).
    D = int(globals()["D"])
    D_E = int(globals()["D_E"])
    D_A = int(globals()["D_A"])
    HEADS = int(globals()["HEADS"])
    DEPTH = int(globals()["DEPTH"])
    ATTENTION_MLP_HIDDENS = _parse_mlp_hiddens(globals()["ATTENTION_MLP_HIDDENS"])
    ATTENTION_KIND = str(globals()["ATTENTION_KIND"]).strip().lower()
    GRPE_MAX_HOP = int(globals()["GRPE_MAX_HOP"])
    # Dense is the v12 default (fast on IEEE 118); env forces dense or sparse.
    dense_env = os.environ.get("GENCO_TPGT_EXPHORMER_DENSE", "").strip().lower()
    if dense_env in ("0", "false", "no"):
        EXPHORMER_DENSE = False
    elif dense_env in ("1", "true", "yes"):
        EXPHORMER_DENSE = True
    else:
        EXPHORMER_DENSE = True
    d_exp_env = os.environ.get("GENCO_TPGT_EXPHORMER_D_EXP", "").strip()
    if d_exp_env:
        EXPHORMER_D_EXP = int(d_exp_env)
    else:
        EXPHORMER_D_EXP = 4
    if ATTENTION_KIND not in ("grit", "joint_mlp", "grpe", "grpe_elec", "exphormer"):
        raise ValueError(
            f"unknown ATTENTION_KIND {ATTENTION_KIND!r}; "
            "use grit, joint_mlp, grpe, grpe_elec, or exphormer"
        )
    if GRPE_MAX_HOP < 0:
        raise ValueError(f"GRPE_MAX_HOP must be >= 0, got {GRPE_MAX_HOP}")
    if EXPHORMER_D_EXP < 2 or EXPHORMER_D_EXP % 2 != 0:
        raise ValueError(f"EXPHORMER_D_EXP must be even and >= 2, got {EXPHORMER_D_EXP}")
    RUN_DIR_NAME = f"tpgt_task11_{RUN_TAG}"
    if DEPTH < 1 or HEADS < 1 or D < 1 or D_E < 1 or D_A < 1:
        raise ValueError(
            f"invalid architecture dims D={D} D_E={D_E} D_A={D_A} HEADS={HEADS} "
            f"DEPTH={DEPTH} ATTENTION_MLP_HIDDENS={ATTENTION_MLP_HIDDENS}"
        )
    apply_case_config()
    return name


def architecture_summary() -> str:
    node_in = NODE_FEATURE_DIM if USE_NODE_SHUNT else NODE_SPECIFIED_DIM
    node_txt = "P,Q,V,θ,busBS" if USE_NODE_SHUNT else "spec"
    case_txt = f" case={CASE_NAME}" if CASE_NAME != "case118" else ""
    h1, h2 = ATTENTION_MLP_HIDDENS
    score_in = 2 * D_A + RPE_DIM
    hop_bins = grpe_num_hop_bins()
    if ATTENTION_KIND == "joint_mlp":
        attn_txt = f"joint_mlp({score_in}->{h1}->{h2}->1, raw_rpe={RPE_DIM})"
    elif ATTENTION_KIND == "grpe":
        attn_txt = (
            f"grpe(Park P/E hop_bins={hop_bins} edge_types={GRPE_NUM_EDGE_TYPES})/"
            f"sqrt({D_A})"
        )
    elif ATTENTION_KIND == "grpe_elec":
        attn_txt = (
            f"grpe_elec(Park P/E + stage U r)/sqrt({D_A}), raw_rpe={RPE_DIM}, "
            f"hop_bins={hop_bins}"
        )
    elif ATTENTION_KIND == "exphormer":
        kernel = "dense" if EXPHORMER_DENSE else "sparse"
        attn_txt = (
            f"exphormer({kernel}, d_exp={EXPHORMER_D_EXP}, pairs=15, "
            f"mlp {score_in}->{h1}->{h2}->1)"
        )
    else:
        attn_txt = "grit"
    return (
        f"arch {RUN_TAG}: DEPTH={DEPTH} D={D} D_E={D_E} D_A={D_A} HEADS={HEADS} "
        f"node_in={node_in} ({node_txt}) attn={attn_txt}{case_txt} "
        f"checkpoint_version={CHECKPOINT_VERSION}"
    )


def count_parameters(model: nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


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


def effective_eval_batch_size(bus_count: int, requested: int) -> int:
    """Cap forward batch for large grids (500-bus attention is O(buses²) per case)."""
    if bus_count <= 118:
        return requested
    if bus_count <= 200:
        cap = min(requested, 32)
    elif bus_count <= 350:
        cap = min(requested, 16)
    else:
        cap = min(requested, 8)
    if cap < requested:
        print(
            f"  eval batch capped {requested} -> {cap} for {bus_count}-bus grid (set GENCO_TPGT_EVAL_BATCH lower to silence)",
            flush=True,
        )
    return max(1, cap)


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


def mix_topologies_per_step(network_count: int) -> bool:
    """Whether one optimizer step may batch cases from different topologies (global shuffle)."""
    for index, arg in enumerate(sys.argv):
        if arg == "--mix-topologies":
            return True
        if arg == "--no-mix-topologies":
            return False
    env = os.environ.get("GENCO_TPGT_MIX_TOPOLOGIES", "").strip().lower()
    if env in ("0", "false", "no"):
        return False
    if env in ("1", "true", "yes"):
        return True
    return network_count <= MIX_TOPOLOGIES_THRESHOLD


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


def merge_many(
    splits: list[Split],
    name: str | None = None,
    rpe_scales: tuple[float, float] | None = None,
) -> Split:
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

    unify_network_rpe(merged_networks, rpe_scales)
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
    rpe_scales: tuple[float, float] | None = None,
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
    bus_count = parquet_bus_count(raw)
    branch_rows = pq.ParquetFile(raw / "branch_data.parquet").metadata.num_rows
    if branch_rows % full != 0:
        raise RuntimeError(f"{raw} branch rows are not one block per scenario.")
    branch_count = branch_rows // full
    bus = pq.read_table(
        raw / "bus_data.parquet",
        columns=["scenario", "bus", "Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF", "BS"],
    ).slice(start * bus_count, count * bus_count)
    bus_ids = _blocks(bus.column("bus").to_numpy(), bus_count)
    scenario_blocks = _blocks(bus.column("scenario").to_numpy(), bus_count)
    if not np.all(scenario_blocks[:, :1] == scenario_blocks):
        raise RuntimeError(f"{raw} bus rows are not grouped by scenario.")
    if not np.all(bus_ids == np.arange(bus_count)):
        raise RuntimeError(f"{raw} buses are not ordered 0..{bus_count - 1} within each scenario.")
    pq_flag = _blocks(bus.column("PQ").to_numpy(), bus_count)
    pv_flag = _blocks(bus.column("PV").to_numpy(), bus_count)
    ref_flag = _blocks(bus.column("REF").to_numpy(), bus_count)
    type_code = np.zeros((count, bus_count), dtype=np.float32)
    type_code[pv_flag > 0.5] = 1
    type_code[ref_flag > 0.5] = 2
    if np.any((pq_flag > 0.5) & (pv_flag > 0.5)) or np.any((type_code == 0) & (pq_flag <= 0.5)):
        raise RuntimeError(f"{name} has a bus that is not exactly one of PQ, PV, or slack.")
    power_p = (_blocks(bus.column("Pg").to_numpy(), bus_count) - _blocks(bus.column("Pd").to_numpy(), bus_count)) / BASE_MVA
    power_q = (_blocks(bus.column("Qg").to_numpy(), bus_count) - _blocks(bus.column("Qd").to_numpy(), bus_count)) / BASE_MVA
    voltage = _blocks(bus.column("Vm").to_numpy(), bus_count)
    angle = np.deg2rad(_blocks(bus.column("Va").to_numpy(), bus_count))
    # Parquet ``BS`` is already p.u. bus-shunt susceptance (device only).
    shunt_b = _blocks(bus.column("BS").to_numpy(), bus_count).astype(np.float32)

    branch = pq.read_table(
        raw / "branch_data.parquet",
        columns=["scenario", "idx", "from_bus", "to_bus", "r", "x", "br_status"],
    ).slice(start * branch_count, count * branch_count)
    branch_scenario = _blocks(branch.column("scenario").to_numpy(), branch_count)
    bus_scenario = _blocks(bus.column("scenario").to_numpy(), bus_count)
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
        ybus = ybus_from_entries(y_by_scenario[scenario_id], bus_count)
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
            name,
            representatives,
            scenario_ids,
            type_code,
            y_by_scenario,
            status,
            origin,
            destination,
            resistance,
            reactance,
            branch_count,
            shunt_b,
            rpe_scales=rpe_scales,
            bus_count=bus_count,
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
    shunt_b: np.ndarray,
    rpe_scales: tuple[float, float] | None = None,
    bus_count: int = 118,
) -> list[Network]:
    rep_specs: list[tuple[int, np.ndarray, list[tuple[int, int, float, float]], np.ndarray]] = []
    for local in representatives:
        types = type_code[local].astype(np.int64)
        if int(np.sum(types == 2)) != 1:
            raise RuntimeError(f"{name} scenario {scenario_ids[local]} has {int(np.sum(types == 2))} slack buses.")
        scenario_id = scenario_ids[local]
        ybus = ybus_from_entries(y_by_scenario[scenario_id], bus_count)
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
        rep_specs.append((local, types, branches, ybus))

    if rpe_scales is not None:
        global_re, global_im = rpe_scales
        print(
            f"  Zeff RPE uses train scales s_R={global_re:.6g} s_X={global_im:.6g} "
            f"({len(rep_specs)} topology(ies) in {name})",
            flush=True,
        )
    else:
        global_re, global_im = global_zeff_rpe_scales([branches for _local, _types, branches, _ybus in rep_specs])
        print(
            f"  global Zeff RPE scales (train-fit): s_R={global_re:.6g} s_X={global_im:.6g} "
            f"over {len(rep_specs)} topology(ies) ({name}); meshed_rpe once per topology (not per scenario)",
            flush=True,
        )

    networks: list[Network] = []
    for local, types, branches, ybus in rep_specs:
        features, component = meshed_rpe(bus_count, branches, ybus, re_scale=global_re, im_scale=global_im)
        hop_id, edge_id = build_grpe_graph_ids(bus_count, branches)
        network = Network(
            ybus=torch.from_numpy(ybus),
            rpe=torch.from_numpy(features),
            component=torch.from_numpy(component.astype(np.int64)),
            types=torch.from_numpy(types),
            branches=tuple(branches),
            shunt_b=torch.from_numpy(np.asarray(shunt_b[local], dtype=np.float32)),
            rpe_re_scale=global_re,
            rpe_im_scale=global_im,
            hop_id=torch.from_numpy(hop_id),
            edge_id=torch.from_numpy(edge_id),
        )
        if ATTENTION_KIND == "exphormer":
            attach_exphormer_graph(network)
        networks.append(network)
    return networks


def log_counts(data_root: Path, results: Path, spec: TaskSpec | None = None) -> dict:
    spec = spec or TASKS["1.1"]
    report = {}
    if spec.is_task11:
        folders = [("train", "n", "feasible"), ("valid", "n", "feasible")]
    else:
        folders = [("train", *folder) for folder in task_folders(data_root, "train")]
        folders.extend(("valid", *folder) for folder in task_folders(data_root, "valid"))
    for grid in eval_case_names(spec):
        for _label, topology, condition in TEST_PARTS:
            folders.append(("test", topology, condition, grid))
    for item in folders:
        if len(item) == 3:
            split, topology, condition = item
            grid = CASE_NAME
        else:
            split, topology, condition, grid = item
        raw = raw_dir(data_root, split, topology, condition, grid)
        count = scenario_count(raw)
        key = (
            f"{split}/{topology}/{condition}"
            if grid == CASE_NAME and spec.task_id != "3.1"
            else f"{split}/{grid}/{topology}/{condition}"
        )
        if not spec.is_task11:
            report[key] = {"loaded": count}
            print(f"count {key}: {count}", flush=True)
            continue
        expected = EXPECTED_COUNTS.get((split, topology, condition))
        report[key] = {"loaded": count, "expected": expected}
        if expected is None:
            print(f"count {key}: {count}", flush=True)
        else:
            print(f"count {key}: {count} (expected {expected})", flush=True)
            if count != expected:
                print(
                    f"  count differs from the PFΔ Task 1.1 split. Using the {count} loaded scenarios.",
                    flush=True,
                )
    path = results / f"task{spec.file_tag}_tpgt_counts.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def raw_dir(
    data_root: Path,
    split: str,
    topology: str,
    condition: str,
    grid: str | None = None,
) -> Path:
    """PFΔ parquet folder: ``<root>/<CASE_NAME>/<split>/<grid>/<topology>/<condition>/raw``.

    Default IEEE-118 packs (including Task 3.1 cross-grid tests) use outer ``case118``.
    Native IEEE-57 Task 1.1 packs use outer ``case57``.
    """
    return data_root / CASE_NAME / split / (grid or CASE_NAME) / topology / condition / "raw"


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


def colab_local_cache_dir(spec: TaskSpec | None = None) -> Path:
    """Colab SSD mirror path; must use ``RUN_DIR_NAME`` after ``apply_architecture_config()``."""
    if spec is None or spec.is_task11:
        return Path(f"/content/{RUN_DIR_NAME}_cache")
    return Path(f"/content/{spec.run_dir_name}_cache")


def split_cache_dirs(checkpoints: Path, spec: TaskSpec | None = None) -> tuple[Path, Path]:
    """Return (fast_cache, persistent_cache). On Colab, fast_cache is local SSD."""
    persistent = checkpoints / "cache"
    if in_colab():
        return colab_local_cache_dir(spec), persistent
    return persistent, persistent


_SPLIT_CACHE_MIRRORED = False


def _mirror_cache_file(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            dest_stat = destination.stat()
            src_stat = source.stat()
            if dest_stat.st_size == src_stat.st_size and dest_stat.st_mtime >= src_stat.st_mtime:
                return False
        except OSError:
            pass
    shutil.copy2(source, destination)
    return True


def sync_split_cache_from_drive(persistent: Path, active: Path, *, force: bool = False) -> None:
    """Copy Drive split caches to local disk so ``torch.load`` is fast on Colab."""
    global _SPLIT_CACHE_MIRRORED
    if persistent == active or not persistent.is_dir():
        return
    if _SPLIT_CACHE_MIRRORED and not force:
        return
    active.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(persistent.iterdir()):
        if path.is_file() and _mirror_cache_file(path, active / path.name):
            copied += 1
    _SPLIT_CACHE_MIRRORED = True
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


def persist_split_caches(active: Path, persistent: Path, *, label: str = "") -> None:
    """Flush Colab SSD caches to Drive and log how many artifacts are stored."""
    sync_split_cache_to_drive(active, persistent)
    if persistent == active:
        root = persistent
    else:
        root = persistent
    if not root.is_dir():
        return
    stamps = sum(1 for _ in root.glob("*.stamp.json"))
    tensors = sum(1 for _ in root.glob("*.pt"))
    extras = sum(1 for _ in root.glob("*.json"))
    tag = f" ({label})" if label else ""
    print(
        f"Split caches on Drive{tag}: {tensors} .pt, {stamps} stamps, {extras} json under {root}",
        flush=True,
    )


def split_cache_stamp_for_raw(raw: Path, limit: int) -> dict:
    return {
        "version": CACHE_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "bytes": raw.joinpath("bus_data.parquet").stat().st_size,
        "scenario_limit": limit,
        "data_fraction": float(DATA_FRACTION),
    }


LEGACY_STAMP_REASONS = frozenset(
    {
        "missing data_fraction in stamp (legacy cache)",
        "missing RPE scales in stamp (legacy cache)",
    }
)


def split_cache_stamp_mismatch(
    saved: dict,
    expected: dict,
    *,
    rpe_scales: tuple[float, float] | None,
    is_train: bool,
) -> str | None:
    if saved.get("version") != expected["version"]:
        return "cache version"
    if saved.get("checkpoint_version") != expected["checkpoint_version"]:
        return "checkpoint_version"
    if saved.get("bytes") != expected["bytes"]:
        return "dataset bytes"
    if saved.get("scenario_limit") != expected["scenario_limit"]:
        return "scenario_limit"
    if saved.get("data_fraction") is None:
        return "missing data_fraction in stamp (legacy cache)"
    if not np.isclose(float(saved["data_fraction"]), float(expected["data_fraction"])):
        return "DATA_FRACTION"
    if "rpe_re_scale" not in saved or "rpe_im_scale" not in saved:
        return "missing RPE scales in stamp (legacy cache)"
    if not is_train and rpe_scales is not None:
        if not np.isclose(float(saved["rpe_re_scale"]), rpe_scales[0], rtol=0, atol=1e-9):
            return "rpe_re_scale vs train"
        if not np.isclose(float(saved["rpe_im_scale"]), rpe_scales[1], rtol=0, atol=1e-9):
            return "rpe_im_scale vs train"
    return None


def parse_split_cache_name(stem: str) -> tuple[str, str, str] | None:
    """``train_n-1_feasible`` → (train, n-1, feasible)."""
    if "_" not in stem:
        return None
    split_name, rest = stem.split("_", 1)
    if "_" not in rest:
        return None
    topology, condition = rest.rsplit("_", 1)
    if split_name not in ("train", "valid", "test"):
        return None
    return split_name, topology, condition


def patch_legacy_split_stamp(
    stamp_path: Path,
    expected: dict,
    rpe_scales: tuple[float, float] | None,
) -> bool:
    """Fill legacy stamp fields when core keys still match ``expected`` (keeps ``.pt`` on disk)."""
    try:
        saved = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    is_train = stamp_path.stem.startswith("train_")
    reason = split_cache_stamp_mismatch(saved, expected, rpe_scales=rpe_scales, is_train=is_train)
    if reason not in LEGACY_STAMP_REASONS:
        return False
    changed = False
    if saved.get("data_fraction") is None:
        saved["data_fraction"] = expected["data_fraction"]
        changed = True
    if ("rpe_re_scale" not in saved or "rpe_im_scale" not in saved) and rpe_scales is not None:
        saved["rpe_re_scale"] = rpe_scales[0]
        saved["rpe_im_scale"] = rpe_scales[1]
        changed = True
    if not changed:
        return False
    stamp_path.write_text(json.dumps(saved), encoding="utf-8")
    return True


def parse_rebuild_cache() -> bool:
    for arg in sys.argv:
        if arg in ("--rebuild-cache",):
            return True
    env = os.environ.get("GENCO_TPGT_REBUILD_CACHE", "").strip().lower()
    return env in ("1", "true", "yes")


def purge_split_caches(active: Path, persistent: Path, reason: str) -> int:
    """Remove split ``.pt`` / ``.json`` caches (and train RPE scales) on both cache roots."""
    removed = 0
    for root in {active, persistent}:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            if path.name == RPE_SCALES_FILE or path.suffix in {".pt", ".json"}:
                path.unlink()
                removed += 1
    if removed:
        global _SPLIT_CACHE_MIRRORED
        _SPLIT_CACHE_MIRRORED = False
        print(
            f"Purged {removed} split-cache file(s) ({reason}); splits will rebuild on load.",
            flush=True,
        )
    return removed


def refresh_split_rpe(
    split: Split,
    persistent: Path,
    *,
    is_train: bool,
    rpe_scales: tuple[float, float] | None,
) -> None:
    """Ensure cached ``network.rpe`` matches ``meshed_rpe`` for stored branch lists and scales."""
    if is_train:
        re_scale, im_scale = unify_rpe_normalization(split)
        save_rpe_global_scales(persistent, re_scale, im_scale)
        return
    scales = rpe_scales if rpe_scales is not None else load_rpe_global_scales(persistent)
    if scales is None:
        return
    unify_rpe_normalization(split, scales)


def reconcile_split_caches(active: Path, persistent: Path, data_root: Path) -> None:
    """Drop stale caches when ``DATA_FRACTION``, architecture, or train RPE scales changed."""
    global _SPLIT_CACHE_MIRRORED
    if parse_rebuild_cache():
        purge_split_caches(active, persistent, "GENCO_TPGT_REBUILD_CACHE=1")
        _SPLIT_CACHE_MIRRORED = False
        return
    folders = task_folders(data_root, "train")
    if not folders:
        return
    topology, condition = folders[0]
    raw = raw_dir(data_root, "train", topology, condition)
    if not raw.joinpath("bus_data.parquet").is_file():
        return
    name = f"train_{topology}_{condition}"
    full = scenario_count(raw)
    limit = scenario_limit(raw)
    if limit is None:
        limit = full
    expected = split_cache_stamp_for_raw(raw, limit)
    scales = load_rpe_global_scales(persistent) or load_rpe_global_scales(active)
    patched = 0
    for root in (persistent, active):
        if not root.is_dir():
            continue
        for stamp_path in sorted(root.glob("*.json")):
            if stamp_path.name == RPE_SCALES_FILE:
                continue
            parsed = parse_split_cache_name(stamp_path.stem)
            if parsed is None:
                continue
            split_name, topo, cond = parsed
            folder_raw = raw_dir(data_root, split_name, topo, cond)
            if not folder_raw.joinpath("bus_data.parquet").is_file():
                continue
            folder_limit = scenario_limit(folder_raw)
            if folder_limit is None:
                folder_limit = scenario_count(folder_raw)
            stamp_expected = split_cache_stamp_for_raw(folder_raw, folder_limit)
            stamp_scales = None if split_name == "train" else scales
            if patch_legacy_split_stamp(stamp_path, stamp_expected, stamp_scales):
                patched += 1
    if patched:
        print(f"Patched {patched} legacy split-cache stamp(s) in place (reusing existing .pt).", flush=True)

    for root in (active, persistent):
        stamp_path = root / f"{name}.json"
        if not stamp_path.is_file():
            continue
        try:
            saved = json.loads(stamp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            purge_split_caches(active, persistent, f"unreadable stamp {stamp_path.name}")
            return
        reason = split_cache_stamp_mismatch(saved, expected, rpe_scales=None, is_train=True)
        if reason:
            purge_split_caches(active, persistent, reason)
            return

    scales = load_rpe_global_scales(persistent)
    if scales is None:
        return
    re_scale, im_scale = scales
    for root in (active, persistent):
        if not root.is_dir():
            continue
        for stamp_path in sorted(root.glob("*.json")):
            if stamp_path.name == RPE_SCALES_FILE:
                continue
            try:
                saved = json.loads(stamp_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if stamp_path.stem.startswith("train_"):
                continue
            if "rpe_re_scale" not in saved:
                purge_split_caches(active, persistent, f"non-train stamp missing scales ({stamp_path.stem})")
                return
            if not np.isclose(float(saved["rpe_re_scale"]), re_scale, rtol=0, atol=1e-9) or not np.isclose(
                float(saved["rpe_im_scale"]), im_scale, rtol=0, atol=1e-9
            ):
                purge_split_caches(
                    active,
                    persistent,
                    f"cached RPE scales in {stamp_path.stem} != {RPE_SCALES_FILE}",
                )
                return


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
    rpe_scales: tuple[float, float] | None = None,
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
    stamp = split_cache_stamp_for_raw(raw, limit)
    stamp_path = active / f"{name}.json"
    tensor_path = active / f"{name}.pt"
    is_train = name.startswith("train")
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
        stamp_reason = split_cache_stamp_mismatch(
            saved,
            stamp,
            rpe_scales=rpe_scales,
            is_train=is_train,
        )
        if version_ok and limit == old_limit and stamp_reason in LEGACY_STAMP_REASONS:
            scales_for_stamp = None if is_train else rpe_scales
            if patch_legacy_split_stamp(stamp_path, stamp, scales_for_stamp):
                saved = json.loads(stamp_path.read_text(encoding="utf-8"))
                stamp_reason = split_cache_stamp_mismatch(
                    saved,
                    stamp,
                    rpe_scales=rpe_scales,
                    is_train=is_train,
                )
                peer = persistent / stamp_path.name
                if persistent != active and peer.is_file():
                    patch_legacy_split_stamp(peer, stamp, scales_for_stamp)
                elif persistent != active:
                    _mirror_cache_file(stamp_path, peer)
        if version_ok and limit == old_limit and stamp_reason is None:
            load_started = time.perf_counter()
            print(f"Loading cached {name} split from {tensor_path}", flush=True)
            payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
            elapsed = time.perf_counter() - load_started
            print(f"  loaded {name} split in {elapsed:.1f}s ({tensor_path})", flush=True)
            refresh_split_rpe(payload, persistent, is_train=is_train, rpe_scales=rpe_scales)
            sync_split_cache_to_drive(active, persistent)
            return _compact_if(payload, name, compact)
        if version_ok and limit == old_limit and stamp_reason is not None:
            print(f"Stale split cache for {name} ({stamp_reason}); rebuilding.", flush=True)
        if version_ok and limit > old_limit:
            existing = torch.load(tensor_path, map_location="cpu", weights_only=False)
            if isinstance(existing, Split) and existing.count == old_limit:
                build_started = time.perf_counter()
                split = extend_split_prefix(raw, name, existing, limit)
                print(
                    f"  extended {name} split in {time.perf_counter() - build_started:.1f}s",
                    flush=True,
                )
                if name.startswith("train"):
                    unify_rpe_normalization(split)
                    re_scale, im_scale = split_rpe_scales(split)
                    save_rpe_global_scales(persistent, re_scale, im_scale)
                    stamp["rpe_re_scale"] = re_scale
                    stamp["rpe_im_scale"] = im_scale
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
    if is_train:
        scales_for_read: tuple[float, float] | None = None
    else:
        scales_for_read = rpe_scales if rpe_scales is not None else load_rpe_global_scales(persistent)
        if scales_for_read is None:
            raise RuntimeError(
                f"Missing train RPE scales for {name}; build the train split cache first "
                f"(expected {rpe_scales_path(persistent)})."
            )
    split = read_split(raw, name, prefix_end=limit, rpe_scales=scales_for_read)
    print(f"  built {name} split from parquet in {time.perf_counter() - build_started:.1f}s", flush=True)
    _compact_if(split, name, compact)
    save_started = time.perf_counter()
    re_scale, im_scale = split_rpe_scales(split)
    stamp["rpe_re_scale"] = re_scale
    stamp["rpe_im_scale"] = im_scale
    torch.save(split, tensor_path)
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8")
    print(f"  saved {name} cache in {time.perf_counter() - save_started:.1f}s", flush=True)
    sync_split_cache_to_drive(active, persistent)
    return split


def load_task_split(data_root: Path, split_name: str, active: Path, persistent: Path) -> Split:
    """Union of every ``split_name`` folder of the task (each cached on its own, DATA_FRACTION per folder).

    A task with a single folder (Task 1.1) returns that folder's split unchanged.
   Train splits fit ``s_R,s_X`` and write ``rpe_global_scales.json``; valid/test reuse those scales.
    """
    folders = task_folders(data_root, split_name)
    if not folders:
        raise FileNotFoundError(f"No {split_name} folders under {data_root / 'case118' / split_name}")
    multi = len(folders) > 1
    train_scales = load_rpe_global_scales(persistent) if split_name != "train" else None
    parts = [
        cached_split(
            active,
            persistent,
            raw_dir(data_root, split_name, topology, condition),
            f"{split_name}_{topology}_{condition}",
            compact=multi,
            rpe_scales=train_scales,
        )
        for topology, condition in folders
    ]
    print(
        f"{split_name} mix: " + folder_mix_text([(folder, part.count) for folder, part in zip(folders, parts)]),
        flush=True,
    )
    if not multi:
        split = parts[0]
    else:
        merge_started = time.perf_counter()
        split = merge_many(
            parts,
            name=split_name,
            rpe_scales=None if split_name == "train" else train_scales,
        )
        print(
            f"{split_name}: merged {len(parts)} folders into {split.count} cases, {len(split.networks)} topologies "
            f"in {time.perf_counter() - merge_started:.1f}s",
            flush=True,
        )
    if split_name == "train":
        re_scale, im_scale, raw_re, raw_im = fit_global_zeff_rpe_scales_from_networks(split.networks)
        unify_rpe_normalization(split, (re_scale, im_scale))
        save_rpe_global_scales(
            persistent,
            re_scale,
            im_scale,
            raw_re=raw_re,
            raw_im=raw_im,
            source=f"train_fit_{split_name}",
        )
        log_rpe_scales_provenance(persistent, context=f"saved {split_name}")
        print(
            f"{split_name}: saved RPE scales s_R={re_scale:.6g} s_X={im_scale:.6g} "
            f"(raw peaks {raw_re:.6g}, {raw_im:.6g}) -> {rpe_scales_path(persistent)}",
            flush=True,
        )
    elif train_scales is not None:
        re_scale, im_scale = train_scales
        print(
            f"{split_name}: RPE normalization uses train scales s_R={re_scale:.6g} s_X={im_scale:.6g}",
            flush=True,
        )
    return split


DATASET_RELATIVE = Path("Foundation Model") / "vendor" / "pfdelta_task1.1"
WINDOWS_DRIVE = Path(r"K:\My Drive")
COLAB_DRIVE = Path("/content/drive/MyDrive")


def dataset_relative(spec: TaskSpec | None = None) -> Path:
    spec = spec or TASKS["1.1"]
    if spec.is_task11:
        if CASE_NAME == "case118":
            return DATASET_RELATIVE
        return Path("Foundation Model") / "vendor" / f"pfdelta_task1.1_{CASE_NAME}"
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
    """Return the saved dataset root of ``spec`` when ``CASE_NAME`` is already on Drive.

    This only checks directory names. It does not copy the dataset.
    """
    for candidate in dataset_candidates(drive, spec):
        if (candidate / CASE_NAME).is_dir():
            return candidate
    return None


def dataset_missing_files(data_root: Path, spec: TaskSpec) -> list[Path]:
    """Parquet files of ``spec``'s published train/valid/test folders that are absent under ``data_root``."""
    folders = [
        ("test", topology, condition, grid)
        for grid in eval_case_names(spec)
        for _label, topology, condition in TEST_PARTS
    ]
    for split in ("train", "valid"):
        folders.extend((split, *folder, CASE_NAME) for folder in spec.folders)
    return [
        raw_dir(data_root, split, topology, condition, grid) / name
        for split, topology, condition, grid in folders
        for name in PARQUET_FILES
        if not (raw_dir(data_root, split, topology, condition, grid) / name).is_file()
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
    if CASE_NAME != "case118":
        return ensure_case_task11_dataset(work, drive, CASE_NAME)
    saved = drive_dataset(drive)
    if saved is not None:
        print(f"Dataset already present: {saved}", flush=True)
        print(f"{CASE_NAME} is already on Drive; skipping download.", flush=True)
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


def ensure_case_task11_dataset(work: Path, drive: Path | None, case: str) -> Path:
    """Task 1.1 on a non-118 case: build gridfm-style parquet from MIT PFΔ JSON.

    Parquet pack is written to Drive (reusable). Raw JSON extract stays on local SSD
    under Colab (``/content/pfdelta_raw_<case>``) — extracting the ~1.6GB case tar onto
    Drive often aborts mid-way and leaves a partial ``case/`` folder.
    """
    # ``build_task11_case_dataset`` already no-ops when the full pack exists and resumes
    # missing N-1/N-2 folders. Reload so a long-lived Colab kernel cannot keep a stale module.
    import importlib

    import pfdelta_case_parquet as case_parquet

    case_parquet = importlib.reload(case_parquet)

    relative = dataset_relative(TASKS["1.1"])
    if drive is not None and (in_colab() or (drive / "Foundation Model").is_dir()):
        out_root = drive / relative
    else:
        requested = dataset_request(TASKS["1.1"])
        out_root = requested if requested.is_absolute() else work / relative.name
    if in_colab():
        extract_root = Path(f"/content/pfdelta_raw_{case}")
    else:
        extract_root = work / f"pfdelta_raw_{case}"
    print(f"PFΔ JSON extract root: {extract_root}", flush=True)
    return case_parquet.build_task11_case_dataset(case, out_root, extract_root)


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
    if data.joinpath(CASE_NAME).is_dir() and not dataset_missing_files(data, spec):
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
    shunt = getattr(network, "shunt_b", None)
    hop_id, edge_id = ensure_network_grpe_ids(network)

    def _maybe(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else host_to_device(tensor, device)

    if ATTENTION_KIND == "exphormer":
        allow, tau, n_virt = ensure_network_exphormer(network)
        exph_kw = dict(
            exph_allow=host_to_device(allow, device),
            exph_tau=host_to_device(tau, device),
            exph_n_virt=n_virt,
            exph_recv=_maybe(getattr(network, "exph_recv", None)),
            exph_send=_maybe(getattr(network, "exph_send", None)),
            exph_edge_tau=_maybe(getattr(network, "exph_edge_tau", None)),
            exph_pair_id=_maybe(getattr(network, "exph_pair_id", None)),
        )
    else:
        exph_kw = dict(
            exph_allow=_maybe(getattr(network, "exph_allow", None)),
            exph_tau=_maybe(getattr(network, "exph_tau", None)),
            exph_n_virt=int(getattr(network, "exph_n_virt", 0) or 0),
            exph_recv=_maybe(getattr(network, "exph_recv", None)),
            exph_send=_maybe(getattr(network, "exph_send", None)),
            exph_edge_tau=_maybe(getattr(network, "exph_edge_tau", None)),
            exph_pair_id=_maybe(getattr(network, "exph_pair_id", None)),
        )

    return Network(
        ybus=host_to_device(network.ybus, device),
        rpe=host_to_device(network.rpe, device).to(dtype=torch.float32),
        component=component,
        types=types,
        pair_type_masks=masks,
        pair_embed=pair_embed,
        branches=getattr(network, "branches", None),
        shunt_b=None if shunt is None else host_to_device(shunt, device),
        # Keep train-fit Zeff scales: audit_rpe_correctness recomputes meshed_rpe with these;
        # dropping them falls back to per-component norms and falsely fails recompute_from_branches.
        rpe_re_scale=getattr(network, "rpe_re_scale", None),
        rpe_im_scale=getattr(network, "rpe_im_scale", None),
        hop_id=host_to_device(hop_id, device),
        edge_id=host_to_device(edge_id, device),
        **exph_kw,
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
    """Edge features for this topology (raw RPE for v10); cached in eval on that Network object only."""
    if not model.training and network.pair_embed is not None:
        return network.pair_embed
    encoded = model.encode_pairs(network.rpe)
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
    shunt_b: torch.Tensor | None = None,
    hop_id: torch.Tensor | None = None,
    edge_id: torch.Tensor | None = None,
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
    if shunt_b is None:
        shunt_b = network_shunt_b(network)
    if hop_id is None or edge_id is None:
        hop_id, edge_id = ensure_network_grpe_ids(network)
    prediction, residuals = model(
        specified,
        layout.types,
        network.rpe,
        network.pair_type_masks,
        network.ybus if ybus is None else ybus,
        pair=pair,
        layout=layout,
        shunt_b=shunt_b,
        hop_id=hop_id,
        edge_id=edge_id,
        **exphormer_forward_kwargs(network),
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
    shunt_b: torch.Tensor | None = None,
    hop_id: torch.Tensor | None = None,
    edge_id: torch.Tensor | None = None,
) -> tuple[TaskLossBreakdown, list[torch.Tensor]]:
    if pair is None:
        pair = pair_features(model, network)
    if layout is None:
        layout = network_layout(network)
    prediction, target, residuals = forward_cases(
        model,
        split,
        network,
        index,
        device,
        pair,
        ybus,
        layout,
        shunt_b=shunt_b,
        hop_id=hop_id,
        edge_id=edge_id,
    )
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
    """Pair features (U, N, N, *) via ``encode_pairs`` and complex128 Ybus (U, N, N)."""
    rpe = torch.stack([network.rpe for network in networks])
    rpe = rpe_channel_dropout(rpe, RPE_CHANNEL_DROPOUT, model.training, keep=keep)
    pair = model.encode_pairs(rpe)
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
        self.shunt_b: torch.Tensor | None = None
        self.hop_id: torch.Tensor | None = None
        self.edge_id: torch.Tensor | None = None
        if len(networks) == 1:
            self.kind = "single"
            return
        keys = {network_type_key(network) for network in networks}
        self.kind = "shared" if len(keys) == 1 else "mixed"
        self.ybus = torch.stack([network.ybus for network in networks]).to(torch.complex128)
        self.shunt_b = torch.stack([network_shunt_b(network) for network in networks])
        self.hop_id = torch.stack([ensure_network_grpe_ids(network)[0] for network in networks])
        self.edge_id = torch.stack([ensure_network_grpe_ids(network)[1] for network in networks])
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
        """Pair features for attention: raw RPE (v10) or ``pair_encoder`` output (GRIT)."""
        if self.kind == "single":
            network = self.networks[0]
            if not model.training:
                return pair_features(model, network)
            rpe = rpe_channel_dropout(network.rpe, RPE_CHANNEL_DROPOUT, model.training, keep=keep)
            return model.encode_pairs(rpe)
        rpe = torch.stack([network.rpe for network in self.networks])
        rpe = rpe_channel_dropout(rpe, RPE_CHANNEL_DROPOUT, model.training, keep=keep)
        return model.encode_pairs(rpe)

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
        # Exphormer allow/tau are topology-specific and not yet batched; run one network at a time.
        if ATTENTION_KIND == "exphormer":
            owners = self.owner[start:stop]
            parts: list[tuple[TaskLossBreakdown, float]] = []
            residuals: list[torch.Tensor] = []
            for network_id in sorted(set(int(x) for x in owners.tolist())):
                mask = owners == network_id
                local_cases = chunk[mask]
                # ``pair`` is stacked per topology (U,N,N,E), not per case — index by network_id.
                local_pair = pair[network_id] if pair.dim() == 4 else pair
                breakdown, stage_res = run_batch_residuals(
                    model,
                    split,
                    self.networks[network_id],
                    local_cases,
                    device,
                    pair=local_pair,
                )
                parts.append((breakdown, float(len(local_cases))))
                if not residuals:
                    residuals = list(stage_res)
                else:
                    for i, r in enumerate(stage_res):
                        residuals[i] = torch.cat((residuals[i], r), dim=0)
            return _weighted_loss_breakdown(parts), residuals
        select = self.select[start:stop]
        layout = None if self.member_layout is None else self.member_layout.take(select)
        assert self.shunt_b is not None and self.hop_id is not None and self.edge_id is not None
        return run_batch_residuals(
            model,
            split,
            self.networks[0],
            chunk,
            device,
            pair=pair[select],
            ybus=self.ybus[select],
            layout=layout,
            shunt_b=self.shunt_b[select],
            hop_id=self.hop_id[select],
            edge_id=self.edge_id[select],
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
    # Exphormer kernels force float32; AMP Half activations collide with Float32 RPE/virt writes.
    use_amp = device.type == "cuda" and ATTENTION_KIND != "exphormer"
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
            scaler.unscale_(optimizer)
        if GRAD_CLIP_NORM > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is not None and tuple(grad.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"grad shape {tuple(grad.shape)} != param {tuple(param.shape)} for {name}"
                )
        try:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        except RuntimeError as error:
            # Stale/misaligned Adam buffers after resume: drop momentum and retry once.
            message = str(error)
            if "must match the size of tensor" not in message:
                raise
            mismatch = live_optimizer_state_shape_mismatch(optimizer)
            reset_optimizer_state(optimizer)
            # Grads are already unscaled; do not call scaler.step again.
            optimizer.step()
            if scaler is not None:
                scaler.update()
            print(
                f"Adam state reset after shape error ({mismatch or message.split(chr(10), 1)[0]}); "
                "continued with fresh momentum.",
                flush=True,
            )
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


def pack_step_limits() -> tuple[int, int]:
    """Max topologies and cases when packing same-type grids. Topology cap 1 disables packing."""

    def bounded(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        value = default if not raw else int(raw)
        return max(1, value)

    return (
        bounded("GENCO_TPGT_PACK_TOPOLOGIES", PACK_MAX_TOPOLOGIES),
        bounded("GENCO_TPGT_PACK_CASES", PACK_MAX_CASES),
    )


def pack_same_type_steps(
    chunks: list[tuple[int, np.ndarray]],
    networks: list[Network],
    *,
    max_topologies: int,
    max_cases: int,
) -> list[list[tuple[int, np.ndarray]]]:
    """Group single-topology chunks that share ``network_type_key`` into one step.

    Chunks are already in network-id order and at most one batch each. A chunk with ``max_cases``
    or more cases is its own step (one topology, shared pair embedding). Smaller chunks pack with
    the following same-type chunks until the topology cap or the case cap. A type change flushes,
    so a step never mixes bus types. Consecutive chunks of one topology are concatenated.
    """
    if max_topologies <= 1:
        return [[chunk] for chunk in chunks]
    steps: list[list[tuple[int, np.ndarray]]] = []
    pending: list[tuple[int, np.ndarray]] = []
    pending_cases = 0
    pending_key: bytes | None = None

    def flush() -> None:
        nonlocal pending_cases, pending_key
        if pending:
            steps.append(list(pending))
            pending.clear()
            pending_cases = 0
            pending_key = None

    for network_id, index in chunks:
        count = int(len(index))
        if count >= max_cases:
            flush()
            steps.append([(network_id, index)])
            continue
        if pending and network_id == pending[-1][0]:
            if pending_cases + count <= max_cases:
                last_id, last_index = pending[-1]
                pending[-1] = (last_id, np.concatenate((last_index, index)))
                pending_cases += count
                continue
            flush()
        key = network_type_key(networks[network_id])
        if pending and (
            key != pending_key or len(pending) >= max_topologies or pending_cases + count > max_cases
        ):
            flush()
        pending.append((network_id, index))
        pending_cases += count
        pending_key = key
    flush()
    return steps


def plan_training_steps(
    split: Split,
    seed: int,
    epoch: int,
    batch_size: int | None = None,
    *,
    mix_topologies: bool,
) -> list[list[tuple[int, np.ndarray]]]:
    """One epoch of training steps: a global mix, or same-type packs (see ``mix_topologies``)."""
    if batch_size is None:
        batch_size = BATCH_SIZE
    if mix_topologies:
        return list(iterate_steps(split, seed, epoch, batch_size))
    chunks = list(iterate_indexes(split, seed, epoch, shuffle=True, batch_size=batch_size))
    max_topologies, max_cases = pack_step_limits()
    return pack_same_type_steps(
        chunks, split.networks, max_topologies=max_topologies, max_cases=max_cases
    )


def iterate_training_steps(
    split: Split,
    seed: int,
    epoch: int,
    batch_size: int | None = None,
    *,
    mix_topologies: bool,
):
    """Training steps: global mixed shuffle or same-type packs (see ``mix_topologies``)."""
    yield from plan_training_steps(
        split, seed, epoch, batch_size, mix_topologies=mix_topologies
    )


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
    *,
    progress_label: str | None = None,
) -> float:
    """Mean final-stage PBL over ``split`` in full batches (several topologies and bus types per batch).

    A single-topology batch reuses that topology's cached pair embedding, dropped once its cases are done.
    """
    model.eval()
    clear_pair_embed_cache(networks)
    total = 0.0
    seen = 0
    cached: dict[int, Network] = {}
    report_progress = progress_label is not None and split.count >= 256
    started = time.perf_counter()
    last_progress = started
    batches = 0
    with torch.no_grad():
        for owners, index in iterate_type_batches(split, batch_size=batch_size, mix_patterns=True):
            batches += 1
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
            if report_progress:
                now = time.perf_counter()
                if (
                    batches == 1
                    or seen >= split.count
                    or batches % 10 == 0
                    or now - last_progress >= PROGRESS_EVERY_SECONDS
                ):
                    elapsed = now - started
                    rate = seen / elapsed if elapsed > 0 else 0.0
                    remaining = split.count - seen
                    eta = remaining / rate if rate > 0 else 0.0
                    print(
                        f"  eval {progress_label}: {seen}/{split.count} scenarios "
                        f"({batches} batches)  elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}",
                        flush=True,
                    )
                    last_progress = now
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


def _state_entry_for_id(state_map: dict, param_id: object) -> dict | None:
    """Resolve Adam state for a serialized param id (int or str keys)."""
    entry = state_map.get(param_id)
    if isinstance(entry, dict):
        return entry
    if isinstance(param_id, int):
        entry = state_map.get(str(param_id))
        if isinstance(entry, dict):
            return entry
    for key, value in state_map.items():
        if key == param_id or str(key) == str(param_id):
            return value if isinstance(value, dict) else None
    return None


def optimizer_state_shape_mismatch(
    optimizer: torch.optim.Optimizer,
    state_payload: object,
) -> str | None:
    """Return a short reason if Adam buffers do not match current parameter shapes.

    ``Optimizer.load_state_dict`` keys momentum by param-group order and does not always
    reject transposed/shifted buffers; Adam foreach then crashes on ``_foreach_lerp_``.
    """
    if not isinstance(state_payload, dict):
        return "optimizer payload missing"
    groups = state_payload.get("param_groups")
    state_map = state_payload.get("state")
    if not isinstance(groups, list) or not isinstance(state_map, dict):
        return "optimizer payload malformed"
    payload_ids: list[object] = []
    for group in groups:
        if not isinstance(group, dict) or "params" not in group:
            return "optimizer param_groups malformed"
        payload_ids.extend(group["params"])
    current_params = [param for group in optimizer.param_groups for param in group["params"]]
    if len(payload_ids) != len(current_params):
        return f"optimizer param count {len(payload_ids)} != model {len(current_params)}"
    for param, param_id in zip(current_params, payload_ids):
        entry = _state_entry_for_id(state_map, param_id)
        if entry is None:
            continue
        for buffer_name in ("exp_avg", "exp_avg_sq"):
            buffer = entry.get(buffer_name)
            if buffer is None or not torch.is_tensor(buffer):
                continue
            if tuple(buffer.shape) != tuple(param.shape):
                return (
                    f"{buffer_name} shape {tuple(buffer.shape)} != param {tuple(param.shape)}"
                )
    return None


def live_optimizer_state_shape_mismatch(optimizer: torch.optim.Optimizer) -> str | None:
    """Validate already-loaded Adam buffers against their parameters."""
    for param in (p for group in optimizer.param_groups for p in group["params"]):
        entry = optimizer.state.get(param)
        if not isinstance(entry, dict):
            continue
        for buffer_name in ("exp_avg", "exp_avg_sq"):
            buffer = entry.get(buffer_name)
            if buffer is None or not torch.is_tensor(buffer):
                continue
            if tuple(buffer.shape) != tuple(param.shape):
                return (
                    f"{buffer_name} shape {tuple(buffer.shape)} != param {tuple(param.shape)}"
                )
    return None


def reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Drop Adam momentum; keep current param_group hyperparameters (e.g. resumed lr)."""
    optimizer.state.clear()


def restore_checkpoint_payload(
    model: TPGT,
    optimizer: torch.optim.Optimizer,
    payload: dict | None,
) -> tuple[dict | None, str | None]:
    """Load weights when compatible; return (restored, obsolete_reason).

    Model weights load by name. Optimizer momentum is validated against current parameter
    shapes; on mismatch, weights are kept and Adam state is left fresh (caller logs it).
    """
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
    except RuntimeError as error:
        detail = str(error).split("\n", maxsplit=1)[0]
        return None, f"state_dict incompatible with current model ({detail})"
    opt_payload = payload.get("optimizer")
    optimizer_reset_reason = optimizer_state_shape_mismatch(optimizer, opt_payload)
    if optimizer_reset_reason is None:
        try:
            optimizer.load_state_dict(opt_payload)
            optimizer_reset_reason = live_optimizer_state_shape_mismatch(optimizer)
        except RuntimeError as error:
            detail = str(error).split("\n", maxsplit=1)[0]
            optimizer_reset_reason = f"optimizer load failed ({detail})"
    if optimizer_reset_reason is not None:
        # Keep resumed LR from the checkpoint even when momentum buffers are dropped.
        if isinstance(opt_payload, dict):
            groups = opt_payload.get("param_groups")
            if isinstance(groups, list) and groups and isinstance(groups[0], dict) and "lr" in groups[0]:
                resumed_lr = float(groups[0]["lr"])
                for group in optimizer.param_groups:
                    group["lr"] = resumed_lr
        reset_optimizer_state(optimizer)
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
            "optimizer_reset_reason": optimizer_reset_reason,
        },
        None,
    )


def saves_checkpoint_epoch(epoch_number: int) -> bool:
    """True on epochs 3, 6, ..., 99 and on the final epoch (Drive checkpoint + results CSV)."""
    if epoch_number <= 0:
        return False
    return epoch_number == EPOCHS or epoch_number % CHECKPOINT_EVERY == 0


def seed_report_dir(seed_dir: Path, epoch: int) -> Path:
    """``seed_*/reports/epoch_N/`` on the Drive run folder (created on demand)."""
    path = seed_dir / "reports" / f"epoch_{int(epoch)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_seed_log(seed_dir: Path, text: str, *, also_print: bool = True) -> None:
    """Append console report text to ``seed_*/training_log.txt`` (Drive-backed) and optionally print."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    body = text if text.endswith("\n") else text + "\n"
    with (seed_dir / "training_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
    if also_print:
        print(text, flush=True)


def write_seed_report(seed_dir: Path, epoch: int, filename: str, text: str) -> Path:
    """Write a durable per-checkpoint report file under ``reports/epoch_N/``."""
    path = seed_report_dir(seed_dir, epoch) / filename
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def append_seed_metrics(seed_dir: Path, row: dict) -> None:
    """Append one epoch (or checkpoint) metrics object to ``seed_*/metrics.jsonl``."""
    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_seed_json(seed_dir: Path, epoch: int, filename: str, payload: dict | list) -> Path:
    path = seed_report_dir(seed_dir, epoch) / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_pbl_line(seed: int, epoch_number: int, scores: dict[str, float]) -> str:
    return (
        f"seed {seed} epoch {epoch_number} test PBL  "
        f"N {scores['N']:.6f}  N-1 {scores['N-1']:.6f}  "
        f"N-2 {scores['N-2']:.6f}  C2I {scores['C2I']:.6f}"
    )


def format_duration(seconds: float) -> str:
    """``45s``, ``12m03s``, or ``2h05m``."""
    whole = max(0, int(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


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
        f"epoch_avg state {state_mse / n:.4f}  phys_w {physics_sum / n:.4f}  {stage_text}  "
        f"train {train_total / n:.4f}  val_pbl {val_display}{lr_text}{timing_text}"
    )


def cosine_lr_at_epoch(epoch: int) -> float:
    """Adam LR after ``epoch`` completed epochs (0 = initial ``LEARNING_RATE``)."""
    progress = min(max(int(epoch), 0), EPOCHS) / max(EPOCHS, 1)
    # Half-cosine from 1 → LR_COSINE_MIN_RATIO over the full run.
    scale = LR_COSINE_MIN_RATIO + 0.5 * (1.0 - LR_COSINE_MIN_RATIO) * (1.0 + math.cos(math.pi * progress))
    return LEARNING_RATE * scale


def lr_plateau_enabled() -> bool:
    flag = os.environ.get("GENCO_TPGT_LR_PLATEAU", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return True


def lr_plateau_patience() -> int:
    env = os.environ.get("GENCO_TPGT_LR_PLATEAU_PATIENCE", "").strip()
    if env:
        return max(1, int(float(env)))
    return LR_PLATEAU_PATIENCE


@dataclass
class LearningRateSchedule:
    """Cosine lr ceiling plus optional ReduceLROnPlateau on validation PBL."""

    optimizer: torch.optim.Optimizer
    plateau: torch.optim.lr_scheduler.ReduceLROnPlateau
    plateau_enabled: bool

    def state_dict(self) -> dict:
        return {
            "kind": "plateau_cosine",
            "plateau": self.plateau.state_dict(),
            "plateau_enabled": self.plateau_enabled,
        }

    def load_state_dict(self, state: dict) -> None:
        self.plateau.load_state_dict(state["plateau"])
        self.plateau_enabled = bool(state.get("plateau_enabled", True))

    def apply_cosine_ceiling(self, completed_epoch: int) -> None:
        ceiling = cosine_lr_at_epoch(completed_epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = min(float(group["lr"]), ceiling)

    def sync_completed_epochs(self, completed_epochs: int) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = LEARNING_RATE
        self.apply_cosine_ceiling(completed_epochs)

    def step_epoch(self, completed_epoch: int, val_pbl: float | None) -> None:
        lr_before = current_lr(self.optimizer)
        if self.plateau_enabled and val_pbl is not None:
            self.plateau.step(val_pbl)
        lr_after_plateau = current_lr(self.optimizer)
        if lr_after_plateau < lr_before - 1e-15:
            print(
                f"  lr reduced on val plateau: {lr_before:.1e} -> {lr_after_plateau:.1e}",
                flush=True,
            )
        self.apply_cosine_ceiling(completed_epoch)
        lr_after = current_lr(self.optimizer)
        if lr_after < lr_after_plateau - 1e-15:
            print(
                f"  lr capped by cosine epoch {completed_epoch}: {lr_after_plateau:.1e} -> {lr_after:.1e}",
                flush=True,
            )


def make_lr_schedule(optimizer: torch.optim.Optimizer) -> LearningRateSchedule:
    eta_min = LEARNING_RATE * LR_COSINE_MIN_RATIO
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=lr_plateau_patience(),
        threshold=LR_PLATEAU_THRESHOLD,
        min_lr=eta_min,
    )
    enabled = lr_plateau_enabled()
    return LearningRateSchedule(optimizer=optimizer, plateau=plateau, plateau_enabled=enabled)


def restore_lr_schedule(
    optimizer: torch.optim.Optimizer,
    state: dict | None,
    *,
    completed_epochs: int = 0,
) -> LearningRateSchedule:
    schedule = make_lr_schedule(optimizer)
    if state is None:
        schedule.sync_completed_epochs(completed_epochs)
        return schedule
    if state.get("kind") == "plateau_cosine":
        try:
            schedule.load_state_dict(state)
        except (KeyError, TypeError, ValueError, RuntimeError):
            schedule.sync_completed_epochs(completed_epochs)
        return schedule
    schedule.sync_completed_epochs(completed_epochs)
    return schedule


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

    def log(text: str) -> None:
        append_seed_log(model_dir, text)

    seed_all(seed)
    configure_cuda_training(device)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    model = TPGT().to(device)
    if wants_torch_compile():
        log("torch.compile enabled (GENCO_TPGT_COMPILE=1, GENCO_TPGT_FAST=1, or --compile / --fast)")
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = make_lr_schedule(optimizer)
    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and ATTENTION_KIND != "exphormer",
    )
    plateau_note = (
        f"plateau x{LR_PLATEAU_FACTOR:g} after {lr_plateau_patience()} val checks without gain"
        if scheduler.plateau_enabled
        else "plateau off (GENCO_TPGT_LR_PLATEAU=0)"
    )
    log(
        f"optim: Adam lr {LEARNING_RATE:.1e} → {LEARNING_RATE * LR_COSINE_MIN_RATIO:.1e} "
        f"cosine ceiling/{EPOCHS}ep, {plateau_note}, grad_clip_norm={GRAD_CLIP_NORM:g}"
    )
    cache_budget = device_cache_budget(device)
    train_share = len(train_split.networks) / max(len(train_split.networks) + len(val_split.networks), 1)
    train_networks = DeviceNetworks(train_split.networks, device, budget_bytes=int(cache_budget * train_share))
    val_networks = DeviceNetworks(
        val_split.networks, device, budget_bytes=cache_budget - int(cache_budget * train_share)
    )
    log(
        f"device network cache (budget {cache_budget / 1024**2:.0f} MB): "
        f"train {train_networks.describe()}; valid {val_networks.describe()}"
    )
    train_probe = train_probe_split(train_split, min(val_split.count, TRAIN_PROBE_CASES))
    mix_topologies = mix_topologies_per_step(len(train_split.networks))
    if mix_topologies:
        log(f"train steps: mix topologies (global shuffle, up to {BATCH_SIZE} cases/step)")
    else:
        max_topologies, max_cases = pack_step_limits()
        if max_topologies <= 1:
            log(f"train steps: one topology per step ({len(train_split.networks)} topologies)")
        else:
            log(
                f"train steps: up to {max_topologies} same-type topologies and {max_cases} cases "
                f"per step ({len(train_split.networks)} topologies; a chunk of {max_cases}+ cases "
                f"stays one topology; GENCO_TPGT_PACK_TOPOLOGIES=1 forces one topology per step)"
            )
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
        else:
            scheduler = restore_lr_schedule(optimizer, restored["scheduler"], completed_epochs=start_epoch)
            if restored["scheduler"] is None:
                print(
                    f"Seed {seed}: checkpoint has no LR scheduler state; "
                    f"synced to epoch {start_epoch} (lr {current_lr(optimizer):.1e}).",
                    flush=True,
                )
            elif restored["scheduler"].get("kind") != "plateau_cosine":
                print(
                    f"Seed {seed}: legacy cosine-only scheduler in checkpoint; "
                    f"synced to epoch {start_epoch} (lr {current_lr(optimizer):.1e}).",
                    flush=True,
                )
        opt_reset = restored.get("optimizer_reset_reason")
        if opt_reset:
            print(
                f"Seed {seed}: optimizer state incompatible ({opt_reset}); "
                f"keeping model weights, resetting Adam momentum "
                f"(lr {current_lr(optimizer):.1e}).",
                flush=True,
            )
            log(f"optimizer reset on resume: {opt_reset}")
        else:
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
        log(
            f"Saved seed {seed} epoch {epoch_number} checkpoint to {epoch_path} and {checkpoint_path}"
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
            steps = plan_training_steps(train_split, seed, epoch, mix_topologies=mix_topologies)
            n_steps = len(steps)
            packed_steps = sum(len(groups) > 1 for groups in steps)
            pack_note = f", {packed_steps} multi-topology" if packed_steps else ""
            log(
                f"  seed {seed} epoch {epoch + 1}: {n_steps} steps ({train_split.count} cases{pack_note})"
            )
            batches = 0
            cases_seen = 0
            batch_durations: list[float] = []
            timer = StepTimer(device)
            epoch_started = time.perf_counter()
            last_progress = epoch_started
            for step_groups in steps:
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
                cases = sum(len(index) for _network_id, index in step_groups)
                cases_seen += cases
                batch_durations.append(time.perf_counter() - step_started)
                timer.end_step(len(step_groups), batch_durations[-1])
                losses = (breakdown.total, breakdown.state_mse, breakdown.physics_sum, *breakdown.physics_stages)
                running += torch.stack([value.detach().to(device) for value in losses]).double()
                batches += 1
                now = time.perf_counter()
                if (
                    batches == 1
                    or batches == n_steps
                    or batches % PROGRESS_EVERY_STEPS == 0
                    or now - last_progress >= PROGRESS_EVERY_SECONDS
                ):
                    timing = timer.summary("window")
                    # Progress lines: this step only (not a running epoch mean — that
                    # was misleading when early-epoch spikes polluted later prints).
                    step_total = float(breakdown.total.detach().cpu())
                    step_state = float(breakdown.state_mse.detach().cpu())
                    elapsed = time.perf_counter() - epoch_started
                    eta = elapsed / batches * (n_steps - batches)
                    topologies = "" if len(step_groups) == 1 else f", {len(step_groups)} topologies"
                    log(
                        f"  seed {seed} epoch {epoch + 1} step {batches}/{n_steps} "
                        f"({cases} cases{topologies}; {cases_seen}/{train_split.count} done)  "
                        f"state {step_state:.4f}  train {step_total:.4f}  "
                        f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}  {timing}"
                    )
                    last_progress = time.perf_counter()
            running_total, running_state, running_phys, *running_phys_stages = running.tolist()
            epoch_timing = timer.summary("epoch")
            run_validation = val_every <= 1 or (epoch + 1) % val_every == 0 or (epoch + 1) == EPOCHS
            if run_validation:
                log(f"  seed {seed} epoch {epoch + 1}: validating {val_split.count} cases")
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
            else:
                val_display = "skipped"
            val_for_lr = val_score if run_validation else None
            scheduler.step_epoch(epoch + 1, val_for_lr)
            epoch_line = train_epoch_loss_line(
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
            )
            log(epoch_line)
            n_batches = max(batches, 1)
            epoch_metrics = {
                "seed": seed,
                "epoch": epoch + 1,
                "batches": batches,
                "state_mse": running_state / n_batches,
                "physics_sum": running_phys / n_batches,
                "physics_stages": [value / n_batches for value in running_phys_stages],
                "train_total": running_total / n_batches,
                "val_pbl": None if not run_validation else float(val_score),
                "train_probe_pbl": None if not run_validation else float(train_probe_score),
                "lr": current_lr(optimizer),
                "timing": epoch_timing,
                "checkpoint_epoch": saves_checkpoint_epoch(epoch + 1),
            }
            append_seed_metrics(model_dir, epoch_metrics)
            if saves_checkpoint_epoch(epoch + 1):
                write_seed_report(model_dir, epoch + 1, "epoch_line.txt", epoch_line)
                write_seed_json(model_dir, epoch + 1, "epoch_metrics.json", epoch_metrics)
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
                    shunt_ablation=wants_shunt_ablation(default=True),
                    seed_dir=model_dir,
                    epoch=epoch + 1,
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
    *,
    source: str = "independent_training",
) -> None:
    spec = spec or TASKS["1.1"]
    scores = attach_c2i(scores, counts, spec)
    columns = [*SEED_COLUMNS, "epoch"]
    rows = []
    for category in categories_for(spec.task_id):
        rows.append(
            {
                "model": MODEL_NAME,
                "task": spec.task_id,
                "test_category": category,
                "seed": str(seed),
                "mean_pbl_pu": f"{scores[category]:.8g}",
                "checkpoint_path": str(checkpoint),
                "source": source,
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
    for category in categories_for(spec.task_id):
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


def format_comparison(summary: list[dict], spec: TaskSpec | None = None) -> str:
    spec = spec or TASKS["1.1"]
    categories = categories_for(spec.task_id)
    lines = [f"Task {spec.label} mean power-balance loss (per unit)", "PUBLISHED"]
    for model, scores in PUBLISHED.get(spec.task_id, {}).items():
        parts = [f"{category} {scores[category][0]:.4g} ± {scores[category][1]:.4g}" for category in categories]
        lines.append(f"  {model}: " + "; ".join(parts))
    lines.append("TPGT this run")
    by_category = {row["test_category"]: row for row in summary}
    if any(category not in by_category for category in categories):
        lines.append("  TPGT has no finished reported-category seed yet.")
        return "\n".join(lines)
    parts = []
    for category in categories:
        row = by_category[category]
        parts.append(
            f"{category} {float(row['mean_across_seeds']):.4g} ± {float(row['std_across_seeds']):.4g} (n={row['n_seeds']})"
        )
    lines.append("  TPGT: " + "; ".join(parts))
    return "\n".join(lines)


def print_comparison(
    summary: list[dict],
    spec: TaskSpec | None = None,
    *,
    seed_dir: Path | None = None,
    epoch: int | None = None,
) -> str:
    text = format_comparison(summary, spec)
    if seed_dir is None:
        print(text, flush=True)
    else:
        append_seed_log(seed_dir, text)
        if epoch is not None and saves_checkpoint_epoch(int(epoch)):
            write_seed_report(seed_dir, int(epoch), "comparison.txt", text)
    return text


def comparison_figure_path(results: Path, spec: TaskSpec | None = None) -> Path:
    spec = spec or TASKS["1.1"]
    return results / f"figure4_task{spec.task_id}_tpgt.svg"


def plot_comparison(
    results: Path,
    summary: list[dict],
    spec: TaskSpec | None = None,
    *,
    archive_dir: Path | None = None,
) -> None:
    """Bar plot (SVG + PNG) vs published models; always refresh ``results/figure4_…_tpgt.*``.

    When ``archive_dir`` is set (checkpoint epochs), also copy SVG+PNG there so every
    3-epoch snapshot is kept next to that epoch's bus maps, not only the latest overwrite.
    """
    spec = spec or TASKS["1.1"]
    categories = categories_for(spec.task_id)
    by_category = {row["test_category"]: row for row in summary}
    if any(category not in by_category for category in categories):
        return
    if any(int(by_category[category]["n_seeds"]) < 1 for category in categories):
        return
    tpgt = {
        category: (float(by_category[category]["mean_across_seeds"]), float(by_category[category]["std_across_seeds"]))
        for category in categories
    }
    seeds = min(int(by_category[category]["n_seeds"]) for category in categories)
    output = comparison_figure_path(results, spec)
    tpgt_label = f"TPGT (this run, {seeds} seed(s))"
    plot_label = spec.label if CASE_NAME == "case118" else f"{spec.label} ({CASE_NAME})"
    # Published Task 1.1 baselines are IEEE-118 only; case57 plots TPGT alone.
    plot_task = (
        spec.task_id
        if CASE_NAME == "case118" or spec.task_id == "3.1"
        else f"{spec.task_id}_{CASE_NAME}"
    )
    written = plot_tpgt_comparison(plot_task, plot_label, tpgt, output, tpgt_label=tpgt_label)
    if spec.task_id == "3.1":
        if write_task31_table is None or display_task31_table is None:
            print(
                "Table 4 HTML/Markdown skipped: plot_task1_1.py is behind train_tpgt_task11.py. "
                "Run the Colab preparation cell (git fetch && reset --hard origin/main), then retry.",
                flush=True,
            )
        else:
            written.extend(write_task31_table(results, tpgt, tpgt_label=tpgt_label))
            display_task31_table(results)
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)
        for path in written:
            dest = archive_dir / path.name
            shutil.copy2(path, dest)
        print(f"archived bar plot to {archive_dir}", flush=True)
    try:
        from IPython.display import Image, SVG, display

        display(SVG(filename=str(output)))
        png = output.with_suffix(".png")
        if png.is_file():
            display(Image(filename=str(png)))
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


def score_key(grid: str, label: str, spec: TaskSpec) -> str:
    """Test-score name. Task 3.1 prefixes the grid (``57 N``); other tasks keep ``N``."""
    if spec.task_id != "3.1":
        return label
    short = {"case57": "57", "case118": "118", "case500": "500"}[grid]
    return f"{short} {label}"


def task31_partial_eval_path(results: Path, spec: TaskSpec) -> Path:
    return results / f"task{spec.file_tag}_tpgt_eval_partial.json"


def load_partial_eval(
    results: Path, spec: TaskSpec, seed: int, ckpt_path: Path
) -> tuple[dict[str, float], dict[str, int]]:
    path = task31_partial_eval_path(results, spec)
    if not path.is_file():
        return {}, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}
    if int(payload.get("seed", -1)) != seed or str(payload.get("checkpoint_path", "")) != str(ckpt_path):
        return {}, {}
    scores = {str(k): float(v) for k, v in payload.get("scores", {}).items()}
    counts = {str(k): int(v) for k, v in payload.get("counts", {}).items()}
    if scores:
        print(f"Resuming partial Task 3.1 eval from {path.name} ({len(scores)} score keys)", flush=True)
    return scores, counts


def save_partial_eval(
    results: Path,
    spec: TaskSpec,
    seed: int,
    epoch: int,
    ckpt_path: Path,
    scores: dict[str, float],
    counts: dict[str, int],
) -> None:
    path = task31_partial_eval_path(results, spec)
    path.write_text(
        json.dumps(
            {
                "seed": seed,
                "epoch": epoch,
                "checkpoint_path": str(ckpt_path),
                "scores": scores,
                "counts": counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def grid_scoring_complete(scores: dict[str, float], grid: str, spec: TaskSpec) -> bool:
    if spec.task_id != "3.1":
        return False
    return all(score_key(grid, label, spec) in scores for label, _topology, _condition in TEST_PARTS)


def attach_c2i(scores: dict[str, float], counts: dict[str, int], spec: TaskSpec) -> dict[str, float]:
    """Add the paper's single C2I number (mean of the three nose splits) per reported grid."""
    scores = dict(scores)
    if spec.task_id != "3.1":
        part_counts = {key: counts[key] for key in ("C2I-N", "C2I-N-1", "C2I-N-2")}
        scores["C2I"] = aggregate_c2i(scores, part_counts)
        return scores
    for grid in ("57", "118", "500"):
        part = {f"C2I-{suffix}": scores[f"{grid} C2I-{suffix}"] for suffix in ("N", "N-1", "N-2")}
        weight = {f"C2I-{suffix}": counts[f"{grid} C2I-{suffix}"] for suffix in ("N", "N-1", "N-2")}
        scores[f"{grid} C2I"] = aggregate_c2i(part, weight)
    return scores


def test_model(
    model: TPGT,
    data_root: Path,
    device: torch.device,
    active_cache: Path,
    persistent_cache: Path,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    after_split: Callable[[str, Split, DeviceNetworks, dict[int, list[float]]], None] | None = None,
    spec: TaskSpec | None = None,
    only_grids: tuple[str, ...] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    spec = spec or TASKS["1.1"]
    scores = {}
    counts = {}
    model.eval()
    grids = only_grids if only_grids is not None else eval_case_names(spec)
    for grid in grids:
        for label, topology, condition in TEST_PARTS:
            cache_name = (
                f"test_{grid}_{topology}_{condition}"
                if spec.task_id == "3.1"
                else f"test_{topology}_{condition}"
            )
            split = cached_split(
                active_cache,
                persistent_cache,
                raw_dir(data_root, "test", topology, condition, grid),
                cache_name,
            )
            networks = DeviceNetworks(split.networks, device)
            per_network: dict[int, list[float]] = {}
            key = score_key(grid, label, spec)
            buses = int(split.networks[0].ybus.shape[0]) if split.networks else 0
            step_batch = effective_eval_batch_size(buses, eval_batch_size)
            print(
                f"  scoring {key}: {split.count} scenarios, {len(split.networks)} topology(ies), "
                f"{buses} buses, batch {step_batch}",
                flush=True,
            )
            scores[key] = evaluate_pbl(
                model,
                split,
                networks,
                device,
                batch_size=step_batch,
                per_network_out=per_network,
                progress_label=key,
            )
            counts[key] = split.count
            print(f"test {key}: PBL {scores[key]:.6g} on {split.count} scenarios", flush=True)
            if after_split is not None and grid == "case118":
                after_split(label, split, networks, per_network)
        persist_split_caches(active_cache, persistent_cache, label=grid)
        if device.type == "cuda":
            del networks
            gc.collect()
            torch.cuda.empty_cache()
    return scores, counts


def main() -> None:
    global BATCH_SIZE, DATA_FRACTION, MICRO_BATCH_SIZE, EVAL_BATCH_SIZE, VAL_EVERY
    ensure_runtime_packages()
    apply_architecture_config()
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
    amp_note = (
        "AMP off (exphormer float32)"
        if ATTENTION_KIND == "exphormer"
        else "AMP on CUDA"
    )
    batch_note = (
        f"Training batch {BATCH_SIZE}, micro {MICRO_BATCH_SIZE}, eval batch {EVAL_BATCH_SIZE}, "
        f"val every {VAL_EVERY} epoch(s), {amp_note}."
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
    case_note = f", case {CASE_NAME}" if CASE_NAME != "case118" else ""
    print(
        f"PFΔ Task {spec.label}: {spec.description} (dataset {spec.dataset_repo}{case_note})",
        flush=True,
    )
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
    print(architecture_summary(), flush=True)
    probe = TPGT()
    print(f"Model parameters: {count_parameters(probe):,}", flush=True)
    del probe
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
    train_split, val_split = warm_split_caches(data, active_cache, persistent_cache, spec, load_valid=True)
    assert train_split is not None and val_split is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
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
            seed_dir = checkpoints / f"seed_{seed}"

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
                        seed_dir=seed_dir,
                        epoch=epoch_number,
                    )

            scores, counts = test_model(
                model,
                data,
                device,
                active_cache,
                persistent_cache,
                eval_batch_size=EVAL_BATCH_SIZE,
                after_split=after_test_split,
                spec=spec,
            )
            reported = attach_c2i(scores, counts, spec)
            if spec.task_id == "3.1":
                for grid in ("57", "118", "500"):
                    subset = {category: reported[f"{grid} {category}"] for category in CATEGORIES}
                    line = f"case {grid}  " + test_pbl_line(seed, epoch_number, subset)
                    append_seed_log(seed_dir, line)
                    write_seed_report(seed_dir, epoch_number, f"test_pbl_case{grid}.txt", line)
            else:
                line = test_pbl_line(seed, epoch_number, reported)
                append_seed_log(seed_dir, line)
                write_seed_report(seed_dir, epoch_number, "test_pbl.txt", line)
            write_seed_json(
                seed_dir,
                epoch_number,
                "test_scores.json",
                {"scores": reported, "counts": counts, "seed": seed, "epoch": epoch_number},
            )
            append_seed_metrics(
                seed_dir,
                {
                    "seed": seed,
                    "epoch": epoch_number,
                    "kind": "test",
                    "scores": reported,
                    "counts": counts,
                },
            )
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
            seed_dir = checkpoints / f"seed_{seed}"
            record_seed(results, seed, scores, counts, checkpoint, epoch_number, spec)
            summary = write_summary(results, spec)
            print_comparison(summary, spec, seed_dir=seed_dir, epoch=epoch_number)
            write_seed_json(
                seed_dir,
                epoch_number,
                "summary_snapshot.json",
                {"seed": seed, "epoch": epoch_number, "summary": summary, "checkpoint": str(checkpoint)},
            )
            bar_archive = checkpoints / "bar_plots" / f"seed_{seed}" / f"epoch_{epoch_number}"
            try:
                plot_comparison(results, summary, spec, archive_dir=bar_archive)
            except Exception as error:
                append_seed_log(
                    seed_dir,
                    f"Bar plot skipped for epoch {epoch_number}: {type(error).__name__}: {error}",
                )
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


def cli_eval_only() -> None:
    """Task 3.1 cross-grid test only: load trained IEEE-118 weights and score 57 / 118 / 500."""
    global BATCH_SIZE, DATA_FRACTION, EVAL_BATCH_SIZE
    ensure_runtime_packages()
    apply_architecture_config()
    DATA_FRACTION = parse_data_fraction()
    BATCH_SIZE = parse_batch_size()
    EVAL_BATCH_SIZE = parse_eval_batch_size()
    spec = parse_task()
    if spec.task_id != "3.1":
        raise ValueError("--eval-only supports Task 3.1 only (use --task 3.1).")
    drive = google_drive_root()
    ckpt_root = parse_checkpoint_root(drive)
    if ckpt_root is None or not ckpt_root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {ckpt_root}")
    outputs = drive_output_dirs(drive, spec)
    work, legacy_results = legacy_locations(drive)
    if outputs is not None:
        checkpoints, results = outputs
    else:
        checkpoints = work / spec.run_dir_name
        results = task_results_dir(checkpoints, legacy_results, spec)
    results.mkdir(parents=True, exist_ok=True)
    os.environ["GENCO_RESULTS_DIR"] = str(results)
    from common import use_selected_results

    use_selected_results()
    print(f"PFΔ Task {spec.label}: eval-only (weights from {ckpt_root})", flush=True)
    print(architecture_summary(), flush=True)
    print(f"Eval batch {EVAL_BATCH_SIZE}; results {results}", flush=True)
    data = ensure_dataset(work, drive, spec)
    print(f"Dataset: {data}", flush=True)
    log_counts(data, results, spec)
    active_cache, persistent_cache = split_cache_dirs(checkpoints, spec)
    warm_split_caches(
        data,
        active_cache,
        persistent_cache,
        spec,
        weight_run=ckpt_root,
        load_valid=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    configure_cuda_training(device)
    any_seed = False
    for seed in SEEDS:
        seed_dir = ckpt_root / f"seed_{seed}"
        picked = eval_checkpoint_payload(seed_dir, device)
        if picked is None:
            print(f"Seed {seed}: no compatible last.pt under {seed_dir}, skipping.", flush=True)
            continue
        state, epoch, ckpt_path = picked
        model = TPGT().to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"Seed {seed}: loaded epoch {epoch} from {ckpt_path}", flush=True)
        scores, counts = load_partial_eval(results, spec, seed, ckpt_path)
        for grid in eval_case_names(spec):
            if grid_scoring_complete(scores, grid, spec):
                print(f"Seed {seed}: skip {grid} tests (scores already on Drive partial eval)", flush=True)
                continue
            part_scores, part_counts = test_model(
                model,
                data,
                device,
                active_cache,
                persistent_cache,
                eval_batch_size=EVAL_BATCH_SIZE,
                spec=spec,
                only_grids=(grid,),
            )
            scores.update(part_scores)
            counts.update(part_counts)
            save_partial_eval(results, spec, seed, epoch, ckpt_path, scores, counts)
        persist_split_caches(active_cache, persistent_cache, label="final")
        reported = attach_c2i(scores, counts, spec)
        for grid in ("57", "118", "500"):
            subset = {category: reported[f"{grid} {category}"] for category in CATEGORIES}
            print(f"case {grid}  " + test_pbl_line(seed, epoch, subset), flush=True)
        record_seed(
            results,
            seed,
            scores,
            counts,
            ckpt_path,
            epoch,
            spec,
            source="cross_grid_eval",
        )
        any_seed = True
    if not any_seed:
        raise RuntimeError(f"No seeds evaluated under {ckpt_root}.")
    summary = write_summary(results, spec)
    print_comparison(summary, spec)
    plot_comparison(results, summary, spec)
    partial = task31_partial_eval_path(results, spec)
    if partial.is_file():
        partial.unlink()
    print(f"Done. Tables and figure under {results}", flush=True)


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
    apply_architecture_config()
    DATA_FRACTION = parse_data_fraction()
    spec = parse_task()
    print(f"Diagnose PFΔ Task {spec.label}", flush=True)
    print(architecture_summary(), flush=True)
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
    reconcile_split_caches(active_cache, persistent_cache, data)
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
        shunt_ablation=wants_shunt_ablation(default=True),
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
        raw = raw_dir(data_root, *parts)
        read_split(raw, name, prefix_end=scenario_limit(raw))
    print(f"verify_task11_splits OK ({len(checks)} folders, case={CASE_NAME})", flush=True)


def cli_verify_splits() -> None:
    global DATA_FRACTION
    ensure_runtime_packages()
    apply_architecture_config()
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
        apply_architecture_config()
        drive = google_drive_root()
        work, _legacy = legacy_locations(drive)
        data = ensure_dataset(work, drive)
        sanity_check_meshed_rpe(data)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-attention":
        from tests.test_tpgt_attention import test_batched_messages_match_reference

        test_batched_messages_match_reference()
        print("test_tpgt_attention OK", flush=True)
    elif wants_eval_only():
        cli_eval_only()
    else:
        main()
