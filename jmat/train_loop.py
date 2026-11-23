"""Shared training/eval loops for Phase 3."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jmat.csr_io import scipy_csr_to_torch
from jmat.data.shard_io import load_triple_npz
from jmat.loss import jmat_combined_loss
from jmat.metrics import relative_errors, summarize_errors
from jmat.model import JMAT
from jmat.normalize import normalize_linear_system


@dataclass
class TrainStats:
    epoch: int
    mean_loss: float
    n_steps: int
    val_mean_loss: float | None = None


@dataclass
class EpochLog:
    train: list[TrainStats] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def _step_batch(model: JMAT, paths: list[Path], device: torch.device, *, lam: float = 1.0) -> float:
    losses = []
    for path in paths:
        j, r, dx, _ = load_triple_npz(path)
        crow, col, val, n = scipy_csr_to_torch(j, device=device, dtype=torch.float32)
        rt = torch.tensor(r, dtype=torch.float32, device=device)
        norm = normalize_linear_system(crow, col, val, n, rt)
        if norm is None:
            continue
        z_star = torch.tensor(dx / norm.beta, dtype=torch.float32, device=device)
        z_hat, _ = model.forward_normalized(norm)
        loss, _, _ = jmat_combined_loss(
            z_hat, z_star, norm.a_crow, norm.a_col, norm.a_val, n, norm.b_tilde, lam=lam
        )
        losses.append(loss)
    if not losses:
        return float("nan")
    total = torch.stack(losses).mean()
    return float(total.item())


def train_epochs(
    model: JMAT,
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    epochs: int = 30,
    lr: float = 3e-4,
    device: torch.device | None = None,
    lam: float = 1.0,
) -> EpochLog:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    log = EpochLog()
    best_val = float("inf")
    best_state = None
    for ep in range(epochs):
        model.train()
        losses = []
        order = np.random.permutation(len(train_paths))
        for idx in order:
            path = train_paths[int(idx)]
            j, r, dx, _ = load_triple_npz(path)
            crow, col, val, n = scipy_csr_to_torch(j, device=device, dtype=torch.float32)
            rt = torch.tensor(r, dtype=torch.float32, device=device)
            norm = normalize_linear_system(crow, col, val, n, rt)
            if norm is None:
                continue
            z_star = torch.tensor(dx / norm.beta, dtype=torch.float32, device=device)
            opt.zero_grad()
            z_hat, _ = model.forward_normalized(norm)
            loss, _, _ = jmat_combined_loss(
                z_hat, z_star, norm.a_crow, norm.a_col, norm.a_val, n, norm.b_tilde, lam=lam
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            vloss = _step_batch(model, val_paths, device, lam=lam) if val_paths else float("nan")
        stat = TrainStats(epoch=ep, mean_loss=float(np.mean(losses)) if losses else float("nan"), n_steps=len(losses), val_mean_loss=vloss)
        log.train.append(stat)
        log.val_loss.append(vloss)
        if val_paths and vloss < best_val:
            best_val = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return log


@dataclass
class EvalSliceResult:
    slice_name: str
    summary: dict[str, Any]
    timings_s: list[float]


def evaluate_paths(
    model: JMAT,
    paths: list[Path],
    *,
    slice_name: str,
    device: torch.device | None = None,
) -> EvalSliceResult:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    ex, er, timings = [], [], []
    with torch.no_grad():
        for path in paths:
            j, r, dx_ref, _ = load_triple_npz(path)
            crow, col, val, n = scipy_csr_to_torch(j, device=device, dtype=torch.float32)
            rt = torch.tensor(r, dtype=torch.float32, device=device)
            t0 = time.perf_counter()
            dx_hat = model.forward_csr(crow, col, val, n, rt).cpu().numpy()
            timings.append(time.perf_counter() - t0)
            erx, err = relative_errors(dx_hat, dx_ref, j, r)
            ex.append(erx)
            er.append(err)
    from jmat.metrics import summarize_errors

    return EvalSliceResult(
        slice_name=slice_name,
        summary=summarize_errors(ex, er).__dict__,
        timings_s=timings,
    )