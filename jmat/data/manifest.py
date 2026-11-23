"""Manifest builder with enforced leakage controls (Phase 3 §1, §3)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jmat.data.splits import (
    HOLDOUT_GRIDS,
    SPLIT_HOLDOUT,
    SPLIT_IN_FAMILY_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TRAIN_GRIDS,
    SplitPolicy,
)

EVAL_SLICE_UNSEEN_RHS = "unseen_rhs"
EVAL_SLICE_UNSEEN_MATRIX = "unseen_matrix"
EVAL_SLICE_UNSEEN_DIM = "unseen_dim"
EVAL_SLICE_ILL_COND = "ill_conditioned"
EVAL_SLICE_UNSEEN_FAMILY = "unseen_family"


def topology_hash(meta: dict[str, Any]) -> str:
    key = json.dumps(
        {
            k: meta.get(k)
            for k in ("grid_id", "topology_id", "family", "n", "matrix_id")
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def assign_synthetic_split(item: dict[str, Any], *, val_fraction: float = 0.15) -> str:
    """Scenario = one (matrix, rhs) path; matrix keyed by family/n/matrix_id."""
    mat_key = f"{item['family']}:n{item['n']}:m{item['matrix_id']}"
    h = int(hashlib.md5(mat_key.encode()).hexdigest(), 16)
    bucket = (h % 1000) / 1000.0
    if item.get("split") == "test":
        if item["family"] == "ill_conditioned":
            return EVAL_SLICE_ILL_COND
        if item["n"] >= 96:
            return EVAL_SLICE_UNSEEN_DIM
        return EVAL_SLICE_UNSEEN_MATRIX
    if bucket < val_fraction:
        return SPLIT_VAL
    return SPLIT_TRAIN


def assign_pfdelta_bucket(
    grid_id: str,
    source_split: str,
    scenario_id: str,
    topology_id: str,
) -> str:
    policy = SplitPolicy(grid_id=grid_id, source_split=source_split)
    base = policy.bucket()
    if base == SPLIT_HOLDOUT:
        return SPLIT_HOLDOUT
    if base == SPLIT_TRAIN:
        # scenario-level val: hash scenario within train grids
        h = int(hashlib.md5(f"{grid_id}:{topology_id}:{scenario_id}".encode()).hexdigest(), 16)
        if (h % 100) < 10:
            return SPLIT_VAL
        return SPLIT_TRAIN
    return base


def enforce_manifest(manifest: list[dict[str, Any]], *, phase: str = "pfdelta") -> list[dict[str, Any]]:
    """Reject holdout grids in train/val buckets; attach topology_hash."""
    out = []
    train_grids: set[str] = set()
    for item in manifest:
        grid = item.get("grid_id")
        bucket = item.get("split_bucket") or item.get("split") or SPLIT_TRAIN
        if grid in HOLDOUT_GRIDS and bucket in (SPLIT_TRAIN, SPLIT_VAL):
            raise ValueError(f"holdout grid {grid} in training bucket {bucket}")
        if grid in TRAIN_GRIDS and bucket == SPLIT_TRAIN:
            train_grids.add(grid)
        item = dict(item)
        item["topology_hash"] = topology_hash(item)
        out.append(item)
    if phase == "pfdelta" and train_grids and train_grids <= set(TRAIN_GRIDS):
        assert_holdout_disjoint(out)
    return out


def assert_holdout_disjoint(manifest: list[dict[str, Any]]) -> None:
    scenarios_train = {
        (m.get("grid_id"), m.get("scenario_id"), m.get("topology_id"))
        for m in manifest
        if m.get("split_bucket") in (SPLIT_TRAIN, SPLIT_VAL)
    }
    scenarios_hold = {
        (m.get("grid_id"), m.get("scenario_id"), m.get("topology_id"))
        for m in manifest
        if m.get("split_bucket") == SPLIT_HOLDOUT
    }
    overlap = scenarios_train & scenarios_hold
    if overlap:
        raise ValueError(f"scenario leakage train/holdout: {len(overlap)} keys")


def write_manifest(path: Path, manifest: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "samples": enforce_manifest(manifest)}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return enforce_manifest(raw), {}
    samples = enforce_manifest(raw.get("samples", raw))
    return samples, raw.get("meta", {})