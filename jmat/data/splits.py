"""Train / val / test partition rules for JMAT Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

# Development & training families (PFΔ grids).
TRAIN_GRIDS = ("case14", "case30", "case118", "case500")
HOLDOUT_GRIDS = ("case57", "case2000")

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_IN_FAMILY_TEST = "in_family_test"
SPLIT_HOLDOUT = "holdout"


@dataclass(frozen=True)
class SplitPolicy:
    """Maps (grid_id, split_name) → coarse split bucket for leakage control."""

    grid_id: str
    source_split: str  # PFΔ folder: train / test / val manifest

    def bucket(self) -> str:
        g = self.grid_id.replace("_ieee", "").replace("case", "case")
        if g in HOLDOUT_GRIDS:
            return SPLIT_HOLDOUT
        if self.source_split in ("val", "validation"):
            return SPLIT_VAL
        if self.source_split == "test":
            return SPLIT_IN_FAMILY_TEST
        return SPLIT_TRAIN


def assert_holdout_not_in_training(grid_ids: set[str]) -> None:
    overlap = set(TRAIN_GRIDS) & set(HOLDOUT_GRIDS)
    if overlap:
        raise ValueError(overlap)
    if set(HOLDOUT_GRIDS) & grid_ids and grid_ids <= set(TRAIN_GRIDS):
        raise ValueError("holdout grid in training manifest")