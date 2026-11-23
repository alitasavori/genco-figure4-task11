"""Locate PFΔ ``sample_*.json`` folders (Task 1.1 parquet-pack layout on Drive)."""

from __future__ import annotations

from pathlib import Path

from pfdelta_case_parquet import list_raw_sample_paths


def _has_json_samples(raw_dir: Path) -> bool:
    return bool(list_raw_sample_paths(raw_dir))


def resolve_pfdelta_raw_dir(vendor_root: Path, grid_id: str) -> Path | None:
    """Return a directory containing PFΔ JSON samples for ``grid_id``, or None."""
    vendor_root = vendor_root.resolve()
    if vendor_root.name == "pfdelta_task1.1":
        pack = vendor_root
        vendor_parent = vendor_root.parent
    else:
        pack = vendor_root / "pfdelta_task1.1"
        vendor_parent = vendor_root

    candidates = [
        pack / grid_id / "train" / grid_id / "n" / "feasible" / "raw",
        pack / grid_id / "valid" / grid_id / "n" / "feasible" / "raw",
        pack / grid_id / "test" / grid_id / "n" / "feasible" / "raw",
        pack / grid_id / "n" / "feasible" / "raw",
        vendor_parent / f"pfdelta_task1.1_{grid_id}" / "n" / "feasible" / "raw",
        vendor_parent / f"pfdelta_task1.1_{grid_id}" / "train" / grid_id / "n" / "feasible" / "raw",
    ]
    for raw in candidates:
        if _has_json_samples(raw):
            return raw
    return None