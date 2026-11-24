"""Locate PFΔ ``sample_*.json`` folders (Task 1.1 parquet-pack layout on Drive)."""

from __future__ import annotations

from pathlib import Path

from pfdelta_case_parquet import list_raw_sample_paths


def candidate_raw_dirs(vendor_root: Path, grid_id: str) -> list[Path]:
    """All locations we probe for PFΔ JSON (for diagnostics)."""
    vendor_root = vendor_root.resolve()
    if vendor_root.name == "pfdelta_task1.1":
        pack = vendor_root
        vendor_parent = vendor_root.parent
    elif vendor_root.name.startswith("pfdelta_task1.1_"):
        vendor_parent = vendor_root.parent
        pack = vendor_parent / "pfdelta_task1.1"
        dedicated = vendor_root
        return [
            dedicated / "n" / "feasible" / "raw",
            dedicated / "train" / grid_id / "n" / "feasible" / "raw",
            dedicated / grid_id / "train" / grid_id / "n" / "feasible" / "raw",
            pack / grid_id / "train" / grid_id / "n" / "feasible" / "raw",
            pack / grid_id / "valid" / grid_id / "n" / "feasible" / "raw",
            pack / grid_id / "test" / grid_id / "n" / "feasible" / "raw",
        ]
    else:
        pack = vendor_root / "pfdelta_task1.1"
        vendor_parent = vendor_root

    return [
        pack / grid_id / "train" / grid_id / "n" / "feasible" / "raw",
        pack / grid_id / "valid" / grid_id / "n" / "feasible" / "raw",
        pack / grid_id / "test" / grid_id / "n" / "feasible" / "raw",
        pack / grid_id / "n" / "feasible" / "raw",
        vendor_parent / f"pfdelta_task1.1_{grid_id}" / "n" / "feasible" / "raw",
        vendor_parent / f"pfdelta_task1.1_{grid_id}" / "train" / grid_id / "n" / "feasible" / "raw",
        vendor_parent / f"pfdelta_task1.1_{grid_id}" / grid_id / "train" / grid_id / "n" / "feasible" / "raw",
    ]


def diagnose_raw_dirs(vendor_root: Path, grid_id: str) -> list[dict]:
    rows = []
    for raw in candidate_raw_dirs(vendor_root, grid_id):
        exists = raw.is_dir()
        n_json = len(list_raw_sample_paths(raw)) if exists else 0
        n_parquet = len(list(raw.glob("*.parquet"))) if exists else 0
        rows.append(
            {
                "path": str(raw),
                "exists": exists,
                "n_json": n_json,
                "n_parquet": n_parquet,
                "hint": "parquet_only" if exists and n_json == 0 and n_parquet else "",
            }
        )
    return rows


def resolve_pfdelta_raw_dir(vendor_root: Path, grid_id: str) -> Path | None:
    for raw in candidate_raw_dirs(vendor_root, grid_id):
        if raw.is_dir() and list_raw_sample_paths(raw):
            return raw
    return None


def resolve_vendor_for_grid(vendor_root: Path, grid_id: str) -> list[Path]:
    """Vendor roots to try when JSON is missing (e.g. dedicated case57 pack on Drive)."""
    vendor_root = vendor_root.resolve()
    parent = vendor_root.parent if vendor_root.name.startswith("pfdelta") else vendor_root
    roots = [vendor_root]
    alt = parent / f"pfdelta_task1.1_{grid_id}"
    if alt.is_dir() and alt not in roots:
        roots.append(alt)
    pack = parent / "pfdelta_task1.1"
    if pack.is_dir() and pack not in roots:
        roots.append(pack)
    return roots