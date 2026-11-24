"""Human-readable CSV reports for PFΔ → JMAT dataset generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})
    return len(rows)


def manifest_to_triples_csv(manifest: list[dict[str, Any]], path: Path) -> int:
    fields = [
        "path",
        "grid_id",
        "scenario_id",
        "topology_id",
        "iteration",
        "split_bucket",
        "source_split",
    ]
    return write_csv(path, fields, manifest)