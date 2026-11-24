"""Default JMAT artifact roots: always prefer Google Drive on Colab."""

from __future__ import annotations

import os
from pathlib import Path

DRIVE_JMAT = Path("/content/drive/MyDrive/Foundation Model/results/jmat")


def drive_mounted() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


def pfdelta_output_dir(*, grids: list[str] | None = None, explicit: Path | None = None) -> Path:
    if explicit is not None:
        out = explicit
    elif drive_mounted():
        if grids and len(grids) == 1:
            out = DRIVE_JMAT / f"pfdelta_{grids[0]}"
        else:
            out = DRIVE_JMAT / "pfdelta_pilot"
    else:
        root = Path(__file__).resolve().parents[1]
        out = root / "results" / "jmat" / "pfdelta_pilot"
    out.mkdir(parents=True, exist_ok=True)
    return out


def print_output_location(out: Path) -> None:
    if str(out).startswith("/content/drive/"):
        print(f"*** All NPZ + CSV saved on Google Drive (survives disconnect): {out}", flush=True)
    else:
        print(
            f"*** WARNING: saving under {out} — NOT on Drive. Mount Drive and use --out {DRIVE_JMAT}/...",
            flush=True,
        )
    os.environ.setdefault("JMAT_DATA_OUT", str(out))