"""Download PFΔ raw JSON for Newton logging (Hugging Face pfdelta/pfdelta)."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from pfdelta_case_parquet import HF_REPO, list_raw_sample_paths


def json_cache_dir() -> Path:
    drive = Path("/content/drive/MyDrive/Foundation Model/vendor/pfdelta_json_cache")
    if drive.parent.is_dir():
        drive.mkdir(parents=True, exist_ok=True)
        return drive
    local = Path("/content/pfdelta_json_cache")
    local.mkdir(parents=True, exist_ok=True)
    return local


def ensure_pfdelta_json_raw(grid_id: str, *, min_samples: int = 1, work: Path | None = None) -> Path:
    """
    Ensure ``{work}/{grid_id}/n/raw/sample_*.json`` exists; download ``{grid_id}.tar.gz`` if needed.
    """
    work = work or json_cache_dir()
    work.mkdir(parents=True, exist_ok=True)
    case_dir = work / grid_id
    raw = case_dir / "n" / "raw"
    n = len(list_raw_sample_paths(raw)) if raw.is_dir() else 0
    if n >= min_samples:
        print(f"PFΔ JSON cache: {n} samples at {raw}", flush=True)
        return raw

    if case_dir.is_dir() and n < min_samples:
        print(f"Clearing incomplete {case_dir} ({n} JSON)", flush=True)
        shutil.rmtree(case_dir)

    print(f"Downloading HF {HF_REPO} {grid_id}.tar.gz (first time may take several minutes)...", flush=True)
    archive = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=f"{grid_id}.tar.gz",
            local_dir=str(work),
        )
    )
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(work)
    n = len(list_raw_sample_paths(raw))
    print(f"Extracted {n} JSON samples → {raw}", flush=True)
    if n < min_samples:
        raise FileNotFoundError(f"After extract only {n} JSON under {raw}; need {min_samples}")
    return raw