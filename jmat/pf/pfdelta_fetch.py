"""Download PFΔ raw JSON for Newton logging (Hugging Face pfdelta/pfdelta)."""

from __future__ import annotations

import shutil
import tarfile
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from pfdelta_case_parquet import HF_REPO, list_raw_sample_paths


def json_cache_dir() -> Path:
    """Fast local SSD on Colab; not Google Drive (tar extract to Drive can take 30+ min)."""
    if Path("/content").is_dir():
        local = Path("/content/pfdelta_json_cache")
        local.mkdir(parents=True, exist_ok=True)
        print(f"JSON cache (Colab local, not Drive): {local}", flush=True)
        return local
    drive = Path("/content/drive/MyDrive/Foundation Model/vendor/pfdelta_json_cache")
    if drive.parent.is_dir():
        drive.mkdir(parents=True, exist_ok=True)
        return drive
    return Path.cwd() / "pfdelta_json_cache"


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
        print(f"PFΔ JSON cache hit: {n} samples at {raw}", flush=True)
        return raw

    if case_dir.is_dir() and n < min_samples:
        print(f"Clearing incomplete {case_dir} ({n} JSON)", flush=True)
        shutil.rmtree(case_dir)

    t0 = time.perf_counter()
    print(
        f"Step 1/2: downloading {grid_id}.tar.gz from Hugging Face ({HF_REPO}) "
        f"(full grid archive — often 5–20 min on first run; only {min_samples} OCs needed after extract)...",
        flush=True,
    )
    archive = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=f"{grid_id}.tar.gz",
            local_dir=str(work),
        )
    )
    print(f"  download finished in {time.perf_counter() - t0:.0f}s → {archive}", flush=True)
    t1 = time.perf_counter()
    print("Step 2/2: extracting tarball (local disk)...", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            tar.extract(member, work)
            if i == 0 or i + 1 == total or (i + 1) % 2000 == 0:
                print(f"  extract {i + 1}/{total} ({time.perf_counter() - t1:.0f}s elapsed)", flush=True)
    n = len(list_raw_sample_paths(raw))
    print(f"Extracted {n} JSON samples → {raw} in {time.perf_counter() - t1:.0f}s", flush=True)
    if n < min_samples:
        raise FileNotFoundError(f"After extract only {n} JSON under {raw}; need {min_samples}")
    return raw