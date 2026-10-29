"""Pick the directory used for data and checkpoints before other modules import it.

On Colab this mounts Google Drive and points later scripts at that folder.
A disconnected runtime can then resume finished seeds instead of starting over.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


def mount_and_select() -> tuple[Path, Path]:
    if in_colab():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Google Drive mount skipped: {exc}")
        drive = Path("/content/drive/MyDrive")
        if drive.is_dir():
            work = drive / "genco_figure4_task11"
            print(f"Data, checkpoints, and metrics will be saved on Drive: {work}")
        else:
            work = Path("/content/genco_figure4_task11")
            print(
                "Google Drive is not mounted. A disconnected runtime deletes checkpoints. "
                f"Directory: {work}"
            )
        results = work / "results"
    else:
        work = Path(os.environ.get("GENCO_WORK_ROOT", str(ROOT / "vendor")))
        results = Path(os.environ.get("GENCO_RESULTS_DIR", str(ROOT / "results")))
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    os.environ["GENCO_WORK_ROOT"] = str(work)
    os.environ["GENCO_RESULTS_DIR"] = str(results)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    return work, results
