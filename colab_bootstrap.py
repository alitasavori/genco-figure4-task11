"""Colab helpers: pull repo, mount Drive, install deps. Used by Task1_1.ipynb."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/content/genco-figure4-task11")
REPO_URL = "https://github.com/alitasavori/genco-figure4-task11.git"
COLAB_DRIVE = Path("/content/drive/MyDrive")
COLAB_DATA_TASK11 = COLAB_DRIVE / "Foundation Model" / "vendor" / "pfdelta_task1.1"
WINDOWS_DATA_TASK11 = Path(r"K:\My Drive\Foundation Model\vendor\pfdelta_task1.1")


def on_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def git_head(repo: Path = REPO) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "log", "-1", "--oneline"],
        text=True,
    ).strip()


def pull_latest(*, clone_if_missing: bool = True) -> str:
    """Clone or hard-reset to ``origin/main``; ``chdir`` to the repo. Returns ``git log -1``."""
    if on_colab():
        if not REPO.is_dir():
            if not clone_if_missing:
                raise FileNotFoundError(f"Repo missing at {REPO}; run the Pull latest cell.")
            subprocess.check_call(["git", "clone", REPO_URL, str(REPO)])
        else:
            subprocess.check_call(["git", "-C", str(REPO), "fetch", "origin"])
            subprocess.check_call(["git", "-C", str(REPO), "reset", "--hard", "origin/main"])
        os.chdir(REPO)
        head = git_head(REPO)
        print(f"Repo at {head}", flush=True)
        return head
    here = Path.cwd()
    if (here / "train_tpgt_task11.py").is_file():
        head = git_head(here) if (here / ".git").is_dir() else "local (no git)"
        print(f"Repo at {head}", flush=True)
        return head
    if (here / "genco" / "train_tpgt_task11.py").is_file():
        os.chdir(here / "genco")
        repo = Path.cwd()
        head = git_head(repo) if (repo / ".git").is_dir() else "local (no git)"
        print(f"Repo at {head}", flush=True)
        return head
    raise FileNotFoundError("train_tpgt_task11.py not found; open the genco folder or run on Colab.")


def ensure_training_cwd() -> Path:
    """Use the repo after prep; local dev falls back to genco/."""
    if on_colab():
        if not (REPO / "train_tpgt_task11.py").is_file():
            raise RuntimeError("Run the Pull latest cell first (repo not at /content/genco-figure4-task11).")
        os.chdir(REPO)
        return REPO
    return Path(os.getcwd())


def mount_drive_if_needed() -> Path:
    """Return Drive root on Colab; local Windows path when available."""
    if on_colab():
        if not COLAB_DRIVE.is_dir():
            from google.colab import drive

            drive.mount("/content/drive")
        return COLAB_DRIVE
    if WINDOWS_DATA_TASK11.is_dir():
        return Path(r"K:\My Drive")
    return Path.cwd()


def install_required_packages() -> None:
    missing: list[str] = []
    for module, package in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("matplotlib", "matplotlib"),
        ("pyarrow", "pyarrow"),
        ("huggingface_hub", "huggingface_hub"),
    ):
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("Required packages are already installed.", flush=True)


def vendor_data_id(task: str) -> str:
    return "1.3" if task in ("1.3", "2.1") else task


def configure_dataset_env(task: str, drive: Path) -> str:
    """Set ``GENCO_PFDELTA_*`` for the chosen PFΔ task."""
    data_id = vendor_data_id(task)
    vendor = drive / "Foundation Model" / "vendor" / f"pfdelta_task{data_id}"
    os.environ["GENCO_PFDELTA_TASK" + data_id.replace(".", "")] = str(vendor)
    return data_id


def run_dir_name(task: str, arch: str) -> str:
    data_id = vendor_data_id(task)
    if task == "1.1":
        return f"tpgt_task11_{arch}"
    return f"tpgt_task{data_id}_{arch}"
