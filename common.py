"""Shared paths, pinned commits, and result-file helpers for Task 1.1."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"

# Recorded from the GitHub API on 2026-10-29. Install scripts check these SHAs
# out explicitly so a later branch move does not change the reproduction.
PINNED_COMMITS = {
    "gridfm-paper-genco": {
        "url": "https://github.com/gridfm/gridfm-paper-genco.git",
        "branch": "main",
        "sha": "107ae0be34660fb7de5e629fa74c60ee63afad2e",
    },
    "gridfm-graphkit": {
        "url": "https://github.com/gridfm/gridfm-graphkit.git",
        "branch": "genco-paper-repro-pfdelta",
        "sha": "310e2f989b46058f3e6c5b4a56b7d03de06a3340",
    },
    "pfdelta": {
        "url": "https://github.com/MOSSLab-MIT/pfdelta.git",
        "branch": "main",
        "sha": "0fb3aeda2bbca807418a49cc934e4640f988964a",
    },
}

TASK = "1.1"
CATEGORIES = ("N", "N-1", "N-2", "C2I")
MODELS = ("GENCO Base", "CANOS-PF", "GNS-S", "PFNet")
SEED_COLUMNS = [
    "model",
    "task",
    "test_category",
    "seed",
    "mean_pbl_pu",
    "checkpoint_path",
    "source",
]
SUMMARY_COLUMNS = [
    "model",
    "test_category",
    "mean_across_seeds",
    "std_across_seeds",
    "published_mean",
    "published_std",
    "abs_diff",
    "rel_diff",
    "measured_source",
    "n_seeds",
]

# PFDelta loader (core/datasets/pfdelta_dataset.py): Task 1.1 training pool is
# 54,000 feasible N samples, then a 90/10 shuffle split. The independent test
# split is 2,000 feasible + 200 close-to-infeasible samples per topology.
EXPECTED_TRAIN_POOL_N = 54000
EXPECTED_TRAIN_N = 48600
EXPECTED_VAL_N = 5400
EXPECTED_TEST_COUNTS = {
    ("feasible", "n"): 2000,
    ("feasible", "n-1"): 2000,
    ("feasible", "n-2"): 2000,
    ("near infeasible", "n"): 200,
    ("near infeasible", "n-1"): 200,
    ("near infeasible", "n-2"): 200,
}
EXPECTED_TEST_TOTAL = sum(EXPECTED_TEST_COUNTS.values())


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


def ensure_dirs() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)


def work_root() -> Path:
    override = os.environ.get("GENCO_WORK_ROOT")
    if override:
        path = Path(override)
    elif in_colab() and Path("/content/drive/MyDrive").is_dir():
        path = Path("/content/drive/MyDrive/genco_figure4_task11")
    elif in_colab():
        path = Path("/content/genco_figure4_task11")
    else:
        path = ROOT / "vendor"
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_environment() -> dict:
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_count"] = 0
            info["gpu_name"] = None
    except Exception as exc:  # torch is absent in the published-only path
        info["torch"] = None
        info["torch_error"] = str(exc)
    print("Environment")
    for key, value in info.items():
        print(f"  {key}: {value}")
    ensure_dirs()
    (LOGS / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def run_command(cmd: list[str], cwd: Path | None, log_path: Path, env: dict | None = None) -> None:
    ensure_dirs()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        process = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=merged,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise SystemExit(f"Command failed ({process.returncode}). Log: {log_path}\n{tail}")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
    )


def clone_repo(name: str, dest: Path) -> str:
    spec = PINNED_COMMITS[name]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        cloned = _git(["clone", "--branch", spec["branch"], spec["url"], str(dest)], dest.parent)
        if cloned.returncode != 0:
            raise SystemExit(cloned.stderr or cloned.stdout)
    head = _git(["rev-parse", "HEAD"], dest)
    if head.returncode != 0 or head.stdout.strip() != spec["sha"]:
        fetched = _git(["fetch", "--depth", "1", "origin", spec["sha"]], dest)
        if fetched.returncode != 0:
            fetched = _git(["fetch", "origin", spec["sha"]], dest)
        if fetched.returncode != 0:
            raise SystemExit(fetched.stderr or fetched.stdout)
        checked = _git(["checkout", "--detach", spec["sha"]], dest)
        if checked.returncode != 0:
            raise SystemExit(checked.stderr or checked.stdout)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dest, text=True).strip()
    if sha != spec["sha"]:
        raise SystemExit(f"{name} HEAD {sha} does not match pinned {spec['sha']}")
    print(f"{name} at {sha}")
    return sha


def write_provenance(extra: dict | None = None) -> None:
    ensure_dirs()
    payload = {
        "task": TASK,
        "system": "IEEE 118-bus",
        "pinned_commits": PINNED_COMMITS,
        "canos_pf_loss_correction": {
            "date": "2026-03-26",
            "status": "fixed_in_pinned_pfdelta_commit",
            "detail": (
                "The PFDelta README at 0fb3aeda states that constraint_violations_loss_pf "
                "incorrectly calculated reactive-power mismatch and line-branch-flow mismatch, "
                "and that the bug has been fixed. Full reproduction uses that corrected "
                "implementation. Figure 4 reused the previously published PFDelta baseline "
                "numbers, so retrained CANOS-PF can differ from those bars. The correction "
                "is not reverted."
            ),
        },
        "std_convention": (
            "Sample standard deviation with Bessel correction (numpy.std ddof=1), "
            "matching torch.Tensor.std in scripts/test_error.py."
        ),
        "metric": (
            "Mean over samples of the mean over buses of sqrt(dP^2 + dQ^2), "
            "in per unit on a 100 MVA base."
        ),
    }
    if extra:
        payload.update(extra)
    (RESULTS / "provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_csv(path: Path, columns: list[str], rows: list[dict], key_fields: list[str]) -> None:
    ensure_dirs()
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    index = {tuple(row.get(field, "") for field in key_fields): row for row in existing}
    for row in rows:
        complete = {column: row.get(column, "") for column in columns}
        index[tuple(str(complete.get(field, "")) for field in key_fields)] = complete
    ordered = list(index.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
