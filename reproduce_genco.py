"""Train GENCO Base on Task 1.1 and test the three seeds."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

from checks import write_layout_report
from common import (
    LOGS,
    RESULTS,
    SUMMARY_COLUMNS,
    TASK,
    ensure_dirs,
    in_colab,
    report_environment,
    run_command,
    upsert_csv,
    work_root,
)
from metrics import aggregate_c2i, sample_mean, sample_std
from plot_task1_1 import plot_task11

GENCO_SEEDS = (1, 2, 3)
CATEGORY_FROM_SUFFIX = {
    "n_feasible": "N",
    "n-1_feasible": "N-1",
    "n-2_feasible": "N-2",
    "n_nose": "C2I-N",
    "n-1_nose": "C2I-N-1",
    "n-2_nose": "C2I-N-2",
}


def download_data(root: Path) -> Path:
    from huggingface_hub import snapshot_download

    data = root / "pfdelta_task1.1"
    snapshot_download(repo_id="gridfm/pfdelta_task1.1", repo_type="dataset", local_dir=str(data))
    return data


def case_dir(data_root: Path) -> Path:
    nested = data_root / "case118"
    if (nested / "train").is_dir():
        return nested
    if (data_root / "train").is_dir():
        return data_root
    raise FileNotFoundError(f"Converted Task 1.1 case folder not found under {data_root}.")


def worker_cap() -> int | None:
    import os

    if os.environ.get("GENCO_DATA_WORKERS"):
        return int(os.environ["GENCO_DATA_WORKERS"])
    if in_colab():
        return 2
    return None


def prepare_config(repo: Path, seed: int) -> Path:
    source = repo / "scripts" / "pfdelta" / "config" / f"HGNS_PF_pfdelta_bs64_seed{seed}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    cap = worker_cap()
    if cap is not None and int(config["data"]["workers"]) != cap:
        message = (
            f"data.workers changed from {config['data']['workers']} to {cap} "
            "in a config copy. The YAML inside the clone is unchanged."
        )
        print(message)
        (LOGS / "genco_worker_override.txt").write_text(message + "\n", encoding="utf-8")
        config["data"]["workers"] = cap
    destination = RESULTS / "configs" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def graphkit_env() -> dict[str, str]:
    return {"MLFLOW_ALLOW_FILE_STORE": "true"}


def parse_split_lengths(log_text: str) -> dict:
    counts = {}
    train = re.search(r"Length of train datasets:\s+(\d+)", log_text)
    valid = re.search(r"Length of valid datasets:\s+(\d+)", log_text)
    if train:
        counts["train"] = int(train.group(1))
    if valid:
        counts["valid"] = int(valid.group(1))
    for name, length in re.findall(r"Name: (\S+)\nLength: (\d+)", log_text):
        counts[name] = int(length)
    return counts


def category_of_dataset(name: str) -> str | None:
    prefix = "test_case118_"
    if not name.startswith(prefix):
        return None
    return CATEGORY_FROM_SUFFIX.get(name[len(prefix) :])


def read_pbe(metrics_csv: Path) -> float:
    import pandas as pd

    frame = pd.read_csv(metrics_csv)
    match = frame.loc[frame["Metric"] == "PBE (Mean, MVA)", "Value"]
    if match.empty:
        raise ValueError(f"{metrics_csv} has no 'PBE (Mean, MVA)' row.")
    value = float(match.iloc[0])
    if value != value:
        raise ValueError(f"Non-finite PBE in {metrics_csv}.")
    return value


def collect_eval_metrics(log_dir: Path) -> dict[str, float]:
    newest: dict[str, Path] = {}
    for path in log_dir.rglob("*_metrics.csv"):
        dataset = path.name[: -len("_metrics.csv")]
        category = category_of_dataset(dataset)
        if category is None:
            continue
        previous = newest.get(category)
        if previous is None or path.stat().st_mtime >= previous.stat().st_mtime:
            newest[category] = path
    found = {category: read_pbe(path) for category, path in newest.items()}
    required = ["N", "N-1", "N-2", "C2I-N", "C2I-N-1", "C2I-N-2"]
    missing = [key for key in required if key not in found]
    if missing:
        raise FileNotFoundError(f"Evaluation under {log_dir} is missing {missing}.")
    lengths = {}
    for path in log_dir.rglob("*.log"):
        lengths.update(parse_split_lengths(path.read_text(encoding="utf-8", errors="replace")))
    count_map = {
        "C2I-N": lengths.get("test_case118_n_nose"),
        "C2I-N-1": lengths.get("test_case118_n-1_nose"),
        "C2I-N-2": lengths.get("test_case118_n-2_nose"),
    }
    counts = count_map if all(value is not None for value in count_map.values()) else None
    found["C2I"] = aggregate_c2i(found, counts)
    return found


def export_last_state_dict(log_dir: Path, destination: Path) -> Path:
    import torch

    checkpoints = sorted(log_dir.rglob("last.ckpt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No last.ckpt under {log_dir}.")
    payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    print(f"Exported {checkpoints[-1]} to {destination}")
    return destination


def run_graphkit(command: str, config: Path, data_path: Path, seed: int, model_path: Path | None, log_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "gridfm_graphkit",
        command,
        "--config",
        str(config),
        "--data_path",
        str(data_path),
        "--exp_name",
        "case118_task1.1",
        "--run_name",
        f"task1.1_seed{seed}",
        "--log_dir",
        str(log_dir),
    ]
    if model_path is not None:
        cmd.extend(["--model_path", str(model_path)])
    run_command(cmd, cwd=log_dir.parent, log_path=log_dir / "console.log", env=graphkit_env())


def resume_train(config: Path, data_path: Path, seed: int, log_dir: Path, ckpt: Path) -> None:
    """Call the same objects as gridfm_graphkit train, adding Lightning ckpt_path.

    The released CLI has no resume flag. This does not change the model, loss, or YAML in the clone.
    """
    import random

    import lightning as L
    import numpy as np
    import torch
    from gridfm_graphkit.cli import get_training_callbacks
    from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
    from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
    from lightning.pytorch.loggers import MLFlowLogger

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = MLFlowLogger(save_dir=str(log_dir), experiment_name="case118_task1.1", run_name=f"task1.1_seed{seed}")
    config_args = NestedNamespace(**yaml.safe_load(config.read_text(encoding="utf-8")))
    torch.manual_seed(config_args.seed)
    random.seed(config_args.seed)
    np.random.seed(config_args.seed)
    datamodule = LitGridHeteroDataModule(config_args, str(data_path))
    model = get_task(config_args, datamodule.data_normalizers)
    trainer = L.Trainer(
        logger=logger,
        accelerator=config_args.training.accelerator,
        devices=config_args.training.devices,
        strategy=config_args.training.strategy,
        log_every_n_steps=100,
        default_root_dir=str(log_dir),
        max_epochs=config_args.training.epochs,
        callbacks=get_training_callbacks(config_args),
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(ckpt))


def write_measurement_rows(metrics_by_seed: dict[int, dict[str, float]], paths: dict[int, str]) -> None:
    seed_rows = []
    summary_rows = []
    for seed, metrics in metrics_by_seed.items():
        for category in ("N", "N-1", "N-2", "C2I"):
            seed_rows.append(
                {
                    "model": "GENCO Base",
                    "task": TASK,
                    "test_category": category,
                    "seed": str(seed),
                    "mean_pbl_pu": f"{metrics[category]:.12g}",
                    "checkpoint_path": paths[seed],
                    "source": "independent_training",
                }
            )
    upsert_csv(
        RESULTS / "task1_1_seed_metrics.csv",
        ["model", "task", "test_category", "seed", "mean_pbl_pu", "checkpoint_path", "source"],
        seed_rows,
        ["model", "task", "test_category", "seed", "source"],
    )
    if len(metrics_by_seed) != 3:
        print(f"GENCO has {len(metrics_by_seed)} tested seeds. The figure needs three.")
        return
    for category in ("N", "N-1", "N-2", "C2I"):
        values = [metrics_by_seed[seed][category] for seed in sorted(metrics_by_seed)]
        summary_rows.append(
            {
                "model": "GENCO Base",
                "test_category": category,
                "mean_across_seeds": f"{sample_mean(values):.12g}",
                "std_across_seeds": f"{sample_std(values):.12g}",
                "n_seeds": "3",
            }
        )
    upsert_csv(
        RESULTS / "task1_1_summary.csv",
        SUMMARY_COLUMNS,
        summary_rows,
        ["model", "test_category"],
    )


def evaluate_seeds(repo: Path, data_path: Path, checkpoints: dict[int, Path], source: str) -> None:
    metrics_by_seed = {}
    paths = {}
    for seed, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        log_dir = RESULTS / "logs" / f"genco_{source}_seed{seed}"
        log_dir.mkdir(parents=True, exist_ok=True)
        config = prepare_config(repo, seed)
        run_graphkit("evaluate", config, data_path, seed, checkpoint, log_dir)
        log_text = (log_dir / "console.log").read_text(encoding="utf-8", errors="replace")
        counts = parse_split_lengths(log_text)
        (log_dir / "split_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
        metrics_by_seed[seed] = collect_eval_metrics(log_dir)
        paths[seed] = str(checkpoint)
        print(f"seed {seed} {source}: {metrics_by_seed[seed]}")
    detail = RESULTS / f"genco_{source}_detail.json"
    detail.write_text(json.dumps({str(seed): metrics for seed, metrics in metrics_by_seed.items()}, indent=2), encoding="utf-8")
    write_measurement_rows(metrics_by_seed, paths)


def train_one(repo: Path, data_path: Path, seed: int) -> Path:
    log_dir = RESULTS / "logs" / f"genco_train_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    exported = log_dir / "last.pt"
    done = log_dir / "complete.json"
    if done.exists() and exported.exists():
        print(f"Seed {seed} already has {exported}. Skipping training.")
        return exported
    config = prepare_config(repo, seed)
    existing = sorted(log_dir.rglob("last.ckpt"), key=lambda path: path.stat().st_mtime)
    if existing:
        print(f"Resuming seed {seed} from {existing[-1]}")
        resume_train(config, data_path, seed, log_dir, existing[-1])
    else:
        run_graphkit("train", config, data_path, seed, None, log_dir)
    export_last_state_dict(log_dir, exported)
    done.write_text(json.dumps({"checkpoint": str(exported), "seed": seed}), encoding="utf-8")
    return exported


def main() -> None:
    ensure_dirs()
    report_environment()
    root = work_root()
    repo = root / "gridfm-graphkit"
    if not (repo / "gridfm_graphkit").is_dir():
        raise SystemExit(f"Missing {repo}. Run the install cell first.")
    data_root = download_data(root)
    data_path = case_dir(data_root)
    write_layout_report(data_root if (data_root / "case118").is_dir() else data_path, LOGS / "genco_layout.json")
    trained = {seed: train_one(repo, data_path, seed) for seed in GENCO_SEEDS}
    evaluate_seeds(repo, data_path, trained, "independent_training")
    plot_task11(RESULTS)
    print(f"GENCO test scores: {RESULTS / 'task1_1_summary.csv'}")


if __name__ == "__main__":
    main()
