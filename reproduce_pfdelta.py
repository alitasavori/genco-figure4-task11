"""Train and test PFNet, CANOS-PF, and GNS-S on PFDelta Task 1.1."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import yaml

from common import (
    LOGS,
    RESULTS,
    SUMMARY_COLUMNS,
    TASK,
    ensure_dirs,
    report_environment,
    run_command,
    upsert_csv,
    work_root,
)
from metrics import sample_mean, sample_std
from plot_task1_1 import plot_task11

MODELS = {
    "PFNet": {
        "config": "pfnet_task_1_1",
        "display": "PFNet",
    },
    "CANOS-PF": {
        "config": "canos_task_1_1",
        "display": "CANOS-PF",
    },
    "GNS-S": {
        "config": "gns_task_1_1",
        "display": "GNS-S",
    },
}
# The released YAML files all set functional.seed to 11. The other two paper
# seeds are not published, so the extra initializations are 21 and 31.
SEEDS = (11, 21, 31)
TEST_CATEGORIES = {
    "n": "N",
    "n-1": "N-1",
    "n-2": "N-2",
    "close2inf": "C2I",
}


def prepare_config(repo: Path, config_name: str, seed: int) -> str:
    source = repo / "core" / "configs" / f"{config_name}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.setdefault("functional", {})
    config["functional"]["seed"] = seed
    config["functional"]["checkpoint"] = True
    config["functional"]["run_name"] = f"seed{seed}"
    config["functional"]["run_location"] = f"task11_{config_name}_seed{seed}"
    copy_name = f"{config_name}_seed{seed}.yaml"
    destination = repo / "core" / "configs" / copy_name
    run_parent = repo / "runs" / config["functional"]["run_location"]
    partial = list(run_parent.glob("*/checkpoint.pt")) if run_parent.exists() else []
    complete = list(run_parent.glob("*/model.pt")) if run_parent.exists() else []
    if partial and not complete and destination.exists():
        print(f"Keeping {destination.name} so the upstream trainer can resume.")
        return copy_name
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(
        f"Wrote {destination.name} from {source.name} with seed {seed} and checkpointing. "
        "The released config file was not edited."
    )
    return copy_name


def newest_run(run_parent: Path) -> Path | None:
    runs = [path for path in run_parent.glob("*") if path.is_dir() and (path / "model.pt").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda path: (path / "model.pt").stat().st_mtime)


def archive_extra_runs(run_parent: Path, keep: Path) -> None:
    archive = run_parent.parent / "_archive" / run_parent.name
    for path in run_parent.iterdir():
        if path.resolve() == keep.resolve() or not path.is_dir():
            continue
        archive.mkdir(parents=True, exist_ok=True)
        target = archive / path.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(path), str(target))


def train_model(repo: Path, config_name: str, seed: int) -> Path:
    copy_name = prepare_config(repo, config_name, seed)
    stem = copy_name[: -len(".yaml")]
    run_parent = repo / "runs" / f"task11_{config_name}_seed{seed}"
    existing = newest_run(run_parent) if run_parent.exists() else None
    if existing is not None and not list(run_parent.glob("*/checkpoint.pt")):
        print(f"Using completed run {existing}")
        return existing
    if existing is not None:
        print(f"Completed weights already exist at {existing}. Not starting another run.")
        return existing
    log_path = LOGS / f"{config_name}_seed{seed}.log"
    run_command(
        [sys.executable, "main.py", "--config", stem],
        cwd=repo,
        log_path=log_path,
    )
    finished = newest_run(run_parent)
    if finished is None:
        raise FileNotFoundError(f"Training for {config_name} seed {seed} did not write model.pt.")
    return finished


def evaluate_model(repo: Path, config_name: str, seed: int) -> dict[str, float]:
    run_parent = repo / "runs" / f"task11_{config_name}_seed{seed}"
    keep = newest_run(run_parent)
    if keep is None:
        raise FileNotFoundError(f"No completed run under {run_parent}.")
    archive_extra_runs(run_parent, keep)
    before = repo / "test_errors_p_seeds.json"
    if before.exists():
        before.unlink()
    run_command(
        [sys.executable, "scripts/test_error.py", "--root", run_parent.name],
        cwd=repo,
        log_path=LOGS / f"test_{config_name}_seed{seed}.log",
    )
    payload = json.loads((repo / "test_errors_p_seeds.json").read_text(encoding="utf-8"))
    key = f"runs/{run_parent.name}"
    if key not in payload:
        key = next(iter(payload))
    means = payload[key]["PBL Mean"]
    if any(len(values) != 1 for values in means.values()):
        raise RuntimeError(f"Expected one seed in {key}, found { {name: len(values) for name, values in means.items()} }.")
    parsed = {}
    for raw_name, category in TEST_CATEGORIES.items():
        value = float(means[raw_name][0])
        if value != value:
            raise ValueError(f"Non-finite PBL for {config_name} seed {seed} {category}.")
        parsed[category] = value
    shutil.copy(repo / "test_errors_p_seeds.json", LOGS / f"test_{config_name}_seed{seed}.json")
    return parsed


def count_probe(repo: Path) -> None:
    script = LOGS / "pfdelta_count_probe.py"
    script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "sys.path.insert(0, '.')",
                "from core.datasets.pfdelta_variants import PFDeltaPFNet",
                "root = 'data/pfdelta_data'",
                "rows = {}",
                "for split in ('train', 'val'):",
                "    dataset = PFDeltaPFNet(root_dir=root, case_name='case118', split=split, model='PFNet', task=1.1, add_bus_type=False)",
                "    rows[split] = len(dataset)",
                "tests = {",
                "    'feasible_n': 'separate_case118_test_feasible_n',",
                "    'feasible_n-1': 'separate_case118_test_feasible_n-1',",
                "    'feasible_n-2': 'separate_case118_test_feasible_n-2',",
                "    'nose_n': 'separate_case118_test_near infeasible_n',",
                "    'nose_n-1': 'separate_case118_test_near infeasible_n-1',",
                "    'nose_n-2': 'separate_case118_test_near infeasible_n-2',",
                "}",
                "for name, split in tests.items():",
                "    dataset = PFDeltaPFNet(root_dir=root, case_name='case118', split=split, model='PFNet', task=1.1, add_bus_type=False)",
                "    rows[name] = len(dataset)",
                "print(json.dumps(rows))",
            ]
        ),
        encoding="utf-8",
    )
    destination = LOGS / "pfdelta_split_counts.json"
    run_command([sys.executable, str(script)], cwd=repo, log_path=LOGS / "pfdelta_count_probe.log")
    text = (LOGS / "pfdelta_count_probe.log").read_text(encoding="utf-8", errors="replace")
    line = [item for item in text.splitlines() if item.startswith("{")][-1]
    destination.write_text(line + "\n", encoding="utf-8")
    print(f"PFDelta loader counts: {line}")


def write_rows(display: str, by_seed: dict[int, dict[str, float]], checkpoints: dict[int, Path]) -> None:
    seed_rows = []
    for seed, metrics in by_seed.items():
        for category, value in metrics.items():
            seed_rows.append(
                {
                    "model": display,
                    "task": TASK,
                    "test_category": category,
                    "seed": str(seed),
                    "mean_pbl_pu": f"{value:.12g}",
                    "checkpoint_path": str(checkpoints[seed] / "model.pt"),
                    "source": "independent_training",
                }
            )
    upsert_csv(
        RESULTS / "task1_1_seed_metrics.csv",
        ["model", "task", "test_category", "seed", "mean_pbl_pu", "checkpoint_path", "source"],
        seed_rows,
        ["model", "task", "test_category", "seed", "source"],
    )
    if len(by_seed) != 3:
        print(f"{display} has {len(by_seed)} completed seeds. The figure needs three.")
        return
    summary_rows = []
    for category in ("N", "N-1", "N-2", "C2I"):
        values = [by_seed[seed][category] for seed in sorted(by_seed)]
        summary_rows.append(
            {
                "model": display,
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


def verify_release(repo: Path) -> None:
    readme = (repo / "README.md").read_text(encoding="utf-8")
    if "March 26, 2026" not in readme or "constraint_violations_loss_pf" not in readme:
        raise SystemExit("Pinned PFDelta README does not contain the CANOS-PF loss-correction note.")
    for spec in MODELS.values():
        path = repo / "core" / "configs" / f"{spec['config']}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if config["functional"]["seed"] != 11:
            raise SystemExit(f"{path.name} no longer has the released seed 11.")
        datasets = config["dataset"]["datasets"]
        splits = {item["split"] for item in datasets}
        if splits != {"train", "val"}:
            raise SystemExit(f"{path.name} training config contains {splits}, expected only train and val.")
        if any(item.get("task") not in (1.1, "1.1") for item in datasets):
            raise SystemExit(f"{path.name} is not Task 1.1.")
    print("Released Task 1.1 configs train on train/val only and record seed 11.")
    print("CANOS-PF loss correction is present in this commit and will be used if you train.")


def main() -> None:
    ensure_dirs()
    report_environment()
    repo = work_root() / "pfdelta"
    if not (repo / "main.py").is_file():
        raise SystemExit(f"Missing {repo}. Run the install cell first.")
    verify_release(repo)
    count_probe(repo)
    for spec in MODELS.values():
        metrics = {}
        checkpoints = {}
        for seed in SEEDS:
            run_dir = train_model(repo, spec["config"], seed)
            metrics[seed] = evaluate_model(repo, spec["config"], seed)
            checkpoints[seed] = run_dir
        write_rows(spec["display"], metrics, checkpoints)
    plot_task11(RESULTS)
    print(f"Baseline test scores: {RESULTS / 'task1_1_summary.csv'}")


if __name__ == "__main__":
    main()
