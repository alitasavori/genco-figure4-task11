"""IEEE-57 Task 1.1 case plumbing and JSON→parquet conversion."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pfdelta_case_parquet as conv  # noqa: E402
import train_tpgt_task11 as tpgt  # noqa: E402


def _tiny_pm_case() -> dict:
    """3-bus PowerModels-style solved sample (1-based bus ids)."""
    y12 = 1 / complex(0.02, 0.06)
    network = {
        "baseMVA": 100.0,
        "bus": {
            "1": {"bus_type": 3, "vmin": 0.9, "vmax": 1.1},
            "2": {"bus_type": 2, "vmin": 0.9, "vmax": 1.1},
            "3": {"bus_type": 1, "vmin": 0.9, "vmax": 1.1},
        },
        "load": {"1": {"load_bus": 3, "pd": 0.5, "qd": 0.2}},
        "gen": {
            "1": {"gen_bus": 1, "gen_status": 1, "pmin": 0, "pmax": 2, "qmin": -1, "qmax": 1},
            "2": {"gen_bus": 2, "gen_status": 1, "pmin": 0, "pmax": 2, "qmin": -1, "qmax": 1},
        },
        "shunt": {},
        "branch": {
            "1": {
                "f_bus": 1,
                "t_bus": 2,
                "br_r": 0.02,
                "br_x": 0.06,
                "g_fr": 0.0,
                "b_fr": 0.0,
                "g_to": 0.0,
                "b_to": 0.0,
                "tap": 1.0,
                "shift": 0.0,
                "br_status": 1,
                "rate_a": 100.0,
            },
            "2": {
                "f_bus": 2,
                "t_bus": 3,
                "br_r": 0.03,
                "br_x": 0.08,
                "g_fr": 0.0,
                "b_fr": 0.0,
                "g_to": 0.0,
                "b_to": 0.0,
                "tap": 1.0,
                "shift": 0.0,
                "br_status": 1,
                "rate_a": 100.0,
            },
        },
    }
    # Flat start voltages; gens chosen so residual is not checked here.
    solution = {
        "bus": {
            "1": {"vm": 1.0, "va": 0.0},
            "2": {"vm": 1.0, "va": -0.05},
            "3": {"vm": 0.98, "va": -0.1},
        },
        "gen": {"1": {"pg": 0.6, "qg": 0.1}, "2": {"pg": 0.2, "qg": 0.05}},
        "branch": {
            "1": {"pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": 0.0},
            "2": {"pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": 0.0},
        },
    }
    _ = y12
    return {"network": network, "solution": {"solution": solution}}


def test_apply_case_config_renames_run_dir(monkeypatch):
    monkeypatch.setenv("GENCO_TPGT_CASE", "case57")
    monkeypatch.setattr(sys, "argv", ["train_tpgt_task11.py"])
    monkeypatch.setattr(tpgt, "RUN_TAG", "v9")
    try:
        assert tpgt.apply_case_config() == "case57"
        assert tpgt.CASE_NAME == "case57"
        assert tpgt.RUN_DIR_NAME == "tpgt_task11_case57_v9"
        assert tpgt.eval_case_names(tpgt.TASKS["1.1"]) == ("case57",)
        assert tpgt.dataset_relative(tpgt.TASKS["1.1"]) == Path(
            "Foundation Model/vendor/pfdelta_task1.1_case57"
        )
        assert tpgt.raw_dir(Path("data"), "train", "n", "feasible") == Path(
            "data/case57/train/case57/n/feasible/raw"
        )
    finally:
        monkeypatch.delenv("GENCO_TPGT_CASE", raising=False)
        monkeypatch.setattr(tpgt, "RUN_TAG", "v7")
        tpgt.apply_case_config()


def test_case118_run_dir_unchanged(monkeypatch):
    monkeypatch.delenv("GENCO_TPGT_CASE", raising=False)
    monkeypatch.setattr(sys, "argv", ["train_tpgt_task11.py"])
    monkeypatch.setattr(tpgt, "RUN_TAG", "v9")
    try:
        assert tpgt.apply_case_config() == "case118"
        assert tpgt.RUN_DIR_NAME == "tpgt_task11_v9"
    finally:
        monkeypatch.setattr(tpgt, "RUN_TAG", "v7")
        tpgt.apply_case_config()


def test_task11_pack_complete_requires_all_test_folders(tmp_path):
    case = "case57"
    # Only train N present → incomplete.
    train_raw = tmp_path / case / "train" / case / "n" / "feasible" / "raw"
    train_raw.mkdir(parents=True)
    for name in conv.PARQUET_NAMES:
        (train_raw / name).write_text("x", encoding="utf-8")
    assert not conv.task11_pack_complete(tmp_path, case)
    # Create every required folder.
    for raw in conv.task11_pack_raw_dirs(tmp_path, case):
        raw.mkdir(parents=True, exist_ok=True)
        for name in conv.PARQUET_NAMES:
            (raw / name).write_text("x", encoding="utf-8")
    assert conv.task11_pack_complete(tmp_path, case)


def test_task11_n1_test_only_allows_small_archive(tmp_path):
    """N-1/N-2 Task 1.1 only need test_size samples, not the 54k N train pool."""
    case_dir = tmp_path / "case57"
    raw = case_dir / "n-1" / "raw"
    raw.mkdir(parents=True)
    for i in range(1, 101):
        (raw / f"sample_{i}.json").write_text("{}", encoding="utf-8")
    shuffle_dir = tmp_path / "shuffle_files" / "n-1"
    shuffle_dir.mkdir(parents=True)
    shuffle = {str(i): (i * 7) % 100 for i in range(100)}
    (shuffle_dir / "raw_shuffle.json").write_text(
        __import__("json").dumps(shuffle), encoding="utf-8"
    )
    files = conv.task11_feasible_file_lists(
        case_dir, tmp_path / "shuffle_files", "n-1", train_size=0, test_size=20
    )["test"]
    assert len(files) == 20


def test_list_raw_sample_paths_accepts_both_namings(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "sample_2.json").write_text("{}", encoding="utf-8")
    (raw / "sample_1.json").write_text("{}", encoding="utf-8")
    paths = conv.list_raw_sample_paths(raw)
    assert [p.name for p in paths] == ["sample_1.json", "sample_2.json"]
    assert not conv.feasible_raw_complete(tmp_path, "n", min_samples=3)


def test_scenario_rows_units_and_ybus(tmp_path):
    rows = conv.scenario_rows(_tiny_pm_case(), scenario=0)
    assert rows["bus"]["bus"] == [0, 1, 2]
    assert rows["bus"]["REF"][0] == 1.0
    assert rows["bus"]["PV"][1] == 1.0
    assert rows["bus"]["PQ"][2] == 1.0
    # PowerModels p.u. * baseMVA -> MW in parquet.
    assert rows["bus"]["Pd"][2] == pytest.approx(50.0)
    assert rows["bus"]["Va"][1] == pytest.approx(math.degrees(-0.05))
    ybus = conv.build_ybus(_tiny_pm_case()["network"])
    assert ybus.shape == (3, 3)
    assert np.isfinite(ybus).all()
    conv.write_parquet_folder(
        {"bus": rows["bus"], "branch": rows["branch"], "gen": rows["gen"], "y_bus": rows["y_bus"]},
        tmp_path / "raw",
    )
    for name in conv.PARQUET_NAMES:
        assert (tmp_path / "raw" / name).is_file()
