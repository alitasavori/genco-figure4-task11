"""Checkpoint bus error maps drawn in-process from a loaded model and splits (118-bus synthetic grid)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_bus_error_map  # noqa: E402
import train_tpgt_task11 as tpgt  # noqa: E402
from test_meshed_rpe_vectorized import random_grid  # noqa: E402
from test_multitask import consistent_split  # noqa: E402


def grid118() -> tuple[list, list[int]]:
    branches, _ybus = random_grid(11, 118, outages=0, isolated=0)
    branches = [row for row in branches if row[2] > 0 or row[3] > 0]
    types = [0] * 118
    for bus in range(1, 118, 3):
        types[bus] = 1
    types[0] = 2
    return branches, types


def test_write_bus_maps_from_loaded_model(tmp_path):
    branches, types = grid118()
    generator_out = list(types)
    generator_out[4] = 0
    val, _networks = consistent_split([(branches, types)], [6], seed=1)
    n1, _n1_networks = consistent_split(
        [(branches, types), (branches[:117] + branches[118:], types), (branches, generator_out)],
        [5, 5, 5],
        seed=2,
    )
    torch.manual_seed(0)
    model = tpgt.TPGT()
    written = plot_bus_error_map.write_bus_maps(
        tpgt.bus_map_api(), model, val, n1, torch.device("cpu"), tmp_path / "epoch_3",
        epoch=3, checkpoint_name="epoch_3.pt", val_cases=4, batch=8,
    )
    names = {path.name for path in written}
    assert {"val_N_mean_pbl.png", "val_N_dP_dQ.png", "test_N1_gen_outage_worst.png",
            "test_N1_line_outage_worst.png", "test_N1_line_outage_median.png"} == names
    summary = (tmp_path / "epoch_3" / "summary.txt").read_text(encoding="utf-8")
    assert "validation N: 4 cases" in summary
    data = dict(np.load(tmp_path / "epoch_3" / "per_bus_pbl.npz", allow_pickle=False))
    assert data["topo_pbl"].shape == (3, 118)


def test_checkpoint_bus_map_failure_is_not_fatal(tmp_path, capsys):
    split, _network = __import__("test_diagnostics").tiny_consistent_split(cases=3)
    tpgt.write_checkpoint_bus_maps(
        tpgt.TPGT(), split, split, torch.device("cpu"), tmp_path / "maps", 6, tmp_path / "epoch_6.pt"
    )
    assert "Bus error maps skipped for epoch 6" in capsys.readouterr().out
