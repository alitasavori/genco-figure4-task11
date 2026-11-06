"""RPE channel dropout, 4-stage training step, and versioned run folders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_tpgt_task11 as tpgt  # noqa: E402
from test_diagnostics import tiny_consistent_split  # noqa: E402


def test_channel_dropout_is_identity_outside_training():
    rpe = torch.rand(6, 6, tpgt.RPE_DIM) + 0.1
    assert tpgt.rpe_channel_dropout(rpe, 0.9, training=False) is rpe
    assert tpgt.rpe_channel_dropout(rpe, 0.0, training=True) is rpe


def test_channel_dropout_zeroes_whole_channels_without_rescaling():
    torch.manual_seed(0)
    rpe = torch.rand(6, 6, tpgt.RPE_DIM) + 0.1
    dropped_channels = 0
    draws = 400
    for _ in range(draws):
        out = tpgt.rpe_channel_dropout(rpe, 0.5, training=True)
        for channel in range(tpgt.RPE_DIM):
            plane = out[..., channel]
            if bool((plane == 0).all()):
                dropped_channels += 1
            else:
                assert torch.equal(plane, rpe[..., channel])
    rate = dropped_channels / (draws * tpgt.RPE_DIM)
    assert 0.45 < rate < 0.55


def test_model_has_four_independent_stages():
    model = tpgt.TPGT()
    assert tpgt.DEPTH == 4
    assert len(model.stages) == 4
    pointers = {stage.w_q.data_ptr() for stage in model.stages}
    assert len(pointers) == 4


def test_train_batch_runs_four_stages_with_dropout_only_in_training(monkeypatch):
    tpgt.seed_all(0)
    split, network = tiny_consistent_split(cases=6)
    model = tpgt.TPGT()
    optimizer = torch.optim.Adam(model.parameters(), lr=tpgt.LEARNING_RATE)
    seen: list[torch.Tensor] = []
    model.pair_encoder.register_forward_hook(lambda _module, inputs, _output: seen.append(inputs[0].detach().clone()))
    monkeypatch.setattr(tpgt, "RPE_CHANNEL_DROPOUT", 1.0)

    model.train()
    before = [param.detach().clone() for param in model.parameters()]
    breakdown = tpgt.train_batch(model, optimizer, None, split, network, np.arange(6), torch.device("cpu"))
    assert len(breakdown.physics_stages) == 4
    assert torch.isfinite(breakdown.total)
    assert bool((seen[-1] == 0).all()), "training step should see dropped RPE channels"
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))

    tpgt.evaluate_pbl(model, split, [network], torch.device("cpu"), batch_size=4)
    assert torch.equal(seen[-1], network.rpe), "evaluation must use the full RPE"


def test_train_batch_default_dropout_step():
    tpgt.seed_all(1)
    split, network = tiny_consistent_split(cases=4)
    model = tpgt.TPGT()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=tpgt.LEARNING_RATE)
    breakdown = tpgt.train_batch(model, optimizer, None, split, network, np.arange(4), torch.device("cpu"))
    assert len(breakdown.physics_stages) == tpgt.DEPTH
    assert torch.isfinite(breakdown.total)


def test_run_folders_are_versioned(tmp_path, monkeypatch):
    monkeypatch.setattr(tpgt, "in_colab", lambda: False)
    (tmp_path / "Foundation Model").mkdir()
    checkpoints, results = tpgt.drive_output_dirs(tmp_path)
    assert tpgt.RUN_DIR_NAME == f"tpgt_task11_{tpgt.RUN_TAG}" != "tpgt_task11"
    assert checkpoints == tmp_path / "Foundation Model" / tpgt.RUN_DIR_NAME
    assert results == tmp_path / "Foundation Model" / "results" / tpgt.RUN_DIR_NAME
    assert tpgt.COLAB_LOCAL_CACHE.name == f"{tpgt.RUN_DIR_NAME}_cache"
    assert tpgt.CHECKPOINT_VERSION != "meshed-rpe-10-v6-capacity"
