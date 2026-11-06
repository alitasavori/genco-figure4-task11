"""PFΔ multi-task training: task registry and paths, folder discovery, merging, multi-topology steps."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_tpgt_task11 as tpgt  # noqa: E402
from plot_task1_1 import PUBLISHED, PUBLISHED_SOURCES, PUBLISHED_TASK11, plot_tpgt_comparison  # noqa: E402
from test_diagnostics import tiny_consistent_split  # noqa: E402


def consistent_split(
    topologies: list[tuple[list[tuple[int, int, float, float]], list[int]]],
    cases_per_topology: list[int],
    *,
    seed: int = 0,
    buses: int | None = None,
) -> tuple[tpgt.Split, list[tpgt.Network]]:
    """Split whose labels satisfy AC power flow exactly; one Network per (branches, bus types) entry."""
    rng = np.random.default_rng(seed)
    buses = buses or len(topologies[0][1])
    networks = []
    blocks = {"p": [], "q": [], "v": [], "a": [], "net": []}
    for network_id, ((branches, types), cases) in enumerate(zip(topologies, cases_per_topology)):
        ybus = np.zeros((buses, buses), dtype=np.complex128)
        for origin, destination, resistance, reactance in branches:
            admittance = 1.0 / complex(resistance, reactance)
            ybus[origin, origin] += admittance
            ybus[destination, destination] += admittance
            ybus[origin, destination] -= admittance
            ybus[destination, origin] -= admittance
        features, component = tpgt.meshed_rpe(buses, branches, ybus.astype(np.complex64))
        type_tensor = torch.tensor(types, dtype=torch.long)
        networks.append(
            tpgt.Network(
                ybus=torch.from_numpy(ybus.astype(np.complex64)),
                rpe=torch.from_numpy(features),
                component=torch.from_numpy(component.astype(np.int64)),
                types=type_tensor,
                branches=tuple(branches),
            )
        )
        voltage = 1.0 + 0.02 * rng.standard_normal((cases, buses))
        angle = 0.03 * rng.standard_normal((cases, buses))
        angle[:, np.asarray(types) == 2] = 0.0
        solved = voltage * np.exp(1j * angle)
        injection = solved * np.conj(solved @ ybus.T)
        blocks["p"].append(injection.real)
        blocks["q"].append(injection.imag)
        blocks["v"].append(voltage)
        blocks["a"].append(angle)
        blocks["net"].append(np.full(cases, network_id))
    split = tpgt.Split(
        name="synthetic",
        count=int(sum(cases_per_topology)),
        power_p=torch.from_numpy(np.concatenate(blocks["p"]).astype(np.float32)),
        power_q=torch.from_numpy(np.concatenate(blocks["q"]).astype(np.float32)),
        voltage=torch.from_numpy(np.concatenate(blocks["v"]).astype(np.float32)),
        angle=torch.from_numpy(np.concatenate(blocks["a"]).astype(np.float32)),
        network_index=torch.from_numpy(np.concatenate(blocks["net"]).astype(np.int64)),
        networks=networks,
    )
    return split, networks


def three_bus_topologies() -> list[tuple[list[tuple[int, int, float, float]], list[int]]]:
    """Intact 3-bus mesh, a line outage, and a generator outage (bus 1 PV -> PQ)."""
    mesh = [(0, 1, 0.02, 0.06), (1, 2, 0.03, 0.08), (0, 2, 0.04, 0.1)]
    return [(mesh, [0, 1, 2]), (mesh[1:], [0, 1, 2]), (mesh, [0, 0, 2])]


def test_task11_names_and_paths_are_unchanged(tmp_path, monkeypatch):
    spec = tpgt.task_spec("1.1")
    assert spec is tpgt.TASKS["1.1"] and spec.is_task11
    assert spec.run_dir_name == tpgt.RUN_DIR_NAME == "tpgt_task11_v7"
    assert spec.dataset_env == "GENCO_PFDELTA_TASK11"
    assert spec.dataset_repo == "gridfm/pfdelta_task1.1"
    assert spec.folders == (("n", "feasible"),)
    assert tpgt.dataset_relative(spec) == tpgt.DATASET_RELATIVE == Path("Foundation Model/vendor/pfdelta_task1.1")
    assert tpgt.CHECKPOINT_VERSION == "meshed-rpe-5-d4-v7"
    assert tpgt.CACHE_VERSION == "tpgt-cache-4"
    monkeypatch.setattr(tpgt, "in_colab", lambda: False)
    (tmp_path / "Foundation Model").mkdir()
    default = tpgt.drive_output_dirs(tmp_path)
    explicit = tpgt.drive_output_dirs(tmp_path, spec)
    assert default == explicit == (
        tmp_path / "Foundation Model" / "tpgt_task11_v7",
        tmp_path / "Foundation Model" / "results" / "tpgt_task11_v7",
    )
    checkpoints = default[0]
    assert tpgt.split_cache_dirs(checkpoints, spec) == (checkpoints / "cache", checkpoints / "cache")
    monkeypatch.setattr(tpgt, "in_colab", lambda: True)
    assert tpgt.split_cache_dirs(checkpoints, spec) == (Path("/content/tpgt_task11_v7_cache"), checkpoints / "cache")
    assert tpgt.split_cache_dirs(checkpoints) == tpgt.split_cache_dirs(checkpoints, spec)
    assert tpgt.comparison_figure_path(tmp_path, spec).name == "figure4_task1.1_tpgt.svg"
    assert spec.file_tag == "1_1"


def test_task11_is_the_default_task(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_tpgt_task11.py"])
    monkeypatch.delenv("GENCO_TPGT_TASK", raising=False)
    assert tpgt.parse_task() is tpgt.TASKS["1.1"]
    monkeypatch.setenv("GENCO_TPGT_TASK", "4.2")
    assert tpgt.parse_task().task_id == "4.2"
    monkeypatch.setattr(sys, "argv", ["train_tpgt_task11.py", "--task", "2.1"])
    assert tpgt.parse_task().task_id == "1.3"
    monkeypatch.setattr(sys, "argv", ["train_tpgt_task11.py", "--task=1.2"])
    assert tpgt.parse_task().task_id == "1.2"
    with pytest.raises(ValueError):
        tpgt.task_spec("3.1")


def test_other_tasks_get_their_own_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(tpgt, "in_colab", lambda: False)
    (tmp_path / "Foundation Model").mkdir()
    alias = tpgt.task_spec("2.1")
    assert alias is tpgt.task_spec("1.3")
    assert alias.label == "1.3/2.1"
    assert alias.dataset_repo == "gridfm/pfdelta_task1.3"
    assert alias.dataset_env == "GENCO_PFDELTA_TASK13"
    expected = {
        "1.2": "tpgt_task1.2_v7",
        "1.3": "tpgt_task1.3_v7",
        "2.3": "tpgt_task2.3_v7",
        "4.1": "tpgt_task4.1_v7",
        "4.2": "tpgt_task4.2_v7",
        "4.3": "tpgt_task4.3_v7",
    }
    for task, folder in expected.items():
        spec = tpgt.task_spec(task)
        checkpoints, results = tpgt.drive_output_dirs(tmp_path, spec)
        assert checkpoints == tmp_path / "Foundation Model" / folder
        assert results == checkpoints / "results"
        assert tpgt.dataset_relative(spec) == Path("Foundation Model") / "vendor" / f"pfdelta_task{task}"
        monkeypatch.setattr(tpgt, "in_colab", lambda: True)
        assert tpgt.split_cache_dirs(checkpoints, spec)[0] == Path(f"/content/{folder}_cache")
        monkeypatch.setattr(tpgt, "in_colab", lambda: False)
    assert len({tpgt.task_spec(task).run_dir_name for task in [*expected, "1.1"]}) == 7


def make_tree(root: Path, layout: dict[str, list[tuple[str, str]]]) -> None:
    for split, folders in layout.items():
        for topology, condition in folders:
            raw = tpgt.raw_dir(root, split, topology, condition)
            raw.mkdir(parents=True)
            for name in tpgt.PARQUET_FILES:
                (raw / name).write_bytes(b"x")


def test_folder_discovery_and_missing_files(tmp_path):
    spec = tpgt.task_spec("4.3")
    test = [(topology, condition) for _label, topology, condition in tpgt.TEST_PARTS]
    make_tree(tmp_path, {"train": list(spec.folders), "valid": list(spec.folders), "test": test})
    assert tpgt.task_folders(tmp_path, "train") == [
        ("n", "nose"),
        ("n", "around_nose"),
        ("n-1", "nose"),
        ("n-1", "around_nose"),
        ("n-2", "nose"),
        ("n-2", "around_nose"),
    ]
    assert tpgt.task_folders(tmp_path, "test") == [
        ("n", "feasible"),
        ("n", "nose"),
        ("n-1", "feasible"),
        ("n-1", "nose"),
        ("n-2", "feasible"),
        ("n-2", "nose"),
    ]
    assert tpgt.dataset_missing_files(tmp_path, spec) == []
    (tpgt.raw_dir(tmp_path, "valid", "n-2", "nose") / "y_bus_data.parquet").unlink()
    assert tpgt.dataset_missing_files(tmp_path, spec) == [
        tpgt.raw_dir(tmp_path, "valid", "n-2", "nose") / "y_bus_data.parquet"
    ]
    missing = tpgt.dataset_missing_files(tmp_path, tpgt.task_spec("4.2"))
    assert tpgt.raw_dir(tmp_path, "train", "n", "feasible") / "bus_data.parquet" in missing


def test_dataset_lookup_uses_the_task_env_override(tmp_path, monkeypatch):
    spec = tpgt.task_spec("1.2")
    (tmp_path / "data12" / "case118").mkdir(parents=True)
    monkeypatch.setenv("GENCO_PFDELTA_TASK12", str(tmp_path / "data12"))
    monkeypatch.setenv("GENCO_PFDELTA_TASK11", str(tmp_path / "data11"))
    assert tpgt.drive_dataset(None, spec) == tmp_path / "data12"
    assert tpgt.dataset_request(tpgt.task_spec("1.1")) == tmp_path / "data11"


def test_folder_mix_text_reports_shares():
    text = tpgt.folder_mix_text([(("n", "feasible"), 50), (("n-1", "nose"), 10), (("n-2", "around_nose"), 40)])
    assert "n/feasible 50 (50.0%)" in text
    assert "feasible 50.0% nose 10.0% around_nose 40.0%" in text
    assert "N 50.0% N-1 10.0% N-2 40.0%" in text


def scaled_network(scale: float, types: tuple[int, ...] = (0, 1, 2)) -> tpgt.Network:
    _split, base = tiny_consistent_split(cases=1)
    return tpgt.Network(
        ybus=base.ybus * scale,
        rpe=base.rpe.clone(),
        component=base.component.clone(),
        types=torch.tensor(types, dtype=torch.long),
    )


def split_of(networks: list[tpgt.Network], network_index: list[int]) -> tpgt.Split:
    count = len(network_index)
    return tpgt.Split(
        name="part",
        count=count,
        power_p=torch.arange(count * 3, dtype=torch.float32).view(count, 3),
        power_q=torch.zeros(count, 3),
        voltage=torch.ones(count, 3),
        angle=torch.zeros(count, 3),
        network_index=torch.tensor(network_index, dtype=torch.long),
        networks=networks,
    )


def test_merge_many_reuses_identical_topologies():
    base = [scaled_network(1.0), scaled_network(2.0)]
    other = [scaled_network(2.0), scaled_network(3.0), scaled_network(1.0, types=(0, 0, 2)), scaled_network(1.0)]
    merged = tpgt.merge_many([split_of(base, [0, 1, 1]), split_of(other, [3, 0, 1, 2])], name="train")
    assert merged.name == "train"
    assert merged.count == 7
    assert len(merged.networks) == 4
    assert merged.network_index.tolist() == [0, 1, 1, 0, 1, 2, 3]
    assert merged.networks[0] is base[0] and merged.networks[1] is base[1]
    assert merged.networks[2] is other[1] and merged.networks[3] is other[2]
    assert torch.equal(merged.power_p[3:], split_of(other, [3, 0, 1, 2]).power_p)


def test_merge_splits_keeps_base_networks_and_ids():
    duplicate = [scaled_network(1.0), scaled_network(1.0)]
    merged = tpgt.merge_splits(split_of(duplicate, [0, 1]), split_of([scaled_network(1.0)], [0, 0]))
    assert len(merged.networks) == 2
    assert merged.network_index.tolist() == [0, 1, 0, 0]


def test_merge_many_scales_to_thousands_of_topologies():
    count = 3000
    first = [scaled_network(1.0 + index * 1e-3) for index in range(count)]
    second = [scaled_network(1.0 + index * 1e-3) for index in range(0, 2 * count, 2)]
    started = time.perf_counter()
    merged = tpgt.merge_many([split_of(first, list(range(count))), split_of(second, list(range(count)))])
    elapsed = time.perf_counter() - started
    assert len(merged.networks) == count + count // 2
    expected_second = [index * 2 if index * 2 < count else count + index - count // 2 for index in range(count)]
    assert merged.network_index[count:].tolist() == expected_second
    assert elapsed < 30.0


def test_iterate_steps_reproduces_single_topology_batches():
    split, _network = tiny_consistent_split(cases=75)
    for seed, epoch in ((0, 0), (2, 7)):
        old = [(net, index.tolist()) for net, index in tpgt.iterate_indexes(split, seed, epoch, shuffle=True)]
        new = [groups for groups in tpgt.iterate_steps(split, seed, epoch)]
        assert len(old) == len(new) == 3
        for (old_net, old_index), groups in zip(old, new):
            assert len(groups) == 1
            assert groups[0][0] == old_net == 0
            assert groups[0][1].tolist() == old_index


def test_iterate_steps_mixes_topologies_and_covers_every_case():
    split, _networks = consistent_split(three_bus_topologies(), [40, 25, 15])
    steps = list(tpgt.iterate_steps(split, seed=1, epoch=0))
    seen = np.concatenate([index for groups in steps for _net, index in groups])
    assert sorted(seen.tolist()) == list(range(split.count))
    assert [sum(len(index) for _net, index in groups) for groups in steps] == [32, 32, 16]
    owners = split.network_index.numpy()
    for groups in steps:
        assert len({net for net, _index in groups}) == len(groups)
        for net, index in groups:
            assert set(owners[index].tolist()) == {net}
    assert any(len(groups) > 1 for groups in steps)


def test_iterate_indexes_never_mixes_topologies():
    split, _networks = consistent_split(three_bus_topologies(), [5, 7, 3])
    batches = list(tpgt.iterate_indexes(split, 0, 0, shuffle=False, batch_size=4))
    assert [(net, index.tolist()) for net, index in batches] == [
        (0, [0, 1, 2, 3]),
        (0, [4]),
        (1, [5, 6, 7, 8]),
        (1, [9, 10, 11]),
        (2, [12, 13, 14]),
    ]


def test_multi_topology_step_gradient_is_the_mean_over_all_cases(monkeypatch):
    monkeypatch.setattr(tpgt, "RPE_CHANNEL_DROPOUT", 0.0)
    split, networks = consistent_split(three_bus_topologies(), [6, 4, 2], seed=3)
    for network in networks:
        network.pair_type_masks = tpgt.type_pair_masks(network.types, network.component)
    torch.manual_seed(0)
    model = tpgt.TPGT()
    model.train()
    owners = split.network_index.numpy()
    chosen = np.array([0, 7, 11, 1, 6, 2, 10, 8])
    groups = [(net, chosen[owners[chosen] == net]) for net in (0, 1, 2)]
    reference = []
    model.zero_grad()
    total = 0.0
    for net, index in groups:
        breakdown, _score = tpgt.run_batch(model, split, networks[net], index, torch.device("cpu"))
        (breakdown.total * len(index) / len(chosen)).backward()
        total += float(breakdown.total.detach()) * len(index) / len(chosen)
    reference = [param.grad.clone() for param in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    logged = tpgt.train_step(
        model, optimizer, None, split, [(networks[net], index) for net, index in groups], torch.device("cpu")
    )
    assert float(logged.total) == pytest.approx(total, rel=1e-5)
    for expected, param in zip(reference, model.parameters()):
        assert torch.allclose(param.grad, expected, rtol=1e-4, atol=1e-5)


def test_train_step_micro_batches_within_one_topology(monkeypatch):
    monkeypatch.setattr(tpgt, "RPE_CHANNEL_DROPOUT", 0.0)
    split, network = tiny_consistent_split(cases=6)
    torch.manual_seed(0)
    model = tpgt.TPGT()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    whole = tpgt.train_step(model, optimizer, None, split, [(network, np.arange(6))], torch.device("cpu"))
    grads = [param.grad.clone() for param in model.parameters()]
    monkeypatch.setattr(tpgt, "MICRO_BATCH_SIZE", 4)
    chunked = tpgt.train_step(model, optimizer, None, split, [(network, np.arange(6))], torch.device("cpu"))
    assert float(chunked.total) == pytest.approx(float(whole.total), rel=1e-5)
    for expected, param in zip(grads, model.parameters()):
        assert torch.allclose(param.grad, expected, rtol=1e-4, atol=1e-5)


def test_one_dropout_draw_per_step_for_every_topology(monkeypatch):
    monkeypatch.setattr(tpgt, "RPE_CHANNEL_DROPOUT", 0.5)
    split, cpu_networks = consistent_split(three_bus_topologies(), [3, 3, 3])
    networks = [tpgt.move_to(network, torch.device("cpu")) for network in cpu_networks]
    model = tpgt.TPGT()
    model.train()
    seen: list[torch.Tensor] = []
    model.pair_encoder.register_forward_hook(lambda _module, inputs, _output: seen.append(inputs[0].detach().clone()))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    for _ in range(8):
        seen.clear()
        groups = [(networks[net], np.arange(3) + 3 * net) for net in range(3)]
        tpgt.train_step(model, optimizer, None, split, groups, torch.device("cpu"))
        dropped = [tuple(bool((rpe[..., c] == 0).all()) for c in range(tpgt.RPE_DIM)) for rpe in seen]
        assert len(set(dropped)) == 1


def test_device_networks_move_lazily_and_evict_least_recent():
    _split, networks = consistent_split(three_bus_topologies(), [1, 1, 1])
    networks[1].rpe = networks[1].rpe.to(torch.float16)
    lazy = tpgt.DeviceNetworks(networks, torch.device("cpu"), capacity=2)
    assert len(lazy) == 3 and lazy.resident() == []
    first = lazy[0]
    assert lazy[0] is first
    second = lazy[1]
    assert second.rpe.dtype == torch.float32
    assert second.pair_type_masks is not None and second.pair_type_masks.shape == (9, 3, 3)
    lazy[0]
    lazy[2]
    assert [network is first for network in lazy.resident()] == [True, False]
    assert lazy[1] is not second
    first.pair_embed = torch.zeros(1)
    tpgt.clear_pair_embed_cache(lazy)
    assert all(network.pair_embed is None for network in lazy.resident())


def test_evaluate_pbl_same_with_lazy_and_eager_networks():
    split, networks = consistent_split(three_bus_topologies(), [5, 4, 3], seed=5)
    torch.manual_seed(0)
    model = tpgt.TPGT()
    device = torch.device("cpu")
    eager = [tpgt.move_to(network, device) for network in networks]
    per_eager: dict[int, list[float]] = {}
    per_lazy: dict[int, list[float]] = {}
    expected = tpgt.evaluate_pbl(model, split, eager, device, batch_size=4, per_network_out=per_eager)
    lazy = tpgt.DeviceNetworks(networks, device, capacity=1)
    assert tpgt.evaluate_pbl(model, split, lazy, device, batch_size=4, per_network_out=per_lazy) == expected
    assert per_lazy == per_eager


def test_compact_networks_keeps_dominant_topology_in_float32():
    split, networks = consistent_split(three_bus_topologies(), [2, 5, 1])
    assert tpgt.compact_networks(split, min_networks=4) == 0
    assert tpgt.compact_networks(split, min_networks=3) == 2
    assert networks[1].rpe.dtype == torch.float32
    assert networks[0].rpe.dtype == networks[2].rpe.dtype == torch.float16
    assert tpgt.compact_networks(split, min_networks=3) == 0
    torch.manual_seed(0)
    score = tpgt.evaluate_pbl(tpgt.TPGT(), split, tpgt.DeviceNetworks(networks, torch.device("cpu")), torch.device("cpu"))
    assert np.isfinite(score)


def test_published_numbers_cover_every_task():
    assert PUBLISHED_TASK11 is PUBLISHED["1.1"]
    assert set(PUBLISHED) == {"1.1", "1.2", "1.3", "2.3", "4.1", "4.2", "4.3"}
    assert set(PUBLISHED_SOURCES) == set(PUBLISHED)
    for task, models in PUBLISHED.items():
        assert set(models) == {"GENCO Base", "CANOS-PF", "GNS-S", "PFNet"}, task
        for scores in models.values():
            assert set(scores) == {"N", "N-1", "N-2", "C2I"}
            assert all(mean > 0 and std > 0 for mean, std in scores.values())
    assert PUBLISHED["1.3"]["GENCO Base"]["N"] == (2.7e-3, 2.2e-3)
    assert PUBLISHED["4.3"]["PFNet"]["N-2"] == (1.5, 0.02)
    for spec in tpgt.TASKS.values():
        assert spec.task_id in PUBLISHED


def test_bar_plot_for_a_task_writes_svg_and_png(tmp_path):
    tpgt_scores = {"N": (0.05, 0.01), "N-1": (0.2, 0.05), "N-2": (0.3, 0.05), "C2I": (2.5, 0.4)}
    written = plot_tpgt_comparison("4.2", "4.2", tpgt_scores, tmp_path / "figure4_task4.2_tpgt.svg")
    assert [path.name for path in written] == ["figure4_task4.2_tpgt.svg", "figure4_task4.2_tpgt.png"]
    assert all(path.stat().st_size > 1000 for path in written)
    summary = [
        {"test_category": category, "mean_across_seeds": str(mean), "std_across_seeds": str(std), "n_seeds": "1"}
        for category, (mean, std) in tpgt_scores.items()
    ]
    tpgt.plot_comparison(tmp_path, summary, tpgt.task_spec("2.1"))
    assert (tmp_path / "figure4_task1.3_tpgt.svg").is_file()
    assert (tmp_path / "figure4_task1.3_tpgt.png").is_file()


def test_seed_metrics_and_summary_per_task(tmp_path):
    counts = {"N": 4, "N-1": 4, "N-2": 4, "C2I-N": 2, "C2I-N-1": 2, "C2I-N-2": 2}
    scores = {"N": 0.1, "N-1": 0.2, "N-2": 0.3, "C2I-N": 1.0, "C2I-N-1": 2.0, "C2I-N-2": 3.0}
    spec = tpgt.task_spec("4.1")
    tpgt.record_seed(tmp_path, 0, scores, counts, tmp_path / "epoch_3.pt", 3, spec)
    tpgt.record_seed(tmp_path, 1, scores, counts, tmp_path / "epoch_3.pt", 3, spec)
    summary = tpgt.write_summary(tmp_path, spec)
    assert (tmp_path / "task4_1_tpgt_seed_metrics.csv").is_file()
    assert (tmp_path / "task4_1_tpgt_summary.csv").is_file()
    assert {row["test_category"]: float(row["mean_across_seeds"]) for row in summary}["C2I"] == pytest.approx(2.0)
    assert "4.1" in (tmp_path / "task4_1_tpgt_seed_metrics.csv").read_text(encoding="utf-8")
    tpgt.record_seed(tmp_path, 0, scores, counts, tmp_path / "epoch_3.pt", 3)
    assert (tmp_path / "task1_1_tpgt_seed_metrics.csv").is_file()
