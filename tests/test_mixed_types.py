"""Per-case bus types (generator outages) in one forward pass; device cache; Task 1.1 identity to dba61c2."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_tpgt_task11 as tpgt  # noqa: E402
from test_multitask import consistent_split, float64_split  # noqa: E402
from test_tpgt_attention import random_stage  # noqa: E402

BASELINE_COMMIT = "dba61c2"
CPU = torch.device("cpu")
MESH = [(0, 1, 0.02, 0.06), (1, 2, 0.03, 0.08), (2, 3, 0.04, 0.1), (3, 4, 0.02, 0.05), (0, 4, 0.03, 0.07), (1, 3, 0.05, 0.12)]
INTACT_TYPES = [0, 1, 0, 1, 2]


def outage_topologies() -> list[tuple[list[tuple[int, int, float, float]], list[int]]]:
    """Intact 5-bus mesh, a line outage (same bus types), and two generator outages (PV -> PQ at bus 1 or 3)."""
    return [
        (MESH, INTACT_TYPES),
        (MESH[:-1], INTACT_TYPES),
        (MESH, [0, 0, 0, 1, 2]),
        (MESH, [0, 1, 0, 0, 2]),
    ]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """``train_tpgt_task11.py`` of dba61c2 as a module (skipped when git or the commit is unavailable)."""
    try:
        source = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:train_tpgt_task11.py"], cwd=ROOT, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git show {BASELINE_COMMIT} unavailable")
    path = tmp_path_factory.mktemp("baseline") / f"tpgt_{BASELINE_COMMIT}.py"
    path.write_bytes(source)
    name = f"tpgt_{BASELINE_COMMIT}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def moved64(networks: list[tpgt.Network]) -> list[tpgt.Network]:
    """``move_to`` copies (precomputed layouts) that keep the float64 RPE."""
    moved = [tpgt.move_to(network, CPU) for network in networks]
    for copy, network in zip(moved, networks):
        copy.rpe = network.rpe
    return moved


def mixed_groups(split: tpgt.Split, networks, chosen: np.ndarray, order=(2, 0, 3, 1)):
    owners = split.network_index.numpy()
    return [(networks[net], chosen[owners[chosen] == net]) for net in order]


def per_network_reference(model, split, groups, device=CPU) -> tuple[list[torch.Tensor], float]:
    """Gradient and loss of the per-topology accumulation (one exact single-network forward per topology)."""
    n = sum(len(index) for _network, index in groups)
    model.zero_grad()
    total = 0.0
    for network, index in groups:
        breakdown, _score = tpgt.run_batch(model, split, network, index, device)
        (breakdown.total * len(index) / n).backward()
        total += float(breakdown.total.detach()) * len(index) / n
    return [param.grad.clone() for param in model.parameters()], total


def test_mixed_forward_matches_per_network_forwards():
    split, networks = float64_split(*consistent_split(outage_topologies(), [3, 2, 2, 2], seed=4))
    moved = moved64(networks)
    torch.manual_seed(0)
    model = tpgt.TPGT().double().eval()
    index = np.array([4, 0, 7, 2, 5, 1, 8, 6, 3])
    owner = split.network_index.numpy()[index]
    batch = tpgt.PassBatch(moved, index, owner)
    assert batch.kind == "mixed"
    with torch.no_grad():
        pair = batch.encode(model)
        layout = batch.member_layout.take(batch.select)
        prediction, _target, residuals = tpgt.forward_cases(
            model, split, moved[0], index, CPU, pair[batch.select], batch.ybus[batch.select], layout
        )
        raw_prediction, raw_residuals = model(
            tpgt.pack_case(
                split.power_p[index], split.power_q[index], split.voltage[index], split.angle[index], layout.types
            )[0],
            layout.types,
            None,
            torch.stack([network.pair_type_masks for network in moved])[batch.select],
            batch.ybus[batch.select],
            pair=pair[batch.select],
        )
        assert torch.equal(raw_prediction, prediction)
        for net in range(4):
            rows = np.flatnonzero(owner == net)
            alone, _target, alone_residuals = tpgt.forward_cases(
                model, split, moved[net], index[rows], CPU, model.pair_encoder(moved[net].rpe)
            )
            assert torch.allclose(prediction[rows], alone, rtol=0, atol=1e-12)
            for stage, stage_alone in zip(residuals, alone_residuals):
                assert torch.allclose(stage[rows], stage_alone, rtol=0, atol=1e-12)
            assert torch.allclose(raw_residuals[-1][rows], alone_residuals[-1], rtol=0, atol=1e-12)


def dense_messages(stage, hidden, pair, pair_type_masks):
    """Dense masked attention (``test_tpgt_attention.dense_reference_messages``) with per-case masks and pairs."""
    batch, buses, _ = hidden.shape
    mask_view = pair_type_masks.view(batch, 9, 1, buses, buses)
    query = torch.einsum("bnd,phad->bphna", hidden, stage.w_q)
    key = torch.einsum("bnd,phad->bphna", hidden, stage.w_k)
    mixed = tpgt.rho_nonlinearity(query.unsqueeze(-2) + key.unsqueeze(-3))
    projected = torch.einsum("bnme,phae->bphnma", pair, stage.w_ew)
    bias = torch.einsum("bnme,phae->bphnma", pair, stage.w_eb)
    interaction = torch.relu(mixed * projected + bias)
    all_scores = (interaction * stage.w_a.unsqueeze(0).unsqueeze(-2)).sum(dim=-1)
    scores = all_scores.masked_fill(~mask_view, float("-inf")).amax(dim=1)
    value = torch.einsum("bnd,phad->bphna", hidden, stage.w_v)
    content = value.unsqueeze(3) + torch.einsum("bphnma,phxa->bphnmx", interaction, stage.w_ev)
    values = (content * mask_view.unsqueeze(-1)).sum(dim=1)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    row_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(row_sum > 0, weights / row_sum.clamp(min=1e-12), weights)
    mixed_heads = (weights.unsqueeze(-1) * values).sum(dim=3)
    return torch.einsum("bhna,hda->bnd", mixed_heads, stage.w_o)


def test_per_case_block_messages_match_dense_masked_attention():
    torch.manual_seed(5)
    buses = 9
    types = torch.tensor(
        [
            [0, 1, 2, 0, 1, 0, 0, 1, 0],
            [0, 0, 2, 0, 1, 0, 0, 1, 0],
            [0, 1, 2, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 0, 1, 0, 0, 1, 0],
        ]
    )
    component = torch.tensor([[0] * 9, [0] * 9, [0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1]])
    masks = torch.stack([tpgt.type_pair_masks(t, c) for t, c in zip(types, component)])
    stage = random_stage().double()
    hidden = torch.randn(4, buses, tpgt.D, dtype=torch.float64)
    pair = torch.randn(4, buses, buses, tpgt.D_E, dtype=torch.float64)
    layout = tpgt.per_case_layout(types, masks)
    grads = []
    outputs = []
    for compute in (lambda: dense_messages(stage, hidden, pair, masks), lambda: stage._messages(hidden, pair, layout.blocks)):
        stage.zero_grad()
        out = compute()
        out.square().sum().backward()
        outputs.append(out.detach())
        grads.append({name: param.grad.clone() for name, param in stage.named_parameters() if param.grad is not None})
    assert torch.allclose(outputs[1], outputs[0], rtol=0, atol=1e-12), (outputs[1] - outputs[0]).abs().max()
    for name in ("w_q", "w_k", "w_v", "w_ew", "w_eb", "w_a", "w_ev", "w_o"):
        assert torch.allclose(grads[1][name], grads[0][name], rtol=1e-9, atol=1e-9), name
    for case in range(4):
        alone = stage._messages(hidden[case : case + 1], pair[case], tpgt.type_pair_blocks(types[case], masks[case]))
        assert torch.allclose(outputs[1][case : case + 1], alone.detach(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("split_dominant", [False, True])
@pytest.mark.parametrize("micro", [32, 4])
def test_mixed_step_gradient_equals_per_network_accumulation(monkeypatch, micro, split_dominant):
    monkeypatch.setattr(tpgt, "RPE_CHANNEL_DROPOUT", 0.0)
    monkeypatch.setattr(tpgt, "MICRO_BATCH_SIZE", micro)
    monkeypatch.setattr(tpgt, "SPLIT_DOMINANT_PATTERN", split_dominant)
    split, networks = float64_split(*consistent_split(outage_topologies(), [5, 4, 3, 3], seed=7))
    moved = moved64(networks)
    torch.manual_seed(0)
    model = tpgt.TPGT().double().train()
    chosen = np.array([0, 7, 11, 1, 6, 2, 10, 8, 5, 3, 13, 14, 12])
    groups = mixed_groups(split, moved, chosen)
    reference, total = per_network_reference(model, split, groups)
    forwards: list[int] = []
    model.register_forward_hook(lambda _module, inputs, _output: forwards.append(inputs[0].shape[0]))
    timer = tpgt.StepTimer(CPU)
    logged = tpgt.train_step(model, torch.optim.SGD(model.parameters(), lr=0.0), None, split, groups, CPU, timer=timer)
    timer.end_step(len(groups), 0.0)
    passes = 2 if split_dominant else 1
    assert sum(forwards) == len(chosen)
    assert len(forwards) == sum(-(-size // micro) for size in ([8, 5] if split_dominant else [13]))
    assert f"passes {len(forwards):.1f} topo 4.0" in timer.summary()
    assert passes == len(tpgt.step_passes(groups))
    assert float(logged.total) == pytest.approx(total, rel=1e-6)
    for expected, param in zip(reference, model.parameters()):
        assert torch.allclose(param.grad, expected, rtol=1e-10, atol=1e-12)


def test_mixed_step_matches_dba61c2_grouped_step(monkeypatch, baseline):
    """Production dtype (float32) with RPE dropout: same gradients and loss as the dba61c2 grouped step."""
    monkeypatch.setattr(tpgt, "RPE_CHANNEL_DROPOUT", 0.5)
    monkeypatch.setattr(baseline, "RPE_CHANNEL_DROPOUT", 0.5)
    split, networks = consistent_split(outage_topologies(), [5, 4, 3, 3], seed=9)
    chosen = np.array([0, 7, 11, 1, 6, 2, 10, 8, 5, 3, 13, 14, 12])
    results = []
    for module in (baseline, tpgt):
        moved = [module.move_to(network, CPU) for network in networks]
        torch.manual_seed(1)
        model = module.TPGT().train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        torch.manual_seed(2)
        logged = module.train_step(model, optimizer, None, split, mixed_groups(split, moved, chosen), CPU)
        results.append((float(logged.total), [param.grad.clone() for param in model.parameters()]))
    (old_total, old_grads), (new_total, new_grads) = results
    assert new_total == pytest.approx(old_total, rel=1e-5)
    # float32 round-off only (rho's sqrt amplifies it in w_q / w_k; the float64 test above is exact).
    for old, new in zip(old_grads, new_grads):
        assert float(torch.linalg.vector_norm(new - old)) <= 1e-3 * float(torch.linalg.vector_norm(old)) + 1e-7


def test_mixed_evaluate_matches_per_network_evaluation():
    split, networks = float64_split(*consistent_split(outage_topologies(), [5, 4, 3, 3], seed=5))
    split.network_index = torch.tensor([1, 0, 2, 3, 0, 2, 1, 0, 3, 1, 0, 2, 3, 1, 0])
    moved = moved64(networks)
    torch.manual_seed(0)
    model = tpgt.TPGT().double().eval()
    reference: dict[int, list[float]] = {}
    with torch.no_grad():
        for net, index in tpgt.iterate_indexes(split, 0, 0, shuffle=False, batch_size=4):
            _breakdown, score = tpgt.run_batch(model, split, moved[net], index, CPU)
            bucket = reference.setdefault(net, [0.0, 0.0])
            bucket[0] += float(score) * len(index)
            bucket[1] += len(index)
    forwards: list[int] = []
    model.register_forward_hook(lambda _module, inputs, _output: forwards.append(inputs[0].shape[0]))
    per_network: dict[int, list[float]] = {}
    score = tpgt.evaluate_pbl(model, split, moved, CPU, batch_size=8, per_network_out=per_network)
    assert forwards == [8, 7]
    assert score == pytest.approx(sum(total for total, _count in reference.values()) / split.count, rel=1e-12)
    assert sorted(per_network) == sorted(reference)
    for net, (total, count) in reference.items():
        assert per_network[net][1] == count
        assert per_network[net][0] == pytest.approx(total, rel=1e-12)
    assert all(network.pair_embed is None for network in moved)


def test_device_cache_all_resident_and_lru_give_identical_results(monkeypatch):
    split, networks = consistent_split(outage_topologies(), [6, 5, 4, 4], seed=11)
    resident = tpgt.DeviceNetworks(networks, CPU)
    assert resident.all_resident and resident.describe().startswith("all 4 networks resident")
    small = tpgt.DeviceNetworks(networks, CPU, budget_bytes=2 * tpgt.network_device_bytes(5) + 1)
    assert small.capacity == 2 and small.describe().startswith("LRU capacity 2 of 4 networks")
    outcomes = []
    for cache in (resident, tpgt.DeviceNetworks(networks, CPU, capacity=1)):
        torch.manual_seed(0)
        model = tpgt.TPGT()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        for epoch in range(2):
            model.train()
            for step in tpgt.iterate_steps(split, seed=0, epoch=epoch, batch_size=8):
                logged = tpgt.train_step(model, optimizer, None, split, [(cache[net], index) for net, index in step], CPU)
                losses.append(float(logged.total))
        per_network: dict[int, list[float]] = {}
        score = tpgt.evaluate_pbl(model, split, cache, CPU, batch_size=6, per_network_out=per_network)
        outcomes.append((losses, score, per_network, model.state_dict()))
    assert len(resident.resident()) == 4
    (losses_a, score_a, per_a, state_a), (losses_b, score_b, per_b, state_b) = outcomes
    assert losses_a == losses_b and score_a == score_b and per_a == per_b
    for key, value in state_a.items():
        assert torch.equal(value, state_b[key]), key


def test_move_to_precomputes_type_layout():
    _split, networks = consistent_split(outage_topologies(), [1, 1, 1, 1])
    moved = tpgt.move_to(networks[2], CPU)
    layout = moved.layout
    assert layout is not None and not layout.per_case
    assert layout.present == (True, True, True)
    assert layout.index[1].tolist() == [3] and layout.index[0].tolist() == [0, 1, 2]
    assert layout.angle_index.tolist() == [0, 1, 2, 3]
    assert len(layout.blocks) == 9
    covered = torch.zeros(5, 5, dtype=torch.long)
    for block in layout.blocks:
        covered[block.receivers.view(-1, 1), block.senders.view(1, -1)] += block.mask.long()
    assert torch.equal(covered, torch.ones(5, 5, dtype=torch.long))


def test_step_timer_line_format():
    timer = tpgt.StepTimer(CPU)
    for _ in range(2):
        with timer.phase("fwd"):
            pass
        timer.add_pass()
        timer.end_step(18, 0.5)
    line = timer.summary()
    assert re.fullmatch(
        r"t/step 0\.500s \(move [\d.]+ pair [\d.]+ fwd [\d.]+ bwd [\d.]+ opt [\d.]+\) passes 1\.0 topo 18\.0", line
    ), line
    assert timer.summary().startswith("t/step 0.000s")
    assert timer.summary("epoch").startswith("t/step 0.500s")


def test_task11_two_epochs_bitwise_identical_to_dba61c2(baseline):
    """Single-topology training (Task 1.1 path, RPE dropout on): weights, Adam state and losses bit for bit."""
    split, (network,) = consistent_split([(MESH, INTACT_TYPES)], [70], seed=3)
    runs = []
    for module, timer in ((baseline, None), (tpgt, tpgt.StepTimer(CPU))):
        moved = module.move_to(network, CPU)
        torch.manual_seed(0)
        model = module.TPGT()
        optimizer = torch.optim.Adam(model.parameters(), lr=module.LEARNING_RATE)
        losses = []
        for epoch in range(2):
            model.train()
            for step in module.iterate_steps(split, 0, epoch):
                groups = [(moved, index) for _net, index in step]
                if timer is None:
                    logged = module.train_step(model, optimizer, None, split, groups, CPU)
                else:
                    logged = module.train_step(model, optimizer, None, split, groups, CPU, timer=timer)
                losses.append(
                    (float(logged.total), float(logged.state_mse), float(logged.physics_sum))
                    + tuple(float(stage) for stage in logged.physics_stages)
                )
        score = module.evaluate_pbl(model, split, [moved], CPU, batch_size=16)
        runs.append((losses, score, model.state_dict(), optimizer.state_dict()))
    (old_losses, old_score, old_state, old_opt), (new_losses, new_score, new_state, new_opt) = runs
    assert len(old_losses) == 6
    assert new_losses == old_losses
    assert new_score == old_score
    for key, value in old_state.items():
        assert torch.equal(new_state[key], value), key
    for param_id, state in old_opt["state"].items():
        for key, value in state.items():
            assert torch.equal(new_opt["state"][param_id][key], value), (param_id, key)
