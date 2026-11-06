"""Per-bus power-balance error (PBL) of a trained TPGT checkpoint drawn on the IEEE 118-bus map.

Outputs (PNG, ~150 dpi) in ``--out-dir``:
  val_N_mean_pbl.png                 mean per-bus PBL over validation N cases
  val_N_dP_dQ.png                    mean |dP| and |dQ| per bus
  test_N1_gen_outage_worst.png       worst generator-outage topology (outaged PV bus ringed red)
  test_N1_line_outage_worst.png      worst line-outage topology (tripped branches red dashed)
  test_N1_line_outage_median.png     median line-outage topology
  summary.txt                        printed tables (worst buses, PBL by hop distance from the outage)
  per_bus_pbl.npz                    per-bus arrays; ``--from-npz`` replots without running the model

Example (Colab):
  python plot_bus_error_map.py \
      --checkpoint "/content/drive/MyDrive/Foundation Model/tpgt_task11/seed_0/last.pt" \
      --cache-dir /content/tpgt_task11_cache
Use ``--module`` / ``--module-path`` to import a different copy of the training module
(it must match the checkpoint's architecture).

In-process (training checkpoints): ``compute_from_model`` evaluates an already-loaded model on
already-loaded splits and ``render`` writes the maps; ``write_bus_maps`` does both.
"""

from __future__ import annotations

import argparse
import importlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

# Bus coordinates from pandapower ``networks.case118()`` geodata (bus k at index k-1).
CASE118_XY = (
    (-2.2754, 2.8543), (-2.9368, 2.2122), (-1.8344, 1.7094), (-0.8886, 1.5533), (-0.9633, 0.6947),
    (-0.411, 1.7676), (-1.4836, 2.262), (-0.6793, -0.5813), (0.4922, 0.0395), (1.5086, 0.7538),
    (-1.8085, 1.0783), (-2.4521, 1.1888), (-2.534, 0.2882), (-3.0868, 0.3407), (-2.3297, -0.6795),
    (-1.4955, 0.1152), (-1.0489, -1.0911), (-1.8357, -0.8293), (-2.6758, -1.2563), (-1.2717, -1.4628),
    (0.0735, -2.1573), (0.9345, -3.4895), (0.3017, -3.7998), (-0.2779, -5.1102), (0.9631, -2.9811),
    (-0.1984, -2.3375), (1.9446, -2.228), (2.4541, -1.222), (1.4972, -0.7254), (-1.2374, -2.03),
    (0.3831, -1.2751), (0.9036, -2.4716), (-2.9834, -1.6492), (-3.6197, -2.0173), (-4.3348, -2.208),
    (-4.7429, -1.3946), (-2.9559, -2.7113), (-1.9468, -3.1511), (-3.6415, -3.1156), (-2.8057, -3.766),
    (-3.1525, -4.4159), (-2.2746, -5.0451), (-4.1875, -3.1293), (-3.5915, -4.421), (-2.63, -5.604),
    (-3.1344, -6.547), (-2.3187, -6.7142), (-2.0246, -7.1401), (-1.7242, -6.2584), (-1.1556, -7.0862),
    (-1.2192, -7.6504), (-0.7486, -8.8466), (0.1459, -8.3262), (-0.4574, -6.983), (0.8612, -7.4176),
    (0.3693, -7.6867), (-0.3863, -8.2012), (-0.0965, -8.546), (0.6559, -6.6867), (1.4936, -6.2503),
    (0.793, -5.6494), (0.0835, -5.7404), (0.5977, -5.3868), (-0.6252, -4.7707), (-1.9502, -4.5115),
    (-1.2079, -5.3569), (-0.1248, -5.0526), (-3.1811, -5.4511), (-2.6513, -6.4304), (-1.2851, -6.3036),
    (0.0746, -7.0345), (1.0237, -5.9102), (1.1747, -8.1088), (-1.7822, -7.1334), (-2.57, -7.3039),
    (-3.0946, -8.595), (-3.6187, -7.4081), (-3.6928, -8.1637), (-4.7266, -7.7222), (-4.854, -6.9947),
    (-4.4266, -5.9494), (-4.1989, -8.5699), (-4.1284, -9.8586), (-4.0286, -10.8456), (-4.9873, -10.7009),
    (-4.6399, -11.9116), (-4.4755, -13.0052), (-5.7173, -11.3184), (-6.2787, -10.59), (-7.0445, -11.5144),
    (-7.55, -10.5464), (-7.0121, -9.4334), (-6.4799, -9.6952), (-6.4131, -8.5909), (-5.7195, -8.8463),
    (-5.334, -8.0607), (-5.4897, -6.9548), (-6.0865, -7.3113), (-6.2121, -6.992), (-7.0667, -7.9857),
    (-7.6658, -8.7968), (-7.9822, -9.8881), (-8.2365, -7.2233), (-7.9601, -7.6514), (-9.084, -7.9346),
    (-8.1036, -8.2327), (-9.06, -8.8273), (-10.2223, -8.3102), (-10.4379, -7.4563), (-9.3587, -6.8229),
    (-10.4402, -6.5782), (-10.0489, -5.9289), (0.0211, -1.4982), (2.2631, -2.7869), (3.1644, -2.3854),
    (-4.1529, -4.9349), (-3.5172, 1.7774), (-2.1458, -8.5467),
)
BUS_COUNT = 118
Y_EPS = 1e-6
TYPE_MARKERS = ((0, "o", 22, "PQ"), (1, "s", 30, "PV"), (2, "*", 170, "slack"))


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-dir", type=Path, help="Directory with the split .pt caches.")
    parser.add_argument("--from-npz", type=Path, help="Replot from a saved per_bus_pbl.npz (no model run).")
    parser.add_argument("--val-cache", default="valid_n_feasible.pt")
    parser.add_argument("--n1-cache", default="test_n-1_feasible.pt")
    parser.add_argument("--out-dir", type=Path, default=here / "results" / "bus_error_maps")
    parser.add_argument("--module", default="train_tpgt_task11", help="Training module providing TPGT etc.")
    parser.add_argument("--module-path", action="append", default=[], help="Extra sys.path entry (repeatable).")
    parser.add_argument("--layout", choices=("geodata", "kamada"), default="geodata")
    parser.add_argument("--val-cases", type=int, default=256)
    parser.add_argument("--min-cases", type=int, default=5, help="Min cases for an N-1 topology to be picked.")
    parser.add_argument("--batch", type=int, default=0, help="Eval batch (default 64 on CUDA, 16 on CPU).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_module(name: str, extra_paths: list[str]):
    for path in reversed(extra_paths):
        sys.path.insert(0, str(Path(path).resolve()))
    module = importlib.import_module(name)
    import __main__

    __main__.Split = module.Split  # split caches were pickled from a __main__ run
    __main__.Network = module.Network
    return module


def per_bus_residual(t, model, split, network_id: int, index: np.ndarray, device, batch: int) -> np.ndarray:
    """Final-stage (dP, dQ) in p.u., shape (cases, buses, 2)."""
    import torch

    net = t.move_to(split.networks[network_id], device)
    net.pair_embed = None
    out = []
    with torch.no_grad():
        pair = t.pair_features(model, net)
        for start in range(0, len(index), batch):
            chosen = torch.as_tensor(index[start : start + batch], dtype=torch.long)
            specified, _target = t.pack_case(
                split.power_p[chosen].to(device),
                split.power_q[chosen].to(device),
                split.voltage[chosen].to(device),
                split.angle[chosen].to(device),
                net.types,
            )
            _prediction, residuals = model(specified, net.types, net.rpe, net.pair_type_masks, net.ybus, pair=pair)
            out.append(residuals[-1].float().cpu().numpy())
    return np.concatenate(out, axis=0)


def pbl(residual: np.ndarray) -> np.ndarray:
    return np.sqrt(residual[..., 0] ** 2 + residual[..., 1] ** 2)


def branch_pairs(ybus: np.ndarray) -> list[tuple[int, int]]:
    mag = np.abs(ybus)
    i, j = np.nonzero(np.triu(mag > Y_EPS, k=1))
    return list(zip(i.tolist(), j.tolist()))


def branch_admittance(ybus: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array([ybus[i, j] for i, j in pairs])


def changed_branches(branch_y: np.ndarray, reference_y: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[list, list]:
    """(fully removed pairs, pairs whose admittance changed but stay connected, e.g. one parallel circuit)."""
    removed, weakened = [], []
    for pair, ref, new in zip(pairs, reference_y, branch_y):
        if abs(new - ref) <= 1e-3 * abs(ref):
            continue
        (removed if abs(new) <= Y_EPS else weakened).append(pair)
    return removed, weakened


def layout(kind: str, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, str]:
    if kind == "geodata":
        return np.asarray(CASE118_XY, dtype=float), "pandapower case118 geodata"
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(BUS_COUNT))
    graph.add_edges_from(pairs)
    pos = nx.kamada_kawai_layout(graph)
    return np.array([pos[b] for b in range(BUS_COUNT)]), "kamada-kawai layout (no geodata)"


def hop_distances(pairs: list[tuple[int, int]], sources: list[int]) -> np.ndarray:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(BUS_COUNT))
    graph.add_edges_from(pairs)
    lengths = nx.multi_source_dijkstra_path_length(graph, set(sources))
    return np.array([lengths.get(b, np.inf) for b in range(BUS_COUNT)])


def draw_map(ax, xy, pairs, ybus, types, values, norm, cmap, *, top_k=10, removed=(), weakened=(), rings=()):
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection

    mag = np.log10(np.array([abs(ybus[i, j]) for i, j in pairs]) + 1e-12)
    width = 0.3 + 1.5 * (mag - mag.min()) / max(mag.max() - mag.min(), 1e-9)
    segments = [(xy[i], xy[j]) for i, j in pairs]
    ax.add_collection(LineCollection(segments, colors="0.6", linewidths=width, alpha=0.7, zorder=1))
    if removed:
        ax.add_collection(
            LineCollection([(xy[i], xy[j]) for i, j in removed], colors="red", linewidths=2.2,
                           linestyles=(0, (4, 2)), zorder=2)
        )
    if weakened:
        ax.add_collection(
            LineCollection([(xy[i], xy[j]) for i, j in weakened], colors="red", linewidths=1.6,
                           linestyles=(0, (1, 1.5)), zorder=2)
        )
    for type_id, marker, size, _label in TYPE_MARKERS:
        mask = types == type_id
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], c=values[mask], norm=norm, cmap=cmap, marker=marker, s=size,
                       edgecolors="k", linewidths=0.35, zorder=3)
    if len(rings):
        ring = np.asarray(rings)
        ax.scatter(xy[ring, 0], xy[ring, 1], s=260, facecolors="none", edgecolors="red", linewidths=2.0, zorder=4)
    for bus in np.argsort(-values)[:top_k]:
        ax.annotate(str(bus + 1), xy[bus], xytext=(4, 4), textcoords="offset points", fontsize=7, weight="bold",
                    zorder=5, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")


def legend_handles(*, outage_line=False, outage_gen=False, weakened=False):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], marker=m, linestyle="none", markerfacecolor="0.8", markeredgecolor="k",
               markersize=10 if m == "*" else 6, label=label)
        for _t, m, _s, label in TYPE_MARKERS
    ]
    if outage_line:
        handles.append(Line2D([], [], color="red", linestyle=(0, (4, 2)), linewidth=2, label="tripped branch"))
    if weakened:
        handles.append(Line2D([], [], color="red", linestyle=(0, (1, 1.5)), linewidth=1.6,
                              label="one parallel circuit tripped"))
    if outage_gen:
        handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="red",
                              markeredgewidth=2, markersize=11, label="outaged generator bus"))
    return handles


def save_map(path, xy, pairs, ybus, types, values, norm, cmap, title, dpi, cbar_label, verbose=True, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    draw_map(ax, xy, pairs, ybus, types, values, norm, cmap, **kwargs)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.01, label=cbar_label)
    set_log_ticks(cbar, norm)
    ax.legend(
        handles=legend_handles(outage_line=bool(kwargs.get("removed")), outage_gen=bool(len(kwargs.get("rings", ()))),
                               weakened=bool(kwargs.get("weakened"))),
        loc="lower left", fontsize=7, frameon=True,
    )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"wrote {path}")


def hop_table(label: str, hops: np.ndarray, values: np.ndarray, baseline: np.ndarray) -> None:
    print(f"\n{label}: per-bus mean PBL by hop distance from the outaged element (N graph)")
    print(f"  {'hops':>6s} {'buses':>5s} {'N-1 PBL':>9s} {'N PBL':>9s} {'ratio':>7s}")
    bins = (("0", hops == 0), ("1", hops == 1), ("2", hops == 2), ("3", hops == 3), (">=4", hops >= 4))
    for name, mask in bins:
        if not mask.any():
            continue
        a, b = float(values[mask].mean()), float(baseline[mask].mean())
        print(f"  {name:>6s} {int(mask.sum()):5d} {a:9.4f} {b:9.4f} {a / max(b, 1e-12):7.1f}")
    a, b = float(values.mean()), float(baseline.mean())
    print(f"  {'all':>6s} {BUS_COUNT:5d} {a:9.4f} {b:9.4f} {a / max(b, 1e-12):7.1f}")


def describe_n(types: np.ndarray, values: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    print("\nValidation N: per-bus mean PBL by bus type")
    for type_id, _m, _s, label in TYPE_MARKERS:
        mask = types == type_id
        print(f"  {label:6s} {int(mask.sum()):3d} buses  mean {values[mask].mean():.4f}  max {values[mask].max():.4f}")
    gens = [int(b) for b in np.nonzero(types >= 1)[0]]
    to_gen = hop_distances(pairs, gens)
    to_slack = hop_distances(pairs, [int(np.nonzero(types == 2)[0][0])])
    pq = types == 0
    for name, dist in (("PQ buses by hops to nearest generator", to_gen), ("all buses by hops to slack", to_slack)):
        print(f"  {name}:")
        scope = pq if "PQ" in name else np.ones_like(pq)
        for hop in range(int(dist[scope].max()) + 1):
            mask = scope & (dist == hop)
            if mask.any():
                print(f"    {hop:2d} hops  {int(mask.sum()):3d} buses  mean {values[mask].mean():.4f}")
    degree = np.zeros(BUS_COUNT)
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1
    print(f"  corr(PBL, degree) = {np.corrcoef(values, degree)[0, 1]:+.3f}")


def compute(args: argparse.Namespace) -> dict:
    """Evaluate the checkpoint; returns the small per-bus arrays the plots need (saved as an .npz)."""
    import torch

    t = load_module(args.module, args.module_path)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    batch = args.batch or (64 if device.type == "cuda" else 16)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = t.TPGT().to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    epoch = payload.get("epoch", "?")
    print(f"checkpoint {args.checkpoint} epoch {epoch} on {device}", flush=True)
    val = torch.load(args.cache_dir / args.val_cache, map_location="cpu", weights_only=False)
    n1 = torch.load(args.cache_dir / args.n1_cache, map_location="cpu", weights_only=False)
    return compute_from_model(
        t, model, val, n1, device, epoch=epoch, checkpoint_name=args.checkpoint.name,
        val_cases=args.val_cases, batch=batch,
    )


def compute_from_model(
    t,
    model,
    val,
    n1,
    device,
    *,
    epoch,
    checkpoint_name: str,
    val_cases: int = 256,
    batch: int = 64,
    progress: bool = True,
) -> dict:
    """Per-bus arrays for ``model`` on the dominant validation topology and every test N-1 topology.

    ``t`` provides move_to, pair_features, pack_case, dominant_network_id, network_case_counts and
    contingency_class (the training module or a namespace with those functions).
    """
    model.eval()
    ref_id = t.dominant_network_id(val)
    reference = val.networks[ref_id]
    pairs = branch_pairs(reference.ybus.detach().cpu().numpy())
    on_ref = np.nonzero(val.network_index.numpy() == ref_id)[0]
    take = np.unique(np.linspace(0, len(on_ref) - 1, min(val_cases, len(on_ref))).round().astype(int))
    val_res = per_bus_residual(t, model, val, ref_id, on_ref[take], device, batch)
    if progress:
        print(f"validation N: {len(take)} cases done", flush=True)

    counts = dict(t.network_case_counts(n1))
    net_index = n1.network_index.numpy()
    ids = np.array(sorted(counts), dtype=np.int64)
    topo_pbl = np.zeros((len(ids), BUS_COUNT), dtype=np.float32)
    for row, net_id in enumerate(ids):
        idx = np.nonzero(net_index == net_id)[0]
        topo_pbl[row] = pbl(per_bus_residual(t, model, n1, int(net_id), idx, device, batch)).mean(axis=0)
        if progress and (row + 1) % 25 == 0:
            print(f"test N-1: {row + 1}/{len(ids)} topologies", flush=True)
    return {
        "epoch": str(epoch),
        "checkpoint": checkpoint_name,
        "val_cases": len(take),
        "val_pbl": pbl(val_res).mean(axis=0),
        "val_dp": np.abs(val_res[..., 0]).mean(axis=0),
        "val_dq": np.abs(val_res[..., 1]).mean(axis=0),
        "ref_ybus": reference.ybus.detach().cpu().numpy(),
        "ref_types": reference.types.cpu().numpy(),
        "topo_ids": ids,
        "topo_counts": np.array([counts[int(n)] for n in ids]),
        "topo_classes": np.array([t.contingency_class(n1.networks[int(n)], reference) for n in ids]),
        "topo_pbl": topo_pbl,
        "topo_branch_y": np.stack([branch_admittance(n1.networks[int(n)].ybus.detach().cpu().numpy(), pairs)
                                   for n in ids]),
        "topo_types": np.stack([n1.networks[int(n)].types.cpu().numpy() for n in ids]),
    }


def set_log_ticks(cbar, norm) -> None:
    low, high = int(np.floor(np.log10(norm.vmin))), int(np.ceil(np.log10(norm.vmax)))
    ticks = [m * 10.0**k for k in range(low, high + 1) for m in (1, 2, 5) if norm.vmin <= m * 10.0**k <= norm.vmax]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:g}" for v in ticks])
    cbar.ax.minorticks_off()


MAP_FILES = (
    "val_N_mean_pbl.png",
    "val_N_dP_dQ.png",
    "test_N1_gen_outage_worst.png",
    "test_N1_line_outage_worst.png",
    "test_N1_line_outage_median.png",
)


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_npz:
        data = dict(np.load(args.from_npz, allow_pickle=False))
    else:
        if args.checkpoint is None or args.cache_dir is None:
            raise SystemExit("--checkpoint and --cache-dir are required unless --from-npz is given")
        data = compute(args)
        np.savez_compressed(args.out_dir / "per_bus_pbl.npz", **data)
    print(render(data, args.out_dir, layout_kind=args.layout, min_cases=args.min_cases, dpi=args.dpi))


def write_bus_maps(t, model, val, n1, device, out_dir: Path, *, epoch, checkpoint_name: str,
                   val_cases: int = 256, batch: int = 64) -> list[Path]:
    """Evaluate ``model`` in-process and write the maps, ``per_bus_pbl.npz`` and ``summary.txt`` to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = compute_from_model(t, model, val, n1, device, epoch=epoch, checkpoint_name=checkpoint_name,
                              val_cases=val_cases, batch=batch, progress=False)
    np.savez_compressed(out_dir / "per_bus_pbl.npz", **data)
    render(data, out_dir, verbose=False)
    return [out_dir / name for name in MAP_FILES if (out_dir / name).is_file()]


def render(data: dict, out_dir: Path, *, layout_kind: str = "geodata", min_cases: int = 5, dpi: int = 150,
           verbose: bool = True) -> str:
    """Draw every map from ``data`` (see ``compute_from_model``) into ``out_dir``; returns the summary text."""
    from matplotlib.colors import LogNorm

    out_dir.mkdir(parents=True, exist_ok=True)
    epoch = str(data["epoch"])
    val_pbl, val_dp, val_dq = data["val_pbl"], data["val_dp"], data["val_dq"]
    ref_y, ref_types = data["ref_ybus"], data["ref_types"]
    val_cases = int(data["val_cases"])
    pairs = branch_pairs(ref_y)
    xy, layout_name = layout(layout_kind, pairs)

    row_of = {int(n): row for row, n in enumerate(data["topo_ids"])}
    counts = {int(n): int(c) for n, c in zip(data["topo_ids"], data["topo_counts"])}
    classes = {int(n): str(c) for n, c in zip(data["topo_ids"], data["topo_classes"])}
    per_topology = {n: data["topo_pbl"][row_of[n]] for n in counts}
    overall_n1 = sum(per_topology[n].mean() * c for n, c in counts.items()) / sum(counts.values())

    def ranked(name):
        return sorted(
            (float(per_topology[n].mean()), n) for n in counts if classes[n] == name and counts[n] >= min_cases
        )

    gens = ranked("generator outage")
    lines = ranked("line outage")
    picks = []
    if gens:
        picks.append(("gen_outage_worst", "worst generator outage", gens[-1][1]))
    if lines:
        picks.append(("line_outage_worst", "worst line outage", lines[-1][1]))
        picks.append(("line_outage_median", "median line outage", lines[len(lines) // 2][1]))

    maps = [val_pbl] + [per_topology[n] for _k, _l, n in picks]
    floor = 1e-4
    norm = LogNorm(vmin=max(floor, min(float(m.min()) for m in maps)), vmax=max(float(m.max()) for m in maps))
    scale = f"shared log color scale {norm.vmin:.2g}-{norm.vmax:.2g} p.u."

    report = io.StringIO()
    with redirect_stdout(report):
        print(f"checkpoint {data['checkpoint']} epoch {epoch}; layout: {layout_name}; {scale}")
        print(f"validation N: {val_cases} cases, mean PBL {val_pbl.mean():.4f} p.u.")
        print("worst 10 buses (1-based):", ", ".join(f"{b + 1}({val_pbl[b]:.3f})" for b in np.argsort(-val_pbl)[:10]))
        describe_n(ref_types, val_pbl, pairs)
        print(f"\ntest N-1: {sum(counts.values())} cases, {len(counts)} topologies, mean PBL {overall_n1:.4f}")
        for name in sorted(set(classes.values())):
            members = [n for n in counts if classes[n] == name]
            cases = sum(counts[n] for n in members)
            mean = sum(per_topology[n].mean() * counts[n] for n in members) / cases
            print(f"  {name:16s} {len(members):4d} topologies {cases:6d} cases  mean PBL {mean:.4f}")

    cmap = "magma"
    title_tail = f"\nepoch {epoch} | {layout_name} | {scale}"
    save_map(out_dir / "val_N_mean_pbl.png", xy, pairs, ref_y, ref_types, val_pbl, norm, cmap,
             f"Validation N ({val_cases} cases): mean per-bus PBL {val_pbl.mean():.4f} p.u.{title_tail}",
             dpi, "mean per-bus PBL (p.u.)", verbose=verbose)

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    pq_norm = LogNorm(vmin=max(1e-5, min(val_dp.min(), val_dq.min())), vmax=max(val_dp.max(), val_dq.max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.6), layout="constrained")
    for ax, values, name in ((axes[0], val_dp, "|ΔP|"), (axes[1], val_dq, "|ΔQ|")):
        draw_map(ax, xy, pairs, ref_y, ref_types, values, pq_norm, "viridis")
        ax.set_title(f"mean {name} per bus (overall {values.mean():.4f} p.u.)", fontsize=10)
    axes[0].legend(handles=legend_handles(), loc="lower left", fontsize=7)
    cbar = fig.colorbar(ScalarMappable(norm=pq_norm, cmap="viridis"), ax=axes, shrink=0.7, pad=0.01,
                        label="mean |mismatch| (p.u., log)")
    set_log_ticks(cbar, pq_norm)
    fig.suptitle(f"Validation N ({val_cases} cases), epoch {epoch} | {layout_name}", fontsize=11)
    fig.savefig(out_dir / "val_N_dP_dQ.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"wrote {out_dir / 'val_N_dP_dQ.png'}")

    for key, label, net_id in picks:
        types = data["topo_types"][row_of[net_id]]
        removed, weakened = changed_branches(data["topo_branch_y"][row_of[net_id]], branch_admittance(ref_y, pairs),
                                             pairs)
        lost_gens = [int(b) for b in np.nonzero((ref_types >= 1) & (types == 0))[0]]
        values = per_topology[net_id]
        sources = sorted({b for edge in removed + weakened for b in edge} | set(lost_gens))
        what = []
        if removed or weakened:
            what.append("branch " + ", ".join(f"{i + 1}-{j + 1}" for i, j in removed + weakened))
        if lost_gens:
            what.append("generator at bus " + ", ".join(str(b + 1) for b in lost_gens))
        outage = "; ".join(what) or "no visible change"
        save_map(out_dir / f"test_N1_{key}.png", xy, pairs, ref_y, types, values, norm, cmap,
                 f"Test N-1 {label}: topology {net_id} ({counts[net_id]} cases), out: {outage}\n"
                 f"mean PBL {values.mean():.4f} p.u.{title_tail}",
                 dpi, "mean per-bus PBL (p.u.)", removed=removed, weakened=weakened, rings=lost_gens,
                 verbose=verbose)
        with redirect_stdout(report):
            print(f"\n{label}: topology {net_id}, {counts[net_id]} cases, out: {outage}, mean PBL {values.mean():.4f}")
            print("  worst 10 buses:", ", ".join(f"{b + 1}({values[b]:.3f})" for b in np.argsort(-values)[:10]))
            if sources:
                hop_table(label, hop_distances(pairs, sources), values, val_pbl)

    text = report.getvalue()
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    main()
