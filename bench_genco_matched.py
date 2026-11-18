"""Time GENCO Tiny/Small/Base on the same GPU + batch sizes as TPGT §5.4.

Uses graphkit ``genco-paper-repro`` (random-init HGNS; wall time is weight-independent
per the paper runtime guide) and HF ``gridfm/genco-pf-runtime`` processed graphs.

Protocol mirrors our TPGT timer: preload, ``WARMUP_BATCHES`` excluded, then amortized
``wall_ms / num_samples`` for at least ``MIN_BENCH_SECONDS`` at the **fixed** batch
sizes taken from the TPGT bench JSON (not GENCO's own best-batch sweep).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent

# Keep in sync with plot_runtime_fig8 TPGT timer.
WARMUP_BATCHES = 10
MIN_BENCH_SECONDS = 20.0

GENCO_MODELS: tuple[tuple[str, int], ...] = (
    ("tiny", 12),
    ("small", 24),
    ("base", 48),
)

NETWORK_BY_BUS: dict[int, str] = {
    57: "case57_ieee",
    118: "case118_ieee",
    500: "case500_goc",
}

HF_RUNTIME_DATASET = "gridfm/genco-pf-runtime"
GRAPHKIT_BRANCH = "genco-paper-repro"
GRAPHKIT_URL = "https://github.com/gridfm/gridfm-graphkit.git"

BASE_CONFIG: dict[str, Any] = {
    "task": {"task_name": "PowerFlow"},
    "data": {
        "baseMVA": 100,
        "mask_value": 0.0,
        "mask_ref_vm": True,
        "normalization": "HeteroDataMVANormalizer",
        "networks": ["case14_ieee"],
    },
    "model": {
        "attention_head": 8,
        "edge_dim": 10,
        "hidden_size": 48,
        "input_bus_dim": 15,
        "input_gen_dim": 6,
        "output_bus_dim": 2,
        "output_gen_dim": 1,
        "num_layers": 12,
        "type": "GNS_heterogeneous",
    },
    "seed": 0,
}


def default_genco_data_root() -> Path:
    override = os.environ.get("GENCO_RUNTIME_DATA")
    if override:
        return Path(override)
    drive = Path("/content/drive/MyDrive")
    if drive.is_dir():
        return drive / "Foundation Model" / "vendor" / "genco_pf_runtime"
    return REPO_ROOT / "vendor" / "genco_pf_runtime"


def default_graphkit_root() -> Path:
    override = os.environ.get("GENCO_GRAPHKIT_ROOT")
    if override:
        return Path(override)
    if Path("/content").is_dir():
        return Path("/content/gridfm-graphkit-runtime")
    return REPO_ROOT / "vendor" / "gridfm-graphkit-runtime"


def release_cuda_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def ensure_graphkit(repo: Path | None = None) -> Path:
    """Clone ``genco-paper-repro`` and ``pip install -e`` if ``gridfm_graphkit`` is missing."""
    repo = repo or default_graphkit_root()
    try:
        import gridfm_graphkit  # noqa: F401

        return Path(gridfm_graphkit.__file__).resolve().parents[1]
    except ImportError:
        pass

    repo.parent.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").is_dir():
        print(f"Cloning gridfm-graphkit ({GRAPHKIT_BRANCH}) → {repo}", flush=True)
        subprocess.check_call(
            ["git", "clone", "--branch", GRAPHKIT_BRANCH, "--depth", "1", GRAPHKIT_URL, str(repo)]
        )
    else:
        subprocess.check_call(["git", "fetch", "origin", GRAPHKIT_BRANCH], cwd=repo)
        subprocess.check_call(["git", "checkout", GRAPHKIT_BRANCH], cwd=repo)
        subprocess.check_call(["git", "pull", "--ff-only", "origin", GRAPHKIT_BRANCH], cwd=repo)

    print(f"Installing gridfm-graphkit from {repo}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo), "-q"])
    try:
        import torch

        tag = torch.__version__ if torch.version.cuda else f"{torch.__version__}+cpu"
        url = f"https://data.pyg.org/whl/torch-{tag}.html"
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch-scatter", "-f", url, "-q"]
        )
    except Exception as exc:
        print(f"torch-scatter install skipped/failed ({exc}); continuing", flush=True)

    import importlib

    importlib.invalidate_caches()
    import gridfm_graphkit  # noqa: F401

    return repo


def _network_ready(data_root: Path, network: str) -> bool:
    processed = data_root / network / "processed"
    if not processed.is_dir():
        return False
    # Need enough samples to fill large batches; paper preloads up to 10k.
    return (processed / "data_index_0.pt").is_file() and (processed / "data_index_63.pt").is_file()


def ensure_runtime_networks(
    data_root: Path,
    buses: list[int],
    *,
    max_samples: int = 2048,
) -> Path:
    """Download + extract HF tars for the grids we time (57 / 118 / 500)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
        from huggingface_hub import hf_hub_download

    data_root.mkdir(parents=True, exist_ok=True)
    needed = sorted({NETWORK_BY_BUS[b] for b in buses if b in NETWORK_BY_BUS})
    for network in needed:
        if _network_ready(data_root, network):
            print(f"GENCO runtime data ready: {network}", flush=True)
            continue
        tar_name = f"{network}.tar"
        print(f"Downloading {HF_RUNTIME_DATASET}/{tar_name} …", flush=True)
        tar_path = Path(
            hf_hub_download(
                repo_id=HF_RUNTIME_DATASET,
                repo_type="dataset",
                filename=tar_name,
                local_dir=str(data_root / "_hf"),
            )
        )
        print(f"Extracting {tar_path.name} → {data_root}", flush=True)
        with tarfile.open(tar_path, "r") as handle:
            # Prefer members under <network>/processed/; keep only first max_samples.
            members = [
                m
                for m in handle.getmembers()
                if "/processed/data_index_" in m.name.replace("\\", "/") and m.isfile()
            ]

            def index_of(name: str) -> int:
                stem = Path(name).stem  # data_index_123
                try:
                    return int(stem.rsplit("_", 1)[-1])
                except ValueError:
                    return 10**9

            members = sorted(members, key=lambda m: index_of(m.name))[:max_samples]
            handle.extractall(data_root, members=members)
        if not _network_ready(data_root, network):
            # Some tars extract as ./case57_ieee/... already under data_root.
            raise FileNotFoundError(
                f"After extract, missing processed samples for {network} under {data_root}"
            )
        print(f"Ready: {network} ({max_samples} samples kept)", flush=True)
    return data_root


def _write_config(network: str, hidden_size: int, path: Path) -> None:
    import copy

    import yaml

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["data"]["networks"] = [network]
    cfg["model"]["hidden_size"] = hidden_size
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _preload_samples(processed_dir: Path, n: int):
    import torch
    from torch_geometric.data import HeteroData
    from tqdm import tqdm

    samples = []
    for idx in tqdm(range(n), desc=f"Preload {processed_dir.parent.name}", leave=False):
        path = processed_dir / f"data_index_{idx}.pt"
        if not path.is_file():
            break
        data_dict = torch.load(path, map_location="cpu", weights_only=True)
        samples.append(HeteroData.from_dict(data_dict))
    if not samples:
        raise FileNotFoundError(f"No data_index_*.pt under {processed_dir}")
    return samples


def _build_model_and_pipe(config_path: Path, samples, device):
    import random

    import torch
    import yaml
    from gridfm_graphkit.datasets.globals import VN_KV
    from gridfm_graphkit.io.param_handler import (
        NestedNamespace,
        get_task_transforms,
        load_model,
        load_normalizer,
    )

    with config_path.open("r", encoding="utf-8") as handle:
        config = NestedNamespace(**yaml.safe_load(handle))

    normalizer = load_normalizer(config)
    random_base_mva = random.Random(config.seed).uniform(100.0, 150.0)
    vn_kv_max = max(sample.x_dict["bus"][:, VN_KV].max().item() for sample in samples)
    normalizer.fit_from_dict(
        {
            "baseMVA_orig": torch.tensor(float(getattr(config.data, "baseMVA", 100.0))),
            "baseMVA": torch.tensor(float(random_base_mva)),
            "vn_kv_max": torch.tensor(float(vn_kv_max)),
        }
    )
    transforms = get_task_transforms(config)
    model = load_model(config)
    model.eval()
    model.to(device)
    return model, normalizer, transforms


def _collate_batch(samples, indices, device):
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([samples[i % len(samples)] for i in indices])
    return batch.to(device)


def _run_pipeline(model, normalizer, transforms, batch) -> None:
    import torch

    normalizer.transform(batch)
    batch = transforms(batch)
    with torch.inference_mode():
        output = model(
            x_dict=batch.x_dict,
            edge_index_dict=batch.edge_index_dict,
            edge_attr_dict=batch.edge_attr_dict,
            mask_dict=batch.mask_dict,
        )
        normalizer.inverse_output(output, batch)


def time_genco_batch(
    *,
    samples,
    model,
    normalizer,
    transforms,
    device,
    batch_size: int,
) -> float:
    """Return amortized ms/instance at a fixed batch size (TPGT-matched protocol)."""
    import torch

    n = len(samples)
    if batch_size > n:
        raise ValueError(f"batch_size {batch_size} > preloaded samples {n}")
    indices = list(range(batch_size))

    for _ in range(WARMUP_BATCHES):
        batch = _collate_batch(samples, indices, device)
        _run_pipeline(model, normalizer, transforms, batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_samples = 0
    wall_start = time.perf_counter()
    while time.perf_counter() - wall_start < MIN_BENCH_SECONDS:
        batch = _collate_batch(samples, indices, device)
        _run_pipeline(model, normalizer, transforms, batch)
        total_samples += batch_size
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
    return elapsed_ms / max(total_samples, 1)


def run_genco_matched_benchmark(
    *,
    batch_by_bus: dict[int, int],
    data_root: Path | None = None,
    results_path: Path | None = None,
    models: tuple[tuple[str, int], ...] = GENCO_MODELS,
    max_preload: int = 2048,
    gpu_name: str | None = None,
) -> dict[str, Any]:
    """Time each GENCO size at TPGT's batch per bus. Returns serializable payload."""
    import torch

    if not batch_by_bus:
        raise ValueError("batch_by_bus is empty; run TPGT timing first.")

    ensure_graphkit()
    data_root = data_root or default_genco_data_root()
    buses = sorted(int(b) for b in batch_by_bus if int(b) in NETWORK_BY_BUS)
    ensure_runtime_networks(data_root, buses, max_samples=max(max_preload, max(batch_by_bus.values())))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GENCO matched timing requires CUDA.")
    if gpu_name is None:
        gpu_name = torch.cuda.get_device_name(0)

    cfg_dir = REPO_ROOT / "results" / "configs" / "genco_matched"
    out: dict[str, Any] = {
        "protocol": {
            "warmup_batches": WARMUP_BATCHES,
            "min_bench_seconds": MIN_BENCH_SECONDS,
            "matched_to": "TPGT best batch per grid",
            "compile": False,
            "weights": "random init (paper: wall time independent of checkpoint)",
        },
        "gpu": gpu_name,
        "data_root": str(data_root),
        "by_model": {},
    }

    for model_name, hidden in models:
        model_rows: dict[str, Any] = {}
        for bus_count in buses:
            network = NETWORK_BY_BUS[bus_count]
            batch_size = int(batch_by_bus[bus_count])
            processed = data_root / network / "processed"
            preload_n = min(max_preload, max(batch_size * 4, 256))
            # Need at least batch_size distinct slots in the ring; preload_n >= batch_size.
            preload_n = max(preload_n, batch_size)
            print(
                f"GENCO {model_name} / {network}: batch={batch_size}, preload={preload_n}",
                flush=True,
            )
            samples = _preload_samples(processed, preload_n)
            cfg_path = cfg_dir / f"config_{network}_{model_name}.yaml"
            _write_config(network, hidden, cfg_path)
            release_cuda_memory()
            model, normalizer, transforms = _build_model_and_pipe(cfg_path, samples, device)
            try:
                ms = time_genco_batch(
                    samples=samples,
                    model=model,
                    normalizer=normalizer,
                    transforms=transforms,
                    device=device,
                    batch_size=batch_size,
                )
            except torch.OutOfMemoryError:
                release_cuda_memory()
                print(f"  OOM at batch {batch_size}", flush=True)
                model_rows[str(bus_count)] = {
                    "ms_per_instance": None,
                    "batch_size": batch_size,
                    "network": network,
                    "status": "oom",
                }
                del model, normalizer, transforms, samples
                release_cuda_memory()
                continue

            print(f"  {ms:.4f} ms/instance @ batch {batch_size}", flush=True)
            model_rows[str(bus_count)] = {
                "ms_per_instance": float(ms),
                "batch_size": batch_size,
                "network": network,
                "status": "ok",
                "hidden_size": hidden,
            }
            del model, normalizer, transforms, samples
            release_cuda_memory()
        out["by_model"][model_name] = model_rows

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {results_path}", flush=True)
    return out


def batches_from_tpgt_bench(tpgt_bench: dict[str, Any] | Path) -> dict[int, int]:
    if isinstance(tpgt_bench, Path):
        tpgt_bench = json.loads(tpgt_bench.read_text(encoding="utf-8"))
    out: dict[int, int] = {}
    for key, payload in tpgt_bench.items():
        buses = int(key)
        batch = int(float(payload["batch_size"]))
        if buses in NETWORK_BY_BUS:
            out[buses] = batch
    return out
