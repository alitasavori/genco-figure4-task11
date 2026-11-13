"""Convert MIT PFΔ PowerModels JSON cases into the gridfm-style parquet layout TPGT expects.

Downloads ``caseXX.tar.gz`` + ``shuffle_files.tar.gz`` from ``pfdelta/pfdelta`` and writes:

``<out>/<case>/<split>/<case>/<topology>/<condition>/raw/{bus,branch,gen,y_bus}_data.parquet``

Task 1.1 uses the same 90/10 train/val cut of the first ``train_size`` shuffled N-feasible
samples and the last ``test_size`` samples for test (MIT ``PFDeltaDataset`` convention).
"""

from __future__ import annotations

import json
import math
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

PARQUET_NAMES = ("bus_data.parquet", "branch_data.parquet", "gen_data.parquet", "y_bus_data.parquet")
TASK11_TRAIN_N = 54000
TASK11_TEST_N = 2000
TASK11_TEST_NOSE = 200
HF_REPO = "pfdelta/pfdelta"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _network_and_solution(pm_case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float]:
    if "solved_net" in pm_case:
        network = pm_case["solved_net"]
        solution = pm_case["solved_net"]
    else:
        network = pm_case["network"]
        solution = pm_case["solution"]["solution"]
    base_mva = float(network.get("baseMVA", 100.0))
    return network, solution, base_mva


def _bus_shunt_bs(network: dict[str, Any], bus_id: int) -> float:
    total = 0.0
    for shunt in network.get("shunt", {}).values():
        if int(shunt["shunt_bus"]) == bus_id:
            total += float(shunt.get("bs", 0.0))
    return total


def _bus_load(network: dict[str, Any], bus_id: int) -> tuple[float, float]:
    pd = qd = 0.0
    for load in network.get("load", {}).values():
        if int(load["load_bus"]) == bus_id:
            pd += float(load.get("pd", 0.0))
            qd += float(load.get("qd", 0.0))
    return pd, qd


def _bus_gen(
    network: dict[str, Any],
    solution: dict[str, Any],
    bus_id: int,
) -> tuple[float, float]:
    pg = qg = 0.0
    for gen_id, gen in network.get("gen", {}).items():
        if int(gen["gen_bus"]) != bus_id:
            continue
        if int(gen.get("gen_status", 1)) != 1:
            continue
        gen_sol = solution.get("gen", {}).get(str(gen_id), solution.get("gen", {}).get(gen_id))
        if gen_sol is None:
            continue
        pg += float(gen_sol.get("pg", 0.0))
        qg += float(gen_sol.get("qg", 0.0))
    return pg, qg


def _branch_y(
    resistance: float,
    reactance: float,
    g_fr: float,
    b_fr: float,
    g_to: float,
    b_to: float,
    tap: float,
    shift: float,
) -> tuple[complex, complex, complex, complex]:
    """PowerModels π-model branch admittance (from/to)."""
    z = complex(resistance, reactance)
    if abs(z) < 1e-18:
        y = 0.0j
    else:
        y = 1.0 / z
    tap = float(tap) if tap else 1.0
    tr = tap * math.cos(shift)
    ti = tap * math.sin(shift)
    tt = tr * tr + ti * ti
    yff = (y + complex(g_fr, b_fr)) / tt
    yft = -y * complex(tr, -ti) / tt
    ytf = -y * complex(tr, ti) / tt
    ytt = y + complex(g_to, b_to)
    return yff, yft, ytf, ytt


def build_ybus(network: dict[str, Any]) -> np.ndarray:
    buses = sorted(int(b) for b in network["bus"])
    n = len(buses)
    index = {bus_id: i for i, bus_id in enumerate(buses)}
    ybus = np.zeros((n, n), dtype=np.complex128)
    for branch in network.get("branch", {}).values():
        if int(branch.get("br_status", 1)) != 1:
            continue
        f = index[int(branch["f_bus"])]
        t = index[int(branch["t_bus"])]
        yff, yft, ytf, ytt = _branch_y(
            float(branch.get("br_r", 0.0)),
            float(branch.get("br_x", 0.0)),
            float(branch.get("g_fr", 0.0)),
            float(branch.get("b_fr", 0.0)),
            float(branch.get("g_to", 0.0)),
            float(branch.get("b_to", 0.0)),
            float(branch.get("tap", 1.0) or 1.0),
            float(branch.get("shift", 0.0)),
        )
        ybus[f, f] += yff
        ybus[f, t] += yft
        ybus[t, f] += ytf
        ybus[t, t] += ytt
    for bus_id_str, bus in network["bus"].items():
        i = index[int(bus_id_str)]
        # Device shunt only (gs + j bs); line charging is already in branch g_fr/b_fr.
        bs = _bus_shunt_bs(network, int(bus_id_str))
        gs = 0.0
        for shunt in network.get("shunt", {}).values():
            if int(shunt["shunt_bus"]) == int(bus_id_str):
                gs += float(shunt.get("gs", 0.0))
        ybus[i, i] += complex(gs, bs)
        # Ignore unused bus dict fields.
        _ = bus
    return ybus


def scenario_rows(pm_case: dict[str, Any], scenario: int) -> dict[str, list]:
    network, solution, base_mva = _network_and_solution(pm_case)
    bus_ids = sorted(int(b) for b in network["bus"])
    bus_rows: dict[str, list] = {
        "scenario": [],
        "bus": [],
        "Pd": [],
        "Qd": [],
        "Pg": [],
        "Qg": [],
        "Vm": [],
        "Va": [],
        "PQ": [],
        "PV": [],
        "REF": [],
        "BS": [],
    }
    for local, bus_id in enumerate(bus_ids):
        bus = network["bus"][str(bus_id)]
        bus_sol = solution["bus"][str(bus_id)]
        pd_pu, qd_pu = _bus_load(network, bus_id)
        pg_pu, qg_pu = _bus_gen(network, solution, bus_id)
        bus_type = int(bus["bus_type"])
        if bus_type == 2 and pg_pu == 0.0 and qg_pu == 0.0:
            bus_type = 1
        vm = float(bus_sol["vm"])
        va_rad = float(bus_sol["va"])
        bus_rows["scenario"].append(scenario)
        bus_rows["bus"].append(local)
        bus_rows["Pd"].append(pd_pu * base_mva)
        bus_rows["Qd"].append(qd_pu * base_mva)
        bus_rows["Pg"].append(pg_pu * base_mva)
        bus_rows["Qg"].append(qg_pu * base_mva)
        bus_rows["Vm"].append(vm)
        bus_rows["Va"].append(math.degrees(va_rad))
        bus_rows["PQ"].append(1.0 if bus_type == 1 else 0.0)
        bus_rows["PV"].append(1.0 if bus_type == 2 else 0.0)
        bus_rows["REF"].append(1.0 if bus_type == 3 else 0.0)
        bus_rows["BS"].append(_bus_shunt_bs(network, bus_id))

    branch_rows: dict[str, list] = {
        "scenario": [],
        "idx": [],
        "from_bus": [],
        "to_bus": [],
        "r": [],
        "x": [],
        "br_status": [],
    }
    index = {bus_id: i for i, bus_id in enumerate(bus_ids)}
    for idx, (_bid, branch) in enumerate(sorted(network.get("branch", {}).items(), key=lambda x: int(x[0]))):
        branch_rows["scenario"].append(scenario)
        branch_rows["idx"].append(idx)
        branch_rows["from_bus"].append(index[int(branch["f_bus"])])
        branch_rows["to_bus"].append(index[int(branch["t_bus"])])
        branch_rows["r"].append(float(branch.get("br_r", 0.0)))
        branch_rows["x"].append(float(branch.get("br_x", 0.0)))
        branch_rows["br_status"].append(float(branch.get("br_status", 1)))

    gen_rows: dict[str, list] = {"scenario": [], "idx": [], "gen_bus": [], "Pg": [], "Qg": []}
    for idx, (gen_id, gen) in enumerate(sorted(network.get("gen", {}).items(), key=lambda x: int(x[0]))):
        gen_sol = solution.get("gen", {}).get(str(gen_id), solution.get("gen", {}).get(gen_id))
        pg = float(gen_sol.get("pg", 0.0)) * base_mva if gen_sol else 0.0
        qg = float(gen_sol.get("qg", 0.0)) * base_mva if gen_sol else 0.0
        gen_rows["scenario"].append(scenario)
        gen_rows["idx"].append(idx)
        gen_rows["gen_bus"].append(index[int(gen["gen_bus"])])
        gen_rows["Pg"].append(pg)
        gen_rows["Qg"].append(qg)

    ybus = build_ybus(network)
    y_rows: dict[str, list] = {"scenario": [], "index1": [], "index2": [], "G": [], "B": []}
    ii, jj = np.nonzero(np.abs(ybus) > 0)
    for i, j in zip(ii.tolist(), jj.tolist()):
        y_rows["scenario"].append(scenario)
        y_rows["index1"].append(i)
        y_rows["index2"].append(j)
        y_rows["G"].append(float(ybus[i, j].real))
        y_rows["B"].append(float(ybus[i, j].imag))
    return {"bus": bus_rows, "branch": branch_rows, "gen": gen_rows, "y_bus": y_rows}


def _append_dict(dst: dict[str, list], src: dict[str, list]) -> None:
    for key, values in src.items():
        dst.setdefault(key, []).extend(values)


def write_parquet_folder(rows: dict[str, dict[str, list]], raw_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    raw_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "bus_data.parquet": rows["bus"],
        "branch_data.parquet": rows["branch"],
        "gen_data.parquet": rows["gen"],
        "y_bus_data.parquet": rows["y_bus"],
    }
    for name, columns in mapping.items():
        table = pa.table({key: pa.array(values) for key, values in columns.items()})
        pq.write_table(table, raw_dir / name)


def _sample_index(path: Path) -> int:
    """Parse ``sample_12.json`` or ``sample12.json`` → 12."""
    stem = path.stem
    if stem.startswith("sample_"):
        return int(stem.split("_", 1)[1])
    if stem.startswith("sample"):
        return int(stem[len("sample") :])
    raise ValueError(f"unrecognized sample name {path.name}")


def list_raw_sample_paths(raw_path: Path) -> list[Path]:
    """Sorted feasible JSON samples under ``…/raw`` (supports ``sample_N`` and ``sampleN``)."""
    if not raw_path.is_dir():
        return []
    underscored = list(raw_path.glob("sample_*.json"))
    if underscored:
        return sorted(underscored, key=_sample_index)
    plain = [p for p in raw_path.glob("sample*.json") if p.stem != "sample"]
    return sorted(plain, key=_sample_index)


def feasible_raw_complete(case_dir: Path, topology: str = "n", *, min_samples: int = 56000) -> bool:
    """True when ``case/topology/raw`` looks fully extracted (N feasible has 56k samples)."""
    paths = list_raw_sample_paths(case_dir / topology / "raw")
    return len(paths) >= min_samples


def download_pfdelta_case(case: str, work: Path) -> tuple[Path, Path]:
    """Ensure ``caseXX/`` and ``shuffle_files/`` exist under ``work``; return those paths.

    Re-extracts when the case folder exists but is incomplete (common when a previous
    extract to Google Drive was interrupted). Prefer a local SSD ``work`` on Colab.
    """
    import shutil

    from huggingface_hub import hf_hub_download

    work.mkdir(parents=True, exist_ok=True)
    case_dir = work / case
    shuffle_dir = work / "shuffle_files"

    need_case = not feasible_raw_complete(case_dir)
    if need_case and case_dir.is_dir():
        n_raw = len(list_raw_sample_paths(case_dir / "n" / "raw"))
        print(
            f"{case_dir} incomplete ({n_raw} N-raw JSON files; need >= 56000). Re-extracting.",
            flush=True,
        )
        shutil.rmtree(case_dir)

    if need_case:
        archive = Path(
            hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=f"{case}.tar.gz", local_dir=str(work))
        )
        print(f"Extracting {archive.name} into {work} ...", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(work)
        n_raw = len(list_raw_sample_paths(case_dir / "n" / "raw"))
        print(f"  {case}/n/raw: {n_raw} JSON samples", flush=True)
        if not feasible_raw_complete(case_dir):
            raise FileNotFoundError(
                f"{case}/n/raw has {n_raw} samples after extract (expected >= 56000). "
                f"Check {case_dir}"
            )

    if not shuffle_dir.is_dir():
        archive = Path(
            hf_hub_download(
                repo_id=HF_REPO, repo_type="dataset", filename="shuffle_files.tar.gz", local_dir=str(work)
            )
        )
        print(f"Extracting {archive.name} ...", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(work)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"expected {case_dir} after extract")
    if not shuffle_dir.is_dir():
        raise FileNotFoundError(f"expected {shuffle_dir} after extract")
    return case_dir, shuffle_dir


def _shuffle_map(shuffle_dir: Path, topology: str, case: str) -> dict[int, int]:
    name = "raw_shuffle_2000.json" if case == "case2000" else "raw_shuffle.json"
    path = shuffle_dir / topology / name
    payload = _load_json(path)
    return {int(k): int(v) for k, v in payload.items()}


def task11_feasible_file_lists(
    case_dir: Path,
    shuffle_dir: Path,
    topology: str,
    *,
    train_size: int = TASK11_TRAIN_N,
    test_size: int = TASK11_TEST_N,
) -> dict[str, list[Path]]:
    """MIT Task 1.1 feasible splits for one topology."""
    shuffle_map = _shuffle_map(shuffle_dir, topology, case_dir.name)
    raw_path = case_dir / topology / "raw"
    raw_fnames = list_raw_sample_paths(raw_path)
    if not raw_fnames:
        raise FileNotFoundError(f"no sample*.json under {raw_path}")
    max_idx = max(shuffle_map.values())
    if max_idx >= len(raw_fnames):
        raise FileNotFoundError(
            f"{raw_path}: shuffle index {max_idx} but only {len(raw_fnames)} JSON files "
            f"(first={raw_fnames[0].name}, last={raw_fnames[-1].name}). "
            "Re-run after a full extract of the case archive."
        )
    # Same as MIT PFDeltaDataset: permute the contiguous sample list by shuffle values.
    keys = sorted(shuffle_map.keys())
    shuffled = [raw_fnames[shuffle_map[i]] for i in keys]
    missing = [path for path in shuffled[:3] + shuffled[-3:] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"shuffle resolved missing files, e.g. {missing[0]}")
    if train_size + test_size > len(shuffled):
        raise RuntimeError(
            f"{case_dir.name}/{topology}: need {train_size}+{test_size} samples, have {len(shuffled)}"
        )
    return {
        "train": shuffled[: int(0.9 * train_size)],
        "valid": shuffled[int(0.9 * train_size) : int(train_size)],
        "test": shuffled[-int(test_size) :],
    }


def task11_nose_test_files(case_dir: Path, topology: str, n: int = TASK11_TEST_NOSE) -> list[Path]:
    nose_test = case_dir / topology / "nose" / "test"
    files = sorted(nose_test.glob("*.json"))[:n]
    if len(files) < n:
        raise FileNotFoundError(f"{nose_test}: need {n} nose test JSON files, found {len(files)}")
    return files


def convert_file_list(files: list[Path], raw_out: Path, *, progress_every: int = 500) -> None:
    if all((raw_out / name).is_file() for name in PARQUET_NAMES):
        print(f"  reusing {raw_out}", flush=True)
        return
    rows = {"bus": {}, "branch": {}, "gen": {}, "y_bus": {}}
    for local, path in enumerate(files):
        pm_case = _load_json(path)
        piece = scenario_rows(pm_case, scenario=local)
        for key in rows:
            _append_dict(rows[key], piece[key])
        if (local + 1) % progress_every == 0 or local + 1 == len(files):
            print(f"  {raw_out.parent.parent.name}/{raw_out.parent.name}: {local + 1}/{len(files)}", flush=True)
    write_parquet_folder(rows, raw_out)


def task11_pack_raw_dirs(out_root: Path, case: str) -> list[Path]:
    """All ``…/raw`` folders required for Task 1.1 on ``case`` (train N + full test suite)."""
    dirs = [
        out_root / case / "train" / case / "n" / "feasible" / "raw",
        out_root / case / "valid" / case / "n" / "feasible" / "raw",
        out_root / case / "test" / case / "n" / "feasible" / "raw",
    ]
    for topology in ("n", "n-1", "n-2"):
        if topology != "n":
            dirs.append(out_root / case / "test" / case / topology / "feasible" / "raw")
        dirs.append(out_root / case / "test" / case / topology / "nose" / "raw")
    return dirs


def task11_pack_complete(out_root: Path, case: str) -> bool:
    """True when every Task 1.1 parquet folder has all four tables."""
    for raw in task11_pack_raw_dirs(out_root, case):
        if not all((raw / name).is_file() for name in PARQUET_NAMES):
            return False
    return True


def build_task11_case_dataset(case: str, out_root: Path, work: Path) -> Path:
    """Download PFΔ ``case`` and write Task 1.1 parquet under ``out_root``."""
    if task11_pack_complete(out_root, case):
        print(f"Parquet Task 1.1 pack already present: {out_root}", flush=True)
        return out_root

    case_dir, shuffle_dir = download_pfdelta_case(case, work)
    missing = [
        raw
        for raw in task11_pack_raw_dirs(out_root, case)
        if not all((raw / name).is_file() for name in PARQUET_NAMES)
    ]
    print(
        f"Building Task 1.1 parquet for {case} -> {out_root} "
        f"({len(missing)} folder(s) still missing)",
        flush=True,
    )
    # Task 1.1 trains on N only (54k pool → 90/10 train/valid). N-1/N-2 have smaller
    # archives (e.g. 29k / 20k) and are test-only (last 2000 + nose), matching MIT.
    for topology in ("n", "n-1", "n-2"):
        if topology == "n":
            splits = task11_feasible_file_lists(
                case_dir, shuffle_dir, topology, train_size=TASK11_TRAIN_N, test_size=TASK11_TEST_N
            )
            for split_name, files in splits.items():
                raw_out = out_root / case / split_name / case / topology / "feasible" / "raw"
                convert_file_list(files, raw_out)
        else:
            files = task11_feasible_file_lists(
                case_dir, shuffle_dir, topology, train_size=0, test_size=TASK11_TEST_N
            )["test"]
            raw_out = out_root / case / "test" / case / topology / "feasible" / "raw"
            convert_file_list(files, raw_out)
        nose = task11_nose_test_files(case_dir, topology)
        raw_out = out_root / case / "test" / case / topology / "nose" / "raw"
        convert_file_list(nose, raw_out)

    if not task11_pack_complete(out_root, case):
        still = [
            str(raw.relative_to(out_root))
            for raw in task11_pack_raw_dirs(out_root, case)
            if not all((raw / name).is_file() for name in PARQUET_NAMES)
        ]
        raise FileNotFoundError(f"conversion unfinished; missing {still[:5]}")
    print(f"Wrote Task 1.1 parquet pack at {out_root}", flush=True)
    return out_root
