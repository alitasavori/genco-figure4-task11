#!/usr/bin/env python3
"""Refresh ``data/pf_runtime_published.json`` from GENCO / PowerModels paper repro CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "pf_runtime_published.json"

GENCO_BASE = (
    "https://raw.githubusercontent.com/gridfm/gridfm-graphkit/genco-paper-repro/scripts/runtime"
)
PM_BASE = (
    "https://raw.githubusercontent.com/gridfm/gridfm-datakit/genco-paper-repro"
    "/scripts/runtime/outputs_julia/full_matrix"
)
RESIDUALS_URL = (
    "https://raw.githubusercontent.com/gridfm/gridfm-graphkit/genco-paper-repro"
    "/scripts/datakit_pf/results/pf_eval_aggregated.csv"
)

NETWORKS = (
    ("case14_ieee", 14, "ieee", "small"),
    ("case30_ieee", 30, "ieee", "small"),
    ("case57_ieee", 57, "ieee", "small"),
    ("case118_ieee", 118, "ieee", "small"),
    ("case500_goc", 500, "goc", "small"),
    ("case2000_goc", 2000, "goc", "large"),
    ("case10000_goc", 10000, "goc", "large"),
)
GENCO_MODELS = ("base", "small", "tiny")


LOCAL_MIRROR = REPO_ROOT / "data" / "runtime_csv_mirror"


def fetch_text(url: str) -> str:
    if "genco_pf_in_memory" in url:
        local = LOCAL_MIRROR / "genco_pf_in_memory" / url.split("genco_pf_in_memory/")[1]
    elif "full_matrix" in url:
        local = LOCAL_MIRROR / "powermodels" / url.split("full_matrix/")[1]
    elif "pf_eval_aggregated.csv" in url:
        local = LOCAL_MIRROR / "pf_eval_aggregated.csv"
    if local.is_file():
        return local.read_text(encoding="utf-8")
    with urllib.request.urlopen(url, timeout=120) as handle:
        return handle.read().decode()


def best_genco(network: str, model: str, group: str) -> tuple[int, float]:
    url = f"{GENCO_BASE}/genco_pf_in_memory/{group}/benchmark_{network}_{model}.csv"
    rows = list(csv.DictReader(fetch_text(url).splitlines()))
    curve: list[tuple[int, float]] = []
    for row in rows:
        if row["status"] != "ok":
            continue
        ms = float(row["outer_elapsed_ms"]) / int(row["num_samples"])
        curve.append((int(row["batch_size"]), ms))
    if not curve:
        raise ValueError(f"No successful GENCO rows in {url}")
    return min(curve, key=lambda point: point[1])


def best_powermodels(network: str, scope: str, mode: str) -> tuple[int, float]:
    url = f"{PM_BASE}/{scope}/setup1/benchmark_{network}_{mode}.csv"
    rows = list(csv.DictReader(fetch_text(url).splitlines()))
    curve = [
        (int(row["p"]), 1_000.0 * float(row["pf_elapsed_s"]) / int(row["n_pfs"])) for row in rows
    ]
    if not curve:
        raise ValueError(f"No PowerModels rows in {url}")
    return min(curve, key=lambda point: point[1])


def load_residuals() -> dict[tuple[int, str], dict[str, float]]:
    residuals: dict[tuple[int, str], dict[str, float]] = {}
    for row in csv.DictReader(fetch_text(RESIDUALS_URL).splitlines()):
        grid = int(row["grid"].replace("case", ""))
        model = row["model"]
        residuals[(grid, model)] = {
            "genco_active_mw": float(row["GENCO Avg. active res. (MW)"]),
            "dc_active_mw": float(row["DC Avg. active res. (MW)"]),
        }
    return residuals


def build() -> dict:
    residuals = load_residuals()
    out: dict = {
        "source": "gridfm-graphkit/datakit genco-paper-repro runtime CSVs",
        "metric": "amortized_ms_per_instance",
        "grids": {},
    }
    for network, buses, group, scope in NETWORKS:
        entry: dict = {"bus_count": buses, "genco": {}, "powermodels": {}}
        for model in GENCO_MODELS:
            batch_size, ms = best_genco(network, model, group)
            res = residuals.get((buses, model), {})
            dc = res.get("dc_active_mw")
            genco = res.get("genco_active_mw")
            ratio = (dc / genco) if dc is not None and genco and genco > 0 else None
            entry["genco"][model] = {
                "batch_size": batch_size,
                "ms_per_instance": ms,
                "dc_over_genco_active_res": ratio,
            }
        for mode in ("pf", "dcpf"):
            workers, ms = best_powermodels(network, scope, mode)
            entry["powermodels"][mode] = {"workers": workers, "ms_per_instance": ms}
        out["grids"][str(buses)] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} ({len(payload['grids'])} grids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
