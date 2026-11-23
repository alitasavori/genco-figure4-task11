#!/usr/bin/env python3
"""Benchmark JMAT and reference solvers on NPZ shards (Tasks 4, 6)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.csr_io import scipy_csr_to_torch  # noqa: E402
from jmat.data.shard_io import load_triple_npz  # noqa: E402
from jmat.metrics import relative_errors, summarize_errors  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.solvers.linear import solve_gmres, solve_superlu  # noqa: E402


def _eval_jmat(j, r, dx_ref, model, device) -> tuple[np.ndarray, float]:
    crow, col, val, n = scipy_csr_to_torch(j, device=device, dtype=torch.float32)
    rt = torch.tensor(r, dtype=torch.float32, device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        dx = model.forward_csr(crow, col, val, n, rt).cpu().numpy()
    latency = time.perf_counter() - t0
    return dx, latency


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "jmat" / "baseline_eval.csv")
    args = p.parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    if isinstance(entries, dict) and "samples" in entries:
        entries = entries["samples"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.checkpoint and args.checkpoint.is_file():
        model = JMAT(JMATConfig()).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        model.eval()
    rows = []
    er_x: dict[str, list[float]] = {}
    er_r: dict[str, list[float]] = {}
    for idx, item in enumerate(entries):
        if args.limit and idx >= args.limit:
            break
        path = args.data_root / item["path"] if "path" in item else args.data_root / item
        if isinstance(item, str):
            path = Path(item)
        j, r, dx_ref, meta = load_triple_npz(path)
        for method, solver in (
            ("superlu", lambda: solve_superlu(j, r, ref_dx=dx_ref)),
            ("gmres", lambda: solve_gmres(j, r, ref_dx=dx_ref)),
            ("gmres_ilu", lambda: solve_gmres(j, r, use_ilu=True, ref_dx=dx_ref)),
        ):
            res = solver()
            erx, err = relative_errors(res.dx, dx_ref, j, r)
            key = method
            er_x.setdefault(key, []).append(erx)
            er_r.setdefault(key, []).append(err)
            rows.append(
                {
                    "path": str(path),
                    "method": method,
                    "er_x": erx,
                    "er_r": err,
                    "setup_s": res.setup_time_s,
                    "solve_s": res.solve_time_s,
                    **meta,
                }
            )
        if model is not None:
            dx, lat = _eval_jmat(j, r, dx_ref, model, device)
            erx, err = relative_errors(dx, dx_ref, j, r)
            er_x.setdefault("jmat", []).append(erx)
            er_r.setdefault("jmat", []).append(err)
            rows.append(
                {
                    "path": str(path),
                    "method": "jmat",
                    "er_x": erx,
                    "er_r": err,
                    "inference_s": lat,
                    **meta,
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["method"])
        w.writeheader()
        w.writerows(rows)
    summary = {m: summarize_errors(er_x[m], er_r[m]).__dict__ for m in er_x}
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()