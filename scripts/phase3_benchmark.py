#!/usr/bin/env python3
"""Phase 3 §6 — one-pass JMAT vs direct/Krylov at documented precision."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.csr_io import scipy_csr_to_torch  # noqa: E402
from jmat.data.manifest import load_manifest  # noqa: E402
from jmat.data.shard_io import load_triple_npz  # noqa: E402
from jmat.metrics import relative_errors, summarize_errors  # noqa: E402
from jmat.model import JMAT, JMATConfig  # noqa: E402
from jmat.solvers.backends import probe_backends, solve_direct  # noqa: E402
from jmat.solvers.backends import solve_gmres_wrapped  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()
    samples, _ = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jmat = None
    if args.checkpoint and args.checkpoint.is_file():
        jmat = JMAT(JMATConfig()).to(device)
        jmat.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        jmat.eval()
    backends = probe_backends()
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "backends": [b.__dict__ for b in backends],
    }
    rows = []
    for item in samples[: args.limit]:
        path = args.data_root / item["path"]
        j, r, dx_ref, meta = load_triple_npz(path)
        for backend in backends:
            if not backend.available or backend.name == "klu":
                continue
            if backend.name not in ("superlu", "umfpack"):
                continue
            t0 = time.perf_counter()
            res = solve_direct(j, r, backend.name, ref_dx=dx_ref)  # type: ignore[arg-type]
            e2e = time.perf_counter() - t0
            rows.append(
                {
                    "method": res.method,
                    "er_x": res.relative_solution_error,
                    "er_r": res.relative_residual,
                    "e2e_s": e2e,
                    "setup_s": res.setup_time_s,
                    "solve_s": res.solve_time_s,
                    "mode": "direct_matched_accuracy",
                    **meta,
                }
            )
        for use_ilu, tag in ((False, "gmres"), (True, "gmres_ilu")):
            res = solve_gmres_wrapped(j, r, use_ilu=use_ilu, ref_dx=dx_ref)
            rows.append(
                {
                    "method": res.method,
                    "er_x": res.relative_solution_error,
                    "er_r": res.relative_residual,
                    "e2e_s": res.setup_time_s + res.solve_time_s,
                    "setup_s": res.setup_time_s,
                    "solve_s": res.solve_time_s,
                    "mode": "iterative",
                    **meta,
                }
            )
        if jmat is not None:
            crow, col, val, n = scipy_csr_to_torch(j, device=device, dtype=torch.float32)
            rt = torch.tensor(r, dtype=torch.float32, device=device)
            t0 = time.perf_counter()
            with torch.no_grad():
                dx = jmat.forward_csr(crow, col, val, n, rt).cpu().numpy()
            e2e = time.perf_counter() - t0
            erx, err = relative_errors(dx, dx_ref, j, r)
            rows.append(
                {
                    "method": "jmat_one_pass",
                    "er_x": erx,
                    "er_r": err,
                    "e2e_s": e2e,
                    "mode": "learned_approximation",
                    **meta,
                }
            )
    by_method: dict[str, list] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = {}
    for method, items in by_method.items():
        er_r = [x["er_r"] for x in items if x["er_r"] is not None]
        er_x = [x["er_x"] for x in items if x["er_x"] is not None]
        summary[method] = summarize_errors(er_x, er_r).__dict__
        summary[method]["latency_median_s"] = float(np.median([x["e2e_s"] for x in items]))
    out = ROOT / "results" / "jmat" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    payload = {"env": env, "summary": summary, "rows": rows}
    (out / "benchmark_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"env": env, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()