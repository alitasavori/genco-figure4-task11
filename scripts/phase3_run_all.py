#!/usr/bin/env python3
"""Orchestrate Phase 3 deliverables (--pilot for Colab / smoke)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    py = Path(cmd[0]).name if cmd else ""
    if py.endswith("python") or "python" in py:
        cmd = [cmd[0], "-u", *cmd[1:]]
    if len(cmd) > 2:
        script = Path(cmd[2] if cmd[1] == "-u" else cmd[1])
        if script.suffix == ".py" and not (ROOT / script).is_file():
            raise FileNotFoundError(
                f"Missing {ROOT / script}. Re-run notebook cell 1 (git pull) — Phase 3 must be on origin/main."
            )
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def _pfdelta_has_samples(root: Path) -> bool:
    man = root / "manifest.json"
    if not man.is_file():
        return False
    import json

    raw = json.loads(man.read_text(encoding="utf-8"))
    samples = raw.get("samples", raw if isinstance(raw, list) else [])
    return bool(samples)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true", help="Small data + short training (Colab-friendly)")
    p.add_argument("--vendor-root", type=Path, default=None, help="PFΔ raw JSON vendor tree")
    p.add_argument("--skip-pfdelta", action="store_true")
    p.add_argument("--skip-ablation", action="store_true")
    args = p.parse_args()
    py = sys.executable
    pilot = ["--pilot"] if args.pilot else []
    pfdelta_root = ROOT / "results" / "jmat" / "pfdelta_pilot"
    print("JMAT Phase 3 run_all starting", flush=True)
    print(f"  pilot={args.pilot} skip_pfdelta={args.skip_pfdelta} vendor={args.vendor_root}", flush=True)

    run([py, "scripts/phase3_verify.py"])
    run([py, "scripts/phase3_synthetic.py", *pilot, "--ablation", "jmat_full"])

    do_pfdelta = not args.skip_pfdelta
    if do_pfdelta:
        pf_cmd = [py, "scripts/phase3_pfdelta_pilot.py", *pilot]
        if args.vendor_root:
            pf_cmd.extend(["--vendor-root", str(args.vendor_root)])
        run(pf_cmd)
        if not _pfdelta_has_samples(pfdelta_root):
            print("No PFΔ samples found — skipping PFΔ train/benchmark holdout (set --vendor-root).", flush=True)
            do_pfdelta = False
    if do_pfdelta and _pfdelta_has_samples(pfdelta_root):
        run([py, "scripts/phase3_train_pfdelta.py", "--mode", "case118", *pilot])
        run([py, "scripts/phase3_train_pfdelta.py", "--mode", "joint", *pilot])
    elif do_pfdelta:
        do_pfdelta = False

    ckpt = ROOT / "results" / "jmat" / "phase3" / ("pfdelta_joint.pt" if do_pfdelta else "synthetic_jmat_full.pt")
    if not ckpt.is_file():
        ckpt = ROOT / "results" / "jmat" / "phase3" / "synthetic_jmat_full.pt"
    man = ROOT / "results" / "jmat" / "pfdelta_pilot" / "manifest.json"
    data = ROOT / "results" / "jmat" / "pfdelta_pilot"
    if not man.is_file() or not _pfdelta_has_samples(data):
        man = ROOT / "results" / "jmat" / "synthetic" / "manifest.json"
        data = ROOT / "results" / "jmat" / "synthetic"
        print(f"benchmark: using synthetic manifest ({man})", flush=True)
    run(
        [
            py,
            "scripts/phase3_benchmark.py",
            "--manifest",
            str(man),
            "--data-root",
            str(data),
            "--checkpoint",
            str(ckpt),
            "--limit",
            "20" if args.pilot else "80",
        ]
    )
    if not args.skip_ablation and not args.pilot:
        run([py, "scripts/phase3_ablation.py"])
    elif args.pilot and not args.skip_ablation:
        print("Pilot mode: skipping 6× ablation train (use --no-skip-ablation to enable).", flush=True)

    if ckpt.is_file():
        run([py, "scripts/phase3_zeroshot.py", "--checkpoint", str(ckpt), *([] if not do_pfdelta else ["--suitesparse"])])


if __name__ == "__main__":
    main()