#!/usr/bin/env python3
"""Phase 3 §5 — train all six ablations on the same synthetic manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jmat.ablation import ABLATION_ORDER  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    results = {}
    for name in ABLATION_ORDER:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "phase3_synthetic.py"),
            "--ablation",
            name,
            "--epochs",
            str(args.epochs),
        ]
        if args.pilot:
            cmd.append("--pilot")
        subprocess.check_call(cmd, cwd=ROOT)
        report = json.loads((ROOT / "results" / "jmat" / "phase3" / f"synthetic_{name}_report.json").read_text())
        results[name] = report
    out = ROOT / "results" / "jmat" / "phase3" / "ablation_summary.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()