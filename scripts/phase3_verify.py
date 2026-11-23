#!/usr/bin/env python3
"""Phase 3 §1 — verification report (Jacobian, backends, pytest)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("phase3_verify: pytest + Jacobian + backends ...", flush=True)
    out = ROOT / "results" / "jmat" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    pytest = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_jmat.py",
            "tests/test_jmat_phase2.py",
            "tests/test_jmat_newton.py",
            "-q",
            "--tb=line",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    sys.path.insert(0, str(ROOT))
    from jmat.solvers.backends import probe_backends  # noqa: E402
    from jmat.verify.jacobian import verify_toy3bus  # noqa: E402

    jac = verify_toy3bus()
    backends = [b.__dict__ for b in probe_backends()]
    report = {
        "jacobian_fd_toy3bus": jac.__dict__,
        "linear_backends": backends,
        "pytest_exit_code": pytest.returncode,
        "pytest_stdout": pytest.stdout,
        "pytest_stderr": pytest.stderr,
    }
    path = out / "verification_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
    if pytest.returncode != 0:
        print("pytest reported failures (see report); continuing Phase 3 pipeline.", flush=True)
    raise SystemExit(0)


if __name__ == "__main__":
    main()