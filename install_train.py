"""Install both Task 1.1 training stacks into separate Python 3.11 environments.

GENCO's graphkit commit requires torch>=2.7.1,<2.9. PFDelta pins torch==2.1.2,
and that pin has no Python 3.12 wheels. One Colab runtime cannot import both,
so this script builds two environments and leaves the notebook kernel alone
except for the packages used to display the final table.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from storage import ROOT, in_colab, mount_and_select


def venv_root() -> Path:
    if in_colab():
        path = Path("/content/task11-venvs")
    else:
        path = ROOT / ".venvs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def interpreter(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def ensure_uv() -> None:
    from common import LOGS, run_command

    run_command(
        [sys.executable, "-m", "pip", "install", "uv"],
        cwd=ROOT,
        log_path=LOGS / "pip_uv.log",
    )


def create_venv(name: str) -> Path:
    from common import LOGS, run_command

    venv = venv_root() / name
    python = interpreter(venv)
    if python.exists():
        print(f"Reusing {python}")
        return python
    if venv.exists():
        print(f"Removing incomplete environment {venv}")
        import shutil

        shutil.rmtree(venv)
    run_command(
        [sys.executable, "-m", "uv", "venv", "--python", "3.11", "--seed", str(venv)],
        cwd=ROOT,
        log_path=LOGS / f"uv_venv_{name}.log",
    )
    if not python.exists():
        raise SystemExit(f"Python 3.11 environment was not created at {python}")
    return python


def run_installer(python: Path, script: str) -> None:
    from common import LOGS, run_command

    marker = python.parent.parent / ".install_complete"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == script:
        print(f"Skipping {script}. This environment is already installed.")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    run_command(
        [str(python), str(ROOT / script)],
        cwd=ROOT,
        log_path=LOGS / f"{script}.log",
        env=env,
    )
    marker.write_text(script + "\n", encoding="utf-8")


def main() -> None:
    print("Installing the two environments used to train and test all four models.")
    print("GENCO Base uses torch 2.8. PFNet, CANOS-PF, and GNS-S use torch 2.1.2.")
    mount_and_select()
    from common import use_selected_results

    use_selected_results()
    from common import LOGS, run_command

    ensure_uv()
    run_command(
        [sys.executable, "-m", "uv", "python", "install", "3.11"],
        cwd=ROOT,
        log_path=LOGS / "uv_python_311.log",
    )
    run_command(
        [sys.executable, "-m", "pip", "install", "pandas", "matplotlib"],
        cwd=ROOT,
        log_path=LOGS / "pip_notebook_display.log",
    )
    genco = create_venv("genco")
    pfdelta = create_venv("pfdelta")
    (LOGS / "venvs.json").write_text(
        json.dumps({"genco": str(genco), "pfdelta": str(pfdelta)}, indent=2),
        encoding="utf-8",
    )
    print("Installing GENCO graphkit into the first environment.")
    run_installer(genco, "install_genco.py")
    print("Installing PFDelta into the second environment.")
    run_installer(pfdelta, "install_pfdelta.py")
    print("Install finished. Run the next cell to train and test.")
    print(f"GENCO Python: {genco}")
    print(f"PFDelta Python: {pfdelta}")


if __name__ == "__main__":
    main()
