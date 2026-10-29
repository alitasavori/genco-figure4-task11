"""Install the packages needed to show the published Task 1.1 figure.

This does not install GENCO or PFDelta and does not download model weights.
"""

from __future__ import annotations

import sys

from common import run_command, LOGS, ensure_dirs


def main() -> None:
    ensure_dirs()
    run_command(
        [sys.executable, "-m", "pip", "install", "pandas", "matplotlib"],
        cwd=None,
        log_path=LOGS / "pip_view.log",
    )
    print("Installed pandas and matplotlib.")
    print("No model was installed and no training will start.")


if __name__ == "__main__":
    main()
