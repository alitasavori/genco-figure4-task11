"""Install the PFDelta baseline stack used for Task 1.1.

Keep this runtime separate from the GENCO notebook. This repository pins
torch==2.1.2 in requirements.txt. The installer does not replace that pin
with a newer torch if the pin fails to see the GPU.
"""

from __future__ import annotations

import sys

from common import LOGS, clone_repo, ensure_dirs, in_colab, report_environment, run_command, work_root


def main() -> None:
    ensure_dirs()
    info = report_environment()
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 12):
        print(
            "PFDelta pins torch==2.1.2, which does not publish cp312 wheels. "
            f"This runtime is Python {major}.{minor}. The installer will still try the "
            "pinned requirements and will stop if that pin cannot be installed. "
            "It will not upgrade torch to make the install succeed."
        )
    if in_colab():
        try:
            from google.colab import drive

            drive.mount("/content/drive")
        except Exception as exc:
            print(f"Google Drive mount skipped: {exc}")
    run_command(
        [sys.executable, "-m", "pip", "install", "pandas", "matplotlib", "seaborn", "pyyaml", "numpy"],
        cwd=None,
        log_path=LOGS / "pip_pfdelta_plot.log",
    )
    root = work_root()
    repo = root / "pfdelta"
    clone_repo("pfdelta", repo)
    print("Installing pinned PFDelta requirements. This replaces the current torch with 2.1.2.")
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=repo,
        log_path=LOGS / "pip_pfdelta_requirements.log",
    )
    report_environment()
    (LOGS / "pfdelta_compatibility.txt").write_text(
        "\n".join(
            [
                "Pinned commit includes the 2026-03-26 CANOS-PF loss correction.",
                "constraint_violations_loss_pf was fixed upstream. This install does not revert it.",
                "Published Figure 4 baseline bars predate that correction and can disagree with a retraining.",
                "requirements.txt is installed as released, including torch==2.1.2.",
                "Seed-specific YAML copies are written under core/configs by the reproduction script.",
                "Those copies set functional.checkpoint=true so an interrupted run can resume.",
                "The released canos_task_1_1.yaml, gns_task_1_1.yaml, and pfnet_task_1_1.yaml are not edited.",
                "Checkpointing rewrites the seed-specific copy. That is upstream trainer behavior.",
                f"Python at install: {info['python']}",
                f"Work directory: {root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"PFDelta install finished. Work directory: {root}")


if __name__ == "__main__":
    main()
