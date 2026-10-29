"""Train and test GENCO Base, PFNet, CANOS-PF, and GNS-S for Task 1.1.

full mode trains three seeds of each model and evaluates those weights.
Finished seeds are skipped, so an interrupted Colab run continues on the next
execution. The retrained figure is written only after all four models have
three finite test results. Published table numbers are not copied into it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from storage import ROOT, mount_and_select


def interpreters() -> tuple[Path, Path]:
    from common import LOGS

    manifest = LOGS / "venvs.json"
    if not manifest.exists():
        raise SystemExit("No training environments are recorded. Run the install cell first.")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    genco = Path(payload["genco"])
    pfdelta = Path(payload["pfdelta"])
    missing = [str(path) for path in (genco, pfdelta) if not path.exists()]
    if missing:
        raise SystemExit(
            "The training environments are not on this runtime: "
            + ", ".join(missing)
            + ". Run the install cell again. Checkpoints already saved on Drive are kept."
        )
    return genco, pfdelta


def run_stack(python: Path, script: str, mode: str, log_name: str) -> None:
    from common import LOGS, run_command

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    run_command(
        [str(python), str(ROOT / script), "--mode", mode],
        cwd=ROOT,
        log_path=LOGS / log_name,
        env=env,
    )


def show_results() -> None:
    from common import resolve_results

    results = resolve_results()
    summary_path = results / "task1_1_summary.csv"
    retrained = results / "figure4_task1.1_retrained.png"
    note = results / "figure4_task1.1_retrained_NOT_RUN.txt"
    print(f"Summary file: {summary_path}")
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    if retrained.exists():
        print(f"Retrained figure: {retrained}")
    elif note.exists():
        print(note.read_text(encoding="utf-8"))
    try:
        from IPython import get_ipython
        from IPython.display import Image, Markdown, display
        import pandas as pd
    except Exception:
        return
    if get_ipython() is None or not summary_path.exists():
        return
    frame = pd.read_csv(summary_path)
    trained = frame[frame["measured_source"] == "independent_training"] if "measured_source" in frame.columns else frame.iloc[0:0]
    if retrained.exists():
        display(Markdown("**Test results after training.** Each bar is the mean power-balance loss of three trained seeds."))
        display(trained if not trained.empty else frame)
        display(Image(filename=str(retrained)))
        return
    display(Markdown(
        "**Training has not produced a full four-model figure yet.** "
        "The table below is whatever has been written. Published rows were copied from the paper. "
        "Rows marked `independent_training` are measurements from this run."
    ))
    display(frame)
    if not trained.empty:
        display(Markdown("Completed independent measurements so far:"))
        display(trained)
    if note.exists():
        display(Markdown("```\n" + note.read_text(encoding="utf-8") + "\n```"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test the four Task 1.1 models.")
    parser.add_argument("--mode", choices=("published", "quick", "full"), default="full")
    args = parser.parse_args()
    mount_and_select()
    from common import use_selected_results

    use_selected_results()
    if args.mode != "full":
        print(f"MODE is {args.mode}. Set MODE = \"full\" to train and test the four models.")
    else:
        print("Training and testing all four models.")
        print("Order: GENCO Base (seeds 1, 2, 3), then PFNet, CANOS-PF, and GNS-S (seeds 11, 21, 31).")
        print("GENCO trains for 300 epochs, PFNet for 100, CANOS-PF for 200, and GNS-S for 25.")
        print("A Colab session can end before that finishes. Run both cells again to resume.")
        print("A seed that already has its checkpoint and test metrics is skipped.")
    genco, pfdelta = interpreters()
    print(f"GENCO interpreter: {genco}")
    run_stack(genco, "reproduce_genco.py", args.mode, "train_genco.log")
    print(f"PFDelta interpreter: {pfdelta}")
    run_stack(pfdelta, "reproduce_pfdelta.py", args.mode, "train_pfdelta.log")
    show_results()


if __name__ == "__main__":
    main()
