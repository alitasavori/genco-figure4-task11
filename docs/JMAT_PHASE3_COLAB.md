# JMAT Phase 3 — Colab

## Cell 1 (notebook top only)

Use **`Task1_1.ipynb` cell 1**: “Colab preparation (one pull for all cells below)”.  
That is the **only** git clone / pull cell for the whole notebook (TPGT, JMAT, figures).

## Cell 2 — Phase 3 everything

Run the **last** notebook cell (“JMAT Phase 3 — one cell”) or:

```python
# After prep cell; same logic as Task1_1.ipynb
import os, subprocess, sys
from pathlib import Path

REPO = Path("/content/genco-figure4-task11")
cwd = REPO
os.environ.setdefault("JMAT_VENDOR_ROOT", "/content/drive/MyDrive/Foundation Model/vendor")

cmd = [sys.executable, "scripts/phase3_run_all.py", "--pilot"]
vendor = Path(os.environ["JMAT_VENDOR_ROOT"])
if vendor.is_dir():
    cmd += ["--vendor-root", str(vendor)]
else:
    cmd += ["--skip-pfdelta"]

subprocess.check_call(cmd, cwd=cwd)
```

Set `FULL_RUN = True` in the notebook cell (and drop `--pilot`) for full 500/grid training.

PFΔ dataset + CSV reports (default on Drive in notebook):

`My Drive/Foundation Model/results/jmat/pfdelta_pilot/reports/`

| CSV | Meaning |
|-----|---------|
| `grid_summary.csv` | Per grid: requested OCs, saved shards, failures |
| `oc_summary.csv` | Per PFΔ scenario: Newton iters, saved iters, converged |
| `iteration_log.csv` | Per saved Newton step: iter, nnz, NPZ path |
| `training_triples_index.csv` | Flat index of all training shards |
| `generation_run.csv` | One-line run metadata |

Training matrices stay in `.npz` (sparse Jacobians); use CSVs to interpret coverage.

Dataset-only generation:

```bash
python scripts/phase3_pfdelta_pilot.py \
  --vendor-root "/content/drive/MyDrive/Foundation Model/vendor/pfdelta_task1.1" \
  --num-oc 500 \
  --out "/content/drive/MyDrive/Foundation Model/results/jmat/pfdelta_pilot"
```