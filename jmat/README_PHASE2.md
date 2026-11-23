# JMAT Phase 2

## Layout

| Component | Path |
|-----------|------|
| Safeguards & complexity | `jmat/safeguards.py`, `jmat/complexity.py` |
| Synthetic benchmark | `jmat/data/synthetic.py`, `scripts/prepare_synthetic_jmat.py` |
| PFΔ Newton traces | `jmat/pf/`, `scripts/generate_pfdelta_newton_traces.py` |
| Linear baselines | `jmat/solvers/linear.py`, `scripts/evaluate_jmat_baselines.py` |
| Training / ablations | `scripts/train_jmat_phase2.py` |

## Splits (Task 3)

- **Train grids:** IEEE 14/30/118, GOC 500 (`jmat/data/splits.py`)
- **Zero-shot holdout:** IEEE 57, GOC 2000 — do not train or tune on these buckets

## Commands

```bash
# Task 2 — synthetic shards
python scripts/prepare_synthetic_jmat.py --out results/jmat/synthetic

# Task 3 — PFΔ (requires extracted raw JSON under vendor/pfdelta)
python scripts/generate_pfdelta_newton_traces.py --raw-dir <path/to/raw> --grid-id case118

# Task 4/6 — baselines (+ optional JMAT checkpoint)
python scripts/evaluate_jmat_baselines.py --manifest results/jmat/synthetic/manifest.json --data-root results/jmat/synthetic

# Task 5 — train (ablation flags)
python scripts/train_jmat_phase2.py --manifest results/jmat/synthetic/manifest.json --data-root results/jmat/synthetic --no-at
```