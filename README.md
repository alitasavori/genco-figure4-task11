# Reproduce GENCO Figure 4, Task 1.1

This folder reproduces the IEEE 118-bus Task 1.1 panel of Figure 4 from
[GENCO](https://arxiv.org/abs/2608.09921) on the PFΔ benchmark. The notebooks
only install dependencies and start a script. Training, evaluation, checks,
and plotting live in the Python files.

## Open in Colab

Open this notebook on a GPU runtime. The first cell runs `install_train.py`, which builds two Python 3.11 environments because the codebases pin different PyTorch versions. The second cell runs `train_and_test_task11.py` with `MODE = "full"`, which trains three seeds of GENCO Base, PFNet, CANOS-PF, and GNS-S, tests them, and displays the measurements.

- [Task1_1.ipynb](https://colab.research.google.com/github/alitasavori/genco-figure4-task11/blob/main/Task1_1.ipynb)

Or clone it yourself:

```bash
git clone https://github.com/alitasavori/genco-figure4-task11.git
cd genco-figure4-task11
```

## What each result means

| `source` / `measured_source` | Meaning |
| --- | --- |
| `published_figure4_table` | Rounded mean and standard deviation transcribed from the authors' Figure 4 table. Not a new measurement. |
| `released_checkpoint` | Evaluation of the three GENCO weights released on Hugging Face. |
| `independent_training` | A model trained in this workspace. The retrained figure uses only these rows, and only when all four models have three finite seeds. |

The published figure is `results/figure4_task1.1_published.pdf`. The retrained
figure is written only after independent training finishes for GENCO Base,
CANOS-PF, GNS-S, and PFNet. If it is missing, `results/figure4_task1.1_retrained_NOT_RUN.txt`
says which models are absent. Paper numbers are not copied into that figure.

## Two environments, one notebook

PFΔ pins `torch==2.1.2`. The GENCO graphkit commit pins `torch>=2.7.1,<2.9`.
`install_train.py` puts each pin in its own Python 3.11 environment. `train_and_test_task11.py` runs GENCO in the first environment and the three baselines in the second.

`Task1_1.ipynb` is the notebook. Its install cell runs `install_train.py`. Its second cell runs `train_and_test_task11.py` with `MODE = "full"`, which trains three seeds, tests them, and writes the figure. Finished seeds are skipped.

## Execution sequence

1. Open `Task1_1.ipynb` on a GPU runtime.
2. Run the install cell. Accept the Google Drive prompt so checkpoints survive a disconnect.
3. Run the training cell. Leave `MODE = "full"`.
4. If Colab disconnects, run both cells again. Completed seeds are not repeated.

Command-line equivalents, from this directory:

```bash
python install_train.py
python train_and_test_task11.py --mode full
python plot_task1_1.py
python -m unittest tests.test_task11
```

`GENCO_WORK_ROOT` overrides the directory used for clones, data, and checkpoints.
On Colab the default is Google Drive when Drive mounts, otherwise `/content/genco_figure4_task11`.
Outside Colab the default is `genco/vendor`.

## Pinned commits

Checked out by the install scripts:

| Repository | Branch | Commit |
| --- | --- | --- |
| [gridfm/gridfm-paper-genco](https://github.com/gridfm/gridfm-paper-genco) | `main` | `107ae0be34660fb7de5e629fa74c60ee63afad2e` |
| [gridfm/gridfm-graphkit](https://github.com/gridfm/gridfm-graphkit) | `genco-paper-repro-pfdelta` | `310e2f989b46058f3e6c5b4a56b7d03de06a3340` |
| [MOSSLab-MIT/pfdelta](https://github.com/MOSSLab-MIT/pfdelta) | `main` | `0fb3aeda2bbca807418a49cc934e4640f988964a` |

GENCO data: [gridfm/pfdelta_task1.1](https://huggingface.co/datasets/gridfm/pfdelta_task1.1).
GENCO weights: [gridfm/genco-pfdelta](https://huggingface.co/gridfm/genco-pfdelta), `task_1.1/seed{1,2,3}/last.pt`.
The official YAML seeds are 0, 42, and 1234 for seed files 1, 2, and 3. Those values are not changed.

PFΔ training uses the JSON dataset downloaded by that repository, not the converted parquet.
The released configs are `pfnet_task_1_1`, `canos_task_1_1`, and `gns_task_1_1`.
Each file sets `functional.seed` to 11. The other two seeds used for the paper are not in the repository.
Full mode therefore trains seeds 11, 21, and 31, and changes no other hyperparameter.
Seed-specific copies live next to the released YAML so the originals stay intact.
Those copies set `functional.checkpoint: true`. The upstream trainer can resume from `checkpoint.pt`, and it rewrites the seed-specific copy when it does.

## CANOS-PF loss correction

The pinned PFΔ README records a 26 March 2026 bug in `constraint_violations_loss_pf`: reactive-power mismatch and line-branch-flow mismatch were calculated incorrectly, and the bug is fixed in this commit. Full mode uses the corrected loss. Figure 4 reused the older published baseline numbers, so a retrained CANOS-PF run can disagree with the published bars. The fix is not reverted. `results/provenance.json` records this.

## Metric

Mean power-balance loss on a 100 MVA base:

`PBL_bus = sqrt((ΔP)^2 + (ΔQ)^2)`, then the mean over buses and over samples in the category.

N, N−1, and N−2 are the feasible test sets. C2I is the close-to-infeasible (nose) test set.
PFΔ's `scripts/test_error.py` averages the three nose topology means. Those topologies each contain 200 official test samples, so that average matches a sample-weighted mean. This code uses the sample-weighted mean when the evaluated counts differ, and it leaves C2I empty rather than inserting the paper value.

The error bar is the unbiased sample standard deviation across the three run-level means (`numpy.std(..., ddof=1)`), the same convention as `torch.Tensor.std` in `scripts/test_error.py`.

`scripts/test_error.py` sums per-batch mean losses. Its batch size is 2000, which covers each official Task 1.1 test category in one batch. Do not lower that batch size inside the upstream script; the sum would no longer be the dataset mean.

GENCO evaluation is `gridfm_graphkit evaluate`. The test CSV labels the residual `PBE (Mean, MVA)`. The converted PFΔ parquet is already per unit, and the normalizer divides and multiplies by the same base, so the stored number is the per-unit residual used for the comparison. The raw value is kept. It is not replaced by the table when the two differ.

The published bars use the rounded LaTeX table parsed by `make_pbl_barplots.py`, not the longer GENCO floats. Those floats are in `results/genco_task1.1_metrics_summary_unrounded.csv`. For example the table prints GENCO N as `1.4e-3 ± 0.4e-3`, while `metrics_summary.csv` has `0.00143654 ± 0.00038566`.

## Data split

Task 1.1 trains on normal-topology feasible samples only. The PFΔ loader's training pool is 54,000 samples, then a 90/10 split: 48,600 train and 5,400 validation. The official test split has 2,000 feasible and 200 close-to-infeasible samples for each of N, N−1, and N−2, 6,600 samples in total. The reproduction logs the counts actually loaded. A mismatch is reported; it is not overwritten with these expected numbers.

N−1, N−2, and C2I samples are not used for training or for choosing hyperparameters. The released training configs contain train and validation only. Test categories are passed in only by `scripts/test_error.py` after training.

## Compatibility changes

Recorded under `results/logs/` when they happen:

- GENCO `data.workers` is set to 2 on Colab, or to `GENCO_DATA_WORKERS`, in a copy under `results/configs/`. The clone's YAML is unchanged.
- torch is installed only inside the range the selected repository declares.
- PFΔ checkpointing is enabled on the seed-specific config copies.
- GENCO resume calls the same Lightning trainer as the released CLI and passes `ckpt_path`. The released CLI has no resume argument. Final weights are exported from Lightning `last.ckpt` to `last.pt` because `gridfm_graphkit evaluate` loads a state dict. The paper's Task 1.1 checkpoints are the final `last.pt` weights. The upstream callback also saves best-validation snapshots; those are not substituted for `last.pt`.

## Compute

Quick GENCO evaluation needs the converted dataset (about 690 MB), three checkpoints (about 80 MB each), and graphkit's preprocessing of every split. Use a GPU. A Colab T4 is a reasonable size for evaluation.

Full training is much longer. GENCO uses 300 epochs, batch size 64, on the 48,600-sample training split. PFNet uses 100 epochs, CANOS-PF 200 epochs, and GNS-S 25 epochs, each for three seeds. Plan on more than one Colab session. Checkpoint files let a rerun continue. Disk use is dominated by the PFΔ JSON archive plus the processed `.pt` graphs; several tens of gigabytes is a practical allowance.

## Outputs

- `results/task1_1_seed_metrics.csv` — one row per model, category, seed, and source
- `results/task1_1_summary.csv` — mean, sample standard deviation, published value, absolute difference, and relative difference
- `results/figure4_task1.1_published.pdf` and `.png`
- `results/figure4_task1.1_retrained.pdf` and `.png` after all four models have three trained seeds
- `results/provenance.json` — commits, the CANOS correction, and the metric definition
- `results/logs/` — install logs, training logs, and split counts

## Tests

`python -m unittest tests.test_task11` checks the published-table parser, the unbiased standard deviation, power-balance loss on a solved line, inclusion of the reference bus, rejection of NaNs, the 54,000 / 6,600 split counts, train/test id overlap, Task 1.1 directory selection, and that the retrained figure is not created from the published table.
