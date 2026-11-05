# GENCO Figure 4, Task 1.1

Train and test GENCO Base, PFNet, CANOS-PF, and GNS-S on the IEEE 118-bus Task 1.1 split, then draw the Figure 4 panel from those test scores.

The model code and the data are the authors'. This folder downloads them and runs their Task 1.1 configs.

[Open Task1_1.ipynb in Colab](https://colab.research.google.com/github/alitasavori/genco-figure4-task11/blob/main/Task1_1.ipynb)

Use a GPU runtime. Run the install cell, accept Google Drive, then run the training cell. A finished seed is skipped, so a disconnected session continues on the next run.

GENCO uses torch 2.8. The three baselines are pinned to torch 2.1.2, so the install cell builds two Python 3.11 environments. Epoch caps in the released configs are GENCO 300, PFNet 100, CANOS-PF 200, and GNS-S 25, three seeds each. The pinned PFΔ commit includes the 26 March 2026 CANOS-PF loss correction.

| File | Role |
| --- | --- |
| `Task1_1.ipynb` | Install cell, then train-and-test cell |
| `install_train.py` | Builds the two environments |
| `train_and_test_task11.py` | Trains, tests, and displays the figure |
| `plot_task1_1.py` | Writes `results/figure4_task1.1.pdf` after all four models have three test seeds |

Clones, data, and checkpoints go to Google Drive on Colab (`MyDrive/genco_figure4_task11`), or to `vendor/` outside Colab.
