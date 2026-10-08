"""Experiment 01: aug none, 30 epochs (baseline).

Why: Baseline: photos only squashed to 256 x 256, no augmentation. The number to beat.
Starts from: random weights
Result: val 0.8950, Kaggle 0.88986 (10/5). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp01_none_ep30.py              # --dry-run prints the train.py command only
    nohup python experiments/exp01_none_ep30.py > runs/exp01.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "01",
    "name": "01_aug-none_ep30",
    "title": "aug none, 30 epochs (baseline)",
    "train_args": ["--aug", "none", "--epochs", "30"],
    "chain": ["01"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
