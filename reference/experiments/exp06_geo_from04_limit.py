"""Experiment 06: aug geo, continued from 04 until the limit.

Why: Find how far geo can go: plateau lr, stop after 12 epochs without a gain.
Starts from: runs/04_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9155, early stop at epoch 41 (10/7). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp06_geo_from04_limit.py              # --dry-run prints the train.py command only
    nohup python experiments/exp06_geo_from04_limit.py > runs/exp06.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "06",
    "name": "06_aug-geo_from04",
    "title": "aug geo, continued from 04 until the limit",
    "init": "04",
    "train_args": ["--aug", "geo", "--lr", "3e-4", "--epochs", "100", "--sched", "plateau", "--early-stop", "12"],
    "chain": ["04", "06"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
