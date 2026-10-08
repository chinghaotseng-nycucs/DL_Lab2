"""Experiment 12: run 08 + 18 epochs, nothing else changed (control for 11, 13, 14, 15).

Why: Control for 11, 13, 14 and 15: what 18 more epochs give by themselves.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9175, no gain over 08 (10/7). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp12_geo_from08_control.py              # --dry-run prints the train.py command only
    nohup python experiments/exp12_geo_from08_control.py >> runs/exp12.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "12",
    "name": "12_aug-geo_from08_control",
    "title": "run 08 + 18 epochs, nothing else changed (control for 11, 13, 14, 15)",
    "init": "08",
    "train_args": ["--aug", "geo", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10"],
    "chain": ["04", "06", "08", "12"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
