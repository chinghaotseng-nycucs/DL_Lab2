"""Experiment 04: aug geo, 50 epochs.

Why: 02 was still improving at epoch 30.
Starts from: random weights
Result: val 0.9113 (10/6). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp04_geo_ep50.py              # --dry-run prints the train.py command only
    nohup python experiments/exp04_geo_ep50.py >> runs/exp04.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "04",
    "name": "04_aug-geo_ep50",
    "title": "aug geo, 50 epochs",
    "train_args": ["--aug", "geo", "--epochs", "50"],
    "chain": ["04"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
