"""Experiment 02: aug geo, 30 epochs.

Why: Geometric augmentation (random crop, flip, rotation) against memorising the training photos.
Starts from: random weights
Result: val 0.8981 (10/6). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp02_geo_ep30.py              # --dry-run prints the train.py command only
    nohup python experiments/exp02_geo_ep30.py >> runs/exp02.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "02",
    "name": "02_aug-geo_ep30",
    "title": "aug geo, 30 epochs",
    "train_args": ["--aug", "geo", "--epochs", "30"],
    "chain": ["02"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
