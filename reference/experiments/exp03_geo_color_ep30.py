"""Experiment 03: aug geo_color, 30 epochs.

Why: 02 plus colour jitter, grayscale and blur, for other cameras and lighting.
Starts from: random weights
Result: val 0.8886 (10/6). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp03_geo_color_ep30.py              # --dry-run prints the train.py command only
    nohup python experiments/exp03_geo_color_ep30.py > runs/exp03.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "03",
    "name": "03_aug-geo_color_ep30",
    "title": "aug geo_color, 30 epochs",
    "train_args": ["--aug", "geo_color", "--epochs", "30"],
    "chain": ["03"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
