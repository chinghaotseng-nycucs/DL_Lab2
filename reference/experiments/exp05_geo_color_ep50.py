"""Experiment 05: aug geo_color, 50 epochs.

Why: 03 was undertrained at 30 epochs; 04 showed that 50 epochs helps.
Starts from: random weights
Result: val 0.9051 (10/6). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp05_geo_color_ep50.py              # --dry-run prints the train.py command only
    nohup python experiments/exp05_geo_color_ep50.py > runs/exp05.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "05",
    "name": "05_aug-geo_color_ep50",
    "title": "aug geo_color, 50 epochs",
    "train_args": ["--aug", "geo_color", "--epochs", "50"],
    "chain": ["05"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
