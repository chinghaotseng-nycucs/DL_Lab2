"""Experiment 13: run 08 + 18 epochs with EMA 0.999.

Why: Weight averaging (EMA) against run-to-run instability. Compare with 12.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9178: smoother, no higher peak (10/8). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp13_geo_from08_ema.py              # --dry-run prints the train.py command only
    nohup python experiments/exp13_geo_from08_ema.py >> runs/exp13.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "13",
    "name": "13_aug-geo_from08_ema",
    "title": "run 08 + 18 epochs with EMA 0.999",
    "init": "08",
    "train_args": ["--aug", "geo", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10", "--ema", "0.999"],
    "chain": ["04", "06", "08", "13"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
