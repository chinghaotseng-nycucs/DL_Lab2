"""Experiment 14: run 08 + 18 epochs, aug geo_light.

Why: geo + brightness/contrast only, for backlit and dark photos (hard_cases/). Compare with 12.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9193, ahead of 12 at all 18 epochs (10/8). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp14_geo_light_from08.py              # --dry-run prints the train.py command only
    nohup python experiments/exp14_geo_light_from08.py > runs/exp14.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "14",
    "name": "14_aug-geo_light_from08",
    "title": "run 08 + 18 epochs, aug geo_light",
    "init": "08",
    "train_args": ["--aug", "geo_light", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10"],
    "chain": ["04", "06", "08", "14"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
