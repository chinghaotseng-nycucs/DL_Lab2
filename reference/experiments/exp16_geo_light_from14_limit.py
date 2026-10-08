"""Experiment 16: aug geo_light, continued from 14 until the limit.

Why: 14 beat its control at every epoch and was still improving at its last epochs.
Starts from: runs/14_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: not run yet.

    cd Lab2/reference
    python experiments/exp16_geo_light_from14_limit.py              # --dry-run prints the train.py command only
    nohup python experiments/exp16_geo_light_from14_limit.py > runs/exp16.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "16",
    "name": "16_aug-geo_light_from14",
    "title": "aug geo_light, continued from 14 until the limit",
    "init": "14",
    "train_args": ["--aug", "geo_light", "--lr", "1e-4", "--dice-weight", "10", "--epochs", "100", "--sched", "plateau", "--early-stop", "12"],
    "chain": ["04", "06", "08", "14", "16"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
