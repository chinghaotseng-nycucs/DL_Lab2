"""Experiment 07: aug geo_color, continued from 05 until the limit.

Why: The same as 06 for geo_color.
Starts from: runs/05_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9117, early stop at epoch 42 (10/7). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp07_geo_color_from05_limit.py              # --dry-run prints the train.py command only
    nohup python experiments/exp07_geo_color_from05_limit.py > runs/exp07.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "07",
    "name": "07_aug-geo_color_from05",
    "title": "aug geo_color, continued from 05 until the limit",
    "init": "05",
    "train_args": ["--aug", "geo_color", "--lr", "3e-4", "--epochs", "100", "--sched", "plateau", "--early-stop", "12"],
    "chain": ["05", "07"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
