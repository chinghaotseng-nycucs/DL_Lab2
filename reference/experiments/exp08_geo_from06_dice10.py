"""Experiment 08: run 06 fine-tuned 10 epochs, dice weight 10.

Why: Does a larger soft-Dice weight help? lr = 06's final lr. Compare with 09 (the control).
Starts from: runs/06_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9175, ahead of 09 at all 10 epochs (10/7). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp08_geo_from06_dice10.py              # --dry-run prints the train.py command only
    nohup python experiments/exp08_geo_from06_dice10.py >> runs/exp08.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "08",
    "name": "08_aug-geo_from06_dice10",
    "title": "run 06 fine-tuned 10 epochs, dice weight 10",
    "init": "06",
    "train_args": ["--aug", "geo", "--epochs", "10", "--lr", "3.75e-5", "--dice-weight", "10"],
    "chain": ["04", "06", "08"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
