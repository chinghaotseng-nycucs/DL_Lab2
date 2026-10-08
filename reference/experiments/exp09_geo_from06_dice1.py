"""Experiment 09: run 06 fine-tuned 10 epochs, dice weight 1 (control for 08).

Why: Control for 08: the same fine-tune with the normal loss.
Starts from: runs/06_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9163 (10/7). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp09_geo_from06_dice1.py              # --dry-run prints the train.py command only
    nohup python experiments/exp09_geo_from06_dice1.py > runs/exp09.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "09",
    "name": "09_aug-geo_from06_dice1",
    "title": "run 06 fine-tuned 10 epochs, dice weight 1 (control for 08)",
    "init": "06",
    "train_args": ["--aug", "geo", "--epochs", "10", "--lr", "3.75e-5", "--dice-weight", "1"],
    "chain": ["04", "06", "09"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
