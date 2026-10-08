"""Experiment 15: run 08 + 18 epochs, aug geo_erase.

Why: geo + random erasing on the image only, against texture reliance and occlusion. Compare with 12.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9178: no reliable gain (10/8). Rerunning makes a new folder; the old one is kept.

    cd Lab2/reference
    python experiments/exp15_geo_erase_from08.py              # --dry-run prints the train.py command only
    nohup python experiments/exp15_geo_erase_from08.py >> runs/exp15.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "15",
    "name": "15_aug-geo_erase_from08",
    "title": "run 08 + 18 epochs, aug geo_erase",
    "init": "08",
    "train_args": ["--aug", "geo_erase", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10"],
    "chain": ["04", "06", "08", "15"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
