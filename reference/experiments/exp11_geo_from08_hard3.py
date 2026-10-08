"""Experiment 11: run 08 + 18 epochs, hard training images 3x.

Why: draw the training images run 08 still gets wrong more often (hard-example mining). Compare with 12.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9185 as a one-epoch spike; no reliable gain over 12 (10/8). Rerunning makes a new folder; the old one is kept.

Before training, prepare_hard_weights scores every training image with run 08 (score_train.py) and writes
review/sample_weights_auto.csv: train Dice 0.3-0.8 -> weight 3. Below 0.3 stays at 1, because such a low score
most often means a bad label (e.g. boxer_150), and training harder on a wrong label teaches the wrong answer.

    cd Lab2/reference
    python experiments/exp11_geo_from08_hard3.py              # --dry-run prints the train.py command only
    nohup python experiments/exp11_geo_from08_hard3.py > runs/exp11.log 2>&1 &     # on a server
"""

import csv
import os

from _pipeline import REF, RUNS, py, run


def prepare_hard_weights(parent_dir):
    py("score_train.py", "--weights", os.path.join(parent_dir, "best.pth"), "--worst", "300")
    with open(os.path.join(RUNS, "train_scores.csv")) as f:
        rows = list(csv.DictReader(f))
    hard = [r["image_id"] for r in rows if 0.3 <= float(r["dice"]) < 0.8]
    os.makedirs(os.path.join(REF, "review"), exist_ok=True)
    with open(os.path.join(REF, "review", "sample_weights_auto.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "weight"])
        w.writerows([i, 3] for i in hard)
    print(f"hard examples (train Dice 0.3-0.8): {len(hard)} weighted 3x", flush=True)


EXPERIMENT = {
    "id": "11",
    "name": "11_aug-geo_from08_hard3",
    "title": "run 08 + 18 epochs, hard training images 3x",
    "init": "08",
    "prep": prepare_hard_weights,
    "train_args": ["--aug", "geo", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10",
                   "--sample-weights", "review/sample_weights_auto.csv"],
    "chain": ["04", "06", "08", "11"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
