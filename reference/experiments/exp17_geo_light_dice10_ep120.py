"""Experiment 17: aug geo_light + dice weight 10 from scratch, one cosine schedule, 120 epochs.

Why: a one-command recipe for the best settings so far. The best model (run 14) came from a 4-step chain
(04 -> 06 -> 08 -> 14, about 120 epochs with 4 lr restarts) that used geo_light and dice weight 10 only near the end.
Here both are on from the first epoch, with a single smooth schedule of about the same total length.
Compare with 18d (the same chain rebuilt on the same machine), not with run 14 from the Mac.
Starts from: random weights
Result: val 0.9223 (dog 0.9181), best so far; beats run 14 on clean val (+0.0030, CI +0.0004 to +0.0057) and stress val (+0.0038) (10/8, RTX 4090, 69 min).

    cd Lab2/reference
    python experiments/exp17_geo_light_dice10_ep120.py              # --dry-run prints the train.py command only
    nohup python experiments/exp17_geo_light_dice10_ep120.py >> runs/exp17.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "17",
    "name": "17_aug-geo_light_dice10_ep120",
    "title": "aug geo_light + dice weight 10 from scratch, cosine, 120 epochs",
    "train_args": ["--aug", "geo_light", "--epochs", "120", "--lr", "1e-3", "--sched", "cosine", "--dice-weight", "10"],
    "chain": ["17"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
