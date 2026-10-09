"""Experiment 23: run 08 + 18 epochs, aug geo_light, training masks resized around pixel centres (--exact-masks).

Why: torchvision resizes a Mask with NEAREST, which samples each output pixel at the corner of its block, while
the photo is resized around pixel centres. Every training mask so far sat about half a pixel down-right of its photo.
On val, masks made that way can reach at most Dice 0.980 after the official resize back; NEAREST_EXACT masks 0.991.
Same random crops as run 14 (only the mask resampling changes), so 23 - 14 isolates the alignment.
Compare with 14.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: val 0.9196 (Mac, 10/9). vs 14: clean +0.0003 (CI -0.0006 to +0.0012), stress -0.0005: no measurable
effect. Predictions moved as intended (pet-centre offset 1.9 -> 1.1 px). Keep as a free correctness fix.

    cd Lab2/reference
    python experiments/exp23_geo_light_exact_masks_from08.py              # --dry-run prints the train.py command only
    nohup python experiments/exp23_geo_light_exact_masks_from08.py >> runs/exp23.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "23",
    "name": "23_aug-geo_light_exact_masks_from08",
    "title": "run 08 + 18 epochs, aug geo_light, masks resized around pixel centres",
    "init": "08",
    "train_args": ["--aug", "geo_light", "--exact-masks", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10"],
    "chain": ["04", "06", "08", "23"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
