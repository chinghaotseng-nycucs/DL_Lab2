"""Experiment 22: run 08 + 18 epochs, aug geo_light + copy-paste (p 0.5).

Why: the remaining hard images are mostly texture mistakes: cushions, blankets, bark and people taken for pet
(hard_cases/). Copy-paste puts pets from other training photos onto each photo (pet + border band copied, only
trimap 1 labelled pet), so the same pet appears on many backgrounds and context stops being a shortcut
(Ghiasi et al., CVPR 2021). Exactly run 14's setting otherwise, so 22 - 14 isolates copy-paste.
Compare with 14 on clean val, the stress val set and the hard images.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: use last.pth (best.pth = run 08). last.pth vs 14 (Mac, 10/9): clean -0.0036, stress +0.0002, hard
cases 0.684 vs 0.706; slightly worse on people and camouflage. No gain from copy-paste at p 0.5.

    cd Lab2/reference
    python experiments/exp22_geo_light_copypaste_from08.py              # --dry-run prints the train.py command only
    nohup python experiments/exp22_geo_light_copypaste_from08.py >> runs/exp22.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "22",
    "name": "22_aug-geo_light_copypaste_from08",
    "title": "run 08 + 18 epochs, aug geo_light + copy-paste p 0.5",
    "init": "08",
    "train_args": ["--aug", "geo_light", "--copy-paste", "0.5", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10"],
    "chain": ["04", "06", "08", "22"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
