"""Experiment 21: run 08 + 18 epochs, aug geo_light_cam (geo_light + blur, JPEG, noise).

Why: on the stress val set run 14 (geo_light) was weakest on blur (0.785), low resolution and noise, where 07
(geo_color, which blurs) was far more robust. geo_light_cam keeps geo_light and adds blur, JPEG compression and
sensor noise, each with p 0.2. Exactly run 14's setting otherwise, so 21 - 14 isolates the camera steps.
Compare with 14 on clean val and on the stress val set.
Starts from: runs/08_*/best.pth (on a server, copy that folder first: weights are not in git)
Result: use last.pth (no epoch beat the start on clean val, so best.pth = run 08). last.pth vs 14 (Mac, 10/9):
clean -0.0029, stress +0.0212 (0.895, best so far; blur +0.093), 23 hard cases 0.714 vs 0.706. Works for robustness.

    cd Lab2/reference
    python experiments/exp21_geo_light_cam_from08.py              # --dry-run prints the train.py command only
    nohup python experiments/exp21_geo_light_cam_from08.py >> runs/exp21.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "21",
    "name": "21_aug-geo_light_cam_from08",
    "title": "run 08 + 18 epochs, aug geo_light_cam (geo_light + blur, JPEG, noise)",
    "init": "08",
    "train_args": ["--aug", "geo_light_cam", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10"],
    "chain": ["04", "06", "08", "21"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
