"""Experiment 30: milder camera finish on run 27 (blur, JPEG, noise at p 0.1 each instead of 0.2).

Why: the camera finish (28b on the 4090, 29d on the Mac) buys a lot of robustness (stress val about +0.025) but costs
about -0.004 on clean val, half of it from fine-tuning at all (29a: -0.002). With the camera steps at half the chance
the clean cost may shrink while most of the robustness stays. Same start, seed and settings as 29a-d (18 epochs,
lr 1e-4 cosine, dice 10, cleaned labels, repeat sampler, final epoch kept).
Compare with 29a (plain finish) and 29d (camera at p 0.2): python stress_eval.py 29a 30 29d
Starts from: runs/27_*/best.pth; about 11 min on the RTX 4090, 2.2 h on the Mac
Result: val 0.9207 (Mac, 10/10), stress 0.9055. vs 29a (plain finish): clean -0.0010 (n.s.), stress +0.0242; vs 29d
(camera at p 0.2): the same robustness (stress 0.9055 vs 0.9057) for half the clean cost (-0.0010 vs -0.0020).
vs 27: clean -0.0029, stress +0.0256 (blur 0.768 -> 0.884). The mild camera finish is the better choice.

    cd Lab2/reference
    python experiments/exp30_mild_camera_finish_from27.py              # --dry-run prints the train.py command only
    nohup python experiments/exp30_mild_camera_finish_from27.py >> runs/exp30.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "30",
    "name": "30_finish_from27_mild_camera",
    "title": "27 + 18 epochs, aug geo_light_cam_mild (camera effects at p 0.1)",
    "init": "27",
    "train_args": ["--aug", "geo_light_cam_mild", "--epochs", "18", "--lr", "1e-4", "--sched", "cosine",
                   "--dice-weight", "10", "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv",
                   "--sampler", "repeat", "--select", "last"],
    "chain": ["27", "30"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
