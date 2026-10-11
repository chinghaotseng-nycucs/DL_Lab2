"""Experiment 34: the mild camera finish (exp30) on run 32, to choose between 31a and 31b.

Why: exp31 makes the final model on train + val, where val can no longer judge it: 31a is run 32's recipe, 31b adds
the 18-epoch mild camera finish. That finish was only tested on 27 (exp30: clean -0.0029, stress +0.0256). 32 is
already more robust (stress 0.897 vs 27's 0.880), so the finish may be worth less on it. This is 31b's last stage on
the 90/10 split, with a fair val score: same settings as exp30 (geo_light_cam_mild, 18 epochs, lr 1e-4 cosine,
dice 10, cleaned labels, repeat sampler, final epoch kept), started from 32.
Decide (set before the run): submit 31b if 34 - 32 is at least +0.010 on stress val and at most -0.003 on clean val;
otherwise submit 31a.
Compare: python stress_eval.py 32 34
Starts from: runs/32_*/best.pth; about 11 min on the RTX 4090, 35 min on the Mac
Result: not run yet.

    cd Lab2/reference
    python experiments/exp34_mild_camera_finish_from32.py              # --dry-run prints the train.py command only
    nohup python experiments/exp34_mild_camera_finish_from32.py >> runs/exp34.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "34",
    "name": "34_mild_camera_finish_from32",
    "title": "32 + 18 epochs, aug geo_light_cam_mild (exp30's finish; decides 31a vs 31b)",
    "init": "32",
    "train_args": ["--aug", "geo_light_cam_mild", "--epochs", "18", "--lr", "1e-4", "--sched", "cosine",
                   "--dice-weight", "10", "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv",
                   "--sampler", "repeat", "--select", "last"],
    "chain": ["32", "34"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
