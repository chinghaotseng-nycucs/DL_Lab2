"""Experiment 24: run 17's one-run recipe with aug geo_light_cam (geo_light + blur, JPEG, noise), 120 epochs.

Why: 17 (geo_light, 120 epochs from scratch) is the best clean model (0.9223) but weak on blur (0.779). In exp21 the
camera steps added +0.021 on the stress val set (blur +0.093) for -0.003 on clean val, as an 18-epoch fine-tune.
Here they are on from the first epoch, so the model has the whole schedule to absorb them.
Compare with 17 on clean val and stress val (python stress_eval.py 17 24).
Starts from: random weights
Result: val 0.9152 (4090, 10/9). vs 17: clean -0.0072 (CI -0.0096 to -0.0048), stress +0.0227 (0.900, best so far;
blur +0.111, low_res +0.040, jpeg +0.033, noise +0.025, lighting -0.005 to -0.009). best.pth and last.pth are
the same within 0.0003. A trade-off, not a win: more robust, less accurate on clean Oxford photos.

    cd Lab2/reference
    python experiments/exp24_geo_light_cam_dice10_ep120.py              # --dry-run prints the train.py command only
    nohup python experiments/exp24_geo_light_cam_dice10_ep120.py >> runs/exp24.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "24",
    "name": "24_aug-geo_light_cam_dice10_ep120",
    "title": "aug geo_light_cam + dice weight 10 from scratch, cosine, 120 epochs",
    "train_args": ["--aug", "geo_light_cam", "--epochs", "120", "--lr", "1e-3", "--sched", "cosine", "--dice-weight", "10"],
    "chain": ["24"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
