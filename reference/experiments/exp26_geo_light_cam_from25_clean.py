"""Experiment 26: run 25 + 18 epochs with camera augmentation (geo_light_cam), cleaned labels, final epoch kept.

Why: camera augmentation for the whole 120 epochs (exp24) bought robustness (stress val +0.023) but cost -0.007 on
clean val. As a short fine-tune of an already trained model (exp21: 08 + 18 epochs) it cost only -0.003 for a
similar stress gain. Here the same fine-tune is added on top of 25: geo_light_cam (geo_light + blur, JPEG, noise,
each p 0.2), lr 1e-4 cosine, dice weight 10, the same cleaned labels and sampler as 25.
--select last: best.pth is the final epoch, not the epoch with the highest clean val Dice (for exp21 and exp22 that
was epoch 0, i.e. the starting weights). Compare with 25 on clean and stress val (python stress_eval.py 25 26).
Starts from: runs/25_*/best.pth (run exp25 first, on the same machine); needs lists/exclude_v1.txt, lists/weights_v1.csv
Result: val 0.9177 (4090, 10/9, final epoch). vs 25: clean -0.0022 (CI -0.0039 to -0.0005), stress +0.0273 (0.901,
best so far; blur +0.118, low_res +0.046). vs 24 (camera effects for all 120 epochs): clean +0.0026, stress +0.0009,
both n.s.: the same robustness for less clean cost, in 11 min.

    cd Lab2/reference
    python experiments/exp26_geo_light_cam_from25_clean.py              # --dry-run prints the train.py command only
    nohup python experiments/exp26_geo_light_cam_from25_clean.py >> runs/exp26.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "26",
    "name": "26_aug-geo_light_cam_from25_clean",
    "title": "run 25 + 18 epochs, aug geo_light_cam, cleaned labels, final epoch kept",
    "init": "25",
    "train_args": ["--aug", "geo_light_cam", "--epochs", "18", "--lr", "1e-4", "--sched", "cosine", "--dice-weight", "10",
                   "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv", "--sampler", "repeat",
                   "--select", "last"],
    "chain": ["25", "26"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
