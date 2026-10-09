"""Experiment 27: exp25 with 200 epochs instead of 120.

Why: 120 epochs was chosen to match the old 4-stage chain (50 + 41 + 10 + 18), never tested as the best length.
In 17 the training loss was still falling at epoch 120 (0.86 -> 0.80 over the last 30 epochs), and going from about
50 epochs (run 04) to 120 (run 17) had helped a lot. Otherwise exactly 25's settings (cleaned labels, repeat
sampler), so 27 - 25 isolates the training length.
Compare with 25 on clean and stress val (python stress_eval.py 25 27). About 2 h on the RTX 4090.
Starts from: random weights; needs lists/exclude_v1.txt and lists/weights_v1.csv (in git)
Result: val 0.9236 (4090, 10/9, epoch 170; last-20-epoch mean 0.9224). vs 25: clean +0.0036 (CI +0.0001 to +0.0070),
stress +0.0059 (CI +0.0029 to +0.0089): longer training helps, mostly by epoch ~140 (120: 0.919, 140: 0.922, then
flat). vs 17: clean +0.0012, stress +0.0022, both n.s. Best clean model so far; still weak on blur (0.768).

    cd Lab2/reference
    python experiments/exp27_geo_light_dice10_ep200_clean.py              # --dry-run prints the train.py command only
    nohup python experiments/exp27_geo_light_dice10_ep200_clean.py >> runs/exp27.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "27",
    "name": "27_aug-geo_light_dice10_ep200_clean",
    "title": "exp25 with 200 epochs: geo_light, dice 10, cleaned labels",
    "train_args": ["--aug", "geo_light", "--epochs", "200", "--lr", "1e-3", "--sched", "cosine", "--dice-weight", "10",
                   "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv", "--sampler", "repeat"],
    "chain": ["27"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
