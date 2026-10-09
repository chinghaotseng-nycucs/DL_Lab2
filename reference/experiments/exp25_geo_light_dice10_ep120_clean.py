"""Experiment 25: run 17's one-run recipe on cleaned labels (lists/*_v1).

Why: the label check (10/9) found 23 clearly wrong training labels (empty or inverted) and 16 sloppy ones (most of
the visible pet marked as background); they are left out (lists/exclude_v1.txt, 39 images). The 8 training photos
with two or more pets that are all labelled are drawn half as often (lists/weights_v1.csv), because the TA's private
photos show one pet each and the model should focus on the main pet. Otherwise exactly 17's settings, so 25 - 17
isolates the label cleaning. --sampler repeat keeps 17's "every image once per epoch" sampling for all other
images (the default weighted sampler would draw with replacement and skip about a third each epoch). Val is never changed.
Compare with 17 on clean val and stress val (python stress_eval.py 17 25).
Starts from: random weights; needs lists/exclude_v1.txt and lists/weights_v1.csv (in git)
Result: not run yet.

    cd Lab2/reference
    python experiments/exp25_geo_light_dice10_ep120_clean.py              # --dry-run prints the train.py command only
    nohup python experiments/exp25_geo_light_dice10_ep120_clean.py >> runs/exp25.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "25",
    "name": "25_aug-geo_light_dice10_ep120_clean",
    "title": "run 17's recipe (geo_light, dice 10, 120 epochs) on cleaned labels: 39 left out, 8 multi-pet photos x0.5",
    "train_args": ["--aug", "geo_light", "--epochs", "120", "--lr", "1e-3", "--sched", "cosine", "--dice-weight", "10",
                   "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv", "--sampler", "repeat"],
    "chain": ["25"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
