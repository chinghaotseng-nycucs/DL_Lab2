"""Experiment 19: Carvana-pretrained start, aug geo_light + dice weight 10, cosine, 30 epochs.

Why: every run so far starts from random weights. The milesial Carvana UNet has the same design as model.py, so
all of it (encoder, decoder, head after a small conversion) is a starting point (TA forum 10/4: pretrained starting
weights are allowed). lr 3e-4 instead of 1e-3, so the first epochs don't wipe out the pretrained features.
Compare with 17 (same recipe from random weights, 120 epochs) and with run 04's curve at the same epochs.
Starts from: init/carvana_unet.pth (made by make_init_weights.py; the prep step makes it if missing)
Result: val 0.9015 (Mac, 10/9). Ahead of random weights for the first ~10 epochs, then level; about the same as
from-scratch 30-epoch runs and less robust (stress 0.848 vs 14's 0.874). No gain from the Carvana start.

    cd Lab2/reference
    python experiments/exp19_geo_light_carvana_init_ep30.py              # --dry-run prints the train.py command only
    nohup python experiments/exp19_geo_light_carvana_init_ep30.py >> runs/exp19.log 2>&1 &     # on a server
"""

from _pipeline import run


def prep(_):
    from make_init_weights import make_carvana

    make_carvana()


EXPERIMENT = {
    "id": "19",
    "name": "19_aug-geo_light_carvana_init_ep30",
    "title": "Carvana-pretrained start, aug geo_light + dice weight 10, lr 3e-4 cosine, 30 epochs",
    "prep": prep,
    "train_args": ["--aug", "geo_light", "--epochs", "30", "--lr", "3e-4", "--sched", "cosine", "--dice-weight", "10",
                   "--init", "init/carvana_unet.pth"],
    "chain": ["19"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
