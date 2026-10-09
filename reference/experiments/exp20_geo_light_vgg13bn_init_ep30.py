"""Experiment 20: ImageNet VGG13-BN encoder start, aug geo_light + dice weight 10, cosine, 30 epochs.

Why: VGG13-BN's first four stages have exactly the shapes of inc, down1, down2 and down3 (8 conv + BN, 15% of
the weights). down4 and the decoder stay random. ImageNet has 118 dog breeds and 5 cat classes, so the encoder
features are the most general start available (TA forum 10/4: allowed). Same settings as 19 for a fair comparison.
Compare with 19 (Carvana start) and 17 (random start).
Starts from: init/vgg13bn_encoder.pth (made by make_init_weights.py; the prep step makes it if missing)
Result: not run yet.

    cd Lab2/reference
    python experiments/exp20_geo_light_vgg13bn_init_ep30.py              # --dry-run prints the train.py command only
    nohup python experiments/exp20_geo_light_vgg13bn_init_ep30.py >> runs/exp20.log 2>&1 &     # on a server
"""

from _pipeline import run


def prep(_):
    from make_init_weights import make_vgg13bn

    make_vgg13bn()


EXPERIMENT = {
    "id": "20",
    "name": "20_aug-geo_light_vgg13bn_init_ep30",
    "title": "ImageNet VGG13-BN encoder start, aug geo_light + dice weight 10, lr 3e-4 cosine, 30 epochs",
    "prep": prep,
    "train_args": ["--aug", "geo_light", "--epochs", "30", "--lr", "3e-4", "--sched", "cosine", "--dice-weight", "10",
                   "--init", "init/vgg13bn_encoder.pth"],
    "chain": ["20"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
