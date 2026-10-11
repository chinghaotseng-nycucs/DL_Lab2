"""Experiment 32: run 27's recipe (200 epochs, cleaned labels) from the ImageNet VGG13-BN encoder start, paired with 27.

Why: ImageNet has 118 dog breeds and 5 cat classes, so its encoder features are the most general start allowed (TA
forum 10/4). The Carvana start (19) failed, but Carvana has no animals. 200 epochs, the length of the final recipe,
rather than exp20's 30: a pretrained start mainly speeds up early training, and the gap to a random start often
closes with a long schedule (He et al., "Rethinking ImageNet Pre-training", ICCV 2019), so a shorter test would
overstate it. A win is also a usable model with a fair val score.
Paired with 27: init/vgg13bn_encoder.pth is 27's random start (seed 0) with inc, down1, down2 and down3 replaced by
VGG13-BN stages 1-4 (4.7 M of 31 M weights, checked 10/10); down4 and the decoder are the same random weights. The
same seed gives the same batches and augmentation, and lr 1e-3 is kept because 85% of the weights start random, so
the start is the only difference.
Adopt it for the final model (exp31 IMAGENET_START = True) if clean val beats 27 by at least 0.003 with the CI
above 0, the last-20-epoch mean beats 27's 0.9224, and stress val is not lower.
Compare: python stress_eval.py 27 32   (and log.csv against 27's: is 32 ahead only early, or still at the end?)
Starts from: init/vgg13bn_encoder.pth (the prep step makes it if missing; it downloads torchvision's VGG13-BN)
about 1 h 50 min on the RTX 4090, on the same machine as 27
Result: val 0.9314 (4090, 10/11; ran alongside 33, 214 min). Adopted for the final model (exp31 IMAGENET_START).
vs 27: clean +0.0078 (CI +0.0042 to +0.0114), last-20-epoch mean 0.9309 vs 0.9224, stress 0.897 vs 0.880 (+0.0173;
blur +0.059, low_res +0.020, low_contrast +0.018, dark +0.012; jpeg +0.001 n.s.). Ahead from the first epoch (+0.094
over epochs 1-20); the lead shrank to about +0.009 by epoch 140 and then held to the end. The photos exp33 targets
(blankets, toys, clothes, snow; 31 val photos) +0.030, CI above 0; camouflage +0.029. Not better: person touching
-0.004 and hairless breeds -0.009 (both n.s.); boxer_150 (dog behind a girl) 0.04.

    cd Lab2/reference
    python experiments/exp32_imagenet_init_ep200_clean.py              # --dry-run prints the train.py command only
    nohup python experiments/exp32_imagenet_init_ep200_clean.py >> runs/exp32.log 2>&1 &     # on a server
"""

from _pipeline import run


def prep(_):
    from make_init_weights import make_vgg13bn

    make_vgg13bn()


EXPERIMENT = {
    "id": "32",
    "name": "32_imagenet_init_ep200_clean",
    "title": "run 27's recipe (geo_light, dice 10, 200 epochs, cleaned labels) from the ImageNet VGG13-BN encoder start",
    "prep": prep,
    "train_args": ["--aug", "geo_light", "--epochs", "200", "--lr", "1e-3", "--sched", "cosine", "--dice-weight", "10",
                   "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv", "--sampler", "repeat",
                   "--init", "init/vgg13bn_encoder.pth"],
    "chain": ["32"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
