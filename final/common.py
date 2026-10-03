import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# Override with env vars when the code runs elsewhere (lab server, unzipped submission).
DATA_ROOT = os.environ.get("PET_ROOT", os.path.join(HERE, "..", "data", "oxford-iiit-pet"))
CSV_DIR = os.environ.get("PET_CSV_DIR", os.path.join(HERE, "..", "material"))
VAL_IDS_PATH = os.path.join(HERE, "val_ids.txt")

SIZE = 256
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_ids(csv_name):
    with open(os.path.join(CSV_DIR, csv_name)) as f:
        lines = [line.strip() for line in f]
    assert lines[0] == "image_id", lines[0]
    return [line for line in lines[1:] if line]


def is_dog(image_id):
    # Oxford-IIIT Pet: cat breeds are capitalised, dog breeds are lowercase
    return image_id[0].islower()


def load_image(image_id):
    return Image.open(os.path.join(DATA_ROOT, "images", f"{image_id}.jpg")).convert("RGB")


def summarize(scores):
    dog = [d for i, d in scores.items() if is_dog(i)]
    cat = [d for i, d in scores.items() if not is_dog(i)]
    return {
        "all": float(np.mean(list(scores.values()))),
        "dog": float(np.mean(dog)) if dog else float("nan"),
        "cat": float(np.mean(cat)) if cat else float("nan"),
    }


# Stage 1 — dataset and official-style validation


def load_mask(image_id):
    """Trimap -> (h, w) uint8 array of 0/1.

    Trimap PNG: DATA_ROOT/annotations/trimaps/<image_id>.png, values 1 = pet, 2 = background,
    3 = boundary. Only 1 is foreground.
    """
    # TODO(Stage 1)
    raise NotImplementedError


def get_split(val_frac=0.1, seed=0):
    """90/10 split of non_test.csv -> (train_ids, val_ids).

    The first call shuffles with a fixed seed and writes the val IDs to VAL_IDS_PATH.
    Every later call reads that file back, so every run validates on the same images.
    """
    # TODO(Stage 1)
    raise NotImplementedError


def preprocess(img):
    """PIL RGB image -> (3, 256, 256) float tensor, exactly like the TA's pipeline:
    squash to 256 x 256 (aspect ratio NOT kept), [0, 1], ImageNet normalisation."""
    # TODO(Stage 1)
    raise NotImplementedError


@torch.no_grad()
def predict_masks(model, imgs, device):
    """Official pipeline on a list of PIL images -> list of (h, w) uint8 0/1 masks,
    each at its own image's ORIGINAL size.

    preprocess -> batch -> model -> sigmoid -> threshold 0.5 -> resize back with NEAREST.
    """
    # TODO(Stage 1)
    raise NotImplementedError


def dice(pred, gt):
    """Dice of two (h, w) 0/1 arrays. Decide (and write down) what both-empty returns."""
    # TODO(Stage 1)
    raise NotImplementedError


def official_val_dice(model, ids, device, batch_size=16):
    """Mean-able per-image Dice at the original size, like the TA. Returns {image_id: dice}.

    Remember model.eval() here, and model.train() in the training loop afterwards.
    """
    # TODO(Stage 1)
    raise NotImplementedError


# Stage 3 — Kaggle CSV


def rle_encode(mask):
    """(h, w) 0/1 mask -> 'start length start length ...', column-major, 1-based. Empty mask -> ''."""
    # TODO(Stage 3)
    raise NotImplementedError


def rle_decode(text, shape):
    """Inverse of rle_encode: text + (h, w) -> (h, w) uint8 mask."""
    # TODO(Stage 3)
    raise NotImplementedError
