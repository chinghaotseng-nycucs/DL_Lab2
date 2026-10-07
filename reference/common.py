import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# Override with env vars when the code runs elsewhere (Colab, lab server, unzipped submission).
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


def get_split(val_frac=0.1, seed=0):
    """90/10 split of non_test.csv. val_ids.txt is written once and reused by every run."""
    ids = read_ids("non_test.csv")
    if not os.path.exists(VAL_IDS_PATH):
        rng = random.Random(seed)
        shuffled = sorted(ids)
        rng.shuffle(shuffled)
        val = sorted(shuffled[: round(len(ids) * val_frac)])
        with open(VAL_IDS_PATH, "w") as f:
            f.write("\n".join(val) + "\n")
    with open(VAL_IDS_PATH) as f:
        val_ids = [line.strip() for line in f if line.strip()]
    val_set = set(val_ids)
    train_ids = [i for i in ids if i not in val_set]
    return train_ids, val_ids


def load_image(image_id):
    return Image.open(os.path.join(DATA_ROOT, "images", f"{image_id}.jpg")).convert("RGB")


def load_mask(image_id):
    # Trimap: 1 = pet, 2 = background, 3 = boundary. Only 1 is foreground.
    trimap = np.array(Image.open(os.path.join(DATA_ROOT, "annotations", "trimaps", f"{image_id}.png")))
    return (trimap == 1).astype(np.uint8)


def preprocess(img):
    """Official inference preprocessing: squash to 256x256, [0, 1], ImageNet normalisation."""
    x: Image.Image = TF.resize(img, [SIZE, SIZE])  # type: ignore[assignment]  # PIL in -> PIL out
    return TF.normalize(TF.to_tensor(x), MEAN, STD)


@torch.no_grad()
def predict_masks(model, imgs, device):
    """Official pipeline on a list of PIL images -> list of (h, w) uint8 masks at original size."""
    x = torch.stack([preprocess(img) for img in imgs]).to(device)
    prob = torch.sigmoid(model(x))
    pred = (prob > 0.5).float().cpu()
    masks = []
    for img, p in zip(imgs, pred):
        w, h = img.size  # PIL gives (width, height)
        p = TF.resize(p, [h, w], interpolation=TF.InterpolationMode.NEAREST)
        masks.append(p[0].numpy().astype(np.uint8))
    return masks


def dice(pred, gt):
    total = pred.sum() + gt.sum()
    # Both empty: treat as perfect. Never happens on Oxford, may on the TA's photos.
    return 1.0 if total == 0 else 2.0 * float((pred * gt).sum()) / float(total)


def official_val_dice(model, ids, device, batch_size=16):
    """Mean per-image Dice at the original size, exactly like the TA. Returns {id: dice}."""
    model.eval()
    scores = {}
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        preds = predict_masks(model, [load_image(i) for i in chunk], device)
        for i, pred in zip(chunk, preds):
            scores[i] = dice(pred, load_mask(i))
    return scores


def summarize(scores):
    dog = [d for i, d in scores.items() if is_dog(i)]
    cat = [d for i, d in scores.items() if not is_dog(i)]
    return {
        "all": float(np.mean(list(scores.values()))),
        "dog": float(np.mean(dog)) if dog else float("nan"),
        "cat": float(np.mean(cat)) if cat else float("nan"),
    }


def rle_encode(mask):
    """(h, w) 0/1 mask -> 'start length ...', column-major, 1-based. Empty mask -> ''."""
    pixels = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def rle_decode(text, shape):
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    nums = list(map(int, text.split()))
    for start, length in zip(nums[0::2], nums[1::2]):
        flat[start - 1 : start - 1 + length] = 1
    return flat.reshape(shape, order="F")
