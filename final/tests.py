import importlib.util
import os
import sys

import numpy as np
import torch

import common
from model import UNet

# python tests.py          all stages
# python tests.py 1        only Stage 1 (also 0, 2, 3, 4)
# TODO = stub not written yet · OK = passed · FAIL = wrong answer · ERR = crashed
# "vs ref" checks run your function and the reference's on the same input and compare.

ONLY = sys.argv[1] if len(sys.argv) > 1 else None
counts = {}


def check(stage, name, fn):
    if ONLY is not None and str(stage) != ONLY:
        return
    try:
        passed, detail = fn()
        status = "OK  " if passed else "FAIL"
    except NotImplementedError:
        status, detail = "TODO", ""
    except Exception as e:
        status, detail = "ERR ", f"{type(e).__name__}: {e}"
    counts[status] = counts.get(status, 0) + 1
    print(f"[Stage {stage}] {status} {name}  {detail}")


def load_reference():
    path = os.path.join(common.HERE, "..", "reference", "common.py")
    spec = importlib.util.spec_from_file_location("ref_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.VAL_IDS_PATH = os.path.join(common.HERE, "..", "reference", "val_ids.txt")
    return mod


ref = load_reference()
IDS = common.read_ids("non_test.csv")[:8]


# Stage 0: the model
def t_shape():
    out = UNet(in_channels=3, out_channels=1)(torch.randn(2, 3, 256, 256))
    return out.shape == (2, 1, 256, 256), str(tuple(out.shape))


check(0, "model(randn(2,3,256,256)) shape", t_shape)


# Stage 1: mask, split, preprocessing, Dice, official validation
def t_mask_values():
    m = common.load_mask(IDS[0])
    return m.dtype == np.uint8 and set(np.unique(m).tolist()) <= {0, 1}, f"dtype {m.dtype}, values {np.unique(m).tolist()}"


def t_mask_ref():
    same = all(np.array_equal(common.load_mask(i), ref.load_mask(i)) for i in IDS)
    return same, "boundary (3) must be 0"


def t_split():
    train, val = common.get_split()
    total = len(common.read_ids("non_test.csv"))
    ok = not set(train) & set(val) and len(train) + len(val) == total and len(val) == round(total * 0.1)
    return ok, f"train {len(train)}, val {len(val)}"


def t_split_stable():
    a, b = common.get_split(), common.get_split()
    return a[1] == b[1], "two calls must return the same val IDs"


def t_preprocess():
    x = common.preprocess(common.load_image(IDS[0]))
    return tuple(x.shape) == (3, 256, 256) and x.dtype == torch.float32, f"{tuple(x.shape)} {x.dtype}"


def t_preprocess_ref():
    img = common.load_image(IDS[0])
    diff = (common.preprocess(img) - ref.preprocess(img)).abs().max().item()
    return diff < 1e-5, f"max |diff| {diff:.2e}"


def t_dice():
    t = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]])
    p = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]])
    cases = [(p, t, 8 / 11), (t, t, 1.0), (np.zeros_like(t), t, 0.0), (np.ones_like(t), t, 12 / 22)]
    got = [float(common.dice(a, b)) for a, b, _ in cases]
    return all(abs(g - e) < 1e-9 for g, (_, _, e) in zip(got, cases)), "got " + ", ".join(f"{g:.3f}" for g in got) + " (want 0.727, 1, 0, 0.545)"


def t_predict_masks():
    torch.manual_seed(0)
    model = UNet().eval()
    imgs = [common.load_image(i) for i in IDS[:3]]
    masks = common.predict_masks(model, imgs, torch.device("cpu"))
    sizes = [(m.shape, (img.size[1], img.size[0])) for m, img in zip(masks, imgs)]
    return all(s == e for s, e in sizes), "mask (h, w) must equal each image's original (h, w)"


def t_val_ref():
    torch.manual_seed(0)
    model = UNet()
    dev = torch.device("cpu")
    mine = common.official_val_dice(model, IDS, dev)
    theirs = ref.official_val_dice(model, IDS, dev)
    diff = max(abs(mine[i] - theirs[i]) for i in IDS)
    return set(mine) == set(IDS) and diff < 1e-6, f"max |diff| {diff:.2e} over {len(IDS)} images (random weights)"


check(1, "load_mask: uint8 of 0/1", t_mask_values)
check(1, "load_mask vs ref", t_mask_ref)
check(1, "get_split: disjoint 90/10", t_split)
check(1, "get_split: stable across calls", t_split_stable)
check(1, "preprocess: (3,256,256) float32", t_preprocess)
check(1, "preprocess vs ref", t_preprocess_ref)
check(1, "dice: worked examples", t_dice)
check(1, "predict_masks: original size", t_predict_masks)
check(1, "official_val_dice vs ref", t_val_ref)


def t_dataset(aug):
    def run():
        from dataset import AUGS, PetDataset

        if aug not in AUGS:
            raise NotImplementedError
        img, mask = PetDataset(IDS[:2], aug)[0]
        vals = mask.unique().tolist()
        ok = img.shape == (3, 256, 256) and mask.shape == (1, 256, 256) and img.dtype == mask.dtype == torch.float32 and set(vals) <= {0.0, 1.0}
        return ok, f"img {tuple(img.shape)}, mask {tuple(mask.shape)} values {vals}"

    return run


def t_dataset_none_ref():
    from dataset import PetDataset

    img, mask = PetDataset(IDS[:1], "none")[0]
    # Same squash + normalise as the official preprocessing, so this should be close (tensor vs PIL resize)
    diff = (img - ref.preprocess(common.load_image(IDS[0]))).abs().mean().item()
    return diff < 0.05, f"mean |diff| vs official preprocess {diff:.3f}"


check(1, "dataset[none]: shapes, dtype, binary mask", t_dataset("none"))
check(1, "dataset[none] close to official preprocess", t_dataset_none_ref)


# Stage 2: loss
def t_soft_dice():
    from train import soft_dice_loss

    y = torch.tensor([0.9, 0.6, 0.3, 0.1])
    logits = torch.log(y / (1 - y)).view(1, 1, 2, 2)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 1, 2, 2)
    a, b = soft_dice_loss(logits, mask, c=0).item(), soft_dice_loss(logits, mask, c=1).item()
    return abs(a - 0.2308) < 1e-3 and abs(b - 0.1837) < 1e-3, f"c=0 {a:.3f} (want 0.231), c=1 {b:.3f} (want 0.184)"


def t_total_loss():
    from train import make_loss

    y = torch.tensor([0.9, 0.6, 0.3, 0.1])
    logits = torch.log(y / (1 - y)).view(1, 1, 2, 2)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 1, 2, 2)
    got = make_loss(1.0)(logits, mask).item()
    return abs(got - (0.2695 + 0.1837)) < 1e-3, f"{got:.3f} (want BCE 0.270 + softDice(c=1) 0.184 = 0.453)"


def t_soft_dice_per_image():
    from train import soft_dice_loss

    # Image 0 perfect, image 1 empty prediction on a full mask: per-image mean is ~0.5
    logits = torch.full((2, 1, 4, 4), -20.0)
    logits[0] = 20.0
    mask = torch.ones(2, 1, 4, 4)
    got = soft_dice_loss(logits, mask, c=1).item()
    return abs(got - 0.4706) < 1e-3, f"{got:.3f} (want 0.471: mean of per-image losses 0 and 16/17)"


check(2, "soft_dice_loss: worked example", t_soft_dice)
check(2, "soft_dice_loss: per image, then mean", t_soft_dice_per_image)
check(2, "make_loss: BCE + soft Dice", t_total_loss)


# Stage 3: RLE
def t_rle_examples():
    m = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]])
    got = [common.rle_encode(m), common.rle_encode(np.array([[1, 0, 1], [1, 0, 0]])), common.rle_encode(np.zeros((5, 7))), common.rle_encode(np.ones((5, 7)))]
    want = ["4 2 7 3", "1 2 5 1", "", "1 35"]
    return got == want, f"got {got}, want {want}"


def t_rle_roundtrip():
    rng = np.random.default_rng(0)
    masks = [rng.integers(0, 2, (rng.integers(1, 40), rng.integers(1, 40))).astype(np.uint8) for _ in range(200)]
    masks += [common.load_mask(i) for i in IDS]
    return all(np.array_equal(common.rle_decode(common.rle_encode(m), m.shape), m) for m in masks), "200 random + 8 real masks"


def t_rle_ref():
    return all(common.rle_encode(common.load_mask(i)) == ref.rle_encode(ref.load_mask(i)) for i in IDS), ""


check(3, "rle_encode: worked examples", t_rle_examples)
check(3, "rle round trip", t_rle_roundtrip)
check(3, "rle_encode vs ref", t_rle_ref)


# Stage 4: augmentation presets you added to AUGS
def t_aug_presets():
    from dataset import AUGS

    if not [a for a in AUGS if a != "none"]:
        raise NotImplementedError
    return True, f"presets: {list(AUGS)}"


check(4, "AUGS has presets beyond 'none'", t_aug_presets)
for aug in ["geo", "geo_color"]:
    check(4, f"dataset[{aug}]: shapes, dtype, binary mask", t_dataset(aug))

print("\n" + " · ".join(f"{k.strip()} {v}" for k, v in sorted(counts.items())))
