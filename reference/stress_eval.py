import argparse
import glob
import io
import os
import zlib

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from common import HERE, dice, get_device, get_split, load_image, load_mask, predict_masks
from model import UNet

# python stress_eval.py 14 23        -> each run's val Dice on clean photos and under 8 photo distortions, and the
#                                       paired difference of every later run against the first (95% bootstrap CI)
# Runs are given by number (runs/<NN>_*/best.pth) or as a path to a .pth. Inference only, official pipeline;
# about 5 min per run on the Mac. Writes runs/<run>/stress_<weights>.csv (per-image Dice) next to each run's weights.
# If review/analysis/tags.csv exists (hard_traits.py), it also prints the difference per hard characteristic.

KINDS = ["clean", "dark", "bright", "low_contrast", "warm_cast", "jpeg", "low_res", "noise", "blur"]
TAGS = ["small_pet", "label_in_pieces", "person_touching", "blanket_or_cushion", "plush_toy_detected", "with_plush_toy",
        "wearing_clothes", "camouflage", "in_snow", "over_exposed", "low_resolution", "hairless_breed"]
# the rare hard groups drawn twice as often in lists/weights_v2.csv (exp33); also reported together
UPWEIGHTED = ["blanket_or_cushion", "with_plush_toy", "plush_toy_detected", "wearing_clothes", "in_snow"]


def corrupt(img, kind, image_id):
    """img: PIL RGB at original size -> distorted PIL RGB. Same output every time for (kind, image_id)."""
    if kind == "clean":
        return img
    if kind == "dark":  # under-exposed: outside geo_light's 0.6-1.4 training range
        return ImageEnhance.Brightness(img).enhance(0.4)
    if kind == "bright":  # over-exposed, washed out
        return ImageEnhance.Brightness(img).enhance(1.8)
    if kind == "low_contrast":
        return ImageEnhance.Contrast(img).enhance(0.4)
    if kind == "warm_cast":  # indoor tungsten light
        a = np.asarray(img).astype(np.float32) * np.array([1.2, 1.0, 0.75])
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if kind == "jpeg":  # messaging-app compression
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=15)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "low_res":  # small or zoomed-in photo
        w, h = img.size
        return img.resize((max(1, w // 4), max(1, h // 4)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if kind == "noise":  # dark-room sensor noise, fixed seed per image
        rng = np.random.default_rng(zlib.crc32(image_id.encode()))
        a = np.asarray(img).astype(np.float32) + rng.normal(0, 20, (img.size[1], img.size[0], 3))
        return Image.fromarray(a.clip(0, 255).astype(np.uint8))
    if kind == "blur":  # out of focus or motion
        return img.filter(ImageFilter.GaussianBlur(radius=3))
    raise ValueError(kind)


def weights_path(run):
    if run.endswith(".pth"):
        return run
    found = sorted(glob.glob(os.path.join(HERE, "runs", f"{run}_*", "best.pth")))
    if not found:
        raise SystemExit(f"no runs/{run}_*/best.pth")
    return found[-1]


def score(path, val_ids, imgs, gts, device):
    """{kind: per-image Dice array}, cached in stress.csv next to the weights."""
    cache = os.path.join(os.path.dirname(path), f"stress_{os.path.splitext(os.path.basename(path))[0]}.csv")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        data = np.genfromtxt(cache, delimiter=",", names=True, dtype=None, encoding=None)
        return {k: np.asarray(data[k], dtype=float) for k in KINDS}
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(path, map_location=device), strict=True)
    model.eval()
    out = {}
    for kind in KINDS:
        s = []
        for st in range(0, len(val_ids), 16):
            chunk = val_ids[st : st + 16]
            preds = predict_masks(model, [corrupt(imgs[i], kind, i) for i in chunk], device)
            s += [dice(p, gts[i]) for i, p in zip(chunk, preds)]
        out[kind] = np.array(s)
    with open(cache, "w") as f:
        f.write("image_id," + ",".join(KINDS) + "\n")
        for n, i in enumerate(val_ids):
            f.write(i + "," + ",".join(f"{out[k][n]:.4f}" for k in KINDS) + "\n")
    return out


def ci(d, n=5000):
    rng = np.random.default_rng(0)
    return np.percentile(d[rng.integers(0, len(d), (n, len(d)))].mean(1), [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run numbers (e.g. 14 23) or .pth paths; later runs are compared with the first")
    args = p.parse_args()
    device = get_device()
    _, val_ids = get_split()
    imgs = {i: load_image(i) for i in val_ids}
    gts = {i: load_mask(i) for i in val_ids}
    res = {r: score(weights_path(r), val_ids, imgs, gts, device) for r in args.runs}
    for r in res:
        res[r]["stress"] = np.mean([res[r][k] for k in KINDS[1:]], axis=0)

    print(f"{'':<14}" + "".join(f"{r:>10}" for r in res))
    for k in KINDS + ["stress"]:
        print(f"{k:<14}" + "".join(f"{res[r][k].mean():>10.4f}" for r in res))
    base = args.runs[0]
    for r in args.runs[1:]:
        print(f"\n{r} - {base} (paired, 95% CI):")
        for k in KINDS + ["stress"]:
            d = res[r][k] - res[base][k]
            lo, hi = ci(d)
            print(f"  {k:<14}{d.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        tags_csv = os.path.join(HERE, "review", "analysis", "tags.csv")
        if os.path.exists(tags_csv):
            import pandas as pd

            t = pd.read_csv(tags_csv, index_col=0).reindex(val_ids)
            d = res[r]["clean"] - res[base]["clean"]
            print(f"  clean val by characteristic ({r} - {base}):")
            for tag in TAGS:
                m = (t[tag] == 1).to_numpy()
                if m.sum() >= 3:
                    lo, hi = ci(d[m])
                    print(f"    {tag:<22} n={int(m.sum()):>3}  {d[m].mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")
            m = (t[UPWEIGHTED] == 1).any(axis=1).to_numpy()
            lo, hi = ci(d[m])
            print(f"    {'any x2 group (exp33)':<22} n={int(m.sum()):>3}  {d[m].mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
