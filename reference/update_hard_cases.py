import argparse
import csv
import glob
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import DATA_ROOT, HERE, dice, get_device, get_split, load_image, load_mask, official_val_dice, predict_masks
from model import UNet

# python update_hard_cases.py              -> hard_cases/: every val image that is in any run's worst 10,
#                                             with scores.csv (Dice of every run) and <id>_compare.png (best run)
# python update_hard_cases.py --model 14   -> draw the compare figures with run 14 instead of the best run
# Inference only, official pipeline; about 30 s per run on the Mac. Prints the README table at the end.

OUT = os.path.join(HERE, "hard_cases")
# Runs shown in the README table and figure titles; scores.csv has every run
KEY = ["01", "04", "07", "08", "11", "12", "13", "14", "15"]


def find_runs():
    """{"04": "runs/04_aug-geo_ep50_val0.9113", ...}: every run folder with a best.pth, by run number."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(HERE, "runs", "*", "best.pth"))):
        name = os.path.basename(os.path.dirname(path))
        m = re.match(r"(\d+[a-z]?)_", name)
        if m:
            runs[m.group(1)] = os.path.dirname(path)
    return runs


def load(run_dir, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pth"), map_location=device), strict=True)
    return model.eval()


def compare_figure(image_id, pred, scores, key_runs, model_id, path):
    """photo | true mask | prediction | errors (red = missed pet, blue = predicted pet where there is none)."""
    img, gt = load_image(image_id), load_mask(image_id)
    err = np.zeros((*gt.shape, 3), dtype=np.uint8)
    err[(gt == 1) & (pred == 0)] = [230, 40, 40]
    err[(gt == 0) & (pred == 1)] = [30, 110, 255]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    panels = [(img, image_id), (gt, "true mask"), (pred, f"run {model_id} prediction (Dice {dice(pred, gt):.3f})"),
              (err, "errors: red = missed pet, blue = false pet")]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray" if im is gt or im is pred else None, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Dice per run:  " + "   ".join(f"{r} {scores[r][image_id]:.3f}" for r in key_runs), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))  # keep the panel titles clear of the suptitle on square photos
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="", help="run number for the compare figures; default: best val Dice")
    p.add_argument("--worst", type=int, default=10)
    args = p.parse_args()

    device = get_device()
    _, val_ids = get_split()
    runs = find_runs()
    scores, means = {}, {}
    for r, run_dir in runs.items():
        scores[r] = official_val_dice(load(run_dir, device), val_ids, device)
        means[r] = float(np.mean(list(scores[r].values())))
        print(f"run {r}: val Dice {means[r]:.4f}  ({os.path.basename(run_dir)})")

    # Best run: highest val Dice; on a tie the earlier run (16 continued 14 without a gain, so 16 = 14)
    model_id = args.model or max(runs, key=lambda r: (round(means[r], 4), -int(re.sub(r"\D", "", r))))
    hard = sorted({i for r in runs for i in sorted(scores[r], key=scores[r].get)[: args.worst]},
                  key=lambda i: scores[model_id][i])
    os.makedirs(OUT, exist_ok=True)

    with open(os.path.join(OUT, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id"] + [f"run_{r}" for r in runs])
        for i in hard:
            w.writerow([i] + [f"{scores[r][i]:.4f}" for r in runs])

    key_runs = [r for r in KEY if r in runs]
    if model_id not in key_runs:
        key_runs.append(model_id)
    model = load(runs[model_id], device)
    for i in hard:
        src = os.path.join(DATA_ROOT, "images", f"{i}.jpg")
        shutil.copy(src, os.path.join(OUT, f"{i}.jpg"))
        pred = predict_masks(model, [load_image(i)], device)[0]
        compare_figure(i, pred, scores, key_runs, model_id, os.path.join(OUT, f"{i}_compare.png"))
    print(f"\n{len(hard)} hard images (worst {args.worst} of any of {len(runs)} runs) -> {OUT}; figures use run {model_id}\n")

    print("| Image | " + " | ".join(key_runs) + " |")
    print("|---" * (len(key_runs) + 1) + "|")
    for i in hard:
        cells = [f"**{scores[r][i]:.3f}**" if r == model_id else f"{scores[r][i]:.3f}" for r in key_runs]
        print(f"| {i} | " + " | ".join(cells) + " |")
    print("| **val mean** | " + " | ".join(f"{means[r]:.4f}" for r in key_runs) + " |")


if __name__ == "__main__":
    main()
