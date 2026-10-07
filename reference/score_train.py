import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import HERE, get_device, get_split, is_dog, load_image, load_mask, official_val_dice, predict_masks
from model import UNet

# python score_train.py --weights runs/<run>/best.pth --worst 300
# Official-pipeline Dice of every TRAINING image -> runs/train_scores.csv (worst first),
# plus review sheets of the worst ones in review/ for checking their labels by eye.
# Validation images are never touched, so validation Dice stays honest.

PER_SHEET = 12  # 6 rows x 2 images, each image = true-mask panel + prediction panel


def overlay(img, mask, rgb):
    a = np.asarray(img, dtype=np.float32) / 255
    a[mask == 1] = 0.55 * a[mask == 1] + 0.45 * np.array(rgb)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--worst", type=int, default=300)
    args = p.parse_args()

    device = get_device()
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)

    train_ids, _ = get_split()
    scores = official_val_dice(model, train_ids, device)  # same official pipeline, on training IDs
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    with open(os.path.join(HERE, "runs", "train_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "dice", "dog"])
        w.writerows([i, f"{d:.4f}", int(is_dog(i))] for i, d in ranked)
    d = np.array([s for _, s in ranked])
    print(f"train Dice {d.mean():.4f} | below 0.8: {(d < 0.8).sum()} | below 0.5: {(d < 0.5).sum()} | "
          f"worst {args.worst} are below {ranked[args.worst - 1][1]:.3f}")

    out = os.path.join(HERE, "review")
    os.makedirs(out, exist_ok=True)
    worst = ranked[: args.worst]
    model.eval()
    for s in range(0, len(worst), PER_SHEET):
        chunk = worst[s : s + PER_SHEET]
        fig, axes = plt.subplots(6, 4, figsize=(16, 22))
        for k, ax_pair in enumerate(axes.reshape(-1, 2)):
            for a in ax_pair:
                a.axis("off")
            if k >= len(chunk):
                continue
            i, dsc = chunk[k]
            img = load_image(i)
            pred = predict_masks(model, [img], device)[0]
            ax_pair[0].imshow(overlay(img, load_mask(i), (0, 1, 0)))
            ax_pair[0].set_title(f"#{s + k + 1}  {i}  LABEL (green)", fontsize=9)
            ax_pair[1].imshow(overlay(img, pred, (1, 0, 1)))
            ax_pair[1].set_title(f"prediction (magenta), Dice {dsc:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"sheet_{s // PER_SHEET + 1:02d}.png"), dpi=60)
        plt.close(fig)

    with open(os.path.join(out, "worst_train.txt"), "w") as f:
        f.write("\n".join(i for i, _ in worst) + "\n")
    bad = os.path.join(out, "bad_labels.txt")
    if not os.path.exists(bad):
        with open(bad, "w") as f:
            f.write("# One image ID per line whose GREEN label is wrong (misses the pet, covers background or a person).\n"
                    "# These keep normal weight; every other ID in worst_train.txt gets extra weight.\n")
    print(f"{len(worst)} images on {(len(worst) + PER_SHEET - 1) // PER_SHEET} sheets in {out}")


if __name__ == "__main__":
    main()
