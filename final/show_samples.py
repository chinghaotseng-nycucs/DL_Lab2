import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from common import DATA_ROOT, HERE, MEAN, STD, get_split, load_image, load_mask
from dataset import PetDataset

# python show_samples.py              -> figures/samples.png    (trimap vs binary mask: boundary ring must be black)
# python show_samples.py --aug geo_color  -> figures/aug_geo_color.png (4 random draws of the same image)


def denorm(x):
    x = x * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aug", default=None)
    p.add_argument("--n", type=int, default=3)
    args = p.parse_args()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    train_ids, _ = get_split()

    if args.aug is None:
        ids = train_ids[: args.n]
        fig, axes = plt.subplots(len(ids), 3, figsize=(10, 3.3 * len(ids)))
        for row, i in zip(axes, ids):
            trimap = np.array(Image.open(os.path.join(DATA_ROOT, "annotations", "trimaps", f"{i}.png")))
            for ax, im, title in zip(row, [load_image(i), trimap, load_mask(i)], [i, "trimap (1 pet, 2 bg, 3 boundary)", "mask = (trimap == 1)"]):
                ax.imshow(im, cmap=None if ax is row[0] else "gray")
                ax.set_title(title, fontsize=9)
                ax.axis("off")
        out = os.path.join(HERE, "figures", "samples.png")
    else:
        ds = PetDataset(train_ids[:1], args.aug)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for k in range(4):
            img, mask = ds[0]
            axes[0, k].imshow(denorm(img))
            axes[1, k].imshow(mask[0], cmap="gray")
            axes[0, k].axis("off")
            axes[1, k].axis("off")
        fig.suptitle(f"{args.aug}: 4 draws of {train_ids[0]}")
        out = os.path.join(HERE, "figures", f"aug_{args.aug}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    print("saved", out)


if __name__ == "__main__":
    main()
