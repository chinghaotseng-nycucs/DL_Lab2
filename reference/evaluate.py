import argparse

import torch

from common import get_device, get_split, official_val_dice, summarize
from model import UNet

# python evaluate.py --weights runs/baseline/best.pth
# Official-style val Dice (original size, nearest upsampling) plus the worst images.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--worst", type=int, default=10)
    args = p.parse_args()

    device = get_device()
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)

    _, val_ids = get_split()
    scores = official_val_dice(model, val_ids, device)
    s = summarize(scores)
    print(f"val Dice {s['all']:.4f} | dog {s['dog']:.4f} | cat {s['cat']:.4f} | n={len(scores)}")
    print(f"worst {args.worst}:")
    for i, d in sorted(scores.items(), key=lambda kv: kv[1])[: args.worst]:
        print(f"  {d:.4f}  {i}")


if __name__ == "__main__":
    main()
