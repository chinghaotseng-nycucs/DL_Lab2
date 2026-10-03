import argparse
import csv

import torch

from common import get_device, load_image, predict_masks, read_ids, rle_encode
from model import UNet

# python predict.py --weights runs/baseline/best.pth --out submission.csv
# Official pipeline on every test.csv ID -> image_id,encoded_mask (RLE, column-major, 1-based,
# at the original image size). Empty mask -> empty field.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    device = get_device()
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)
    model.eval()

    ids = read_ids("test.csv")
    rows, empty = [], 0
    for start in range(0, len(ids), args.batch_size):
        chunk = ids[start : start + args.batch_size]
        for i, mask in zip(chunk, predict_masks(model, [load_image(i) for i in chunk], device)):
            rle = rle_encode(mask)
            empty += rle == ""
            rows.append([i, rle])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "encoded_mask"])
        w.writerows(rows)
    assert [r[0] for r in rows] == ids
    print(f"wrote {len(rows)} rows to {args.out} ({empty} empty masks)")


if __name__ == "__main__":
    main()
