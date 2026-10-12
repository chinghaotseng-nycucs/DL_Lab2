import argparse
import csv

import torch

from common import TTA_SIZES, get_device, load_image, predict_masks, predict_masks_tta, read_ids, rle_encode
from model import UNet

# python predict.py --weights runs/baseline/best.pth --out submission.csv
# Official pipeline on every test.csv ID -> image_id,encoded_mask (RLE, column-major, 1-based,
# at the original image size). Empty mask -> empty field.
# --tta: test-time augmentation for the Kaggle CSV (mirrored + several input sizes, averaged; common.predict_masks_tta).
# The submitted .pth is unchanged: the TA's private evaluation always runs the official single pass.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tta", action="store_true", help=f"test-time augmentation: mirrored + sizes {TTA_SIZES}, averaged")
    args = p.parse_args()
    predict = predict_masks_tta if args.tta else predict_masks

    device = get_device()
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)
    model.eval()

    ids = read_ids("test.csv")
    rows, empty = [], 0
    for start in range(0, len(ids), args.batch_size):
        chunk = ids[start : start + args.batch_size]
        for i, mask in zip(chunk, predict(model, [load_image(i) for i in chunk], device)):
            rle = rle_encode(mask)
            empty += rle == ""
            rows.append([i, rle])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "encoded_mask"])
        w.writerows(rows)
    assert [r[0] for r in rows] == ids
    print(f"wrote {len(rows)} rows to {args.out} ({empty} empty masks)"
          + (f"; TTA: mirrored + sizes {TTA_SIZES}" if args.tta else "; official pipeline"))


if __name__ == "__main__":
    main()
