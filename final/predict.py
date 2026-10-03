import argparse
import csv

import torch

from common import get_device, load_image, predict_masks, read_ids, rle_encode
from model import UNet

# python predict.py --weights runs/baseline/best.pth --out submission.csv
# Official pipeline on every test.csv ID -> image_id,encoded_mask (RLE). Empty mask -> empty field.


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    # TODO(Stage 3):
    #   UNet(in_channels=3, out_channels=1), load the weights with strict=True, eval()
    #   for test.csv IDs in batches: predict_masks -> rle_encode
    #   write the header + one row per ID, in test.csv order (csv module, not pandas)
    #   print the row count and how many masks came out empty
    raise NotImplementedError


if __name__ == "__main__":
    main()
