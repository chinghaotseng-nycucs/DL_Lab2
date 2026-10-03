import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import HERE, get_device, get_split, official_val_dice, seed_everything, summarize
from dataset import AUGS, PetDataset
from model import UNet

# Train:       python train.py --run baseline --aug none --epochs 30
# Smoke test:  python train.py --smoke     (16 images, 50 steps; loss must fall toward 0)
# Each run writes runs/<run>/{config.json, log.csv, best.pth, last.pth}
# and appends one line to runs/run_log.csv.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="baseline")
    p.add_argument("--aug", default="none")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--sched", default="cosine", choices=["cosine", "none"])
    p.add_argument("--dice-weight", type=float, default=1.0, help="L = BCE + w * softDice")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--note", default="")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def soft_dice_loss(logits, mask, c=1.0):
    """logits, mask: (B, 1, H, W). Soft Dice per image, then mean over the batch."""
    # TODO(Stage 2)
    raise NotImplementedError


def make_loss(dice_weight):
    """-> function(logits, mask) returning BCE-with-logits + dice_weight * soft Dice."""
    # TODO(Stage 2)
    raise NotImplementedError


def smoke_test(args, device):
    """16 training images, one batch, 50 optimizer steps. Print the loss every 10 steps,
    then the official-pipeline Dice on those same 16 images (should be high)."""
    # TODO(Stage 2)
    raise NotImplementedError


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print("device:", device)
    if args.smoke:
        smoke_test(args, device)
        return

    out_dir = os.path.join(HERE, "runs", args.run)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_dice", "val_dog", "val_cat", "seconds"])

    # TODO(Stage 2) setup: split, DataLoader (shuffle, drop_last), UNet on device, AdamW,
    # CosineAnnealingLR when args.sched == "cosine", loss = make_loss(args.dice_weight).
    # AMP (autocast + GradScaler) only when device.type == "cuda".

    best, best_epoch, best_val, t_start = -1.0, 0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # TODO(Stage 2) one epoch:
        #   model.train(); for each batch: zero_grad -> forward -> loss -> backward -> step
        #   keep the mean train loss in `train_loss`, the lr used this epoch in `lr`
        #   scheduler step (once per epoch)
        #   val = summarize(official_val_dice(model, val_ids, device))
        #   if val["all"] > best: update best / best_epoch / best_val and
        #       torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
        raise NotImplementedError

        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{lr:.2e}", f"{train_loss:.4f}", f"{val['all']:.4f}", f"{val['dog']:.4f}", f"{val['cat']:.4f}", f"{secs:.0f}"]
            )
        print(f"epoch {epoch:3d} | lr {lr:.2e} | loss {train_loss:.4f} | val Dice {val['all']:.4f} "
              f"(dog {val['dog']:.4f}, cat {val['cat']:.4f}) | {secs:.0f}s")
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pth"))

    run_log = os.path.join(HERE, "runs", "run_log.csv")
    new = not os.path.exists(run_log)
    with open(run_log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "aug", "epochs", "lr", "batch", "sched", "dice_w", "seed", "best_epoch", "val_dice", "val_dog", "val_cat", "minutes", "note"])
        w.writerow(
            [args.run, args.aug, args.epochs, args.lr, args.batch_size, args.sched, args.dice_weight, args.seed,
             best_epoch, f"{best:.4f}", f"{best_val['dog']:.4f}", f"{best_val['cat']:.4f}", f"{(time.time() - t_start) / 60:.0f}", args.note]
        )
    print(f"best val Dice {best:.4f} at epoch {best_epoch} -> {os.path.join(out_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
