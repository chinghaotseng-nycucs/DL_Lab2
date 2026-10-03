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
    p.add_argument("--aug", default="none", choices=list(AUGS))
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
    # Per image, then mean over the batch, to match the mean per-image Dice
    prob = torch.sigmoid(logits)
    inter = (prob * mask).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + c) / (total + c)).mean()


def make_loss(dice_weight):
    bce = nn.BCEWithLogitsLoss()  # model outputs logits: no sigmoid in model.py
    return lambda logits, mask: bce(logits, mask) + dice_weight * soft_dice_loss(logits, mask)


def smoke_test(args, device):
    train_ids, _ = get_split()
    ids = train_ids[:16]
    loader = DataLoader(PetDataset(ids, "none"), batch_size=16, shuffle=True)
    model = UNet(in_channels=3, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = make_loss(args.dice_weight)
    img, mask = next(iter(loader))
    print("batch", tuple(img.shape), img.dtype, "| mask", tuple(mask.shape), mask.unique().tolist())
    img, mask = img.to(device), mask.to(device)
    model.train()
    for step in range(1, 51):
        opt.zero_grad()
        loss = loss_fn(model(img), mask)
        loss.backward()
        opt.step()
        if step == 1 or step % 10 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}")
    d = summarize(official_val_dice(model, ids, device))["all"]
    print(f"official-pipeline Dice on the same 16 images: {d:.4f}  (should be high)")


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

    train_ids, val_ids = get_split()
    print(f"train {len(train_ids)} | val {len(val_ids)} | aug {args.aug}")
    loader = DataLoader(
        PetDataset(train_ids, args.aug),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.sched == "cosine" else None
    loss_fn = make_loss(args.dice_weight)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_dice", "val_dog", "val_cat", "seconds"])

    best, best_epoch, t_start = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for img, mask in bar:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(img), mask)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * img.size(0)
            n += img.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        lr = opt.param_groups[0]["lr"]
        if sched:
            sched.step()

        val = summarize(official_val_dice(model, val_ids, device))
        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{lr:.2e}", f"{total / n:.4f}", f"{val['all']:.4f}", f"{val['dog']:.4f}", f"{val['cat']:.4f}", f"{secs:.0f}"]
            )
        mark = ""
        if val["all"] > best:
            best, best_epoch, best_val = val["all"], epoch, val
            # Plain state dict of the un-wrapped UNet: what the TA's strict load expects
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
            mark = "  * best"
        print(
            f"epoch {epoch:3d} | lr {lr:.2e} | loss {total / n:.4f} | "
            f"val Dice {val['all']:.4f} (dog {val['dog']:.4f}, cat {val['cat']:.4f}) | {secs:.0f}s{mark}"
        )
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
