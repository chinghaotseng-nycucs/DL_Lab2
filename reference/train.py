import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    p.add_argument(
        "--sched",
        default="cosine",
        choices=["cosine", "plateau", "none"],
        help="plateau = halve the lr after 3 epochs without val improvement",
    )
    p.add_argument("--init", default="", help="start from these weights (a .pth state dict) instead of random")
    p.add_argument(
        "--early-stop", type=int, default=0, help="stop after N epochs without val gain > --min-delta; 0 = off"
    )
    p.add_argument("--min-delta", type=float, default=0.0005)
    p.add_argument("--dice-weight", type=float, default=1.0, help="L = BCE + w * softDice")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights, e.g. 0.999; 0 = off")
    p.add_argument("--sample-weights", default="", help="CSV image_id,weight: how often each training image is drawn")
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
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.sample_weights:
        # Images listed in the file are drawn in proportion to their weight; unlisted ones get 1
        with open(args.sample_weights) as f:
            given = {r["image_id"]: float(r["weight"]) for r in csv.DictReader(f)}
        weights = [given.get(i, 1.0) for i in train_ids]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ids), replacement=True, generator=generator)
        print(f"sample weights: {sum(w != 1.0 for w in weights)} images reweighted from {args.sample_weights}")
    loader = DataLoader(
        PetDataset(train_ids, args.aug),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        drop_last=True,
        persistent_workers=args.workers > 0,
        generator=generator,
    )

    model = UNet(in_channels=3, out_channels=1).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device), strict=True)
        print("init weights:", args.init)

    # EMA: a running average of the weights, validated and saved instead of the raw weights
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True) if args.ema else None
    eval_model = ema.module if ema else model

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs) if args.sched == "cosine" else None

    plateau = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3, min_lr=1e-6)
        if args.sched == "plateau"
        else None
    )
    loss_fn = make_loss(args.dice_weight)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_dice", "val_dog", "val_cat", "seconds"])

    best, best_epoch, t_start = -1.0, 0, time.time()
    if args.init:
        # Epoch 0 = the starting weights; best.pth starts as them, so it never gets worse
        best_val = summarize(official_val_dice(eval_model, val_ids, device))
        best = best_val["all"]
        torch.save(eval_model.state_dict(), os.path.join(out_dir, "best.pth"))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([0, "-", "-", f"{best:.4f}", f"{best_val['dog']:.4f}", f"{best_val['cat']:.4f}", 0])
        print(f"epoch   0 | start weights | val Dice {best:.4f} (dog {best_val['dog']:.4f}, cat {best_val['cat']:.4f})")
    ref, stale, stopped = best, 0, ""
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
            if ema:
                ema.update_parameters(model)
            total += loss.item() * img.size(0)
            n += img.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        lr = opt.param_groups[0]["lr"]

        val = summarize(official_val_dice(eval_model, val_ids, device))
        if cosine:
            cosine.step()
        if plateau:
            plateau.step(val["all"])
        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{lr:.2e}",
                    f"{total / n:.4f}",
                    f"{val['all']:.4f}",
                    f"{val['dog']:.4f}",
                    f"{val['cat']:.4f}",
                    f"{secs:.0f}",
                ]
            )
        mark = ""
        if val["all"] > best:
            best, best_epoch, best_val = val["all"], epoch, val
            # Plain state dict of the un-wrapped UNet: what the TA's strict load expects
            torch.save(eval_model.state_dict(), os.path.join(out_dir, "best.pth"))
            mark = "  * best"
        print(
            f"epoch {epoch:3d} | lr {lr:.2e} | loss {total / n:.4f} | "
            f"val Dice {val['all']:.4f} (dog {val['dog']:.4f}, cat {val['cat']:.4f}) | {secs:.0f}s{mark}"
        )
        if val["all"] > ref + args.min_delta:
            ref, stale = val["all"], 0
        else:
            stale += 1
        if args.early_stop and stale >= args.early_stop:
            stopped = f"early stop at epoch {epoch}: no gain > {args.min_delta} for {stale} epochs"
            print(stopped)
            break
    torch.save(eval_model.state_dict(), os.path.join(out_dir, "last.pth"))

    run_log = os.path.join(HERE, "runs", "run_log.csv")
    new = not os.path.exists(run_log)
    with open(run_log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "run",
                    "aug",
                    "epochs",
                    "lr",
                    "batch",
                    "sched",
                    "dice_w",
                    "seed",
                    "best_epoch",
                    "val_dice",
                    "val_dog",
                    "val_cat",
                    "minutes",
                    "note",
                ]
            )
        w.writerow(
            [
                args.run,
                args.aug,
                args.epochs,
                args.lr,
                args.batch_size,
                args.sched,
                args.dice_weight,
                args.seed,
                best_epoch,
                f"{best:.4f}",
                f"{best_val['dog']:.4f}",
                f"{best_val['cat']:.4f}",
                f"{(time.time() - t_start) / 60:.0f}",
                "; ".join(s for s in (args.note, stopped) if s),
            ]
        )
    print(f"best val Dice {best:.4f} at epoch {best_epoch} -> {os.path.join(out_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
