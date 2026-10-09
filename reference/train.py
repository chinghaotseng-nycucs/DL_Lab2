import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import (HERE, dice, get_device, get_split, load_image, load_mask, official_val_dice, predict_masks,
                    seed_everything, summarize)
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
    p.add_argument("--bce-weight", type=float, default=1.0, help="weight of the BCE term; 0 = soft Dice only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights, e.g. 0.999; 0 = off")
    p.add_argument("--sample-weights", default="", help="CSV image_id,weight: how often each training image is drawn")
    p.add_argument("--select", default="best", choices=["best", "last"],
                   help="which epoch becomes best.pth: best = highest clean val Dice; last = the final epoch")
    p.add_argument("--sampler", default="weighted", choices=["weighted", "repeat"],
                   help="how --sample-weights is applied. weighted: random draws with replacement (run 11); "
                   "repeat: every image once per epoch like plain shuffling, weight w = shown w times on average")
    p.add_argument("--exclude", default="", help="text file of training image IDs to leave out (bad labels); "
                   "one per line, # starts a comment; val is never changed")
    p.add_argument("--copy-paste", type=float, default=0.0,
                   help="chance of pasting another training image's pet onto each training image; 0 = off")
    p.add_argument("--exact-masks", action="store_true",
                   help="resize training masks around pixel centres (NEAREST_EXACT) like the photos; "
                   "off = torchvision's NEAREST, as in runs 01-22")
    p.add_argument("--note", default="")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (offline if not logged in)")
    p.add_argument("--wandb-project", default="DL_Lab2")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# Validation images shown in W&B at the end of a run: four known failure cases (hard_cases/) and four ordinary ones.
# All are in val_ids.txt; log_examples also skips any ID that isn't, so test images can never be loaded here.
WANDB_EXAMPLES = ["basset_hound_191", "newfoundland_11", "shiba_inu_92", "english_setter_194",
                  "Abyssinian_148", "scottish_terrier_23", "english_cocker_spaniel_12", "Ragdoll_166"]


def start_wandb(args):
    """A W&B run for this training, or None. Never blocks and never stops training:
    logged in -> online; not logged in -> offline (upload later with `wandb sync runs/wandb/offline-run-*`)."""
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed (pip install wandb); training continues without it")
        return None
    try:
        mode = os.environ.get("WANDB_MODE") or ("online" if wandb.login(prompt=False) else "offline")
        run = wandb.init(project=args.wandb_project, name=args.run, config=vars(args), notes=args.note,
                         tags=[args.aug, args.sched], dir=os.path.join(HERE, "runs"), mode=mode)
        print(f"wandb: {mode} run {run.id}")
        return run
    except Exception as e:  # a logging problem must never cost a training run
        print(f"wandb disabled: {e}")
        return None


def log_examples(wb, best_path, val_ids, device):
    """Official-pipeline predictions of best.pth on WANDB_EXAMPLES, as images with true and predicted masks."""
    import wandb

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device), strict=True)
    model.eval()
    ids = [i for i in WANDB_EXAMPLES if i in set(val_ids)]
    imgs = [load_image(i) for i in ids]
    labels = {0: "background", 1: "pet"}
    images = []
    for i, img, pred in zip(ids, imgs, predict_masks(model, imgs, device)):
        gt = load_mask(i)
        images.append(wandb.Image(np.asarray(img), caption=f"{i}  Dice {dice(pred, gt):.3f}", masks={
            "ground_truth": {"mask_data": gt, "class_labels": labels},
            "prediction": {"mask_data": pred, "class_labels": labels},
        }))
    wb.log({"examples": images})


class RepeatSampler(torch.utils.data.Sampler):
    """Each epoch, image i appears floor(w) times, plus once more with probability w - floor(w), in shuffled order.
    Weight 1 = exactly once per epoch, the same as shuffle=True; 0.5 = in about half of the epochs; 2 = twice.
    Unlike WeightedRandomSampler(replacement=True), images with weight 1 are never skipped or repeated."""

    def __init__(self, weights, generator):
        self.w = torch.tensor(weights, dtype=torch.float64)
        self.generator = generator

    def __iter__(self):
        base = self.w.floor()
        extra = torch.rand(len(self.w), generator=self.generator, dtype=torch.float64) < (self.w - base)
        idx = torch.repeat_interleave(torch.arange(len(self.w)), (base + extra).long())
        return iter(idx[torch.randperm(len(idx), generator=self.generator)].tolist())

    def __len__(self):
        return int(self.w.sum().round())  # expected length; one epoch may differ by a few images


def soft_dice_loss(logits, mask, c=1.0):
    # Per image, then mean over the batch, to match the mean per-image Dice.
    # float32: under CUDA AMP the logits are float16, whose max (65,504) is below 256 x 256 = 65,536 pixels,
    # so a pet filling the frame would make the sum inf. On the Mac the logits are already float32 (no change).
    prob = torch.sigmoid(logits.float())
    inter = (prob * mask).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + c) / (total + c)).mean()


def make_loss(dice_weight, bce_weight=1.0):
    bce = nn.BCEWithLogitsLoss()  # model outputs logits: no sigmoid in model.py
    # With AdamW only the ratio of the two weights matters, not the overall scale of the loss
    return lambda logits, mask: bce_weight * bce(logits, mask) + dice_weight * soft_dice_loss(logits, mask)


def smoke_test(args, device):
    train_ids, _ = get_split()
    ids = train_ids[:16]
    loader = DataLoader(PetDataset(ids, "none"), batch_size=16, shuffle=True)
    model = UNet(in_channels=3, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = make_loss(args.dice_weight, args.bce_weight)
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
    if args.exclude:
        with open(args.exclude) as f:
            drop = {line.split("#")[0].strip() for line in f} - {""}
        n_before = len(train_ids)
        train_ids = [i for i in train_ids if i not in drop]
        ignored = drop & set(val_ids)  # val stays the same for every run, so listed val IDs are ignored
        print(f"exclude: {n_before - len(train_ids)} training images left out ({args.exclude})"
              + (f"; {len(ignored)} val IDs in the list ignored" if ignored else ""))
    print(f"train {len(train_ids)} | val {len(val_ids)} | aug {args.aug}")
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.sample_weights:
        # Images listed in the file are drawn in proportion to their weight; unlisted ones get 1
        with open(args.sample_weights) as f:
            given = {r["image_id"]: float(r["weight"]) for r in csv.DictReader(f)}
        weights = [given.get(i, 1.0) for i in train_ids]
        if args.sampler == "repeat":
            sampler = RepeatSampler(weights, generator)
        else:
            sampler = WeightedRandomSampler(weights, num_samples=len(train_ids), replacement=True, generator=generator)
        print(f"sample weights: {sum(w != 1.0 for w in weights)} images reweighted from {args.sample_weights} "
              f"({args.sampler} sampler)")
    loader = DataLoader(
        PetDataset(train_ids, args.aug, copy_paste_p=args.copy_paste, exact_masks=args.exact_masks),
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
    loss_fn = make_loss(args.dice_weight, args.bce_weight)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "val_dice", "val_dog", "val_cat", "seconds"])

    wb = start_wandb(args)
    best, best_epoch, t_start = -1.0, 0, time.time()
    if args.init:
        # Epoch 0 = the starting weights; best.pth starts as them, so it never gets worse
        best_val = summarize(official_val_dice(eval_model, val_ids, device))
        best = best_val["all"]
        torch.save(eval_model.state_dict(), os.path.join(out_dir, "best.pth"))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([0, "-", "-", f"{best:.4f}", f"{best_val['dog']:.4f}", f"{best_val['cat']:.4f}", 0])
        print(f"epoch   0 | start weights | val Dice {best:.4f} (dog {best_val['dog']:.4f}, cat {best_val['cat']:.4f})")
        if wb:
            wb.log({"val/dice": best, "val/dog": best_val["dog"], "val/cat": best_val["cat"], "val/best": best}, step=0)
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
        # --select last: best.pth follows the latest epoch, so a short fine-tune that trades a little clean val Dice
        # for robustness is not thrown away in favour of its starting weights (exp21/22 kept epoch 0)
        if val["all"] > best or args.select == "last":
            best, best_epoch, best_val = val["all"], epoch, val
            # Plain state dict of the un-wrapped UNet: what the TA's strict load expects
            torch.save(eval_model.state_dict(), os.path.join(out_dir, "best.pth"))
            mark = "  * best"
        print(
            f"epoch {epoch:3d} | lr {lr:.2e} | loss {total / n:.4f} | "
            f"val Dice {val['all']:.4f} (dog {val['dog']:.4f}, cat {val['cat']:.4f}) | {secs:.0f}s{mark}"
        )
        if wb:
            wb.log({"lr": lr, "train/loss": total / n, "val/dice": val["all"], "val/dog": val["dog"],
                    "val/cat": val["cat"], "val/best": best, "time/epoch_s": secs}, step=epoch)
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
    if wb:
        wb.summary.update({"best_val_dice": best, "best_epoch": best_epoch, "best_val_dog": best_val["dog"],
                           "best_val_cat": best_val["cat"], "stopped": stopped or "ran all epochs"})
        try:
            log_examples(wb, os.path.join(out_dir, "best.pth"), val_ids, device)
        except Exception as e:
            print(f"wandb examples skipped: {e}")
        wb.finish()


if __name__ == "__main__":
    main()
