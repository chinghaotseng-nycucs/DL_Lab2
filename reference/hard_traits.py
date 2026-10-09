import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

from common import HERE, get_device, load_image, load_trimap

# python hard_traits.py   (after tag_images.py) -> review/analysis/:
#   tags.csv            one row per non_test image (train and val): every characteristic below as 0/1
#   tag_summary.md      per characteristic: how many train/val images have it, run 14's Dice with/without it
#   tag_sheets/*.png    examples of each characteristic (val worst first, then train)
# and review/label_check/: candidates.csv (suspicious labels in both sets, for a decision by eye) and sheets.
# Needs: pip install open_clip_torch scipy pandas. The detector and CLIP only describe photos; nothing here trains.

A = os.path.join(HERE, "review", "analysis")
LC = os.path.join(HERE, "review", "label_check")

# CLIP zero-shot characteristics: score = mean similarity to the positive prompts - to the neutral prompts.
# Thresholds set 10/8 by looking at the photos ranked by score; share of the top-ranked photos that really show it:
# blanket ~95%, bars/fence ~70%, clothes ~90%, snow ~100%, plush toy ~85%, grass ~100%. A text/logo prompt was
# tried and dropped: it mostly found photos with people.
CLIP_TAGS = {
    "blanket_or_cushion": (["a photo of a pet lying on a patterned blanket", "a photo of a pet on a fluffy rug",
                            "a photo of a pet on a cushion", "a photo of a pet wrapped in a blanket",
                            "a photo of a pet on a knitted throw"], 0.030),
    "behind_bars_or_fence": (["a photo of a pet behind the bars of a cage", "a photo of a dog behind a fence",
                              "a photo of a pet inside a wire crate"], 0.020),
    "wearing_clothes": (["a photo of a pet wearing clothes", "a photo of a dog wearing a sweater",
                         "a photo of a pet in a costume", "a photo of a pet wearing a hat"], 0.020),
    "in_snow": (["a photo of a pet in the snow"], 0.050),
    "with_plush_toy": (["a photo of a pet with a stuffed toy", "a photo of a pet holding a plush toy"], 0.040),
    "on_grass": (["a photo of a pet lying in tall grass", "a photo of a pet hidden in grass"], 0.028),
}
NEUTRAL = ["a photo of a pet", "a photo of a cat", "a photo of a dog"]


def clip_scores():
    import open_clip

    feats = np.load(os.path.join(A, "clip.npy")).astype(np.float32)
    with open(os.path.join(A, "clip_ids.json")) as f:
        ids = json.load(f)
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        def emb(texts):
            t = model.encode_text(tok(texts)).float()
            return torch.nn.functional.normalize(t, dim=-1).numpy()

        neutral = (feats @ emb(NEUTRAL).T).mean(1)
        out = {name: (feats @ emb(pos).T).mean(1) - neutral for name, (pos, _) in CLIP_TAGS.items()}
    return pd.DataFrame(out, index=ids)


def build_table():
    d = pd.read_csv(os.path.join(A, "basic.csv")).set_index("image_id")
    d = d.join(pd.read_csv(os.path.join(A, "det.csv")).set_index("image_id"))
    d = d.join(pd.read_csv(os.path.join(A, "ceiling.csv")).set_index("image_id"))
    c = clip_scores()
    d = d.join(c.add_prefix("clip_"))
    has_label = d.pet_frac >= 0.005
    t = pd.DataFrame(index=d.index)
    # photo
    t["dark_photo"] = d.lum_mean < 0.25
    t["over_exposed"] = d.frac_clipped > 0.15
    t["low_contrast"] = d.lum_std < 0.13
    t["backlit"] = has_label & ((d.ring_lum - d.pet_lum) > 0.3)
    t["dark_coat"] = has_label & (d.pet_lum < 0.2)
    t["camouflage"] = has_label & (d.pet_ring_overlap > 0.6)  # pet colours match the surroundings
    t["grayscale"] = d.gray_photo == 1
    t["very_saturated"] = d.sat_mean > 0.6
    t["blurry"] = d.blur_var < 16
    t["low_resolution"] = d[["width", "height"]].min(axis=1) < 200
    # pet and label shape
    t["small_pet"] = has_label & (d.pet_frac < 0.075)
    t["cut_by_frame"] = d.border_frac > 0.4
    t["label_in_pieces"] = d.n_parts >= 3
    t["hairless_breed"] = d.breed == "Sphynx"
    # detector (COCO Mask R-CNN)
    t["person_touching"] = d.person_near == 1
    t["second_animal_unlabelled"] = d.n_extra_animals >= 1
    t["plush_toy_detected"] = d.teddy_score >= 0.5
    t["detector_misses_pet"] = has_label & (d.det_found == 0)
    # CLIP
    for name, (_, thr) in CLIP_TAGS.items():
        t[name] = d[f"clip_{name}"] > thr
    t = t.astype(int)
    keep = ["split", "species", "breed", "dice14", "dice07", "ceiling", "pet_frac", "missed14", "extra14",
            "det_label_dice", "det_unet_dice", "n_parts", "band_frac"]
    return d, pd.concat([d[keep], t], axis=1), list(t.columns)


def boot_diff(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    da = a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    db = b[rng.integers(0, len(b), (n, len(b)))].mean(1)
    return np.percentile(da - db, [2.5, 97.5])


def summary(tab, names, hard_ids):
    v, tr = tab.split == "val", tab.split == "train"
    labelled = tab.pet_frac >= 0.005
    fail = v & (tab.dice14 < 0.8)
    rows = []
    for n in names:
        m = tab[n] == 1
        a, b = tab.dice14[v & m & labelled], tab.dice14[v & ~m & labelled]
        lo, hi = boot_diff(a, b) if len(a) >= 3 else (np.nan, np.nan)
        rows.append({
            "characteristic": n, "train_n": int((tr & m).sum()), "train_%": 100 * (tr & m).mean() / tr.mean(),
            "val_n": int((v & m).sum()), "val_Dice": a.mean(), "val_Dice_others": b.mean(), "delta": a.mean() - b.mean(),
            "ci_lo": lo, "ci_hi": hi, "train_Dice": tab.dice14[tr & m & labelled].mean(),
            "ceiling": tab.ceiling[v & m & labelled].mean(), "val_failures": int((fail & m).sum()),
            "hard_cases": int(tab.loc[tab.index.isin(hard_ids), n].sum()),
        })
    s = pd.DataFrame(rows).sort_values("delta")
    s.to_csv(os.path.join(A, "tag_summary.csv"), index=False, float_format="%.4f")
    lines = [f"val failures (run 14 Dice < 0.8): {int(fail.sum())} of {int(v.sum())}; "
             f"run 14 Dice: train {tab.dice14[tr & labelled].mean():.4f}, val {tab.dice14[v].mean():.4f}", "",
             "| characteristic | train n (%) | val n | val Dice with / without | delta (95% CI) | train Dice with | "
             "ceiling | val failures | of 23 hard cases |", "|---|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        ci = "" if np.isnan(r.ci_lo) else f" ({r.ci_lo:+.3f} to {r.ci_hi:+.3f})"
        lines.append(f"| {r.characteristic} | {r.train_n} ({getattr(r, '_3'):.1f}%) | {r.val_n} | {r.val_Dice:.3f} / "
                     f"{r.val_Dice_others:.3f} | {r.delta:+.3f}{ci} | {r.train_Dice:.3f} | {r.ceiling:.3f} | "
                     f"{r.val_failures} | {r.hard_cases} |")
    with open(os.path.join(A, "tag_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return s


def outline(mask, it=2):
    return mask & ~ndimage.binary_erosion(mask, iterations=it)


def pred14(preds, image_id, shape):
    u = np.unpackbits(preds[image_id])[: 256 * 256].reshape(256, 256)
    return np.array(Image.fromarray(u * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 0


def overlay(img, layers):
    """layers: list of (bool mask, rgb, alpha); outlines drawn with alpha 1."""
    a = np.asarray(img, dtype=np.float32) / 255
    for m, rgb, alpha in layers:
        a[m] = (1 - alpha) * a[m] + alpha * np.array(rgb)
    return a


def tag_sheets(tab, names, preds, per=16):
    out = os.path.join(A, "tag_sheets")
    os.makedirs(out, exist_ok=True)
    for n in names:
        m = tab[n] == 1
        vals = tab[m & (tab.split == "val")].sort_values("dice14").index[:10].tolist()
        trs = tab[m & (tab.split == "train")].sample(min(per - len(vals), int((m & (tab.split == "train")).sum())),
                                                     random_state=0).index.tolist()
        ids = vals + trs
        if not ids:
            continue
        cols = 4
        rows = (len(ids) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), ids):
            img, tri = load_image(i), load_trimap(i)
            p = pred14(preds, i, tri.shape)
            ax.imshow(overlay(img, [(outline(tri == 1), (0.1, 0.9, 0.1), 1.0), (outline(p), (1, 0, 1), 1.0)]))
            ax.set_title(f"{i} ({tab.split[i]}) Dice {tab.dice14[i]:.2f}", fontsize=8)
        fig.suptitle(f"{n}: green = label outline, magenta = run 14 outline (val worst first, then random train)",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(os.path.join(out, f"{n}.png"), dpi=70)
        plt.close(fig)


def label_candidates(d, tab):
    """Suspicious labels in both sets, by kind of evidence. The detector never saw Oxford labels, so a label that
    disagrees with both the detector and our UNet (which agree with each other) is the strongest signal."""
    c = pd.DataFrame(index=d.index)
    c["split"] = d.split
    c["dice14"], c["dice07"] = d.dice14, d.dice07
    c["det_label_dice"] = d.det_label_dice.where(d.pet_frac >= 0.005)  # two empty masks would count as 1.0
    c["det_unet_dice"] = d.det_unet_dice
    c["pet_frac"], c["n_parts"], c["band_frac"] = d.pet_frac, d.n_parts, d.band_frac
    reasons = {i: [] for i in d.index}
    score = pd.Series(0.0, index=d.index)
    empty = d.pet_frac < 0.005
    disagree = (d.det_found == 1) & (d.det_label_dice < 0.6) & (d.det_unet_dice > d.det_label_dice + 0.15)
    both_models = (d.dice14 < 0.7) & (d.dice07 < 0.7)
    pieces = d.n_parts >= 4
    second = (d.n_extra_animals >= 1) & (d.extra_animal_score >= 0.85) & ~empty  # with no label every animal is "outside"
    width = band_width(d)
    wide_band = (d.band_frac > 0.3) | (width > width.quantile(0.99))  # most of the pet marked as uncertain border
    c["band_width_pct"] = width
    kinds = [(empty, "empty label: no pet pixels", 10), (disagree, "detector and UNet agree, label differs", 3),
             (both_models, "runs 14 and 07 both below 0.7", 2), (wide_band, "border band unusually wide", 1),
             (pieces, "label in 4+ pieces", 1), (second, "second animal not in the label", 1)]
    for mask, why, w in kinds:
        for i in d.index[mask]:
            reasons[i].append(why)
        score += w * mask
    score += (1 - d.dice14).clip(0, 1)  # tie-break: worse fit first
    c["reasons"] = [", ".join(reasons[i]) for i in d.index]
    c["score"] = score
    c = c[c.reasons != ""].copy()
    # group the sheets by the strongest kind of evidence, so one kind can be reviewed at a time
    c["category"] = [next(k for k, (_, why, _) in enumerate(kinds) if why in r) + 1 for r in c.reasons]
    c["category_name"] = [kinds[k - 1][1].split(":")[0] for k in c.category]
    c = c.sort_values(["split", "category", "score"], ascending=[True, True, False])
    c["my_verdict"] = ""
    mv = os.path.join(LC, "my_verdicts.csv")  # written by hand after looking at the sheets
    if os.path.exists(mv):
        v = pd.read_csv(mv, index_col=0).verdict
        c["my_verdict"] = v.reindex(c.index).fillna("")
    c["your_decision"] = ""  # drop / keep / a weight such as 0.5 (see make_sample_weights.py)
    old = os.path.join(LC, "candidates.csv")
    if os.path.exists(old):  # never wipe decisions already typed in
        prev = pd.read_csv(old, index_col=0, dtype={"your_decision": str}).your_decision.fillna("")
        c["your_decision"] = prev.reindex(c.index).fillna("")
    return c


def band_width(d):
    """Average width of the border band, in % of the photo diagonal (band area / pet outline length)."""
    path = os.path.join(A, "band_width.csv")
    if not os.path.exists(path):
        out = {}
        for i in d.index:
            t = load_trimap(i)
            pet = t == 1
            perim = (pet & ~ndimage.binary_erosion(pet)).sum()
            out[i] = (t == 3).sum() / max(perim, 1) / np.hypot(*t.shape) * 100 if pet.mean() >= 0.005 else 0.0
        pd.Series(out, name="band_width_pct").rename_axis("image_id").to_csv(path)
    return pd.read_csv(path, index_col=0).band_width_pct.reindex(d.index).fillna(0.0)


@torch.no_grad()
def label_sheets(c, preds, per=8):
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    import torchvision.transforms.functional as TF

    w = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    cats = w.meta["categories"]
    device = get_device()
    det = maskrcnn_resnet50_fpn_v2(weights=w).to(device).eval()
    tf = w.transforms()
    animals = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
    os.makedirs(LC, exist_ok=True)
    for f in os.listdir(LC):  # sheets from an earlier run
        if f.endswith(".png"):
            os.remove(os.path.join(LC, f))
    jobs = []  # one set of sheets per split and category: train_1_empty_label_01.png, ...
    for (split, cat), g in c.groupby(["split", "category"], sort=False):
        slug = g.category_name.iloc[0].replace(" ", "_").replace(",", "")
        ids = g.index.tolist()
        jobs += [(f"{split}_{cat}_{slug}_{k // per + 1:02d}.png", ids[k : k + per]) for k in range(0, len(ids), per)]
    order = {i: n + 1 for n, i in enumerate(c.index)}
    for name, chunk in jobs:
        fig, axes = plt.subplots(len(chunk), 3, figsize=(12, 3.3 * len(chunk)))
        axes = np.atleast_2d(axes)
        for row, i in zip(axes, chunk):
            img, tri = load_image(i), load_trimap(i)
            o = det([tf(TF.pil_to_tensor(img)).to(device)])[0]
            dm = np.zeros(tri.shape, bool)
            for lab, sc, m in zip(o["labels"].tolist(), o["scores"].tolist(), o["masks"][:, 0].cpu().numpy()):
                if cats[lab] in animals and sc >= 0.5:
                    dm |= m > 0.5
            p = pred14(preds, i, tri.shape)
            panels = [
                (overlay(img, [(tri == 1, (0.1, 0.9, 0.1), 0.45), (tri == 3, (1, 0.85, 0), 0.35)]),
                 f"#{order[i]} {i} ({c.split[i]}): LABEL green, band yellow"),
                (overlay(img, [(p, (1, 0, 1), 0.45)]), f"run 14 prediction, Dice {c.dice14[i]:.2f}"),
                (overlay(img, [(dm, (0, 0.8, 1), 0.45)]),
                 "COCO detector animals" + ("" if np.isnan(c.det_label_dice[i]) else f", vs label {c.det_label_dice[i]:.2f}")),
            ]
            for ax, (im, title) in zip(row, panels):
                ax.imshow(im)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            row[0].text(0, -0.02, c.reasons[i], transform=row[0].transAxes, fontsize=7, va="top", color="darkred")
        fig.tight_layout()
        fig.savefig(os.path.join(LC, name), dpi=70)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-sheets", action="store_true")
    args = p.parse_args()
    d, tab, names = build_table()
    tab.to_csv(os.path.join(A, "tags.csv"), float_format="%.4f")
    hard = pd.read_csv(os.path.join(HERE, "hard_cases", "scores.csv")).image_id.tolist()
    s = summary(tab, names, hard)
    print(open(os.path.join(A, "tag_summary.md")).read())
    c = label_candidates(d, tab)
    os.makedirs(LC, exist_ok=True)
    c.to_csv(os.path.join(LC, "candidates.csv"), float_format="%.3f")
    print(f"label candidates: {len(c)} ({(c.split == 'train').sum()} train, {(c.split == 'val').sum()} val)")
    if not args.no_sheets:
        preds = np.load(os.path.join(A, "pred14_256.npz"))
        tag_sheets(tab, names, preds)
        label_sheets(c, preds)


if __name__ == "__main__":
    main()
