import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage

from common import DATA_ROOT, HERE, dice, get_device, get_split, is_dog, load_image, preprocess
from model import UNet

# python tag_images.py   -> review/analysis/: per-image features of every non_test image (train and val)
#   basic.csv   label shape, photo statistics, Dice and error types of runs 14 and 07     (about 5 min on the Mac)
#   det.csv     COCO Mask R-CNN: people near the pet, other animals, plush toys, furniture,
#               and how well its own pet mask agrees with the label                       (about 50 min)
#   clip.npy    CLIP ViT-L/14 image embeddings, for zero-shot tags such as "blanket"     (about 10 min)
# Each step is skipped when its file exists. Needs: pip install open_clip_torch scipy
# Analysis only: the detector and CLIP describe the photos; they never train the submitted model.

OUT = os.path.join(HERE, "review", "analysis")
RUNS = {"14": "14_aug-geo_light_from08_val0.9193", "07": "07_aug-geo_color_from05_val0.9117"}
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


def load_trimap(image_id):
    # 1 = pet, 2 = background, 3 = border band
    return np.array(Image.open(os.path.join(DATA_ROOT, "annotations", "trimaps", f"{image_id}.png")))


def all_ids():
    train_ids, val_ids = get_split()
    return [(i, "train") for i in train_ids] + [(i, "val") for i in val_ids]


def breed(image_id):
    return image_id.rsplit("_", 1)[0]


def write_csv(path, rows):
    keys = list(rows[0])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")


def photo_stats(img, tri256):
    """Brightness, contrast, colour and sharpness of the 256 x 256 photo the model sees, and pet-vs-surroundings."""
    a = np.asarray(img.resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    pet = tri256 == 1
    ring = ndimage.binary_dilation(tri256 != 2, iterations=12) & (tri256 == 2)  # background right around the pet
    q = (np.clip(a, 0, 0.999) * 4).astype(int)  # 4 x 4 x 4 colour bins
    codes = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]

    def hist(m):
        h = np.bincount(codes[m], minlength=64).astype(np.float64)
        return h / max(h.sum(), 1)

    return {
        "lum_mean": float(y.mean()),
        "lum_std": float(y.std()),
        "frac_dark": float((y < 0.08).mean()),
        "frac_clipped": float((y > 0.97).mean()),
        "sat_mean": float(sat.mean()),
        "gray_photo": int(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 0.01),
        "pet_lum": float(y[pet].mean()) if pet.any() else -1.0,
        "ring_lum": float(y[ring].mean()) if ring.any() else -1.0,
        "pet_ring_overlap": float(np.minimum(hist(pet), hist(ring)).sum()) if pet.any() and ring.any() else -1.0,
        "blur_var": float(ndimage.laplace(y).var() * 1e4),
    }


def label_shape(tri):
    pet, band = tri == 1, tri == 3
    n = max(int(pet.sum()), 1)
    lab, k = ndimage.label(pet, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:] if k else np.array([])
    border = np.concatenate([tri[0], tri[-1], tri[:, 0], tri[:, -1]])
    return {
        "pet_frac": float(pet.mean()),
        "band_frac": float(band.mean()),
        "n_parts": int((sizes >= 0.02 * n).sum()),
        "border_frac": float((border != 2).mean()),  # share of the photo's edge covered by pet or band
    }


@torch.no_grad()
def step_basic(device):
    path = os.path.join(OUT, "basic.csv")
    if os.path.exists(path):
        return
    models = {}
    for r, name in RUNS.items():
        m = UNet(in_channels=3, out_channels=1).to(device)
        m.load_state_dict(torch.load(os.path.join(HERE, "runs", name, "best.pth"), map_location=device), strict=True)
        models[r] = m.eval()
    ids = all_ids()
    rows, preds = [], {}
    t0 = time.time()
    for s in range(0, len(ids), 16):
        chunk = ids[s : s + 16]
        imgs = [load_image(i) for i, _ in chunk]
        x = torch.stack([preprocess(im) for im in imgs]).to(device)
        probs = {r: torch.sigmoid(m(x))[:, 0].cpu().numpy() for r, m in models.items()}
        for k, ((i, split), img) in enumerate(zip(chunk, imgs)):
            tri = load_trimap(i)
            pet, band = tri == 1, tri == 3
            tri256 = np.array(Image.fromarray(tri).resize((256, 256), Image.NEAREST))
            row = {"image_id": i, "split": split, "species": "dog" if is_dog(i) else "cat", "breed": breed(i),
                   "width": img.size[0], "height": img.size[1]}
            row.update(label_shape(tri))
            row.update(photo_stats(img, tri256))
            n256 = max(int((tri256 == 1).sum()), 1)
            for r, p in probs.items():
                p = p[k]
                pred = TF.resize(torch.from_numpy((p > 0.5).astype(np.float32))[None], list(tri.shape),
                                 interpolation=TF.InterpolationMode.NEAREST)[0].numpy().astype(np.uint8)
                npet = max(int(pet.sum()), 1)
                row[f"dice{r}"] = float(dice(pred, pet.astype(np.uint8)))
                row[f"missed{r}"] = float((pet & (pred == 0)).sum() / npet)  # share of the true pet not predicted
                row[f"extra{r}"] = float(((tri == 2) & (pred == 1)).sum() / npet)  # predicted pet outside pet + band
                # confident disagreement with the label, border band excluded
                row[f"conf_extra{r}"] = float(((p > 0.9) & (tri256 == 2)).sum() / n256)
                row[f"conf_missed{r}"] = float(((p < 0.1) & (tri256 == 1)).sum() / n256)
                if r == "14":
                    preds[i] = np.packbits(p > 0.5)
            rows.append(row)
        if s % 800 == 0:
            print(f"basic {s}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    write_csv(path, rows)
    np.savez_compressed(os.path.join(OUT, "pred14_256.npz"), **preds)
    print("wrote", path, flush=True)


@torch.no_grad()
def step_det(device):
    path = os.path.join(OUT, "det.csv")
    if os.path.exists(path):
        return
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

    w = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    cats = w.meta["categories"]
    model = maskrcnn_resnet50_fpn_v2(weights=w).to(device).eval()
    tf = w.transforms()
    preds = np.load(os.path.join(OUT, "pred14_256.npz"))
    ids = all_ids()
    rows, raw = [], {}
    t0 = time.time()
    for s in range(0, len(ids), 4):
        chunk = ids[s : s + 4]
        imgs = [load_image(i) for i, _ in chunk]
        outs = model([tf(TF.pil_to_tensor(im)).to(device) for im in imgs])
        for (i, split), img, o in zip(chunk, imgs, outs):
            tri = load_trimap(i)
            pet, region = tri == 1, tri != 2
            npet = max(int(pet.sum()), 1)
            r = max(3, round(0.03 * max(tri.shape)))
            near = ndimage.binary_dilation(region, iterations=r)
            u = np.unpackbits(preds[i])[: 256 * 256].reshape(256, 256)
            unet = np.array(Image.fromarray(u * 255).resize((tri.shape[1], tri.shape[0]), Image.NEAREST)) > 0
            keep = o["scores"] >= 0.3
            labels = [cats[k] for k in o["labels"][keep].tolist()]
            scores = o["scores"][keep].cpu().numpy()
            masks = (o["masks"][keep, 0] > 0.5).cpu().numpy()
            dets, animal_union = [], np.zeros_like(pet)
            n_person = person_near = 0
            person_overlap = teddy = furniture = extra_max = 0.0
            n_extra = n_in = 0
            for lab, sc, m in zip(labels, scores, masks):
                area = max(int(m.sum()), 1)
                inside = float((m & region).sum() / area)
                dets.append([lab, round(float(sc), 3), round(area / m.size, 4), round(inside, 3)])
                if lab == "person" and sc >= 0.6:
                    n_person += 1
                    if (m & near).sum() >= 0.01 * npet:
                        person_near = 1
                    person_overlap = max(person_overlap, float((m & region).sum() / npet))
                elif lab in ANIMALS and sc >= 0.5:
                    if inside >= 0.3:
                        animal_union |= m
                        n_in += 1
                    elif sc >= 0.7 and area >= 0.005 * m.size:
                        n_extra += 1
                        extra_max = max(extra_max, float(sc))
                elif lab == "teddy bear":
                    teddy = max(teddy, float(sc))
                elif lab in ("bed", "couch", "chair"):
                    furniture = max(furniture, float(sc))
            raw[i] = dets
            rows.append({
                "image_id": i, "n_person": n_person, "person_near": person_near, "person_overlap": person_overlap,
                "n_animals_on_label": n_in, "n_extra_animals": n_extra, "extra_animal_score": extra_max,
                "teddy_score": teddy, "furniture_score": furniture, "det_found": int(animal_union.any()),
                "det_label_dice": float(dice(animal_union.astype(np.uint8), pet.astype(np.uint8))),
                "det_unet_dice": float(dice(animal_union.astype(np.uint8), unet.astype(np.uint8))),
            })
        if s % 400 == 0:
            print(f"det {s}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
    write_csv(path, rows)
    with open(os.path.join(OUT, "det.json"), "w") as f:
        json.dump(raw, f)
    print("wrote", path, flush=True)


@torch.no_grad()
def step_clip(device):
    path = os.path.join(OUT, "clip.npy")
    if os.path.exists(path):
        return
    import open_clip

    model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()
    ids = all_ids()
    feats = []
    for s in range(0, len(ids), 32):
        batch = []
        for i, _ in ids[s : s + 32]:
            img = load_image(i)
            side = max(img.size)  # pad to a square so CLIP's centre crop keeps the whole photo
            sq = Image.new("RGB", (side, side), (124, 116, 104))
            sq.paste(img, ((side - img.size[0]) // 2, (side - img.size[1]) // 2))
            batch.append(prep(sq))
        f = model.encode_image(torch.stack(batch).to(device))
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float16))
    np.save(path, np.concatenate(feats))
    with open(os.path.join(OUT, "clip_ids.json"), "w") as f:
        json.dump([i for i, _ in ids], f)
    print("wrote", path, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", default="basic,det,clip")
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = get_device()
    for step in args.steps.split(","):
        {"basic": step_basic, "det": step_det, "clip": step_clip}[step](device)


if __name__ == "__main__":
    main()
