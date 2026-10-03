import numpy as np
import torch

from common import dice, get_split, load_mask, rle_decode, rle_encode
from dataset import PetDataset
from model import UNet

# python tests.py
# Road Map gates and sanity checks. Every line should print OK.


def ok(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")
    assert cond, name


# Stage 0 gate: one logit per pixel per image
model = UNet(in_channels=3, out_channels=1)
out = model(torch.randn(2, 3, 256, 256))
n_params = sum(p.numel() for p in model.parameters())
ok("model output shape", out.shape == (2, 1, 256, 256), f"{tuple(out.shape)}, {n_params:,} params")
ok("last layer has no sigmoid", isinstance(model.outc, torch.nn.Conv2d))

# Stage 1: Dice
t = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]])
p = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]])
ok("Dice 4x4 example = 8/11", abs(dice(p, t) - 8 / 11) < 1e-9, f"{dice(p, t):.3f}")
ok("Dice all-ones = 0.545", abs(dice(np.ones_like(t), t) - 12 / 22) < 1e-9)
ok("Dice mask with itself = 1", dice(t, t) == 1.0)
ok("Dice empty prediction = 0", dice(np.zeros_like(t), t) == 0.0)

# Stage 3: RLE
m = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0]])
ok("RLE 3x4 example", rle_encode(m) == "4 2 7 3", repr(rle_encode(m)))
ok("RLE self-test 2x3", rle_encode(np.array([[1, 0, 1], [1, 0, 0]])) == "1 2 5 1")
ok("RLE empty mask", rle_encode(np.zeros((5, 7))) == "")
ok("RLE full mask", rle_encode(np.ones((5, 7))) == "1 35")
rng = np.random.default_rng(0)
ok("RLE round trip, 200 random masks",
   all((rle_decode(rle_encode(r), r.shape) == r).all() for r in (rng.integers(0, 2, (rng.integers(1, 40), rng.integers(1, 40))) for _ in range(200))))

# Needs the dataset
train_ids, val_ids = get_split()
ok("split is disjoint", not set(train_ids) & set(val_ids), f"train {len(train_ids)}, val {len(val_ids)}")
ok("RLE round trip, 20 val masks", all((rle_decode(rle_encode(load_mask(i)), load_mask(i).shape) == load_mask(i)).all() for i in val_ids[:20]))
for aug in ["none", "geo", "geo_color"]:
    img, mask = PetDataset(train_ids[:4], aug)[0]
    vals = mask.unique().tolist()
    ok(f"dataset[{aug}] shapes and binary mask", img.shape == (3, 256, 256) and mask.shape == (1, 256, 256) and set(vals) <= {0.0, 1.0},
       f"img mean {img.mean():+.2f}, mask values {vals}")
