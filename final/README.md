# Lab2 — my version

Workspace for writing the pipeline myself. It has the same files and function names as `../reference/`, so `tests.py` can compare the two. Study guide: Notion → Lab2 → *Study guide: Lab2 reference code*.

## What is mine to write

| Stage | File | TODO |
|---|---|---|
| 1 | `common.py` | `load_mask`, `get_split`, `preprocess`, `predict_masks`, `dice`, `official_val_dice` |
| 1 | `dataset.py` | `AUGS["none"]`, `build_transform`, `PetDataset.__getitem__` |
| 2 | `train.py` | `soft_dice_loss`, `make_loss`, `smoke_test`, the setup and epoch body in `main` |
| 3 | `common.py` | `rle_encode`, `rle_decode` |
| 3 | `predict.py` | `main` body |
| 4 | `dataset.py` | `AUGS["geo"]`, `AUGS["geo_color"]` (or my own presets) |

Given (not the point of the lab): `model.py` (TA, unmodified), paths/constants, `seed_everything`, `get_device`, `read_ids`, `is_dog`, `load_image`, `summarize`, the CSV logging in `train.py`, `check_weights.py` (TA's snippet), `evaluate.py`, `show_samples.py`.

## Loop per stage

1. Write the TODOs for the stage without opening `../reference/`
2. `python tests.py <stage>` until it shows no TODO, FAIL or ERR
3. Then read the matching part of `../reference/` and note what differs

## Commands (run inside `final/`)

```bash
python tests.py            # all stages; python tests.py 1 for Stage 1 only
python show_samples.py     # figures/samples.png: the boundary ring must be black
python train.py --smoke
python train.py --run baseline --aug none --epochs 30
python evaluate.py --weights runs/baseline/best.pth
python predict.py  --weights runs/baseline/best.pth --out submission.csv
python check_weights.py runs/baseline/best.pth
```
