# DL Lab2: binary pet segmentation (fixed UNet)

Training, validation and Kaggle-CSV code around the TA's unmodified `model.py`.
Submitted model: run 31b (`DL_Lab2_315551118_曾敬豪.pth`), Kaggle public 0.93225.

Run every command below inside this folder (`reference/`).

## 1. Setup

```bash
pip install -r ../requirements.txt      # numpy, pillow, torch, torchvision, tqdm, matplotlib, wandb (tested with Python 3.13)

# Oxford-IIIT Pet images and trimaps
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
mkdir -p /path/to/oxford-iiit-pet
tar -xzf images.tar.gz      -C /path/to/oxford-iiit-pet
tar -xzf annotations.tar.gz -C /path/to/oxford-iiit-pet

export PET_ROOT=/path/to/oxford-iiit-pet    # contains images/ and annotations/trimaps/
export PET_CSV_DIR=/path/to/csvs            # contains the TA's non_test.csv and test.csv

python tests.py                             # every line should start with OK
```

Without the two variables, the code looks in `../data/oxford-iiit-pet` and `../material`.

## 2. Predict with the submitted weights

```bash
python check_weights.py DL_Lab2_315551118_曾敬豪.pth                                  # strict load with the official model.py
python predict.py  --weights DL_Lab2_315551118_曾敬豪.pth --tta --out submission.csv   # the Kaggle CSV
python evaluate.py --weights DL_Lab2_315551118_曾敬豪.pth                             # val Dice, official pipeline
```

- **`--tta`** (Kaggle CSV only): each photo is predicted as is and mirrored at 256, 288, 320 and 352 pixels, and the
  8 probability maps are averaged before the 0.5 threshold. The private evaluation runs the `.pth` with the official
  single pass. Without `--tta`, `predict.py` uses the official pipeline
- **Val score:** 31b was trained on train + val, so its val Dice (0.954) is not a fair score. The same recipe trained
  on train only (runs 32 and 34) scores 0.9300, or 0.9374 with `--tta`
- **CSV format:** RLE with 1-based starts, column-major, at the original image size

## 3. Retrain the submitted model

```bash
python experiments/exp31_final_all_data.py --no-wandb     # add --workers 2 on a machine with few CPU cores
```

Two stages, about 2 h 15 min on an RTX 4090 (CPU and Apple GPUs are much slower):

- **31a:** 200 epochs from the ImageNet VGG13-BN encoder start, `init/vgg13bn_encoder.pth` (included in the zip, so no
  download is needed; if it is missing, the script rebuilds it, downloading VGG13-BN from torchvision once)
- **31b:** 31a + 18 epochs with mild camera effects (blur, JPEG, noise at 10% each)

The result is `runs/31b_*/best.pth`, the submitted weights, plus `RESULT.md`, `evaluate.txt` and `submission.csv`
(official single pass). Make the Kaggle CSV from it as in step 2.

Without the experiment script, the same two runs are:

```bash
python -c "from make_init_weights import make_vgg13bn; make_vgg13bn()"
python train.py --run 31a --aug geo_light --epochs 200 --lr 1e-3 --sched cosine --dice-weight 10 \
    --exclude lists/exclude_v1.txt --sample-weights lists/weights_v1.csv --sampler repeat --all-data --select last \
    --init init/vgg13bn_encoder.pth
python train.py --run 31b --aug geo_light_cam_mild --epochs 18 --lr 1e-4 --sched cosine --dice-weight 10 \
    --exclude lists/exclude_v1.txt --sample-weights lists/weights_v1.csv --sampler repeat --all-data --select last \
    --init runs/31a/best.pth
```

## Files

| File | What it does |
|---|---|
| `model.py` | the TA's UNet, unmodified |
| `common.py` | data paths, the fixed split, official inference (`predict_masks`), TTA (`predict_masks_tta`), Dice, RLE |
| `dataset.py` | dataset and augmentation presets (`geo_light`, `geo_light_cam_mild`, ...) |
| `train.py` | training loop; `python train.py --help` lists every option |
| `predict.py`, `evaluate.py`, `check_weights.py`, `tests.py` | Kaggle CSV, val Dice, strict-load check, sanity checks |
| `make_init_weights.py` | pretrained starting weights (ImageNet VGG13-BN, Carvana) |
| `experiments/` | one script per experiment (`exp01`–`exp34`); each docstring holds its settings, reason and result. `--dry-run` prints the `train.py` commands |
| `lists/` | training labels left out (`exclude_v1.txt`: 39 broken masks) and sample weights (`weights_v1.csv`: 8 multi-pet photos at 0.5) |
| `val_ids.txt` | the fixed 10% validation split (665 photos) |
| analysis only | `stress_eval.py`, `score_train.py`, `show_samples.py`, `study.py`, `tag_images.py`, `hard_traits.py`, `make_sample_weights.py`, `update_hard_cases.py`; some also need `pip install pandas scipy open_clip_torch` |

The experiment scripts log to Weights & Biases (online after `wandb login`, offline otherwise; `--no-wandb` turns it
off). Plain `train.py` logs only with `--wandb`. CUDA runs use mixed precision automatically.
