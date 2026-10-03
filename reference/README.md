# Lab2 — Binary Semantic Segmentation (fixed UNet)

Pipeline around the TA's unmodified `model.py`: dataset, augmentation, training, official-style validation, inference, and the Kaggle CSV.

## Layout

```text
Lab2/
  material/                 TA release (spec, model.py, non_test.csv, test.csv, sample_submission.csv)
  data/oxford-iiit-pet/     dataset (git-ignored): images/, annotations/trimaps/
  reference/
    model.py                TA's model.py, byte-identical (shasum f3502c20…)
    common.py               paths, split, mask, official inference pipeline, Dice, RLE
    dataset.py              PetDataset + augmentation presets (none / geo / geo_color)
    train.py                training loop; logs official val Dice every epoch, saves best.pth
    evaluate.py             official-style val Dice of a .pth (all / dog / cat + worst images)
    predict.py              test.csv -> submission.csv
    check_weights.py        TA's strict-load snippet + key/dtype/shape checks
    tests.py                Dice / RLE / mask / dataset sanity checks
    show_samples.py         figures/samples.png (mask check), figures/aug_<name>.png
    val_ids.txt             fixed 10% validation split (665 IDs, seed 0), reused by every run
    runs/<run>/             config.json, log.csv, best.pth, last.pth; runs/run_log.csv = one line per run
```

## Setup

```bash
pip install -r requirements.txt          # numpy, matplotlib, torch, torchvision, tqdm
# dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
mkdir -p data/oxford-iiit-pet
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz
curl -LO https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz
tar -xzf images.tar.gz -C data/oxford-iiit-pet && tar -xzf annotations.tar.gz -C data/oxford-iiit-pet
```

Paths default to `../data/oxford-iiit-pet` and `../material`. Elsewhere (Colab, lab server, the unzipped submission) set:

```bash
export PET_ROOT=/path/to/oxford-iiit-pet     # contains images/ and annotations/trimaps/
export PET_CSV_DIR=/path/to/csvs             # contains non_test.csv and test.csv
```

## Commands (run inside `reference/`)

```bash
python tests.py                                  # every line OK
python show_samples.py                           # boundary ring must be black in the mask column
python train.py --smoke                          # 16 images, 50 steps: loss falls, Dice on them is high

python train.py --run baseline --aug none --epochs 30
python evaluate.py --weights runs/baseline/best.pth
python predict.py  --weights runs/baseline/best.pth --out submission.csv
python check_weights.py runs/baseline/best.pth

# Stage 4: one change per run, same val_ids.txt
python train.py --run geo       --aug geo       --epochs 30
python train.py --run geo_color --aug geo_color --epochs 30
```

Other flags: `--lr`, `--batch-size`, `--weight-decay`, `--sched cosine|none`, `--dice-weight`, `--seed`, `--workers`, `--note "..."`.

## Speed

- Apple M4 Pro (MPS): about 1 s per step at batch 16, so roughly 6–7 min per epoch (374 steps) plus about 20 s of validation. 30 epochs takes about 3.5 h
- CUDA uses AMP (float16 autocast + GradScaler) automatically. The saved state dict stays float32

## Final submission

```bash
cp runs/<best>/best.pth DL_Lab2_315551118_曾敬豪.pth
python check_weights.py DL_Lab2_315551118_曾敬豪.pth    # must print OK
```

## Still open (check on Kaggle)

`sample_submission.csv` has only empty masks, so it can't confirm the RLE convention. `predict.py` assumes **1-based starts at the original image size**. Confirm both on the competition's Evaluation tab before the first submission.
