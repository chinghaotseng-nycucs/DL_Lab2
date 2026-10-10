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
    dataset.py              PetDataset + augmentation presets (none / geo / geo_color / geo_light / geo_erase /
                            geo_light_cam / geo_light_cam_mild; geo_light_rc / geo_light_cam_rc add texture
                            augmentation, RandConv), copy-paste, masks resized around pixel centres (--exact-masks)
    train.py                training loop; logs official val Dice every epoch, saves best.pth
    make_init_weights.py    pretrained starting weights for --init: init/carvana_unet.pth, init/vgg13bn_encoder.pth
    stress_eval.py          val Dice under 8 photo distortions + paired comparison of runs (and per characteristic)
    update_hard_cases.py    hard_cases/: val images in any run's worst 10, with compare figures (git-ignored output)
    tag_images.py           per-image features of all non_test images (photo stats, COCO detector, CLIP) -> review/
    hard_traits.py          characteristics per image, Dice per characteristic, suspicious-label sheets -> review/
    make_sample_weights.py  label decisions + group weights -> lists/weights_*.csv, lists/exclude_*.txt
    experiments/            one script per experiment (expNN_*.py), each runs the whole process end to end;
                            _pipeline.py holds the shared steps
    score_train.py          val-style Dice of every TRAINING image + review sheets (hard examples)
    evaluate.py             official-style val Dice of a .pth (all / dog / cat + worst images)
    predict.py              test.csv -> submission.csv
    check_weights.py        TA's strict-load snippet + key/dtype/shape checks
    tests.py                Dice / RLE / mask / dataset sanity checks
    show_samples.py         figures/samples.png (mask check), figures/aug_<name>.png
    val_ids.txt             fixed 10% validation split (665 IDs, seed 0), reused by every run
    runs/<run>/             config.json, log.csv, best.pth, last.pth, evaluate.txt, submission.csv, RESULT.md;
                            runs/run_log.csv = one line per run; runs/wandb/ = local W&B files
```

## Setup

```bash
pip install -r ../requirements.txt       # Lab2/requirements.txt: numpy, matplotlib, pillow, torch, torchvision, tqdm, wandb
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

Other flags: `--lr`, `--batch-size`, `--weight-decay`, `--sched cosine|plateau|none`, `--dice-weight`, `--seed`, `--workers`, `--note "..."`, plus `--init <best.pth>` (continue from saved weights), `--early-stop N` / `--min-delta`, `--ema <decay>`, `--sample-weights <csv>`, and `--wandb` / `--wandb-project` (Weights & Biases logging, below).

Added 10/8: `--exclude <txt>` (training IDs to leave out, e.g. broken labels; val never changes), `--copy-paste <p>` (paste another training image's pet onto each image with chance p), `--exact-masks` (resize training masks around pixel centres like the photos; plain torchvision NEAREST samples pixel corners, which caps Dice at 0.980 instead of 0.991 after the official resize back). All default off, so runs 01–22 reproduce unchanged. Added 10/9: `--sampler repeat` (with `--sample-weights`: every image once per epoch like plain shuffling, weight 0.5 = in about half the epochs; the default `weighted` draws with replacement as in run 11) and `--select last` (best.pth = the final epoch instead of the highest clean val Dice). `--bce-weight` (default 1; 0 = soft Dice only; with AdamW only the BCE : Dice ratio matters).

`tag_images.py` and `hard_traits.py` are analysis only and need `pip install open_clip_torch scipy pandas`; the COCO detector and CLIP describe photos and never train the submitted model. Files that a training run reads (`--sample-weights`, `--exclude`) belong in `lists/`, which is committed; `review/` is git-ignored.

## Experiments: `experiments/`

Each experiment is its own script, `experiments/expNN_<what>.py`. It holds the full settings of that run, what it starts from, why it exists and its result, and one command runs the whole process: train, `check_weights`, `evaluate`, `predict`, `RESULT.md`, and renaming the folder with its val Dice. The shared steps live in `experiments/_pipeline.py`, which is never run on its own.

```bash
python experiments/exp16_geo_light_from14_limit.py               # one experiment (needs runs/14_*/best.pth)
python experiments/exp16_geo_light_from14_limit.py --dry-run     # print the train.py command only
python experiments/exp16_geo_light_from14_limit.py --no-wandb    # without W&B logging
```

The best model so far is a chain; to rebuild it from scratch, run `exp04`, `exp06`, `exp08`, `exp14` in this order (each script's `chain` field lists its own). A new experiment = a new `expNN_*.py`: copy the closest one and change its `EXPERIMENT` dict.

## Weights & Biases

Every experiment script logs to W&B (project `DL_Lab2`): per epoch `lr`, `train/loss`, `val/dice`, `val/dog`, `val/cat`, `val/best`, and at the end the best scores plus 8 validation examples (4 known hard cases, 4 ordinary) with true and predicted masks.

```bash
pip install wandb
wandb login              # once per machine; paste the API key from https://wandb.ai/authorize
```

- Not logged in: the run logs **offline** into `runs/wandb/` instead of stopping. Upload later with `wandb sync runs/wandb/offline-run-*`
- `WANDB_MODE=offline` (or `disabled`) forces a mode, e.g. on a server without internet
- W&B problems never stop training: any error just turns logging off for that run

## Running on the lab server

Code goes through git; the dataset and weights are gitignored, so copy them once:

```bash
# on the server, once
git clone git@github.com:chinghaotseng-nycucs/DL_Lab2.git ~/DL_Lab2      # later: cd ~/DL_Lab2 && git pull
# (paths below assume ~/DL_Lab2; use your clone's path. git creates an empty reference/runs/)

# on the Mac, inside Lab2/
scp -r data  server:~/DL_Lab2/                  # -> DL_Lab2/data/oxford-iiit-pet, the default path (819 MB)
scp -r reference/runs/14_aug-geo_light_from08_val0.9193  server:~/DL_Lab2/reference/runs/   # run 16 starts from it

# on the server
cd ~/DL_Lab2/reference
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # expect: True, ... 4090
wandb login                                     # once; or skip and sync offline runs later
nohup python experiments/exp16_geo_light_from14_limit.py >> runs/exp16.log 2>&1 &
tail -f runs/exp16.log                          # Ctrl-C stops watching, not the run
```

Then copy the finished `runs/16_*` folder back to the Mac (`scp -r server:~/DL_Lab2/reference/runs/16_* runs/`). With W&B online, the curves are already on wandb.ai.

## Speed

- Apple M4 Pro (MPS): about 1 s per step at batch 16, so roughly 6–7 min per epoch (374 steps) plus about 20 s of validation. 30 epochs takes about 3.5 h
- CUDA uses AMP (float16 autocast + GradScaler) automatically. The saved state dict stays float32

## Final submission

```bash
cp runs/<best>/best.pth DL_Lab2_315551118_曾敬豪.pth
python check_weights.py DL_Lab2_315551118_曾敬豪.pth    # must print OK
```

## RLE convention (confirmed)

`sample_submission.csv` has only empty masks, so it couldn't confirm the RLE convention. `predict.py` uses **1-based starts, column-major, at the original image size**. Confirmed on 2026-10-05: the baseline scored 0.88986 on Kaggle against 0.8950 val Dice; a wrong convention would score near 0.
