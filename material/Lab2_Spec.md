# Lab 2: Binary Semantic Segmentation with a Fixed UNet

TA: 劉子齊 Jonathan

In this lab, you will train a binary semantic segmentation model. Unlike previous years, the model architecture is fixed. Everyone must use the official `model.py` provided by the TA. Your main task is to improve the training pipeline, especially data preprocessing and augmentation.

The Kaggle Public Leaderboard accounts for 30% of the final score. The official private-evaluation score will be computed after the deadline by the TA using your submitted model weights on a private TA-labeled dataset.

## Released Files

Students will receive only:

```text
model.py
non_test.csv
test.csv
sample_submission.csv
Lab2_Spec.md
```

You must write your own training, validation, inference, and Kaggle CSV generation code.

## Dataset

Students should download the Oxford-IIIT Pet Dataset from the official website:

https://www.robots.ox.ac.uk/~vgg/data/pets/

Expected local layout after extraction:

```text
oxford-iiit-pet/
  images/
  annotations/
    trimaps/
```

Kaggle provides two CSV files:

```text
non_test.csv
test.csv
```

`non_test.csv` lists the image IDs that may be used for training and validation. Students may freely split these IDs into train and validation sets.

`test.csv` lists the fixed public Kaggle test image IDs. These images may be used only for inference and CSV generation. The corresponding masks are hidden by Kaggle.

The final official evaluation set is a private dataset labeled by the TA and will not be released.

## Task

You need to train the official fixed UNet model for binary semantic segmentation.

The goal is to segment the pet region as foreground and all other pixels as background.

You should focus on:

- data preprocessing
- data augmentation
- image resizing/cropping strategy during training
- mask preprocessing
- loss function design
- optimizer and learning-rate schedule
- training stability and reproducibility

You must not modify the official model architecture.

## Fixed Model Rule

The TA will release an official `model.py`.

You may import and use this file in your project, but you may not change the architecture, layer names, tensor shapes, or forward behavior.

Your submitted weight file must be a PyTorch `state_dict` compatible with the official UNet. During official private evaluation, the TA will ignore your model implementation and load your weights using the official `model.py`.

The TA will use the following loading procedure:

```python
import torch
from model import UNet

weight_path = "DL_Lab2_YourStudentID_name.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=1).to(device)
state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.eval()
```

If this code cannot successfully load your submitted `.pth` file, your official private evaluation score will be 0.

Do not submit a checkpoint dictionary such as:

```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}
```

Submit only the model state dictionary:

```python
torch.save(model.state_dict(), "DL_Lab2_YourStudentID_name.pth")
```

## Binary Mask Definition

The Oxford-IIIT Pet annotations are trimaps with three labels:

- `1`: foreground pet
- `2`: background
- `3`: boundary

For this lab, convert the trimap into a binary mask:

- pixels with value `1` become foreground `1`
- pixels with value `2` or `3` become background `0`

Boundary pixels must be treated as background. Different label mappings will be considered incorrect.

## Kaggle Competition Registration

Please fill out the Google Form provided by the TA with your name, student ID, and NYCU email address.

```text
https://forms.gle/qgbs8P1FJVntf3qj9
```

After you submit the form, the TA will add you to the Kaggle competition within 48 hours. After being added, you can join the competition using the invitation link below:

```text
https://www.kaggle.com/t/533d77b146294ce2a1f4ab2421f7eb7d
```

## Public Kaggle Leaderboard

The Kaggle Public Leaderboard accounts for 30% of the final score.

Kaggle link: https://www.kaggle.com/t/533d77b146294ce2a1f4ab2421f7eb7d

For Kaggle, you will submit a prediction CSV file:

```text
submission.csv
```

The CSV format is:

```text
image_id,encoded_mask
example_001,1 5 20 3 ...
example_002,
```

The mask must be encoded with run-length encoding (RLE) in column-major order. Empty masks should have an empty `encoded_mask` field.

The public Kaggle score is part of the final score and accounts for 30% of this lab. It also helps you check whether your inference pipeline works.

Your Kaggle team name must follow this format:

```text
StudentID_Name
```

Examples:

```text
311605004_劉子齊
311605004_Jonathan_TC_Liu
```

Every student must have at least one valid submission on the Kaggle Public Leaderboard before the deadline. If you do not have any valid Public Leaderboard submission, the TA will not evaluate your private model-weight submission, and your official private evaluation score will be 0.

Kaggle Public Leaderboard submissions are limited to one submission per day.

## Official Private Evaluation

The official score will be computed by the TA after the deadline.

You must submit your trained model weights through the course submission system:

```text
DL_Lab2_YourStudentID_name.pth
```

The TA will:

1. instantiate the official UNet from `model.py`
2. load your submitted `.pth` file with `model.load_state_dict(state_dict, strict=True)`
3. run inference on a private TA-labeled dataset
4. compute the average Dice score
5. use this Dice score for official grading

If step 2 fails for any reason, your official private evaluation score will be 0.

The private dataset will not be released. It may contain images outside the Oxford-IIIT Pet Dataset, including TA-collected dog photos labeled by the TA.

## Official Inference Preprocessing

For fairness, official private evaluation uses a fixed inference preprocessing pipeline:

- load image as RGB
- resize image to `256 x 256`
- convert to tensor in `[0, 1]`
- normalize with ImageNet mean and standard deviation
- run the official UNet
- apply sigmoid
- threshold at `0.5`
- resize predicted mask back to the original private mask size with nearest-neighbor interpolation before computing Dice

You are free to use different preprocessing and augmentation during training, but your final weights should perform well under the official inference preprocessing above.

## Submission Components

You must submit the following components:

1. Short Technical Summary (`.txt`)
2. Source Code (`.zip`)
3. Trained Model Weights (`.pth`)
4. Kaggle public leaderboard submission (`submission.csv`)

A valid Kaggle Public Leaderboard submission is required. Students without any valid public Kaggle submission will not be evaluated on the private TA-labeled dataset.

### Short Technical Summary

Filename:

```text
DL_Lab2_YourStudentID_name.txt
```

Example:

```text
DL_Lab2_311605004_劉子齊.txt
```

Maximum length: 2000 characters, including spaces.

The summary should include:

- data preprocessing strategy
- augmentation strategy
- training strategy
- loss function
- optimizer and scheduler
- validation result
- public Kaggle score
- key observations
- AI tool usage disclosure, if any

### Source Code

Filename:

```text
DL_Lab2_YourStudentID_name.zip
```

Example:

```text
DL_Lab2_311605004_劉子齊.zip
```

Your zip file must include all code needed to reproduce your training and generate your Kaggle CSV. The TA provides only `model.py`; all other training, dataset, validation, inference, and CSV-generation code must be written by you.

### Trained Model Weights

Filename:

```text
DL_Lab2_YourStudentID_name.pth
```

Example:

```text
DL_Lab2_311605004_劉子齊.pth
```

This file is used for the official private evaluation. It must be exactly loadable by the official UNet using `model.load_state_dict(state_dict, strict=True)`.

## Important Dates

```text
Code, model weights, Kaggle submission, and summary submission deadline:
2026/10/12 (Mon.) 23:55:59 (UTC+8)
```

## Scoring

Final score:

```text
Kaggle Public Leaderboard 30% + Official Private Evaluation 50% + Short Technical Summary 20%
```

The Kaggle Public Leaderboard is part of the final score. The private evaluation is computed separately by the TA after the deadline.

Suggested official private evaluation grading for the private-evaluation portion:

- Top 10%: 100 pts
- Top 25%: 90 pts
- Top 50%: 85 pts
- Top 75%: 80 pts
- Others above baseline: 70 pts
- Below baseline: 0 pts

Private evaluation baseline dice score == 0.85.

## Reproducibility Requirement

The TA may randomly select students for a demo or retraining check.

You must be able to:

- explain your preprocessing and augmentation choices
- retrain your model using your submitted code
- load your submitted `.pth` file with the official `model.py`
- regenerate your Kaggle `submission.csv`
- reproduce a reasonable validation Dice score

If the TA cannot run your code, load your model, or reproduce your result, you may receive 0 points.

## Data Usage Rules

- You may freely split the IDs in `non_test.csv` into training and validation sets.
- You may use the IDs in `test.csv` only for inference and CSV generation.
- You may not use public Kaggle test images for training, fine-tuning, pseudo-labeling, manual annotation, or model selection beyond normal leaderboard submission.
- You may not use hidden public Kaggle test masks in any way.
- You may not use the private official evaluation set in any way.
- You may not modify the official UNet architecture for the submitted weights.
- You may not submit fake, corrupted, or incompatible weights.
- You may not plagiarize another student's code or report.
- If you use AI tools such as ChatGPT, Copilot, or other generative tools, you must disclose this in your summary.

Violation of the data usage rules may result in a score of 0.
