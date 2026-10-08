"""Experiment 18: rebuild the best model's chain from scratch, as one script.

Why: reproducibility. Run 14 (val 0.9193, Mac) came from 04 -> 06 -> 08 -> 14. This runs the same four stages with
exactly their settings, as 18a -> 18b -> 18c -> 18d, each starting from the previous stage's best.pth. 18d is the
rebuilt model: it shows how close a retrain lands (the TA may ask for one) and is the same-machine reference for 17.
Expect close to 0.9193 but not identical: float16 AMP on CUDA vs float32 on the Mac, and GPU kernels that are not
bit-for-bit deterministic.
Starts from: random weights (needs no earlier run folder)
Result: not run yet.

Each stage gets its own folder, RESULT.md and W&B run. If a stage fails, the script stops there; rerunning it starts
again at 18a (finished stages are kept as separate folders).

    cd Lab2/reference
    python experiments/exp18_rebuild_chain_04_06_08_14.py              # --dry-run prints the train.py commands only
    nohup python experiments/exp18_rebuild_chain_04_06_08_14.py >> runs/exp18.log 2>&1 &     # on a server
"""

from _pipeline import run

LIMIT = ["--epochs", "100", "--sched", "plateau", "--early-stop", "12"]

STAGES = [
    {  # = exp04
        "id": "18a",
        "name": "18a_aug-geo_ep50",
        "title": "rebuild 1/4 (= 04): aug geo, 50 epochs",
        "train_args": ["--aug", "geo", "--epochs", "50"],
        "chain": ["18"],
    },
    {  # = exp06
        "id": "18b",
        "name": "18b_aug-geo_from18a",
        "title": "rebuild 2/4 (= 06): aug geo, continued from 18a until the limit",
        "init": "18a",
        "train_args": ["--aug", "geo", "--lr", "3e-4"] + LIMIT,
        "chain": ["18"],
    },
    {  # = exp08
        "id": "18c",
        "name": "18c_aug-geo_from18b_dice10",
        "title": "rebuild 3/4 (= 08): 18b fine-tuned 10 epochs, dice weight 10",
        "init": "18b",
        "train_args": ["--aug", "geo", "--epochs", "10", "--lr", "3.75e-5", "--dice-weight", "10"],
        "chain": ["18"],
    },
    {  # = exp14
        "id": "18d",
        "name": "18d_aug-geo_light_from18c",
        "title": "rebuild 4/4 (= 14): 18c + 18 epochs, aug geo_light",
        "init": "18c",
        "train_args": ["--aug", "geo_light", "--epochs", "18", "--lr", "1e-4", "--dice-weight", "10"],
        "chain": ["18"],
    },
]

if __name__ == "__main__":
    for stage in STAGES:
        run(stage)
