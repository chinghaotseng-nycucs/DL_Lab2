"""Experiment 31: the final model, trained on all non_test images (train + val), in two stages.

Why: the recipe was chosen on the 90/10 split (runs 17-30). The final model uses the same recipe on all 6,651 non_test
images (+11% data, minus the 39 dropped labels). Its val Dice is NOT a fair score (the val photos are in training),
so the model is judged by its recipe and the Kaggle score. Both stages keep the final epoch (--select last).
  31a  run 27's recipe: geo_light, dice 10, 200 epochs from scratch, cleaned labels       (about 2 h on the RTX 4090)
  31b  31a + an 18-epoch mild camera finish (FINISH_AUG below), as exp30                (about 11 min)
Submit 31a if you choose the best clean model, 31b if you choose the robust one.
Set the switches below from exp32 and exp33 before running: IMAGENET_START = True if exp32 wins (31a then starts from
init/vgg13bn_encoder.pth), WEIGHTS = "lists/weights_v2.csv" if exp33 wins.
Starts from: random weights or init/vgg13bn_encoder.pth; needs lists/exclude_v1.txt and the WEIGHTS file (in git)
Result: not run yet.

Each stage gets its own folder, RESULT.md and W&B run. If a stage fails, the script stops there.

    cd Lab2/reference
    python experiments/exp31_final_all_data.py              # --dry-run prints the train.py commands only
    nohup python experiments/exp31_final_all_data.py >> runs/exp31.log 2>&1 &     # on a server
"""

from _pipeline import run

FINISH_AUG = "geo_light_cam_mild"  # camera effects at p 0.1: exp30 had 29d's robustness for half the clean cost
IMAGENET_START = False  # True if exp32 (ImageNet encoder start) beats 27
WEIGHTS = "lists/weights_v1.csv"  # "lists/weights_v2.csv" if exp33 (blankets, toys, clothes, snow x2) beats 27

CLEAN = ["--exclude", "lists/exclude_v1.txt", "--sample-weights", WEIGHTS, "--sampler", "repeat",
         "--all-data", "--select", "last", "--dice-weight", "10"]
START = ["--init", "init/vgg13bn_encoder.pth"] if IMAGENET_START else []


def prep(_):
    if IMAGENET_START:
        from make_init_weights import make_vgg13bn

        make_vgg13bn()


STAGES = [
    {
        "id": "31a",
        "name": "31a_final_all_data_ep200",
        "title": "final base: run 27's recipe on train + val (geo_light, dice 10, 200 epochs, cleaned labels)",
        "prep": prep,
        "train_args": ["--aug", "geo_light", "--epochs", "200", "--lr", "1e-3", "--sched", "cosine"] + CLEAN + START,
        "chain": ["31"],
    },
    {
        "id": "31b",
        "name": "31b_final_all_data_camera_finish",
        "title": f"final: 31a + 18 epochs {FINISH_AUG} on train + val",
        "init": "31a",
        "train_args": ["--aug", FINISH_AUG, "--epochs", "18", "--lr", "1e-4", "--sched", "cosine"] + CLEAN,
        "chain": ["31"],
    },
]

if __name__ == "__main__":
    for stage in STAGES:
        run(stage)
