"""Experiment 31: the final model, trained on all non_test images (train + val), in two stages.

Why: the recipe was chosen on the 90/10 split (runs 17-34). The final model uses the same recipe on all 6,651 non_test
images (+11% data, minus the 39 dropped labels). Its val Dice is NOT a fair score (the val photos are in training),
so the model is judged by its recipe and the Kaggle score. Both stages keep the final epoch (--select last).
  31a  run 32's recipe: ImageNet VGG13-BN encoder start, geo_light, dice 10, 200 epochs, cleaned labels
                                                                                         (about 2 h on the RTX 4090)
  31b  31a + an 18-epoch mild camera finish (FINISH_AUG below), as exp30 and exp34      (about 11 min)
Submit 31b if exp34 (the same finish on 32) is worth it on 32, otherwise 31a.
Switches set 10/11: IMAGENET_START = True (exp32 beat 27: clean +0.0078, stress +0.0173); WEIGHTS stays
lists/weights_v1.csv (exp33's x2 groups: +0.008 on those photos, CI includes 0).
Starts from: init/vgg13bn_encoder.pth (the prep step makes it if missing); needs lists/exclude_v1.txt and the WEIGHTS
file (in git)
Result: not run yet.

Each stage gets its own folder, RESULT.md and W&B run. If a stage fails, the script stops there.

    cd Lab2/reference
    python experiments/exp31_final_all_data.py              # --dry-run prints the train.py commands only
    nohup python experiments/exp31_final_all_data.py >> runs/exp31.log 2>&1 &     # on a server
"""

from _pipeline import run

FINISH_AUG = "geo_light_cam_mild"  # camera effects at p 0.1: exp30 had 29d's robustness for half the clean cost
IMAGENET_START = True  # exp32 (ImageNet encoder start) beat 27; False = random start, as 27
WEIGHTS = "lists/weights_v1.csv"  # exp33's lists/weights_v2.csv (blankets, toys, clothes, snow x2) did not pass

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
        "title": f"final base: run {'32' if IMAGENET_START else '27'}'s recipe on train + val "
                 f"({'ImageNet start, ' if IMAGENET_START else ''}geo_light, dice 10, 200 epochs, cleaned labels)",
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
