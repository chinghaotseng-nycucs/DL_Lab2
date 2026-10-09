"""Experiment 28: camera finish on run 27 (best clean model), with a paired sweep of the BCE : Dice ratio.

Why: 27 (200 epochs, cleaned labels) is the best clean model (val 0.9236) but weak on blurred photos (0.768). An
18-epoch camera finish added robustness cheaply on 25 (exp26: stress +0.027 for clean -0.002). Here the same finish
runs on 27, four times, changing only the loss: BCE + 3 x Dice, BCE + 10 x Dice (control, as in all runs since 08),
BCE + 30 x Dice, Dice only. All four start from 27's best.pth and use the same seed, so they see the same images in
the same order with the same augmentation: the differences between them come from the loss alone (paired, as 08/09).
Each: aug geo_light_cam, lr 1e-4 cosine, cleaned labels (lists/*_v1, repeat sampler), final epoch kept.
Compare each with 28b (python stress_eval.py 28b 28a 28c 28d) and 28b with 27 (python stress_eval.py 27 28b).
Starts from: runs/27_*/best.pth (run exp27 first, on the same machine); about 11 min per stage on the RTX 4090
Result: not run yet.

Each stage gets its own folder, RESULT.md and W&B run. If a stage fails, the script stops there; rerunning it starts
again at 28a (finished stages are kept as separate folders).

    cd Lab2/reference
    python experiments/exp28_cam_finish_from27_dice_sweep.py              # --dry-run prints the train.py commands only
    nohup python experiments/exp28_cam_finish_from27_dice_sweep.py >> runs/exp28.log 2>&1 &     # on a server
"""

from _pipeline import run

FINISH = ["--aug", "geo_light_cam", "--epochs", "18", "--lr", "1e-4", "--sched", "cosine",
          "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv", "--sampler", "repeat",
          "--select", "last"]

STAGES = [
    {
        "id": "28a",
        "name": "28a_cam_finish_from27_dice3",
        "title": "27 + 18 epochs geo_light_cam, loss BCE + 3 x Dice",
        "init": "27",
        "train_args": FINISH + ["--dice-weight", "3"],
        "chain": ["27", "28"],
    },
    {
        "id": "28b",
        "name": "28b_cam_finish_from27_dice10",
        "title": "27 + 18 epochs geo_light_cam, loss BCE + 10 x Dice (control)",
        "init": "27",
        "train_args": FINISH + ["--dice-weight", "10"],
        "chain": ["27", "28"],
    },
    {
        "id": "28c",
        "name": "28c_cam_finish_from27_dice30",
        "title": "27 + 18 epochs geo_light_cam, loss BCE + 30 x Dice",
        "init": "27",
        "train_args": FINISH + ["--dice-weight", "30"],
        "chain": ["27", "28"],
    },
    {
        "id": "28d",
        "name": "28d_cam_finish_from27_dice_only",
        "title": "27 + 18 epochs geo_light_cam, loss Dice only (BCE weight 0)",
        "init": "27",
        "train_args": FINISH + ["--dice-weight", "10", "--bce-weight", "0"],
        "chain": ["27", "28"],
    },
]

if __name__ == "__main__":
    for stage in STAGES:
        run(stage)
