"""Experiment 33: run 27's recipe (200 epochs, cleaned labels) with rare hard groups drawn twice as often.

Why: val Dice is lower on photos with blankets or cushions (-0.064), plush toys (-0.065 to -0.142), clothes (-0.126)
and snow (-0.032), and each group is under 2% of the training photos. Run 11 drew hard photos 3x with no gain, but it
was an 18-epoch fine-tune on the uncleaned labels, chose the photos by low train Dice (which also picks up bad labels)
and used the sampler that skips a third of the photos each epoch. Here the groups come from the photo tags instead,
on the cleaned labels, for a whole 200-epoch run (the final recipe's length, so a win is also a usable model).
lists/weights_v2.csv = lists/weights_v1.csv + 283 training photos x2 (blanket_or_cushion 76, with_plush_toy 32,
plush_toy_detected 72, wearing_clothes 24, in_snow 103); the 8 multi-pet photos stay at 0.5 and the same 39 are left
out. Made by: python make_sample_weights.py --weight blanket_or_cushion=2 --weight with_plush_toy=2
--weight plush_toy_detected=2 --weight wearing_clothes=2 --weight in_snow=2
--kind review/label_check/multi_animals.csv:all_labelled=0.5 --decisions review/label_check/candidates.csv --name v2
Otherwise 27's settings, so 33 - 27 isolates the weighting. Not perfectly paired: the extra draws change the batch
order, and an epoch is about 5% longer.
Adopt it for the final model (exp31 WEIGHTS = lists/weights_v2.csv) if the 31 val photos in these groups improve
(stress_eval.py's "any x2 group" row, CI above 0) and clean val is not lower by more than 0.002.
Compare: python stress_eval.py 27 33
Starts from: random weights; needs lists/exclude_v1.txt and lists/weights_v2.csv (in git)
about 1 h 55 min on the RTX 4090, on the same machine as 27
Result: not run yet.

    cd Lab2/reference
    python experiments/exp33_hard_groups_x2_ep200_clean.py              # --dry-run prints the train.py command only
    nohup python experiments/exp33_hard_groups_x2_ep200_clean.py >> runs/exp33.log 2>&1 &     # on a server
"""

from _pipeline import run

EXPERIMENT = {
    "id": "33",
    "name": "33_hard_groups_x2_ep200_clean",
    "title": "run 27's recipe (geo_light, dice 10, 200 epochs, cleaned labels) + blankets, toys, clothes, snow x2",
    "train_args": ["--aug", "geo_light", "--epochs", "200", "--lr", "1e-3", "--sched", "cosine", "--dice-weight", "10",
                   "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v2.csv", "--sampler", "repeat"],
    "chain": ["33"],  # runs to execute in this order to rebuild it from scratch
}

if __name__ == "__main__":
    run(EXPERIMENT)
