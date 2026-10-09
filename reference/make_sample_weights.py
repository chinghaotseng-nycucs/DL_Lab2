import argparse
import os
import re

import pandas as pd

from common import HERE

# Turn decisions into the two files train.py reads (training images only; val never changes):
#   python make_sample_weights.py --weight small_pet=2 --weight label_in_pieces=2 \
#       --decisions review/label_check/candidates.csv --name v1
#   -> lists/weights_v1.csv  (image_id,weight for --sample-weights)  and  lists/exclude_v1.txt  (for --exclude)
# --weight TAG=W: training images with that characteristic (review/analysis/tags.csv) are drawn W times as often;
#   an image with several weighted characteristics gets the largest W, not the product.
# --kind FILE:KIND=W: training images whose "kind" column in FILE is KIND get weight W, e.g.
#   --kind review/label_check/multi_animals.csv:all_labelled=0.5   (2+ pets, all labelled: focus on one main pet)
# --decisions: the your_decision column of candidates.csv, per image: "drop" (left out), "keep" (stays in,
#   weight unchanged), a number such as "0.5" (that weight), or empty (undecided: stays in, weight unchanged).
# Order: --weight first, then --kind, then --decisions; a later step overrides an earlier one for the same image.
# lists/ is tracked by git on purpose: a run is only reproducible if its weight and exclude files are committed.

OUT = os.path.join(HERE, "lists")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weight", action="append", default=[], help="TAG=W, repeatable")
    p.add_argument("--kind", action="append", default=[], help="FILE:KIND=W, repeatable")
    p.add_argument("--decisions", default="")
    p.add_argument("--name", required=True)
    args = p.parse_args()

    tags = pd.read_csv(os.path.join(HERE, "review", "analysis", "tags.csv"), index_col=0)
    train = tags[tags.split == "train"]
    w = pd.Series(1.0, index=train.index)
    for item in args.weight:
        tag, value = item.split("=")
        if tag not in train.columns:
            raise SystemExit(f"unknown characteristic {tag}; see the columns of review/analysis/tags.csv")
        has = train[tag] == 1
        w[has] = w[has].clip(lower=float(value)) if float(value) >= 1 else w[has].clip(upper=float(value))
        print(f"{tag}: {int(has.sum())} training images x{value}")

    for item in args.kind:
        path, rule = item.rsplit(":", 1)
        kind, value = rule.split("=")
        kinds = pd.read_csv(path, index_col=0).kind
        ids = [i for i in kinds.index[kinds == kind] if i in w.index]  # val IDs are skipped
        if not ids:
            raise SystemExit(f"no training images of kind {kind} in {path}")
        w[ids] = float(value)
        print(f"{kind} ({os.path.basename(path)}): {len(ids)} training images x{value}")

    drop = []
    if args.decisions:
        dec = pd.read_csv(args.decisions, index_col=0, dtype={"your_decision": str}).your_decision.fillna("")
        for i, text in dec.items():
            text = text.strip().lower()
            if i not in w.index:  # val images: decisions there change nothing in training
                continue
            if text == "drop":
                drop.append(i)
            elif re.fullmatch(r"[0-9.]+", text):
                w[i] = float(text)
        print(f"decisions: {len(drop)} dropped, {int((dec != '').sum())} decided of {len(dec)} candidates")

    os.makedirs(OUT, exist_ok=True)
    wpath, xpath = os.path.join(OUT, f"weights_{args.name}.csv"), os.path.join(OUT, f"exclude_{args.name}.txt")
    w.drop(drop)[lambda s: s != 1.0].rename("weight").rename_axis("image_id").to_csv(wpath)
    with open(xpath, "w") as f:
        f.write(f"# made by make_sample_weights.py --name {args.name}\n" + "".join(f"{i}\n" for i in sorted(drop)))
    print(f"wrote {wpath} ({int((w.drop(drop) != 1).sum())} reweighted) and {xpath} ({len(drop)} left out)")
    print(f"train with: --sample-weights {os.path.relpath(wpath, HERE)} --exclude {os.path.relpath(xpath, HERE)}")


if __name__ == "__main__":
    main()
