# Hardest validation images

Copies of the images that appeared in a worst-10 list of any run (01–04). The originals stay in `data/oxford-iiit-pet/images/`: they are validation images, so moving them would change the validation set.

Each `<id>_compare.png` shows: photo | true mask | run 04 prediction | errors (red = missed pet, blue = predicted pet where there is none).

| Image | 01 none-30 | 02 geo-30 | 03 geo_color-30 | 04 geo-50 |
|---|---|---|---|---|
| boxer_150 | 0.402 | 0.362 | 0.366 | 0.188 |
| basset_hound_191 | 0.742 | 0.437 | 0.729 | 0.317 |
| leonberger_38 | 0.532 | 0.346 | 0.413 | 0.442 |
| shiba_inu_92 | 0.468 | 0.451 | 0.469 | 0.459 |
| english_setter_194 | 0.346 | 0.397 | 0.350 | 0.478 |
| miniature_pinscher_16 | 0.515 | 0.453 | 0.636 | 0.523 |
| boxer_161 | 0.294 | 0.594 | 0.373 | 0.552 |
| shiba_inu_103 | 0.569 | 0.529 | 0.531 | 0.569 |
| boxer_149 | 0.341 | 0.468 | 0.432 | 0.578 |
| Siamese_119 | 0.726 | 0.492 | 0.502 | 0.592 |
| Siamese_186 | 0.473 | 0.709 | 0.397 | 0.824 |
| american_pit_bull_terrier_76 | 0.834 | 0.482 | 0.477 | 0.841 |
| wheaten_terrier_113 | 0.724 | 0.648 | 0.487 | 0.863 |
| newfoundland_11 | 0.155 | 0.373 | 0.821 | 0.903 |

## Patterns (from looking at the compare images, 2026-10-06)

1. **Fur-like textures predicted as pet.** Blankets, throws and knitted chair covers get marked as pet: shiba_inu_92 (patterned blanket in front of the dog) and english_setter_194 (brown knitted throw on the chair). This is the most common failure: lots of blue in the error panels
2. **People predicted as pet.** boxer_150: the girl hugging the dog is mostly marked as pet, and part of the dog's head is missed. The true mask itself is fragmented here (sunglasses, the girl's head in front), so even a good model can't score high on this one
3. **Extreme lighting makes the dog disappear.** basset_hound_191 is strongly backlit and washed out, and run 04 misses most of the dog (0.317). Colour augmentation handles this much better: run 03 scores 0.729 on it; newfoundland_11 (dark dog) is 0.373 in 02 vs 0.821 in 03
4. **Hard images are unstable between runs.** The same image can swing by 0.3–0.4 between runs (basset_hound_191: 0.742 / 0.437 / 0.729 / 0.317). The overall average is stable; single hard images are not

Implication: colour augmentation (03) is worse on average at 30 epochs but clearly better on bad lighting, the kind of photo the TA's private set may contain. That makes experiment 05 (geo_color, 50 epochs) the key comparison.
