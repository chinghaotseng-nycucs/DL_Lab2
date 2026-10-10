"""Experiment 29: texture augmentation (RandConv) as an 18-epoch finish on run 27, with and without camera effects.

Why: many remaining mistakes are texture mistakes (fabric, bark and plush toys taken for fur; a hairless cat missed).
CNNs lean on texture (Geirhos et al., ICLR 2019); random convolutions keep shapes but scramble colour and fine
texture, so the model has to use shape (RandConv, Xu et al., ICLR 2021). Blur, which also removes texture, already
fixed several texture mistakes in exp21. Four finishes, all from 27's best.pth with the same seed and settings
(18 epochs, lr 1e-4 cosine, dice 10, cleaned labels, repeat sampler, final epoch kept):
  29a  geo_light               control: what a plain 18-epoch finish does to 27
  29b  geo_light_rc            + texture augmentation (RandConv, p 0.5)        -> 29b - 29a = texture effect
  29c  geo_light_cam_rc        + camera effects + texture augmentation         -> 29c - 29d = texture on top of camera
  29d  geo_light_cam           camera effects only (= 28b's recipe; its control on the same machine)
Compare: python stress_eval.py 29a 29b   and   python stress_eval.py 29d 29c   (and 27 for reference).
Starts from: runs/27_*/best.pth (exp27 first, same machine); about 11 min per stage on the 4090, 2.2 h on the Mac
Result (Mac, 10/10): texture augmentation hurts; camera effects work and reproduce 28b. Clean / stress val:
27 0.9236/0.880, 29a 0.9217/0.881, 29b 0.9172/0.887, 29c 0.9159/0.901, 29d 0.9197/0.906 (28b on the 4090: 0.9197/0.906).
29b-29a: clean -0.0046, stress +0.006 (blur only); blankets -0.039, plush toys -0.024: the texture mistakes got worse.
29c-29d: clean -0.0038, stress -0.005. 29d-29a: clean -0.0020, stress +0.024. 29a-27: clean -0.0018, stress +0.001.

Each stage gets its own folder, RESULT.md and W&B run. If a stage fails, the script stops there; rerunning it starts
again at 29a (finished stages are kept as separate folders).

    cd Lab2/reference
    python experiments/exp29_texture_finish_from27.py              # --dry-run prints the train.py commands only
    nohup python experiments/exp29_texture_finish_from27.py >> runs/exp29.log 2>&1 &     # on a server
"""

from _pipeline import run

FINISH = ["--epochs", "18", "--lr", "1e-4", "--sched", "cosine", "--dice-weight", "10",
          "--exclude", "lists/exclude_v1.txt", "--sample-weights", "lists/weights_v1.csv", "--sampler", "repeat",
          "--select", "last"]

STAGES = [
    {
        "id": "29a",
        "name": "29a_finish_from27_geo_light",
        "title": "27 + 18 epochs, aug geo_light (control for 29b)",
        "init": "27",
        "train_args": ["--aug", "geo_light"] + FINISH,
        "chain": ["27", "29"],
    },
    {
        "id": "29b",
        "name": "29b_finish_from27_texture",
        "title": "27 + 18 epochs, aug geo_light + texture (RandConv)",
        "init": "27",
        "train_args": ["--aug", "geo_light_rc"] + FINISH,
        "chain": ["27", "29"],
    },
    {
        "id": "29c",
        "name": "29c_finish_from27_camera_texture",
        "title": "27 + 18 epochs, aug geo_light_cam + texture (RandConv)",
        "init": "27",
        "train_args": ["--aug", "geo_light_cam_rc"] + FINISH,
        "chain": ["27", "29"],
    },
    {
        "id": "29d",
        "name": "29d_finish_from27_camera",
        "title": "27 + 18 epochs, aug geo_light_cam (control for 29c, = 28b's recipe)",
        "init": "27",
        "train_args": ["--aug", "geo_light_cam"] + FINISH,
        "chain": ["27", "29"],
    },
]

if __name__ == "__main__":
    for stage in STAGES:
        run(stage)
