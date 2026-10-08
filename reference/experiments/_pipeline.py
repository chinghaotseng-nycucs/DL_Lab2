"""Shared steps for the experiment scripts in this folder. Not an experiment itself: each expNN_*.py describes
one run and calls run(EXPERIMENT), which does the whole process with no manual steps in between:
train -> check_weights -> evaluate -> predict -> RESULT.md -> rename the run folder with its val Dice.

Every script accepts:  --workers N  (DataLoader workers, default 8)
                       --dry-run    (print the train.py command only)
                       --no-wandb   (don't log to Weights & Biases)
"""

import argparse
import csv
import glob
import json
import os
import platform
import subprocess
import sys
import time

REF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Lab2/reference
RUNS = os.path.join(REF, "runs")
EXP_DIR = os.path.join(REF, "experiments")


def log(msg):
    print(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def find_run(exp_id):
    """Newest runs/<id>_*/ folder that has a best.pth, or None."""
    dirs = [d for d in glob.glob(os.path.join(RUNS, f"{exp_id}_*")) if os.path.isfile(os.path.join(d, "best.pth"))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def py(*args, capture=False):
    """Run a script of Lab2/reference in a fresh Python process, like the TA would."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", TQDM_DISABLE="1")
    r = subprocess.run([sys.executable, *args], cwd=REF, env=env, check=True, text=True,
                       stdout=subprocess.PIPE if capture else None)
    if capture:
        print(r.stdout, end="", flush=True)
        return r.stdout


def acquire_lock(name):
    """Refuse to start a second copy of an experiment that is already running on this machine: both would write
    into the same runs/<name>/ folder (happened 10/8 with exp16). A lock left by a crashed run is ignored."""
    path = os.path.join(RUNS, f"{name}.lock")
    if os.path.exists(path):
        with open(path) as f:
            fields = f.read().split()
        try:
            pid = int(fields[0])
            os.kill(pid, 0)  # signal 0 sends nothing: it only checks that the process exists
        except (ValueError, IndexError, ProcessLookupError):
            pass  # stale lock: that process is gone
        except PermissionError:  # exists but belongs to another user: treat it as running
            sys.exit(f"experiment {name} seems to be running already (pid {pid}); delete {path} if it isn't")
        else:
            sys.exit(f"experiment {name} is already running (pid {pid}); not starting a second copy.\n"
                     f"If that's wrong, delete {path}")
    os.makedirs(RUNS, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{os.getpid()} {platform.node()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return path


def read_run_log():
    with open(os.path.join(RUNS, "run_log.csv")) as f:
        return [[c.strip() for c in r] for r in csv.reader(f)]  # strip: the file may be column-aligned in an editor


def script_of(exp_id):
    found = sorted(glob.glob(os.path.join(EXP_DIR, f"exp{exp_id}_*.py")))
    return f"experiments/{os.path.basename(found[0])}" if found else f"experiments/exp{exp_id}_*.py"


def write_result(exp, cmd, init_dir, device_name):
    run = exp["name"]
    row = next(r for r in read_run_log()[1:] if r[0] == run)
    best_epoch, val, dog, cat, minutes, note = row[8], row[9], row[10], row[11], row[12], row[13]
    with open(os.path.join(RUNS, run, "log.csv")) as f:
        epochs = list(csv.DictReader(f))
    with open(os.path.join(RUNS, run, "evaluate.txt")) as f:
        worst = f.read().split("worst 10:")[1].strip().splitlines()[:5]
    stop = next((s for s in note.split("; ") if s.startswith("early stop")), f"ran all {row[2]} epochs")
    lines = [
        f"# {exp['id']} · {exp['title']}",
        "",
        f"Command: `{' '.join(cmd)}`",
        f"Machine: {platform.node()} ({device_name}), {minutes} min",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    if init_dir:
        lines.append(f"| Starting val Dice (epoch 0 = run {os.path.basename(init_dir)}) | {epochs[0]['val_dice']} |")
    lines += [
        f"| Best val Dice (epoch {best_epoch}) | **{val}** |",
        f"| Dog / cat | {dog} / {cat} |",
        f"| Epochs trained | {epochs[-1]['epoch']} ({stop}) |",
        "| Kaggle public | not uploaded yet |",
        "| check_weights | OK |",
        "",
    ]
    if init_dir:
        gain = float(val) - float(epochs[0]["val_dice"])
        lines += ["No epoch beat the starting weights, so best.pth is the starting model." if best_epoch == "0"
                  else f"Gain over the starting weights: {gain:+.4f}.", ""]
    lines += [
        "Per epoch: " + ", ".join(f"{r['epoch']}: {r['val_dice']}" for r in epochs),
        "",
        "Worst val images: " + ", ".join(w.split()[1] + " " + w.split()[0] for w in worst),
        "",
        "Reproduce from scratch, in order: " + ", ".join(f"`python {script_of(i)}`" for i in exp["chain"]),
        "",
    ]
    with open(os.path.join(RUNS, run, "RESULT.md"), "w") as f:
        f.write("\n".join(lines))
    return val


def rename_with_val(run, val):
    new = f"{run}_val{val}"
    while os.path.exists(os.path.join(RUNS, new)):  # the same experiment rerun with the same score
        new += "_2"
    rows = read_run_log()
    for r in rows[1:]:
        if r[0] == run:
            r[0] = new
    with open(os.path.join(RUNS, "run_log.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)
    cfg_path = os.path.join(RUNS, run, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["run"] = new
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    os.rename(os.path.join(RUNS, run), os.path.join(RUNS, new))
    return new


def run(exp):
    """The whole process for one experiment dict: id, name, title, train_args, chain; optional init, prep."""
    p = argparse.ArgumentParser(description=f"Experiment {exp['id']}: {exp['title']}")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    init_dir = find_run(exp["init"]) if exp.get("init") else None
    if exp.get("init") and init_dir is None and not args.dry_run:
        sys.exit(f"Run {exp['init']} not found: copy its folder (at least runs/{exp['init']}_*/best.pth) into {RUNS}, "
                 f"or run it first: python {script_of(exp['init'])}")
    cmd = ["python", "train.py", "--run", exp["name"], *exp["train_args"]]
    if exp.get("init"):
        parent = os.path.relpath(init_dir, REF) if init_dir else f"runs/{exp['init']}_<not run yet>"
        cmd += ["--init", os.path.join(parent, "best.pth")]
    full = cmd + ["--workers", str(args.workers), "--note", exp["title"]] + ([] if args.no_wandb else ["--wandb"])
    if args.dry_run:
        print(" ".join(full))
        return

    import torch  # here, so --dry-run and --help work instantly
    sys.path.insert(0, REF)
    from common import CSV_DIR, DATA_ROOT

    for path in (os.path.join(DATA_ROOT, "images"), os.path.join(DATA_ROOT, "annotations", "trimaps"),
                 os.path.join(CSV_DIR, "non_test.csv")):
        if not os.path.exists(path):
            sys.exit(f"missing {path}\nSet PET_ROOT (folder with images/ and annotations/) and PET_CSV_DIR "
                     f"(folder with non_test.csv and test.csv), e.g. export PET_ROOT=~/data/oxford-iiit-pet")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device_name = "Apple MPS"
    else:
        device_name = "CPU"
    lock = acquire_lock(exp["name"])
    log(f"experiment {exp['id']} on {platform.node()} ({device_name})")
    t0 = time.time()
    try:
        _run_steps(exp, full, cmd, init_dir, device_name, t0)
    finally:
        os.remove(lock)


def _run_steps(exp, full, cmd, init_dir, device_name, t0):
    try:
        if exp.get("prep"):
            log(f"{exp['id']}: preparation")
            exp["prep"](init_dir)
        log(f"{exp['id']}: training {exp['name']}" + (f" from {os.path.basename(init_dir)}" if init_dir else ""))
        py(*full[1:])
        log(f"{exp['id']}: checks")
        run_dir = os.path.join(RUNS, exp["name"])
        py("check_weights.py", os.path.join(run_dir, "best.pth"))
        out = py("evaluate.py", "--weights", os.path.join(run_dir, "best.pth"), capture=True)
        with open(os.path.join(run_dir, "evaluate.txt"), "w") as f:
            f.write(out)
        py("predict.py", "--weights", os.path.join(run_dir, "best.pth"), "--out", os.path.join(run_dir, "submission.csv"))
    except subprocess.CalledProcessError as e:
        log(f"{exp['id']}: FAILED ({' '.join(e.cmd[1:3])} exited with {e.returncode})")
        sys.exit(1)
    val = write_result(exp, cmd, init_dir, device_name)
    new = rename_with_val(exp["name"], val)
    log(f"{exp['id']}: done in {(time.time() - t0) / 60:.0f} min -> runs/{new} (best val Dice {val})")
