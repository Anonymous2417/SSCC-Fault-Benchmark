# -*- coding: utf-8 -*-
"""
make_splits.py
--------------------------

Generate a SEALED train/test split at the condition level for fault detection task.

Definition of 'condition' can be found in the paper.

Split Policy 
-----------------------
Given a target velocity (leave_vel):

TRAIN:
    1) All NORMAL combos with vel != leave_vel
    2) A user-specified subset of NORMAL combos with vel == leave_vel
       (selected via --train_leave_combos, sealed by combo)

TEST:
    1) Remaining NORMAL combos with vel == leave_vel
    2) All ABNORMAL combos (label != "normal")

This follows a normal-only training protocol while evaluating cross-velocity
generalization.

Usage Example
-------------
python make_splits_fd.py \
    --csv /path/to/index.csv \
    --outdir /path/to/output_dir \
    --leave_vel 100 \
    --train_leave_combos heavy-clean,med-clean,light-clean \
    --seed 0

Outputs
-------
    outdir/train.ids
    outdir/test.ids
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd


# ====== parse patterns (adapt if your id naming differs) ======
LOAD_RE  = re.compile(r'/(heavy|med|light)_vel', re.I)
VEL_RE   = re.compile(r'vel(\d+)', re.I)
NOISE_RE = re.compile(r'_(clean|noise\w*)', re.I)


def parse_id(s: str):
    """
    Parse load, vel, noise_cls from an id string like:
      normal/heavy_vel100_clean_0001
      abnormal/med_vel80_noiseB_0012
    """
    m_load  = LOAD_RE.search(s)
    m_vel   = VEL_RE.search(s)
    m_noise = NOISE_RE.search(s)

    load  = m_load.group(1).lower() if m_load else ""
    vel   = m_vel.group(1) if m_vel else ""
    noise = m_noise.group(1).lower() if m_noise else "clean"

    # normalize noise domain to {clean, noise}
    noise_cls = "noise" if noise.startswith("noise") else "clean"
    return load, vel, noise_cls


def save_ids(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in arr:
            f.write(str(x) + "\n")


def _parse_train_leave_combos(spec: str):
    """
    spec:
      - 'all'  -> use all existing leave_vel normal combos in train
      - 'none' -> use none of them in train (all go to test)
      - 'heavy-clean,med-clean,light-clean' -> explicit list

    returns: set of (load, noise_cls) pairs, or None to indicate 'all'
    """
    spec = (spec or "").strip().lower()
    if spec in ("all", ""):
        return None  # means all
    if spec == "none":
        return set()

    items = [x.strip() for x in spec.split(",") if x.strip()]
    out = set()
    for it in items:
        if "-" not in it:
            raise ValueError(f"--train_leave_combos item must be like 'heavy-clean', got '{it}'")
        load, noise = it.split("-", 1)
        load = load.strip().lower()
        noise = noise.strip().lower()
        if load not in ("heavy", "med", "light"):
            raise ValueError(f"load must be heavy/med/light, got '{load}' in '{it}'")
        if noise not in ("clean", "noise"):
            raise ValueError(f"noise must be clean/noise, got '{noise}' in '{it}'")
        out.add((load, noise))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="index.csv path")
    ap.add_argument("--outdir", required=True, help="output split directory")
    ap.add_argument("--leave_vel", type=str, default="100", help="target velocity, e.g. 100")
    ap.add_argument("--train_leave_combos", type=str, default="all",
                    help="Which leave_vel NORMAL combos go to TRAIN (sealed by combo). "
                         "Examples: 'all' / 'none' / 'heavy-clean,med-clean,light-clean'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)  # reserved for future (e.g., random selecting combos)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ====== read index.csv ======
    df = pd.read_csv(args.csv)

    # Assumption consistent with your pipeline:
    # col0 = id, col2 = label
    df["id"] = df.iloc[:, 0].astype(str)
    df["label"] = df.iloc[:, 2].astype(str).str.lower()

    # parse fields from id
    loads, vels, noises = [], [], []
    for x in df["id"]:
        l, v, n = parse_id(x)
        loads.append(l)
        vels.append(v)
        noises.append(n)

    df["load"] = loads
    df["vel"] = vels
    df["noise_cls"] = noises

    leave_vel = str(args.leave_vel)

    # ====== define combo key (sealed unit) ======
    # combo_key = (label_class, load, vel, noise_cls)
    df["combo_key"] = list(zip(df["label"], df["load"], df["vel"], df["noise_cls"]))

    # ====== prepare sets of combos ======
    combos_train = set()
    combos_test = set()

    # all abnormal combos -> test
    df_abn = df[df["label"] != "normal"]
    combos_test.update(df_abn["combo_key"].unique().tolist())

    # normal combos
    df_norm = df[df["label"] == "normal"]

    # normal with vel != leave_vel -> train
    df_norm_base = df_norm[df_norm["vel"] != leave_vel]
    combos_train.update(df_norm_base["combo_key"].unique().tolist())

    # leave_vel normal combos (6 possible)
    df_leave_norm = df_norm[df_norm["vel"] == leave_vel]

    target_conditions = [
        ("heavy", "clean"),
        ("heavy", "noise"),
        ("med",   "clean"),
        ("med",   "noise"),
        ("light", "clean"),
        ("light", "noise"),
    ]

    # determine which leave_vel normal combos exist
    existing_leave_combos = []
    for load, noise_cls in target_conditions:
        combo = ("normal", load, leave_vel, noise_cls)
        exists = ((df_leave_norm["load"] == load) & (df_leave_norm["noise_cls"] == noise_cls)).any()
        if exists:
            existing_leave_combos.append(combo)
        else:
            print(f"[WARN] leave_vel normal combo missing (no samples): {combo}")

    # parse user specification
    spec_pairs = _parse_train_leave_combos(args.train_leave_combos)
    if spec_pairs is None:
        # 'all'
        train_leave = set(existing_leave_combos)
    else:
        # explicit subset (sealed)
        train_leave = set()
        for load, noise_cls in spec_pairs:
            train_leave.add(("normal", load, leave_vel, noise_cls))
        # keep only those truly existing to avoid typos silently breaking
        train_leave = train_leave.intersection(set(existing_leave_combos))

    test_leave = set(existing_leave_combos) - train_leave

    # assign
    for c in sorted(train_leave):
        combos_train.add(c)
        print(f"[OK] leave_vel normal combo -> TRAIN: {c}")
    for c in sorted(test_leave):
        combos_test.add(c)
        print(f"[OK] leave_vel normal combo -> TEST : {c}")

    # ====== leakage check at combo level ======
    inter = combos_train.intersection(combos_test)
    if inter:
        raise RuntimeError(f"[LEAK] combos appear in both train and test: {list(inter)[:10]}")

    # ====== expand combos to ids ======
    train_ids = df[df["combo_key"].isin(combos_train)]["id"].tolist()
    test_ids  = df[df["combo_key"].isin(combos_test)]["id"].tolist()

    # de-dup preserve order
    train_ids = list(dict.fromkeys(train_ids))
    test_ids  = list(dict.fromkeys(test_ids))

    save_ids(outdir / "train.ids", train_ids)
    save_ids(outdir / "test.ids", test_ids)

    # ====== print stats ======
    def n_rows_for(cset):
        return int(df["combo_key"].isin(cset).sum())

    # leave_vel normal distribution
    leave_total = int(len(df_leave_norm))
    leave_train = int(df_leave_norm["combo_key"].isin(train_leave).sum())
    leave_test  = int(df_leave_norm["combo_key"].isin(test_leave).sum())

    # test normal/abnormal counts by prefix (best-effort)
    n_test_normal = sum(1 for x in test_ids if str(x).startswith("normal/"))
    n_test_abn    = len(test_ids) - n_test_normal

    print("===== DONE (SEALED by combo, Scheme A) =====")
    print(f"leave_vel: {leave_vel}")
    print("--- Combo counts ---")
    print(f"Train combos: {len(combos_train)}  (rows={n_rows_for(combos_train)})")
    print(f"Test  combos: {len(combos_test)}   (rows={n_rows_for(combos_test)})")
    print("--- Segment(id) counts ---")
    print(f"Train ids: {len(train_ids)}")
    print(f"Test  ids: {len(test_ids)}   (normal={n_test_normal}, abnormal={n_test_abn})")
    print("--- leave_vel NORMAL (vel==leave_vel) ---")
    print(f"leave_vel normal total rows: {leave_total}")
    print(f"leave_vel normal in train  : {leave_train}")
    print(f"leave_vel normal in test   : {leave_test}")


if __name__ == "__main__":
    main()
