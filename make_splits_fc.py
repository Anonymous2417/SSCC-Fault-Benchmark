# -*- coding: utf-8 -*-
"""
make_splits_fc.py
===========================================
Combo-style condition-balanced splits for fault classification (FC)

Purpose
-------
This script generates **condition-level** train/test partitions for a fault
classification dataset where samples belong to operating **conditions** defined by:

    condition = (velocity, load, noise_class)

Key requirements
----------------
1) Leave-one-velocity hold-out (leave_vel):
   - All samples with velocity == leave_vel are assigned to TEST.
   - TRAIN must contain **zero** samples from leave_vel.

2) Condition-balanced split inside the base domain (vel != leave_vel):
   - First, group samples into unique conditions (vel, load, noise_class).
   - Stratify by (load, noise_class), then split conditions per stratum into
     TRAIN vs TEST with an approximate ratio controlled by --train_frac
     (default 0.5, i.e., roughly 1:1 by condition).

3) Optional HARDER+ (sample-level shift):
   - After the condition split, move a subset of samples from BASE-TRAIN to TEST
     if (label in hard_labels) AND (vel in hard_vels).
   - Note: This is **sample-level** movement (not condition-level).

Outputs
-------
- train.ids / test.ids
- train_dist.csv / test_dist.csv
- split_meta.json
"""

import json
import argparse
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# -------------------------
# ID parsing helpers
# -------------------------
LOAD_RE = re.compile(r"/(heavy|med|light)[-_]vel", re.I)
VEL_RE = re.compile(r"vel(\d+)", re.I)
NOISE_RE = re.compile(r"[_/](clean|noise[^_/]*)", re.I)


def parse_id(s: str):
    """
    Parse (load, vel, noise_cls) from an item id / path-like string.
    Expected patterns:
      - load: /heavy-vel..., /med_vel..., /light-vel...
      - vel:  vel80, vel100, ...
      - noise: /clean, /noise*, _clean, _noise*
    """
    m_load = LOAD_RE.search(s)
    m_vel = VEL_RE.search(s)
    m_noise = NOISE_RE.search(s)

    load = m_load.group(1).lower() if m_load else ""
    vel = m_vel.group(1) if m_vel else ""
    noise = m_noise.group(1).lower() if m_noise else "clean"
    noise_cls = "noise" if noise.startswith("noise") else "clean"
    return load, vel, noise_cls


def save_ids(path: Path, arr):
    with open(path, "w", encoding="utf-8") as f:
        for x in arr:
            f.write(str(x) + "\n")


def parse_csv_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def unique_preserve_order(seq):
    return list(dict.fromkeys(seq))


def main():
    ap = argparse.ArgumentParser(description="Condition-level data partitioning for fault classification.")

    ap.add_argument("--csv", required=True, help="Path to index CSV")
    ap.add_argument("--outdir", required=True, help="Output directory for split files")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label_col", type=int, default=2, help="Label column index (0-based)")
    ap.add_argument("--id_col", type=int, default=0, help="ID column index (0-based)")

    # Leave-one-velocity: fully held out from TRAIN
    ap.add_argument("--leave_vel", type=str, default="100")

    # Optional: strengthen domain shift inside base domain (vel != leave_vel)
    ap.add_argument(
        "--exclude_noise_from_base",
        type=str,
        default="False",
        help="True: keep only clean samples in base domain (vel != leave_vel)",
    )
    ap.add_argument(
        "--exclude_load_from_base",
        type=str,
        default="",
        help="Optional: drop one load from base domain entirely: heavy/med/light",
    )

    # HARDER+ (sample-level move from base TRAIN -> TEST)
    ap.add_argument(
        "--hard_labels",
        type=str,
        default="dry,lean",
        help="Comma-separated labels to move from base TRAIN to TEST, e.g., dry,lean",
    )
    ap.add_argument(
        "--hard_vels",
        type=str,
        default="80,100",
        help="Comma-separated velocities for HARDER+ move, e.g., 80,100. "
             "NOTE: leave_vel is already fully held out from TRAIN.",
    )

    # Condition split ratio per stratum
    ap.add_argument(
        "--train_frac",
        type=float,
        default=0.5,
        help="Fraction of base conditions (per load×noise_class) assigned to TRAIN. Default 0.5.",
    )

    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    leave_vel = str(args.leave_vel).strip()
    exclude_noise_from_base = str(args.exclude_noise_from_base).lower() == "true"
    excl_load = args.exclude_load_from_base.strip().lower()

    hard_labels = set(parse_csv_list(args.hard_labels))
    hard_vels = set(parse_csv_list(args.hard_vels))  # e.g., {"80","100"}

    # -------------------------
    # Load & parse CSV
    # -------------------------
    df = pd.read_csv(args.csv)
    df["id"] = df.iloc[:, args.id_col].astype(str)
    df["label"] = df.iloc[:, args.label_col].astype(str).str.lower()

    loads, vels, noises = [], [], []
    for x in df["id"]:
        l, v, n = parse_id(x)
        loads.append(l)
        vels.append(v)
        noises.append(n)

    df["load"] = loads
    df["vel"] = vels
    df["noise_cls"] = noises

    # -------------------------
    # 1) Leave-one-velocity => TEST
    # -------------------------
    df_leave = df[df["vel"] == leave_vel].copy()
    df_base_all = df[df["vel"] != leave_vel].copy()

    if exclude_noise_from_base:
        before = len(df_base_all)
        df_base_all = df_base_all[df_base_all["noise_cls"] == "clean"].copy()
        print(f"[HARD] exclude_noise_from_base=True: base {before} -> {len(df_base_all)} (clean only)")

    if excl_load in ("heavy", "med", "light"):
        before = len(df_base_all)
        df_base_all = df_base_all[df_base_all["load"] != excl_load].copy()
        print(f"[HARD] exclude_load_from_base={excl_load}: base {before} -> {len(df_base_all)} (drop that load)")

    # -------------------------
    # 2) Condition-balanced split in base domain
    # condition = (vel, load, noise_cls)
    # stratify by (load, noise_cls)
    # -------------------------
    df_base_all["cond_key"] = list(zip(df_base_all["vel"], df_base_all["load"], df_base_all["noise_cls"]))
    df_base_all["stratum"] = list(zip(df_base_all["load"], df_base_all["noise_cls"]))

    stratum_to_conds = defaultdict(list)
    for (stratum, cond_key) in df_base_all[["stratum", "cond_key"]].drop_duplicates().itertuples(index=False):
        stratum_to_conds[stratum].append(cond_key)

    train_conds = set()
    test_conds = set()

    for stratum, conds in stratum_to_conds.items():
        conds = list(conds)
        rng.shuffle(conds)

        n = len(conds)
        n_train = int(np.floor(n * float(args.train_frac)))

        train_part = conds[:n_train]
        test_part = conds[n_train:]

        train_conds.update(train_part)
        test_conds.update(test_part)

        print(
            f"[COND] stratum(load,noise)={stratum}: conds={n}, "
            f"train_conds={len(train_part)}, test_conds={len(test_part)}"
        )

    df_base_train = df_base_all[df_base_all["cond_key"].isin(train_conds)].copy()
    df_base_test = df_base_all[df_base_all["cond_key"].isin(test_conds)].copy()

    print(f"[COND] base domain condition split done: base_train={len(df_base_train)} samples, base_test={len(df_base_test)} samples")
    print(f"[COND] #conditions: train={len(train_conds)}, test={len(test_conds)} (target ~ 1:1)")

    # -------------------------
    # 3) HARDER+ (sample-level): move some base TRAIN samples to TEST
    # -------------------------
    extra_test_pool = pd.DataFrame(columns=df.columns)

    if hard_labels and hard_vels:
        mask_hard = df_base_train["label"].isin(hard_labels) & df_base_train["vel"].isin(hard_vels)
        df_hold = df_base_train[mask_hard].copy()
        df_keep = df_base_train[~mask_hard].copy()

        extra_test_pool = df_hold
        df_base_train = df_keep

        print(
            f"[HARD+] move from base TRAIN to TEST: labels={sorted(list(hard_labels))}, "
            f"vels={sorted(list(hard_vels))} -> moved {len(df_hold)} samples; "
            f"base_train now {len(df_base_train)}"
        )

    # -------------------------
    # Assemble final split
    # -------------------------
    train_ids = df_base_train["id"].tolist()

    test_ids = []
    test_ids.extend(df_leave["id"].tolist())      # full leave_vel hold-out
    test_ids.extend(df_base_test["id"].tolist())  # base test conditions
    if len(extra_test_pool) > 0:
        test_ids.extend(extra_test_pool["id"].tolist())

    train_ids = unique_preserve_order(train_ids)
    test_ids = unique_preserve_order(test_ids)

    # -------------------------
    # Sanity checks
    # -------------------------
    df_train_chk = df[df["id"].isin(train_ids)]
    if (df_train_chk["vel"] == leave_vel).any():
        bad = df_train_chk[df_train_chk["vel"] == leave_vel]["id"].head(5).tolist()
        raise RuntimeError(f"[FATAL] TRAIN contains leave_vel={leave_vel} samples, e.g. {bad}")

    inter = set(train_ids) & set(test_ids)
    if inter:
        some = list(inter)[:5]
        raise RuntimeError(f"[FATAL] TRAIN/TEST overlap detected, e.g. {some}")

    # -------------------------
    # Save outputs
    # -------------------------
    save_ids(outdir / "train.ids", train_ids)
    save_ids(outdir / "test.ids", test_ids)

    dist_train = (
        df[df["id"].isin(train_ids)]
        .groupby(["label", "vel", "load", "noise_cls"])
        .size()
        .reset_index(name="n")
    )
    dist_test = (
        df[df["id"].isin(test_ids)]
        .groupby(["label", "vel", "load", "noise_cls"])
        .size()
        .reset_index(name="n")
    )

    dist_train.to_csv(outdir / "train_dist.csv", index=False)
    dist_test.to_csv(outdir / "test_dist.csv", index=False)

    all_labels = sorted(df["label"].unique().tolist())
    train_labels = sorted(df[df["id"].isin(train_ids)]["label"].unique().tolist())
    test_labels = sorted(df[df["id"].isin(test_ids)]["label"].unique().tolist())

    def count_conds(dsub: pd.DataFrame) -> int:
        if len(dsub) == 0:
            return 0
        return len(set(zip(dsub["vel"], dsub["load"], dsub["noise_cls"])))

    meta = {
        "split_name": "combo_condition_balanced_fc",
        "leave_vel": leave_vel,
        "seed": int(args.seed),

        "task": "fault_classification",
        "partition_level": "condition_level",
        "condition_def": "(vel, load, noise_cls)",
        "train_frac_per_stratum": float(args.train_frac),

        "exclude_noise_from_base": bool(exclude_noise_from_base),
        "exclude_load_from_base": excl_load if excl_load else "",

        "hard_labels": sorted(list(hard_labels)),
        "hard_vels": sorted(list(hard_vels)),
        "N_extra_test_from_base_train_moved": int(len(extra_test_pool)),

        "N_train": int(len(train_ids)),
        "N_test": int(len(test_ids)),

        "N_train_conditions": int(count_conds(df[df["id"].isin(train_ids)])),
        "N_test_conditions": int(count_conds(df[df["id"].isin(test_ids)])),

        "train_labels": train_labels,
        "test_labels": test_labels,
        "all_labels": all_labels,
    }
    (outdir / "split_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("===== DONE (Condition-level splits for fault classification) =====")
    print(f"Leave velocity (fully in TEST): {leave_vel}")
    print(f"Train total: {len(train_ids)}  |  Test total: {len(test_ids)}")
    print(f"Train conditions: {meta['N_train_conditions']}  |  Test conditions: {meta['N_test_conditions']}")
    print(f"Extra moved from base TRAIN -> TEST: {len(extra_test_pool)}")
    print(f"Train labels: {train_labels}")
    print(f"Test labels:  {test_labels}")
    print(f"[OK] Saved -> {outdir}")


if __name__ == "__main__":
    main()