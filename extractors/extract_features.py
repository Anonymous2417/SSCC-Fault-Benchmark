# -*- coding: utf-8 -*-
"""
extract_features.py
-------------------

Extract features sample-by-sample from index.csv and save to features.npz.

Two output modes are supported:

1) Standard mode:
   Saves a 2D array:
     X (N, D)

2) Per-channel mode (--per_channel):
   Saves a 1D object array:
     per_channel (N,)
   where:
     per_channel[i] is a list of length = number of channels,
     and each element is a 1D numpy array (variable length allowed).

Dependencies:
  numpy, pandas, torch, tqdm
"""

import os
import sys
import json
import argparse
import importlib.util
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


# ---------- Dynamic extractor loader ----------
def import_extractor_from_file(py_path: str, class_name: str = "FeatureExtractor"):
    py_path = str(py_path)
    spec = importlib.util.spec_from_file_location("user_extractor", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_extractor"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {py_path}")
    return getattr(mod, class_name)


def parse_paths_cell(cell: Any) -> List[str]:
    """Allow 'paths' column to be either list or JSON string."""
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        return json.loads(cell)
    raise ValueError("Column 'paths' must be either a list or a JSON string.")


def sanitize_id_to_filename(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace("\\", "__")


def main():
    parser = argparse.ArgumentParser(description="Feature extraction from index.csv")
    parser.add_argument("--index", required=True, help="Path to index.csv")
    parser.add_argument("--extractor_py", required=True,
                        help="Path to extractor .py file (must contain FeatureExtractor class)")
    parser.add_argument("--class_name", default="FeatureExtractor",
                        help="Extractor class name (default: FeatureExtractor)")
    parser.add_argument("--out_npz", required=True,
                        help="Output .npz file (contains X or per_channel, ids, labels, splits)")
    parser.add_argument("--multi_channel_strategy", default="concatenate",
                        choices=["concatenate", "average", "first", "last"])
    parser.add_argument("--vib_default_sr", type=int, default=None,
                        help="Default vibration sampling rate (if not provided in index.csv)")
    parser.add_argument("--cache_dir", default=None,
                        help="Optional cache directory for per-sample .npy/.npz")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N samples (debugging)")
    parser.add_argument("--extractor_args", default=None,
                        help="Extra JSON arguments passed to FeatureExtractor constructor")
    parser.add_argument("--per_channel", action="store_true",
                        help="Enable per-channel variable-length feature mode")
    parser.add_argument("--debug_first", type=int, default=0,
                        help="Print channel dimension info for first K samples")

    args = parser.parse_args()

    # 1) Load extractor
    Extractor = import_extractor_from_file(args.extractor_py, args.class_name)

    extractor_kwargs: Dict[str, Any] = {
        "multi_channel_strategy": args.multi_channel_strategy
    }

    if args.vib_default_sr is not None:
        extractor_kwargs["vib_default_sr"] = args.vib_default_sr

    if args.extractor_args:
        extra = json.loads(args.extractor_args)
        if not isinstance(extra, dict):
            raise ValueError("--extractor_args must be a JSON dictionary")
        extractor_kwargs.update(extra)

    extractor = Extractor(**extractor_kwargs)

    model = getattr(extractor, "model", None)
    if hasattr(model, "eval"):
        model.eval()

    # 2) Load index.csv
    df = pd.read_csv(args.index)
    required_cols = {"id", "paths", "split", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"index.csv missing required columns: {missing}")

    if "vibration_sr" not in df.columns:
        df["vibration_sr"] = None

    N_total = len(df)
    N = N_total if args.limit is None else min(args.limit, N_total)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    X_list = []
    per_channel_all = []
    ids, labels, splits = [], [], []

    pbar = tqdm(range(N), desc="extracting")

    for i in pbar:
        row = df.iloc[i]
        sid = str(row["id"])
        lab = int(row["label"])
        spl = str(row["split"])

        try:
            paths = parse_paths_cell(row["paths"])
            meta = {
                "paths": paths,
                "vibration_sr": None if pd.isna(row["vibration_sr"])
                                 else int(row["vibration_sr"]),
            }

            if args.per_channel:
                meta["return_per_channel"] = True

            with torch.no_grad():
                out = extractor.extract_features(meta)

            if args.per_channel:
                if isinstance(out, (list, tuple)):
                    feat = [
                        (t.detach().cpu().numpy().reshape(-1)
                         if isinstance(t, torch.Tensor)
                         else np.asarray(t).reshape(-1))
                        for t in out
                    ]
                else:
                    vec = out.detach().cpu().numpy() \
                        if isinstance(out, torch.Tensor) else np.asarray(out)
                    feat = [vec.reshape(-1)]
                per_channel_all.append(feat)
            else:
                if isinstance(out, torch.Tensor):
                    feat = out.detach().cpu().numpy().astype(np.float32).reshape(-1)
                else:
                    feat = np.asarray(out, dtype=np.float32).reshape(-1)
                X_list.append(feat)

            ids.append(sid)
            labels.append(lab)
            splits.append(spl)

        except Exception as e:
            print(f"[ERROR] Feature extraction failed for sample {sid}: {e}")
            traceback.print_exc()
            continue

    if args.per_channel and not per_channel_all:
        raise RuntimeError("No features extracted (per_channel mode).")
    if not args.per_channel and not X_list:
        raise RuntimeError("No features extracted.")

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)

    ids_arr = np.array(ids, dtype=str)
    labels_arr = np.array(labels, dtype=np.int64)
    splits_arr = np.array(splits, dtype=str)

    if args.per_channel:
        arr = np.empty((len(per_channel_all),), dtype=object)
        for i in range(len(per_channel_all)):
            arr[i] = list(per_channel_all[i])
        np.savez(out, per_channel=arr,
                 ids=ids_arr, labels=labels_arr, splits=splits_arr)
        print(f"[DONE] Saved per-channel features to {out}")
    else:
        X = np.stack(X_list, axis=0)
        np.savez(out, X=X,
                 ids=ids_arr, labels=labels_arr, splits=splits_arr)
        print(f"[DONE] Saved features to {out}, shape={X.shape}")


if __name__ == "__main__":
    main()