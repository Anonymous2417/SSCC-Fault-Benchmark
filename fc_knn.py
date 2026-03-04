# -*- coding: utf-8 -*-
"""
fc_knn.py
=========
7-channel kNN fault classification with view-level probability fusion.

Overview
--------
This script performs **multi-class fault classification** using a **per-channel kNN**
pipeline on 7-channel features. Each channel is processed independently:

  1) (Optional) Standardize features with StandardScaler (fit on TRAIN only).
  2) Query k-nearest neighbors in TRAIN for each TEST sample.
  3) Convert neighbor labels to per-class probabilities (distance-weighted vote).

It then produces predictions for multiple **views** by averaging per-channel probabilities:
  - audio_mean: channels [0,1,2]
  - vib_mean:   channels [3,4,5,6]
  - fused_mean: channels [0..6]

Supported feature NPZ formats
-----------------------------
A) Concatenated fixed-dim format:
   - keys: 'X' (N, 7*D), 'ids' (N,)

B) Per-channel variable-dim format (channel dims can differ; each channel fixed internally):
   - key: 'per_channel' as a 1D object array of length N
     where per_channel[i] is a list/tuple of length 7, each element is a 1D ndarray.
   - Enable this mode with: --per_channel
   - No padding is performed. If a channel has inconsistent dims across samples, an error is raised.

Labels
------
Labels are parsed from the sample id/path string:
  - ".../normal/..."                  -> "normal"
  - ".../abnormal/<fault>/..."        -> "<fault>"

Outputs
-------
For each view, a subdirectory is created under --outdir:
  view_<view>/
    - preds_multiclass.csv
    - cm_multiclass.csv
    - metrics_multiclass.csv
    - metrics.json

Additionally:
  - metrics_compare_views_multiclass.csv
  - unseen_test_samples.csv (if TEST contains labels not present in TRAIN)

Neighbor inspection (optional)
------------------------------
To audit which TRAIN samples are frequently retrieved as neighbors for certain TEST labels,
use:
  --inspect_labels dry,lean
  --inspect_view fused_mean
  --inspect_topk 11
  --inspect_mode all|correct|wrong

This generates:
  - inspect_neighbors_<labels>__<view>.csv
  - inspect_train_hit_counts_<labels>__<view>.csv

"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)


# ----------------------------
# label / condition parsing
# ----------------------------
def extract_fault_from_path(path_str: str):
    p = Path(path_str)
    parts = [s.lower() for s in p.parts]

    if "abnormal" in parts:
        idx = parts.index("abnormal")
        if idx + 1 < len(parts):
            return parts[idx + 1]
        return None

    if "normal" in parts:
        return "normal"

    return None


def extract_condition_from_path(path_str: str):
    p = Path(path_str)
    parts = [s.lower() for s in p.parts]

    if "abnormal" in parts:
        idx = parts.index("abnormal")
        if idx + 2 < len(parts):
            return parts[idx + 2]
        return None

    if "normal" in parts:
        idx = parts.index("normal")
        if idx + 1 < len(parts):
            return parts[idx + 1]
        return "normal"

    return parts[-2] if len(parts) >= 2 else None


# ----------------------------
# views
# ----------------------------
VIEW2CH = {
    "audio_mean": [0, 1, 2],
    "vib_mean":   [3, 4, 5, 6],
    "fused_mean": [0, 1, 2, 3, 4, 5, 6],
}


def read_id_list(path: str) -> np.ndarray:
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                buf.append(s)
    return np.array(buf, dtype=str)


def save_json(path: Path, obj: dict):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_knn(metric: str):
    metric = metric.lower()
    if metric in ("l2", "euclidean"):
        return NearestNeighbors(metric="minkowski", p=2, algorithm="auto")
    if metric in ("cos", "cosine"):
        return NearestNeighbors(metric="cosine", algorithm="auto")
    raise ValueError(f"Unknown metric: {metric} (use l2/cosine)")


def probs_from_neighbors(
    neigh_idx: np.ndarray,              # (Nt, k) indices in train set (local)
    neigh_dist: np.ndarray,             # (Nt, k) distances
    y_train: np.ndarray,                # (Ntr,) int labels
    C: int,
    vote: str,
    eps: float,
):
    vote = vote.lower()
    Nt, k = neigh_idx.shape
    probs = np.zeros((Nt, C), dtype=np.float32)

    if vote == "hard":
        for i in range(Nt):
            labs = y_train[neigh_idx[i]]
            cnt = np.bincount(labs, minlength=C).astype(np.float32)
            s = float(cnt.sum())
            probs[i] = cnt / (s if s > 0 else 1.0)
        return probs

    if vote == "weighted":
        w = 1.0 / (neigh_dist.astype(np.float32) + float(eps))
        for i in range(Nt):
            labs = y_train[neigh_idx[i]]
            wi = w[i]
            acc = np.zeros((C,), dtype=np.float32)
            for lab, ww in zip(labs, wi):
                acc[int(lab)] += float(ww)
            s = float(acc.sum())
            probs[i] = acc / (s if s > 0 else 1.0)
        return probs

    raise ValueError(f"Unknown vote: {vote} (use hard/weighted)")


def fuse_view_probs(probs_ch_list, view: str):
    chs = VIEW2CH[view]
    acc = None
    for c in chs:
        acc = probs_ch_list[c] if acc is None else (acc + probs_ch_list[c])
    return acc / float(len(chs))


def parse_csv_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


# ----------------------------
# per_channel loading
# ----------------------------
def load_7ch_blocks_from_npz(data: np.lib.npyio.NpzFile, per_channel: bool):
    """
    Return:
      - X_blocks: List[np.ndarray], length=7, each shape [N, D_ch] (D_ch can differ by channel)
      - ids: np.ndarray shape [N], dtype=str

    Supported formats:
      (A) concat format: keys 'X' [N, D], 'ids' [N], where D % 7 == 0
      (B) per_channel format: key 'per_channel' object array len N,
          where per_channel[i] is list/tuple length 7, each element is 1D array [D_ch]
    """
    if "ids" not in data:
        raise RuntimeError("NPZ must contain 'ids'.")

    ids = data["ids"].astype(str)
    N = len(ids)

    if per_channel:
        if "per_channel" not in data:
            raise RuntimeError(
                "[per_channel] NPZ missing key 'per_channel'. Available keys: "
                f"{list(data.keys())}"
            )

        pc = data["per_channel"]
        if pc.dtype != object or pc.ndim != 1 or len(pc) != N:
            raise RuntimeError(
                f"[per_channel] expected per_channel to be 1D object array of len N={N}, "
                f"but got shape={pc.shape}, dtype={pc.dtype}"
            )

        # infer dims per channel
        dims_by_ch = [None] * 7
        for i in range(N):
            item = pc[i]
            if not isinstance(item, (list, tuple)) or len(item) != 7:
                raise RuntimeError(
                    f"[per_channel] per_channel[{i}] must be list/tuple length 7, "
                    f"but got type={type(item)} len={getattr(item,'__len__',None)}"
                )
            for ch in range(7):
                v = item[ch]
                if not isinstance(v, np.ndarray):
                    v = np.asarray(v)
                if v.ndim != 1:
                    raise RuntimeError(
                        f"[per_channel] per_channel[{i}][{ch}] must be 1D, got shape={v.shape}"
                    )
                d = int(v.shape[0])
                if dims_by_ch[ch] is None:
                    dims_by_ch[ch] = d

        # build 7 matrices [N, D_ch] (no padding)
        X_blocks = []
        for ch in range(7):
            Dch = int(dims_by_ch[ch])
            buf = np.empty((N, Dch), dtype=np.float32)
            for i in range(N):
                v = pc[i][ch]
                if not isinstance(v, np.ndarray):
                    v = np.asarray(v)
                if v.ndim != 1:
                    raise RuntimeError(
                        f"[per_channel] per_channel[{i}][{ch}] not 1D, shape={v.shape}"
                    )
                if v.shape[0] != Dch:
                    raise RuntimeError(
                        f"[per_channel] channel {ch} feature dim mismatch at sample {i}: "
                        f"expected {Dch}, got {v.shape[0]}. (No padding.)"
                    )
                buf[i] = v.astype(np.float32, copy=False)
            X_blocks.append(buf)

        return X_blocks, ids

    # concat mode
    if "X" not in data:
        raise RuntimeError(
            "NPZ must contain 'X' and 'ids' for concat mode. "
            f"Available keys: {list(data.keys())}. "
            "If this is a per-channel NPZ, rerun with --per_channel."
        )

    X = data["X"].astype(np.float32)
    if X.ndim != 2 or X.shape[0] != N:
        raise RuntimeError(f"[concat] X must be [N,D] with N={N}, got {X.shape}")

    D = X.shape[1]
    if D % 7 != 0:
        raise RuntimeError(f"[concat] feature dim {D} not divisible by 7.")

    X_blocks = np.split(X, 7, axis=1)
    return X_blocks, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="NPZ containing features and ids (X+ids or per_channel+ids)")
    ap.add_argument("--train_ids", required=True, help="train.ids (one id per line)")
    ap.add_argument("--test_ids", required=True, help="test.ids (one id per line)")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--knn_k", type=int, default=11)
    ap.add_argument("--metric", type=str, default="l2", help="l2 or cosine")
    ap.add_argument("--vote", type=str, default="weighted", choices=["hard", "weighted"])
    ap.add_argument("--eps", type=float, default=1e-12)

    ap.add_argument("--views", type=str, default="audio_mean,vib_mean,fused_mean")
    ap.add_argument("--standardize", type=str, default="True", help="True/False (fit scaler on TRAIN only)")

    ap.add_argument("--unseen_mode", type=str, default="ignore", choices=["ignore", "error"])

    ap.add_argument(
        "--per_channel",
        action="store_true",
        help="Read per-channel features from npz['per_channel'] (object list-of-7 vectors).",
    )

    # inspection
    ap.add_argument("--inspect_labels", type=str, default="",
                    help="comma-separated TEST labels to inspect, e.g., dry,lean")
    ap.add_argument("--inspect_view", type=str, default="fused_mean",
                    help="view name used for inspection naming and channel subset")
    ap.add_argument("--inspect_topk", type=int, default=0,
                    help="top-k neighbors to export per channel (0 => use --knn_k)")
    ap.add_argument("--inspect_mode", type=str, default="all",
                    choices=["all", "correct", "wrong"],
                    help="inspect which TEST samples: all / only correct / only wrong")

    args = ap.parse_args()

    standardize = args.standardize.lower() == "true"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in VIEW2CH:
            raise ValueError(f"Unknown view: {v}. Allowed: {list(VIEW2CH.keys())}")

    inspect_labels = parse_csv_list(args.inspect_labels)
    inspect_view = (args.inspect_view or "fused_mean").strip()
    if inspect_view not in VIEW2CH:
        raise ValueError(f"--inspect_view must be in {list(VIEW2CH.keys())}, got {inspect_view}")

    inspect_topk = int(args.inspect_topk) if int(args.inspect_topk) > 0 else int(args.knn_k)
    inspect_topk = max(1, inspect_topk)

    # load npz
    data = np.load(args.npz, allow_pickle=True)
    X_blocks, ids_all = load_7ch_blocks_from_npz(data, per_channel=bool(args.per_channel))
    print(f"[info] loaded ids={len(ids_all)} per_channel={bool(args.per_channel)}")
    print("[info] channel dims =", [int(b.shape[1]) for b in X_blocks])

    # map id -> index
    id2idx = {sid: i for i, sid in enumerate(ids_all)}

    # read splits
    train_ids = read_id_list(args.train_ids)
    test_ids = read_id_list(args.test_ids)
    print(f"[info] train_ids={len(train_ids)}, test_ids={len(test_ids)}")

    # filter ids to those present
    miss_tr = [x for x in train_ids if x not in id2idx]
    miss_te = [x for x in test_ids if x not in id2idx]
    if miss_tr:
        print(f"[warn] {len(miss_tr)} train ids not in npz (first 5): {miss_tr[:5]}")
    if miss_te:
        print(f"[warn] {len(miss_te)} test ids not in npz (first 5): {miss_te[:5]}")

    train_ids = np.array([x for x in train_ids if x in id2idx], dtype=str)
    test_ids = np.array([x for x in test_ids if x in id2idx], dtype=str)

    # parse labels
    ytr_str = np.array([extract_fault_from_path(s) for s in train_ids], dtype=object)
    yte_str = np.array([extract_fault_from_path(s) for s in test_ids], dtype=object)

    # drop None labels
    tr_keep = np.array([y is not None for y in ytr_str], dtype=bool)
    te_keep = np.array([y is not None for y in yte_str], dtype=bool)
    if not np.all(tr_keep):
        print(f"[warn] drop {int((~tr_keep).sum())} train samples with label=None (parsing failed)")
    if not np.all(te_keep):
        print(f"[warn] drop {int((~te_keep).sum())} test samples with label=None (parsing failed)")

    train_ids = train_ids[tr_keep]
    ytr_str = ytr_str[tr_keep].astype(str)
    test_ids = test_ids[te_keep]
    yte_str = yte_str[te_keep].astype(str)

    # class mapping from TRAIN only
    classes = sorted(list(set(ytr_str.tolist())))
    class2idx = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    print(f"[info] num_classes(train)={C}, classes={classes}")

    ytr = np.array([class2idx[y] for y in ytr_str], dtype=np.int64)

    # unseen labels in test
    unseen = sorted(list(set([y for y in yte_str.tolist() if y not in class2idx])))
    if unseen:
        msg = f"[warn] unseen labels in test (not in train): {unseen}"
        if args.unseen_mode == "error":
            raise RuntimeError(msg)
        print(msg + " -> ignored in metrics (listed in unseen_test_samples.csv)")

    yte = np.array([class2idx[y] if y in class2idx else -1 for y in yte_str], dtype=np.int64)

    # indices into feature arrays
    tri = np.array([id2idx[s] for s in train_ids], dtype=np.int64)
    tei = np.array([id2idx[s] for s in test_ids], dtype=np.int64)

    # evaluable test subset
    eval_mask = (yte >= 0)
    tei_eval = tei[eval_mask]
    test_ids_eval = test_ids[eval_mask]
    yte_eval = yte[eval_mask]
    yte_eval_str = np.array([s.lower() for s in yte_str[eval_mask].tolist()], dtype=str)

    if len(test_ids_eval) == 0:
        raise RuntimeError("No evaluable test samples (all unseen labels or parsing failed).")

    probs_ch = [np.zeros((len(test_ids_eval), C), dtype=np.float32) for _ in range(7)]
    neigh_idx_ch = [None] * 7
    neigh_dist_ch = [None] * 7

    k = int(args.knn_k)
    for ch in range(7):
        Xi = X_blocks[ch]
        Xtr = Xi[tri]
        Xte = Xi[tei_eval]

        if standardize:
            scaler = StandardScaler().fit(Xtr)
            Xtr = scaler.transform(Xtr).astype(np.float32)
            Xte = scaler.transform(Xte).astype(np.float32)

        nn = build_knn(args.metric)
        nn.fit(Xtr)

        kk = min(max(1, k), len(Xtr))
        dist, idx = nn.kneighbors(Xte, n_neighbors=kk, return_distance=True)

        neigh_idx_ch[ch] = idx.astype(np.int64, copy=False)
        neigh_dist_ch[ch] = dist.astype(np.float32, copy=False)

        probs_ch[ch] = probs_from_neighbors(idx, dist, ytr, C, vote=args.vote, eps=args.eps)
        print(f"[info] ch{ch}: knn_k={kk}, metric={args.metric}, vote={args.vote}, D={Xi.shape[1]}")

    results_rows = []
    preds_by_view = {}

    for view in views:
        vdir = outdir / f"view_{view}"
        vdir.mkdir(parents=True, exist_ok=True)

        Pf = fuse_view_probs(probs_ch, view)
        ypred = Pf.argmax(axis=1).astype(np.int64)
        preds_by_view[view] = ypred

        acc = float(accuracy_score(yte_eval, ypred))
        bal = float(balanced_accuracy_score(yte_eval, ypred))
        f1m = float(f1_score(yte_eval, ypred, average="macro", zero_division=0))
        cm = confusion_matrix(yte_eval, ypred, labels=np.arange(C))

        rep = classification_report(
            yte_eval, ypred, target_names=classes, output_dict=True, zero_division=0
        )
        percls = {f"F1_{c}": float(rep[c]["f1-score"]) for c in classes}

        conds = [extract_condition_from_path(s) for s in test_ids_eval.tolist()]
        df_pred = pd.DataFrame({
            "id": test_ids_eval,
            "condition": conds,
            "y_true": [classes[i] for i in yte_eval],
            "y_pred": [classes[i] for i in ypred],
        })
        for ci, cname in enumerate(classes):
            df_pred[f"p_{cname}"] = Pf[:, ci]
        df_pred.to_csv(vdir / "preds_multiclass.csv", index=False)

        df_cm = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in classes],
            columns=[f"pred_{c}" for c in classes],
        )
        df_cm.to_csv(vdir / "cm_multiclass.csv", index=True)

        row = {
            "view": view,
            "raw_acc": acc,
            "balanced_acc": bal,
            "macro_f1": f1m,
            "N_train": int(len(train_ids)),
            "N_test_eval": int(len(test_ids_eval)),
            "knn_k": int(args.knn_k),
            "metric": args.metric,
            "vote": args.vote,
            "standardize": bool(standardize),
            "per_channel": bool(args.per_channel),
        }
        row.update(percls)
        pd.DataFrame([row]).to_csv(vdir / "metrics_multiclass.csv", index=False)

        meta = {
            "view": view,
            "classes": classes,
            "unseen_test_labels": unseen,
            "args": vars(args),
            "metrics": {
                "raw_acc": acc,
                "balanced_acc": bal,
                "macro_f1": f1m,
            }
        }
        save_json(vdir / "metrics.json", meta)

        results_rows.append(row)
        print(f"[{view}] acc={acc:.4f} bal={bal:.4f} macro_f1={f1m:.4f}")

    pd.DataFrame(results_rows).to_csv(outdir / "metrics_compare_views_multiclass.csv", index=False)

    if unseen:
        unseen_rows = []
        for sid, ystr, yi in zip(test_ids.tolist(), yte_str.tolist(), yte.tolist()):
            if yi >= 0:
                continue
            unseen_rows.append({"id": sid, "y_true": f"UNSEEN:{ystr}"})
        if unseen_rows:
            pd.DataFrame(unseen_rows).to_csv(outdir / "unseen_test_samples.csv", index=False)

    # ----------------------------
    # neighbor inspection
    # ----------------------------
    if inspect_labels:
        want = set([x.lower() for x in inspect_labels])
        sel_label = np.array([lab in want for lab in yte_eval_str], dtype=bool)

        ypred_ins = preds_by_view.get(inspect_view, None)
        if ypred_ins is None:
            print(f"[warn] inspect_view={inspect_view} not in evaluated views; skip inspect.")
        else:
            if args.inspect_mode == "correct":
                sel_mode = (ypred_ins == yte_eval)
            elif args.inspect_mode == "wrong":
                sel_mode = (ypred_ins != yte_eval)
            else:
                sel_mode = np.ones_like(sel_label, dtype=bool)

            sel = sel_label & sel_mode
            n_sel = int(sel.sum())

            if n_sel == 0:
                print(f"[warn] inspect selected 0 test samples. "
                      f"labels={inspect_labels}, mode={args.inspect_mode}, view={inspect_view}.")
            else:
                rows = []
                topk = min(int(inspect_topk), int(args.knn_k))
                topk = max(1, topk)

                chs = VIEW2CH[inspect_view]
                train_ids_list = train_ids.tolist()
                ytr_str_list = ytr_str.tolist()

                for i in np.where(sel)[0]:
                    te_id = str(test_ids_eval[i])
                    te_lab = str(yte_eval_str[i])
                    te_pred = classes[int(ypred_ins[i])]
                    te_true = classes[int(yte_eval[i])]

                    for ch in chs:
                        idx = neigh_idx_ch[ch][i]
                        dist = neigh_dist_ch[ch][i]
                        kk = min(topk, len(idx))
                        for r in range(kk):
                            tr_local = int(idx[r])
                            rows.append({
                                "inspect_view": inspect_view,
                                "inspect_mode": args.inspect_mode,
                                "test_id": te_id,
                                "test_label": te_lab,
                                "test_true": te_true,
                                "test_pred": te_pred,
                                "channel": int(ch),
                                "rank": int(r + 1),
                                "distance": float(dist[r]),
                                "train_id": train_ids_list[tr_local],
                                "train_label": ytr_str_list[tr_local],
                            })

                df_ins = pd.DataFrame(rows)

                labels_tag = "_".join(inspect_labels)
                out_csv = outdir / f"inspect_neighbors_{labels_tag}__{inspect_view}.csv"
                df_ins.to_csv(out_csv, index=False)
                print(f"[OK] saved inspect neighbors -> {out_csv} (test_samples={n_sel}, rows={len(df_ins)})")

                hit = df_ins.groupby(["train_id", "train_label"]).size().reset_index(name="hit_count")
                hit = hit.sort_values("hit_count", ascending=False)
                hit_csv = outdir / f"inspect_train_hit_counts_{labels_tag}__{inspect_view}.csv"
                hit.to_csv(hit_csv, index=False)
                print(f"[OK] saved train hit counts -> {hit_csv}")

    print(f"[done] outputs -> {outdir}")


if __name__ == "__main__":
    main()
