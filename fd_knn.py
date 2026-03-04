# -*- coding: utf-8 -*-
"""
fd_knn.py
================================================================================

kNN-based Fault Detection with Multi-Channel Mean Fusion

Overview
--------
This script performs anomaly scoring using k-Nearest Neighbor distance
as the anomaly score:
    score(x) = mean distance to the k nearest neighbors (Euclidean or cosine)
              (training queries exclude self-match)

Scoring is conducted independently per channel, followed by
channel-level normalization and score averaging.

Channel Views
-------------
The script supports three evaluation views:
    - audiomean
        Uses the first 3 channels (audio).
        Each channel is scored independently.
        Final score = mean over 3 audio channel scores.

    - vibrationmean
        Uses the last 4 channels (vibration).
        Each channel is scored independently.
        Final score = mean over 4 vibration channel scores.

    - fusedmean
        Uses all 7 channels.
        Each channel is scored independently.
        Final score = mean over 7 channel scores.

Training & Evaluation Protocol
-------------------------------
Training:
    - Only NORMAL samples from the training split are used
      to construct the kNN reference set.

Testing:
    - All samples from the test split are scored.
    - Primary metric: AUROC.
    - Optional: threshold-based binary classification with
      confusion matrix and ACC / Precision / Recall / F1.

Split Sources
-------------
Provide either:
    1) --splits_dir
         Directory containing:
            {dev_source_normal.ids}
            {eval_normal.ids}
            {eval_abn.ids}

    OR

    2) --train_ids  {train_ids_file}
       --test_ids   {test_ids_file}

Feature Storage Formats
-----------------------
Two feature storage modes are supported:

1) Concatenated (non per-channel)
   Required keys in NPZ:
       - X   : (N, 7*D)
       - ids : (N,)
       - splits or labels

2) Per-channel mode (--per_channel)
   Supported structures:
       - X_ch0 ... X_ch6  (each shape (N, D_ch))
       - per_channel      object array
       - X_per_channel    object array

Note:
    - Dimensionality may differ across channels.
    - Within each individual channel, dimensionality must be fixed.

Channel-Level Processing
-------------------------
For each channel:

    1) (Optional) Standardization using StandardScaler
       fitted on training NORMAL samples only.
    2) Fit kNN model.
    3) Compute mean distance of k nearest neighbors (exclude self for train).
    4) Normalize per-channel scores using:
         - quantile (ECDF-based percentile mapping), or
         - zscore  (standard score + min-max to [0,1])

Final score:
    Mean over normalized channel scores.

Thresholding (Optional)
------------------------
If --do_cm is enabled, a threshold is selected using one of:

    - youden      : maximize TPR - FPR
    - fpr_at      : select threshold achieving target FPR
    - percentile  : fixed percentile of score distribution
    - score       : user-defined fixed score threshold

Outputs
-------
{OUTDIR}/view_{view_name}/
    - test_scores.csv
    - per_channel_scores.csv
    - metrics.json
    - cm_*.csv              (if enabled)
    - threshold_*.json      (if enabled)
    - roc_curve.csv

Additionally:
    {OUTDIR}/metrics_compare_views.csv

Example Usage
-------------
python fd_knn.py \
    --npz {FEATURE_FILE}.npz \
    --outdir {OUTPUT_DIR} \
    --train_ids {TRAIN_IDS_FILE} \
    --test_ids {TEST_IDS_FILE} \
    --modal_modes vibrationmean,audiomean,fusedmean \
    --knn_k {K_VALUE} \
    --fusion_norm quantile \
    --per_channel
"""

import argparse, json, csv
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

# ----------------- io helpers -----------------
def load_id_list(p: Optional[str]) -> Optional[np.ndarray]:
    if not p:
        return None
    buf = []
    with open(p, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # 去掉路径前缀（只保留最后两段或三段）
            parts = s.split('/')
            if len(parts) >= 3:
                # abnormal/fault_type/sample_id
                s = "/".join(parts[-3:])
            elif len(parts) == 2:
                # normal/sample_id
                s = "/".join(parts[-2:])
            else:
                s = parts[-1]
            buf.append(s)
    return np.array(buf, dtype=str)


def safe_load_ids(path: Path) -> np.ndarray:
    arr = load_id_list(str(path))
    if arr is None or len(arr) == 0:
        return np.array([], dtype=str)
    return arr


def load_from_8plus8_splits_dir(splits_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    sd = Path(splits_dir)
    tr = safe_load_ids(sd / 'dev_source_normal.ids')
    if tr is None or len(tr) == 0:
        raise RuntimeError(f"[splits_dir] missing or empty: {sd/'dev_source_normal.ids'}")
    te_n = safe_load_ids(sd / 'eval_normal.ids')
    te_a = safe_load_ids(sd / 'eval_abn.ids')
    te = np.concatenate([te_n, te_a], axis=0)
    if len(te) == 0:
        raise RuntimeError(f"[splits_dir] missing test files: need eval_normal.ids and/or eval_abn.ids")
    return tr, te


# ----------------- npz loader -----------------
def _stack_object_vectors(vecs, name: str) -> np.ndarray:
    out = []
    lens = []
    for v in vecs:
        a = np.asarray(v).reshape(-1)
        out.append(a.astype(np.float32, copy=False))
        lens.append(a.shape[0])
    if len(set(lens)) != 1:
        uniq = sorted(set(lens))
        raise RuntimeError(
            f"[per_channel] {name}: inconsistent lengths within this channel. "
            f"unique lens={uniq[:20]} (showing up to 20). "
            f"KNN/Scaler require fixed dim per channel."
        )
    return np.stack(out, axis=0)  # (N, D)


def load_7ch_blocks_from_npz(data: np.lib.npyio.NpzFile, per_channel: bool) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Return:
      ids: (N,)
      blocks: list of 7 arrays, blocks[ch] is (N, Dch)
    """
    if "ids" not in data:
        raise RuntimeError("npz must contain 'ids'.")

    ids = data["ids"].astype(str)
    N = len(ids)

    # ---------- non-per_channel: expect X (N, 7*D) ----------
    if not per_channel:
        if "X" not in data:
            raise RuntimeError("npz must contain 'X' and 'ids'.")
        X = data["X"].astype(np.float32)
        if X.ndim != 2:
            raise RuntimeError(f"X must be 2D, got {X.shape}")
        if X.shape[1] % 7 != 0:
            raise RuntimeError(f"Feature dim {X.shape[1]} not divisible by 7. Your npz should be 7ch concat.")
        return ids, np.split(X, 7, axis=1)

    # ---------- per_channel mode ----------
    # Case A: X_ch0..X_ch6
    if all([f"X_ch{i}" in data for i in range(7)]):
        blocks = []
        for i in range(7):
            Xi = data[f"X_ch{i}"].astype(np.float32)
            if Xi.ndim != 2 or Xi.shape[0] != N:
                raise RuntimeError(f"[per_channel] X_ch{i} shape mismatch: {Xi.shape}, expected ({N}, D)")
            blocks.append(Xi)
        return ids, blocks

    # Case B/C: per_channel or X_per_channel
    key = "per_channel" if "per_channel" in data else ("X_per_channel" if "X_per_channel" in data else None)
    if key is None:
        raise RuntimeError(
            "[per_channel] Cannot find per-channel features in npz. Expected either:\n"
            "  - keys: X_ch0..X_ch6 (each [N,D_ch])\n"
            "  - or key: per_channel / X_per_channel\n"
            f"Available keys: {list(data.keys())}"
        )

    pc = data[key]  # object
    if pc.shape[0] != N:
        raise RuntimeError(f"[per_channel] {key} first dim mismatch: {pc.shape[0]} vs ids {N}")

    # B: pc is (N,7) object array
    if pc.ndim == 2 and pc.shape[1] == 7:
        blocks = []
        for ch in range(7):
            vecs = [pc[i, ch] for i in range(N)]
            blocks.append(_stack_object_vectors(vecs, f"{key}[ch{ch}]"))
        return ids, blocks

    # C: pc is (N,) object; each pc[i] is list/tuple length 7
    if pc.ndim == 1:
        first = pc[0]
        if not isinstance(first, (list, tuple)) or len(first) != 7:
            raise RuntimeError(f"[per_channel] {key}[0] should be list/tuple of length 7, got {type(first)}")
        blocks = []
        for ch in range(7):
            vecs = [pc[i][ch] for i in range(N)]
            blocks.append(_stack_object_vectors(vecs, f"{key}[ch{ch}]"))
        return ids, blocks

    raise RuntimeError(f"[per_channel] Unsupported {key} structure: shape={pc.shape}, ndim={pc.ndim}")


# ----------------- channels & knn -----------------
def fit_knn(Xtr: np.ndarray, metric: str = "l2") -> NearestNeighbors:
    metric = (metric or "l2").lower()
    if metric in ("l2", "euclidean"):
        nn = NearestNeighbors(metric='minkowski', p=2, algorithm='auto')
    elif metric in ("cos", "cosine"):
        nn = NearestNeighbors(metric='cosine', algorithm='auto')
    else:
        raise ValueError(f"Unknown metric: {metric} (use l2/cosine)")
    nn.fit(Xtr)
    return nn


def knn_kmean_distance(
    nn: NearestNeighbors,
    Xref: np.ndarray,
    Xq: np.ndarray,
    k: int,
    exclude_self: bool = False
) -> np.ndarray:
    """
    Return mean distance to k nearest neighbors.
    If exclude_self is True, drop the closest neighbor (self) before averaging.
    """
    k = max(1, int(k))
    n_ref = int(len(Xref))
    if n_ref <= 0:
        raise RuntimeError("KNN reference set is empty.")

    if exclude_self:
        if n_ref <= 1:
            raise RuntimeError("Need at least 2 reference samples to exclude self.")
        k_eff = min(k, n_ref - 1)
        n_neighbors = min(k_eff + 1, n_ref)
    else:
        k_eff = min(k, n_ref)
        n_neighbors = k_eff

    dist, _ = nn.kneighbors(Xq, n_neighbors=n_neighbors, return_distance=True)
    if exclude_self:
        if dist.shape[1] <= 1:
            raise RuntimeError("Not enough neighbors after excluding self.")
        dist = dist[:, 1:]

    if dist.shape[1] < k_eff:
        k_eff = dist.shape[1]
    return dist[:, :k_eff].mean(axis=1)


# ----------------- quantile mapping -----------------
def ecdf_quantile(train_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    xs = np.sort(np.asarray(train_scores, dtype=float))
    N = xs.shape[0]
    k = np.searchsorted(xs, np.asarray(query_scores, dtype=float), side='right')
    q = (k + 0.5) / (N + 1.0)
    return np.clip(q, 1e-6, 1 - 1e-6)


# ----------------- core scoring: per-channel -> normalize -> mean -----------------
def mean_score_over_channels(
    X_blocks: List[np.ndarray],         # list of (N, Dch), variable Dch allowed
    ids: np.ndarray,                    # (N,)
    is_normal_all: np.ndarray,          # (N,) bool
    train_ids: np.ndarray,              # ids in train split
    test_ids: np.ndarray,               # ids in test split
    k: int,
    dump_dir: Path = None,
    fusion_norm: str = 'quantile',      # 'quantile' or 'zscore'
    metric: str = "l2",
    standardize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, Dict[str, float]]:
    pos = {sid: i for i, sid in enumerate(ids)}
    tri_all = np.array([pos[sid] for sid in train_ids if sid in pos], dtype=int)
    tei_all = np.array([pos[sid] for sid in test_ids  if sid in pos], dtype=int)

    if len(tri_all) == 0 or len(tei_all) == 0:
        raise RuntimeError(f"[mean_score_over_channels] empty tri_all={len(tri_all)} or tei_all={len(tei_all)}. "
                           f"Check your ids mapping.")

    tri_norm = tri_all[is_normal_all[tri_all]]
    if len(tri_norm) < 2:
        raise RuntimeError(f"[mean_score_over_channels] not enough train normal: {len(tri_norm)}")

    y_test = (~is_normal_all[tei_all]).astype(int)

    ch_scores = []
    for ch, Xi in enumerate(X_blocks):
        if Xi.ndim != 2 or Xi.shape[0] != len(ids):
            raise RuntimeError(f"[channel {ch}] Xi shape invalid: {Xi.shape}, expected (N, Dch)")

        Xtr_raw = Xi[tri_norm]
        Xte_raw = Xi[tei_all]

        if standardize:
            scaler = StandardScaler().fit(Xtr_raw)
            Xtr = scaler.transform(Xtr_raw).astype(np.float32)
            Xte = scaler.transform(Xte_raw).astype(np.float32)
        else:
            Xtr = Xtr_raw.astype(np.float32, copy=False)
            Xte = Xte_raw.astype(np.float32, copy=False)

        nn = fit_knn(Xtr, metric=metric)
        dist_tr = knn_kmean_distance(nn, Xtr, Xtr, k=k, exclude_self=True)
        dist_te = knn_kmean_distance(nn, Xtr, Xte, k=k, exclude_self=False)

        if fusion_norm == 'quantile':
            di_norm = ecdf_quantile(dist_tr, dist_te)
        elif fusion_norm == 'zscore':
            mu = float(dist_tr.mean())
            sigma = float(dist_tr.std(ddof=1)) + 1e-12
            z = (dist_te - mu) / sigma
            z_min, z_max = float(z.min()), float(z.max())
            di_norm = (z - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z, dtype=float)
        else:
            raise ValueError(f"Unknown fusion_norm: {fusion_norm}")

        ch_scores.append(di_norm.astype(float, copy=False))

    # (N_test, #channels)
    S = np.stack(ch_scores, axis=1)
    scores_mean = S.mean(axis=1)

    # AUROC
    auroc = float(roc_auc_score(y_test, scores_mean)) if len(np.unique(y_test)) == 2 else float("nan")

    metrics = {
        "AUROC": auroc,
        "N_train_norm": int(len(tri_norm)),
        "N_test": int(len(tei_all)),
        "N_test_pos": int(int(np.sum(y_test))),
        "fusion_norm": str(fusion_norm),
        "metric": str(metric),
        "standardize": bool(standardize),
        "knn_k": int(k),
        "num_channels_used": int(len(X_blocks)),
    }

    # dump per-channel scores
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        per_ch_path = dump_dir / 'per_channel_scores.csv'
        with open(per_ch_path, 'w', newline='') as f:
            w = csv.writer(f)
            header = ['id', 'y_true'] + [f'ch{j+1}' for j in range(S.shape[1])] + ['score_mean']
            w.writerow(header)
            for sid, y, row, m in zip(ids[tei_all], y_test, S, scores_mean):
                w.writerow([sid, int(y), *[float(v) for v in row.tolist()], float(m)])

    return scores_mean, y_test, tei_all, int(len(tri_norm)), metrics


# ----------------- threshold & confusion matrix -----------------
def pick_threshold(scores: np.ndarray, y_true: np.ndarray, mode: str,
                   fpr_target: float, perc: float, fixed_score: Optional[float]):
    aux = {}
    if mode in ('youden', 'fpr_at'):
        fpr, tpr, thr = roc_curve(y_true, scores, drop_intermediate=True)
        aux = {"roc_points": int(len(thr))}
        if mode == 'youden':
            j = tpr - fpr
            k = int(np.argmax(j))
            return float(thr[k]), {**aux, "youden_index": float(j[k])}
        else:
            ok = np.where(fpr <= float(fpr_target))[0]
            if len(ok) == 0:
                k = int(np.argmin(fpr))
            else:
                k = int(ok[-1])
            return float(thr[k]), {**aux, "achieved_fpr": float(fpr[k]), "achieved_tpr": float(tpr[k])}

    if mode == 'percentile':
        q = float(np.clip(perc, 1e-6, 1 - 1e-6))
        thr = float(np.quantile(scores, q))
        return thr, {"percentile": q}

    if mode == 'score':
        if fixed_score is None:
            raise ValueError("--cm_score must be provided when cm_mode=score")
        return float(fixed_score), {}

    raise ValueError(f"Unknown cm_mode: {mode}")


def apply_threshold_and_dump(view_dir: Path, view_name: str,
                             scores: np.ndarray, y_true: np.ndarray,
                             ids_test: np.ndarray,
                             cm_mode: str, cm_fpr: float, cm_percentile: float, cm_score: Optional[float]):
    thr, aux = pick_threshold(scores, y_true, cm_mode, cm_fpr, cm_percentile, cm_score)
    y_pred = (scores >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    cm_path = view_dir / f"cm_{cm_mode}.csv"
    with open(cm_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['', 'Pred_0', 'Pred_1'])
        w.writerow(['True_0', int(tn), int(fp)])
        w.writerow(['True_1', int(fn), int(tp)])

    thr_path = view_dir / f"threshold_{cm_mode}.json"
    thr_meta = {
        "mode": cm_mode,
        "threshold": float(thr),
        "ACC": acc,
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        **aux
    }
    thr_path.write_text(json.dumps(thr_meta, indent=2, ensure_ascii=False))

    fpr, tpr, thr_all = roc_curve(y_true, scores, drop_intermediate=True)
    with open(view_dir / 'roc_curve.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['FPR', 'TPR', 'THR'])
        for a, b, c in zip(fpr, tpr, thr_all):
            w.writerow([float(a), float(b), float(c)])

    print(f"[{view_name}] CM saved -> {cm_path.name}  thr={thr:.6f}  ACC={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    return thr_meta


# ----------------- main -----------------
def main():
    import sys, time

    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--outdir', required=True)

    # 切分方式（二选一）
    ap.add_argument('--train_ids', type=str, default=None)
    ap.add_argument('--test_ids',  type=str, default=None)
    ap.add_argument('--splits_dir', type=str, default=None, help='指向 8+8 的 splits 目录')

    # 视角
    ap.add_argument('--modal_modes', type=str, default='vibrationmean,audiomean,fusedmean',
                    help='逗号分隔：audiomean, vibrationmean, fusedmean（大小写/别名均可）')

    # kNN
    ap.add_argument('--knn_k', type=int, default=11)
    ap.add_argument('--metric', type=str, default="l2", help="l2 or cosine")

    # 通道融合归一化
    ap.add_argument('--fusion_norm', choices=['quantile', 'zscore'], default='quantile',
                    help='quantile=ECDF分位数；zscore=标准分后再min-max到[0,1]')

    ap.add_argument('--standardize', type=str, default="True", help="True/False (per-channel scaler fit on train normal)")

    # per_channel 模式
    ap.add_argument('--per_channel', action='store_true',
                    help='Use per-channel features from npz (per_channel/X_ch*). Allows different D per channel.')

    # 混淆矩阵（可选）
    ap.add_argument('--do_cm', action='store_true', help='启用阈值→二分类→混淆矩阵/ACC等')
    ap.add_argument('--cm_mode', choices=['youden', 'fpr_at', 'percentile', 'score'], default='youden')
    ap.add_argument('--cm_fpr', type=float, default=0.1, help='cm_mode=fpr_at 时的目标 FPR')
    ap.add_argument('--cm_percentile', type=float, default=0.5, help='cm_mode=percentile 时的分位点(0~1)')
    ap.add_argument('--cm_score', type=float, default=None, help='cm_mode=score 时的固定分数阈值')

    args = ap.parse_args()

    standardize = str(args.standardize).lower() == "true"

    print(f"[info] script={sys.argv[0]}  time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load npz
    data = np.load(args.npz, allow_pickle=True)

    # load ids + 7 blocks
    ids, blocks = load_7ch_blocks_from_npz(data, per_channel=bool(args.per_channel))
    print("[info] loaded ids =", len(ids))
    print("[info] channel dims =", [int(b.shape[1]) for b in blocks])

    # normal/abn labels
    if 'splits' in data:
        splits = data['splits'].astype(str)
        is_normal_all = (np.char.lower(splits) == 'normal')
    elif 'labels' in data:
        labels = data['labels']
        is_normal_all = (labels == 0)
    else:
        raise RuntimeError("npz must contain either 'splits' (normal/abn) or 'labels' (0/1).")

    if len(is_normal_all) != len(ids):
        raise RuntimeError(f"labels length mismatch: {len(is_normal_all)} vs ids {len(ids)}")

    # read splits
    if args.splits_dir:
        tr_ids, te_ids = load_from_8plus8_splits_dir(Path(args.splits_dir))
    else:
        if not (args.train_ids and args.test_ids):
            raise RuntimeError("need either --splits_dir or (--train_ids and --test_ids).")
        tr_ids = load_id_list(args.train_ids)
        te_ids = load_id_list(args.test_ids)
        if tr_ids is None or te_ids is None or len(tr_ids) == 0 or len(te_ids) == 0:
            raise RuntimeError("train_ids/test_ids empty.")

    # modes
    modes = [m.strip().lower() for m in args.modal_modes.split(',') if m.strip()]
    rows = []

    for m in modes:
        if m in ('audiomean', 'audio_mean', 'audio'):
            use_blocks = blocks[:3]
            view_name = 'audiomean'
        elif m in ('vibrationmean', 'vib_mean', 'vibration', 'vib'):
            use_blocks = blocks[3:]
            view_name = 'vibrationmean'
        elif m in ('fusedmean', 'fused_mean', 'fused'):
            use_blocks = blocks
            view_name = 'fusedmean'
        else:
            raise ValueError(f"Unknown modal mode: {m}")

        vdir = outdir / f"view_{view_name}"

        scores, y_test, tei_all, ntr, metrics = mean_score_over_channels(
            use_blocks, ids, is_normal_all, tr_ids, te_ids,
            k=args.knn_k, dump_dir=vdir, fusion_norm=args.fusion_norm,
            metric=args.metric, standardize=standardize
        )

        # save test_scores.csv
        vdir.mkdir(parents=True, exist_ok=True)
        with open(vdir / 'test_scores.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'score', 'y_true'])
            for sid, s, y in zip(ids[tei_all], scores, y_test):
                w.writerow([sid, float(s), int(y)])

        # optional cm
        if args.do_cm and len(np.unique(y_test)) == 2:
            thr_meta = apply_threshold_and_dump(
                vdir, view_name, scores, y_test, ids[tei_all],
                cm_mode=args.cm_mode, cm_fpr=args.cm_fpr,
                cm_percentile=args.cm_percentile, cm_score=args.cm_score
            )
            metrics.update({f"CM_{k}": v for k, v in thr_meta.items()})

        (vdir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

        rows.append({
            "view": view_name,
            "AUROC": metrics["AUROC"],
            "N_train_norm": metrics["N_train_norm"],
            "N_test": metrics["N_test"],
            "N_test_pos": metrics["N_test_pos"],
            "metric": metrics["metric"],
            "fusion_norm": metrics["fusion_norm"],
            "standardize": metrics["standardize"],
            "knn_k": metrics["knn_k"],
            "num_channels_used": metrics["num_channels_used"],
            "per_channel": bool(args.per_channel),
        })

        print(f"[{view_name}] AUROC={metrics['AUROC']:.6f}  N_train_norm={metrics['N_train_norm']}  "
              f"N_test={metrics['N_test']} (pos={metrics['N_test_pos']})")

    # compare table
    cmp_path = outdir / 'metrics_compare_views.csv'
    with open(cmp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] compare table -> {cmp_path}")


if __name__ == "__main__":
    main()


