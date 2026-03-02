# -*- coding: utf-8 -*-
"""
Dataset Index Builder (CSV) for Feature Extraction
--------------------------------------------------

This script scans a dataset organized as:

  audio/audio_part_XXXX.wav
  ios/ios_part_XXXX.mp4
  android/android_part_XXXX.mp4
  vibration/vibration_part_XXXX.csv

and generates a CSV index file for downstream feature extraction pipelines.

Output CSV columns:
  - id: unique sample identifier
  - paths: JSON-encoded list of 7 paths:
      [recorder_wav, ios_mp4, android_mp4, vib_ch1, vib_ch2, vib_ch3, vib_ch4]
    where each vibration channel is encoded as:
      "<absolute_path_to_csv>::col=<channel_index>"
    and the first column of the vibration CSV is assumed to be a timestamp.
  - split: "Normal" or "Anomalous"
  - label: 0 (normal) or 1 (anomalous)
  - vibration_sr: vibration sampling rate (Hz), stored as metadata

Notes:
  - The script expects four subfolders under each condition folder:
      audio/, ios/, android/, vibration/
  - Samples are aligned by the shared index XXXX in filenames.
  - Vibration files must contain at least 1 timestamp column + 4 signal columns.
"""

import argparse
import csv
import json
from pathlib import Path


def detect_vibration_channels(csv_path: Path) -> int:
    """
    Detect how many vibration signal columns exist in a vibration CSV file,
    excluding the first timestamp column.

    Returns:
      int: number of signal columns (>= 0). If detection fails, returns 0.
    """
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                # Assume the first column is timestamp
                return max(0, len(parts) - 1)
    except Exception:
        return 0
    return 0


def collect_rows(condition_dir: Path, split_name: str, label: int, vibration_sr: int, id_prefix: str):
    """
    Collect sample rows for one condition directory.

    Args:
      condition_dir: Path to a condition folder containing audio/ ios/ android/ vibration/
      split_name: "Normal" or "Anomalous"
      label: 0 for normal, 1 for anomalous
      vibration_sr: vibration sampling rate (Hz)
      id_prefix: prefix used in the sample id (e.g., "normal/" or "abnormal/dry/")

    Returns:
      list[dict]: list of CSV rows (as dicts)
    """
    audio_dir = condition_dir / "audio"
    ios_dir = condition_dir / "ios"
    android_dir = condition_dir / "android"
    vibration_dir = condition_dir / "vibration"

    required_dirs = [audio_dir, ios_dir, android_dir, vibration_dir]
    if not all(d.is_dir() for d in required_dirs):
        print(f"[WARN] Skipped (missing subfolders): {condition_dir}")
        return []

    audio_files = sorted(audio_dir.glob("audio_part_*.wav"))
    if not audio_files:
        print(f"[WARN] Skipped (no audio_part files found): {condition_dir}")
        return []

    indices = []
    for p in audio_files:
        # audio_part_0001.wav -> idx="0001"
        stem = p.stem
        idx = stem.replace("audio_part_", "")
        if idx.isdigit():
            indices.append(idx)

    rows = []
    for idx in indices:
        recorder_wav = audio_dir / f"audio_part_{idx}.wav"
        ios_mp4 = ios_dir / f"ios_part_{idx}.mp4"
        android_mp4 = android_dir / f"android_part_{idx}.mp4"
        vibration_csv = vibration_dir / f"vibration_part_{idx}.csv"

        if not (recorder_wav.exists() and ios_mp4.exists() and android_mp4.exists() and vibration_csv.exists()):
            print(f"[WARN] Missing file(s): {condition_dir.name} idx={idx}")
            continue

        n_signal_cols = detect_vibration_channels(vibration_csv)
        if n_signal_cols < 4:
            print(f"[WARN] Skipped (vibration has < 4 signal columns): {condition_dir.name} idx={idx}")
            continue

        vib_paths = [f"{vibration_csv.resolve()}::col={ch}" for ch in range(1, 5)]

        paths = [
            str(recorder_wav.resolve()),
            str(ios_mp4.resolve()),
            str(android_mp4.resolve()),
            vib_paths[0],
            vib_paths[1],
            vib_paths[2],
            vib_paths[3],
        ]

        sample_id = f"{id_prefix}{condition_dir.name}_{idx}"

        rows.append(
            {
                "id": sample_id,
                "paths": json.dumps(paths, ensure_ascii=False),
                "split": split_name,
                "label": label,
                "vibration_sr": vibration_sr,
            }
        )

    return rows


def build_index_csv(dataset_root: str, out_csv: str, vibration_sr: int):
    """
    Build the dataset index CSV for feature extraction.

    The dataset root is expected to contain:
      - normal/<condition_dir>/
      - abnormal/<fault_type>/<condition_dir>/

    Args:
      dataset_root: dataset root directory
      out_csv: output CSV path
      vibration_sr: vibration sampling rate (Hz)
    """
    root = Path(dataset_root)
    all_rows = []

    # Normal: one-level hierarchy under normal/
    normal_root = root / "normal"
    if normal_root.is_dir():
        for condition_dir in sorted(normal_root.iterdir()):
            if condition_dir.is_dir():
                all_rows.extend(
                    collect_rows(
                        condition_dir=condition_dir,
                        split_name="Normal",
                        label=0,
                        vibration_sr=vibration_sr,
                        id_prefix="normal/",
                    )
                )

    # Abnormal: two-level hierarchy abnormal/<fault_type>/<condition_dir>/
    abnormal_root = root / "abnormal"
    if abnormal_root.is_dir():
        for fault_type_dir in sorted(abnormal_root.iterdir()):  # e.g., dry/ lean/ loose/ screwdrop
            if not fault_type_dir.is_dir():
                continue
            for condition_dir in sorted(fault_type_dir.iterdir()):
                if condition_dir.is_dir():
                    all_rows.extend(
                        collect_rows(
                            condition_dir=condition_dir,
                            split_name="Anomalous",
                            label=1,
                            vibration_sr=vibration_sr,
                            id_prefix=f"abnormal/{fault_type_dir.name}/",
                        )
                    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "paths", "split", "label", "vibration_sr"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[DONE] Wrote {len(all_rows)} samples to: {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Build an index CSV for feature extraction.")
    ap.add_argument("--root", type=str, required=True, help="Dataset root directory")
    ap.add_argument("--csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--vib_sr", type=int, default=100000, help="Vibration sampling rate (Hz)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index_csv(args.root, args.csv, vibration_sr=args.vib_sr)