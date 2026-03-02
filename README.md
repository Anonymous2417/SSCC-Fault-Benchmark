# Multimodal Single-Speed Chain Conveyer (SSCC) Dataset Benchmark  

This repository provides a benchmark suite for the **SSCC dataset**, focusing on multimodal representation learning and channel-wise kNN baselines.

Detailed information about the SSCC dataset can be found at:  
https://anonymous2417.github.io/SSCC-Dataset/

# Overview

This benchmark supports two downstream tasks:

1. Fault Detection  
2. Fault Classification  

Both tasks require feature extraction before running the kNN baselines.

Feature extraction follows the workflow defined in **SIREN**:  
https://github.com/yucongzh/SIREN.git

# 1. Feature Extraction

Before running any downstream task, features must be extracted.

## 1.1 Prepare CSV Index

SIREN requires a CSV file listing all data paths. First generate the CSV:

```bash
python make_csv.py \
  --root <your_dataset_root> \
  --csv <your_output_csv_path> \
  --vib_sr 100000
```
## 1.2 Extract Features

Supported encoders (located in `extractors/`):

- BEATs  
- CED-small  
- Dasheng  
- EAT-base  
- ECHO  
- FISHER  

All extractors follow the same interface and are compatible with the SIREN feature extraction pipeline.

### General Command Format

```bash
python extract_features.py \
  --index <your_index_csv_path> \
  --extractor_py <path_to_extractor.py> \
  --out_npz <your_output_feature_path>.npz \
  --multi_channel_strategy concatenate \
  --cache_dir <your_cache_directory>
```
Example (BEATs)
```bash
python extract_features.py \
  --index ./SSCC_dataset/index.csv \
  --extractor_py ./extractors/BEATs_extractor.py \
  --out_npz ./features/BEATs_7ch.npz \
  --multi_channel_strategy concatenate \
  --cache_dir ./feature_cache/BEATs
```
### Output

The extraction step produces a `.npz` file containing:

- `X` — shape `(N, 7D)`, concatenated 7-channel features  
- `ids` — shape `(N,)`, corresponding file identifiers  

For encoders such as **ECHO** and **FISHER**, features may be stored in **per-channel format** instead of concatenated format.

In that case:

- The `.npz` file contains `per_channel` instead of `X`
- Each sample stores 7 independent feature vectors (one per channel)
- Downstream classification must be run with the `--per_channel` flag

After feature extraction is completed, you can proceed to the downstream benchmarks.

# 2. Fault Detection

After feature extraction is completed, you can proceed to the downstream tasks.

This section describes how to run **Fault Detection** using the channel-wise kNN baseline.

Fault Detection evaluates anomaly detection performance under leave-velocity and condition-controlled splits.

## 2.1 Generate Detection Splits

First, create the data partition using `make_splits_fd.py`.

```bash
python make_splits_fd.py \
  --csv <your_index_csv_path> \
  --outdir <your_output_split_dir> \
  --leave_vel 100 \
  --train_leave_combos <your_desired_condition> \
  --seed 0
```
## 2.2 Run Channel-wise kNN Fault Detection

After generating splits, run the detection baseline.
```bash
python fd_knn.py \
  --npz <your_feature_npz_path> \
  --outdir <your_output_dir> \
  --train_ids <path_to_train.ids> \
  --test_ids <path_to_test.ids>
```

# 3. Fault Classification
Fault Classification serves as a classical supervised task. The pipeline is similar to fault detection task.

## 3.1 Generate Classification Splits

First, create the data partition using `make_splits_fc.py`.
```bash
python make_splits_fc.py \
--csv /path/to/your_dataset/index.csv \
--outdir /path/to/your_dataset/fc_splits/leaveXX_condition \
--leave_vel 80 \
--seed 0 \
--hard_labels dry,lean \
--hard_vels 80,100
```
## 3.2 Run Channel-wise kNN Fault Classification

After data partitoning, run the classification baseline.
```bash
python fc_knn.py \
--npz /path/to/features_7ch_concat.npz \
--train_ids /path/to/splits/train.ids \
--test_ids  /path/to/splits/test.ids \
--outdir /path/to/output_dir \
--knn_k 11 \
--vote weighted \
--metric l2 \
--views audio_mean,vib_mean,fused_mean \
--standardize True \
```
