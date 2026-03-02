# SSCC Benchmark  
Multimodal Learning with Channel-wise kNN Baselines

This repository provides a benchmark suite for the **SSCC dataset**, focusing on multimodal representation learning and channel-wise kNN baselines.

Detailed information about the SSCC dataset can be found at:  
https://anonymous2417.github.io/SSCC-Dataset/

---

# Overview

This benchmark supports two downstream tasks:

1. Fault Detection  
2. Fault Classification  

Both tasks require feature extraction before running the kNN baselines.

Feature extraction follows the workflow defined in **SIREN**:  
https://github.com/yucongzh/SIREN.git

---

# 1. Feature Extraction

Before running any downstream task, features must be extracted.

## 1.1 Prepare CSV Index

SIREN requires a CSV file listing all data paths. First generate the CSV:

```bash
python make_csv.py \
  --root <your_dataset_root> \
  --csv <your_output_csv_path> \
  --vib_sr 100000

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

Example (BEATs)
```bash
python extract_features.py \
  --index ./SSCC_dataset/index.csv \
  --extractor_py ./extractors/BEATs_extractor.py \
  --out_npz ./features/BEATs_7ch.npz \
  --multi_channel_strategy concatenate \
  --cache_dir ./feature_cache/BEATs

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
