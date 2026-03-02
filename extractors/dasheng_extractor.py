# dasheng_extractor.py
# -*- coding: utf-8 -*-
"""
Dasheng Multichannel Feature Extractor
--------------------------------------

This module implements multichannel feature extraction using Dasheng models.

Supported inputs:
  - Audio: .wav, .flac, .m4a, .mp3, .mp4
  - Vibration: .txt, .csv, .tsv
      - Header lines such as "timestamp_s,..." are supported
      - Column selection is supported via:
            file.csv::col=K
        where K is 1-based index over the value columns (timestamp column is ignored).

Processing:
  1) Each channel is processed independently:
       waveform -> resample to 16 kHz -> segment into 10-second windows
       -> Dasheng forward -> temporal mean pooling
  2) For long recordings:
       segment embeddings are averaged
  3) Across channels:
       embeddings are aggregated using:
         "concatenate" | "average" | "first" | "last"

Output:
  - A single 1D float32 numpy feature vector per sample.

Requirements:
  - PyTorch
  - torchaudio
  - Dasheng (installed via pip) OR available as a local source tree
  - SIREN BaseFeatureExtractor

Notes:
  - This extractor is configured to run on GPU only.
  - CPU is used only when returning the final numpy array (for downstream npz writing).
"""

import os
import sys
import json
import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio

from siren.core.base_extractor import BaseFeatureExtractor


# ---------------------------------------------------------
# Dasheng import (pip first, then optional local fallback)
# ---------------------------------------------------------
# If pip import fails, set:
#   export DASHENG_LOCAL_SRC=/path/to/local/dasheng/source_root
# so that "<source_root>" contains a "Dasheng" package folder (or equivalent).
try:
    from dasheng import dasheng_base, dasheng_06B, dasheng_12B
except Exception as e_pip:
    local_src = os.environ.get("DASHENG_LOCAL_SRC", "").strip()
    if local_src:
        sys.path.append(local_src)
    try:
        # Common local layout: Dasheng/dasheng.py
        from Dasheng.dasheng import dasheng_base, dasheng_06B, dasheng_12B  # type: ignore
    except Exception as e_local:
        raise ImportError(
            "[dasheng_extractor] Failed to import Dasheng model.\n"
            f"- pip import error: {e_pip}\n"
            f"- local import error: {e_local}\n"
            "Please install Dasheng via `pip install dasheng`, or set:\n"
            "  export DASHENG_LOCAL_SRC=/path/to/local/dasheng/source_root\n"
            "so that Dasheng can be imported from the local source tree."
        )


# ---------------------------------------------------------
# Helpers: robust path parsing + vibration reading
# ---------------------------------------------------------
_COL_RE = re.compile(r"^(.*?)(?:::col=(\d+))?$", re.IGNORECASE)


def _clean_path(p: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    p = p.replace("\ufeff", "").replace("\u200b", "")
    p = re.sub(r"[：:；;，,\s]+$", "", p)
    return p


def _resolve_paths_field(paths_field: Any) -> List[str]:
    """
    The `paths` field in index.csv may be:
      - list[str]
      - JSON string that encodes a list[str]
      - other string (fallback: treat as a single path)
    Returns a clean list[str].
    """
    if isinstance(paths_field, list):
        return [_clean_path(x) for x in paths_field]
    if paths_field is None:
        return []
    s = str(paths_field).strip()
    if not s:
        return []
    if isinstance(paths_field, str):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [_clean_path(x) for x in obj]
        except Exception:
            pass
    return [_clean_path(s)]


def _read_vibration_text(path_with_opt: str) -> np.ndarray:
    """
    Robust vibration reader:
      - supports file.csv::col=K (1-based over value columns)
      - supports a header line such as "timestamp_s,..."
      - supports delimiter:
          .csv -> comma
          .tsv -> tab
          .txt -> whitespace by default, with fallback to comma

    File convention:
      - columns: [timestamp, ch1, ch2, ...]
      - ::col=1 selects ch1 (i.e., first value column)

    Returns:
      float32 vector of shape (T,)
    """
    m = _COL_RE.match(path_with_opt.strip())
    real_path = m.group(1) if m else path_with_opt
    col = int(m.group(2)) if (m and m.group(2) is not None) else None
    real_path = _clean_path(real_path)

    ext = os.path.splitext(real_path)[1].lower()

    # Detect header (e.g., "timestamp_s")
    with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    has_header = any(ch.isalpha() for ch in first)

    # Choose delimiter
    if ext == ".tsv":
        delimiter = "\t"
    elif ext == ".csv":
        delimiter = ","
    else:
        delimiter = None  # whitespace split by default

    def _try_genfromtxt(delim):
        return np.genfromtxt(
            real_path,
            delimiter=delim,
            skip_header=1 if has_header else 0,
            dtype=np.float32,
            invalid_raise=False,
        )

    arr = _try_genfromtxt(delimiter)

    # If txt but actually comma-separated (or other corner cases), retry
    if (arr is None) or (not np.isfinite(np.asarray(arr)).any()):
        if delimiter is None:
            arr = _try_genfromtxt(",")
        else:
            arr = _try_genfromtxt(None)

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        raise ValueError(
            f"{real_path} contains only one column. "
            "Expected at least two columns: timestamp + at least one channel."
        )
    if arr.shape[1] < 2:
        raise ValueError(
            f"{real_path} must contain at least two columns "
            "(timestamp + at least one channel)."
        )

    chmat = arr[:, 1:]  # drop timestamp
    if col is None:
        if chmat.shape[1] == 1:
            sig = chmat[:, 0]
        else:
            raise ValueError(
                f"{real_path} contains {chmat.shape[1]} channels. "
                f"Please specify the desired column using '::col=1..{chmat.shape[1]}'."
            )
    else:
        if not (1 <= col <= chmat.shape[1]):
            raise ValueError(
                f"{real_path} specifies col={col}, but valid range is 1..{chmat.shape[1]}."
            )
        sig = chmat[:, col - 1]

    sig = np.asarray(sig, dtype=np.float32)
    sig = sig[np.isfinite(sig)]
    if sig.size == 0:
        raise ValueError(f"{real_path} vibration signal is empty or contains only NaN values.")
    return sig


# ---------------------------------------------------------
# Main extractor
# ---------------------------------------------------------
class FeatureExtractor(BaseFeatureExtractor):
    """
    Dasheng multichannel feature extractor.

    Args:
      multi_channel_strategy:
        "concatenate" | "average" | "first" | "last"
      model_size:
        "dasheng_base" | "dasheng_06B" | "dasheng_12B"
      expect_channels:
        number of channels (e.g., 7)
      vib_default_sr:
        fallback vibration sampling rate if sample_meta does not provide vibration_sr
      dtype:
        optional precision: "fp16" or "bf16" (to reduce GPU memory)
    """

    def __init__(
        self,
        multi_channel_strategy: str = "concatenate",
        model_size: str = "dasheng_06B",
        expect_channels: int = 7,
        vib_default_sr: Optional[int] = None,
        dtype: Optional[str] = None,
    ):
        super().__init__(multi_channel_strategy)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise RuntimeError(
                "[dasheng_extractor] CUDA is not available. "
                "This extractor is configured to run on GPU only."
            )

        self.expected_channels = int(expect_channels)
        self.target_sample_rate = 16000
        self.max_samples = 160000  # 10 seconds @ 16 kHz
        self.vib_default_sr = vib_default_sr

        # Feature dims per model variant (adjust if your Dasheng implementation differs)
        self._feature_dims: Dict[str, int] = {
            "dasheng_base": 768,
            "dasheng_06B": 1280,
            "dasheng_12B": 1536,
        }
        if model_size not in self._feature_dims:
            raise ValueError(f"Unknown model_size={model_size}, allowed={list(self._feature_dims.keys())}")
        self.model_size = model_size

        self.model = self._load_dasheng_model(model_size)

        # Optional precision
        if dtype:
            d = dtype.lower()
            if d == "fp16":
                self.model = self.model.half()
            elif d == "bf16":
                self.model = self.model.to(dtype=torch.bfloat16)
            else:
                raise ValueError("dtype only supports: fp16 | bf16")

        self.model.to(self.device).eval()
        print(
            f"[dasheng_extractor] loaded {model_size} on {self.device} "
            f"(expected_channels={self.expected_channels}, strategy={self.multi_channel_strategy})"
        )

    def _get_single_channel_feature_dim(self) -> int:
        return self._feature_dims[self.model_size]

    @property
    def feature_dim(self) -> int:
        d = self._get_single_channel_feature_dim()
        if self.multi_channel_strategy == "concatenate":
            return d * self.expected_channels
        return d

    # ---------------- core API ----------------
    def extract_features(self, sample_meta: Dict[str, Any]) -> np.ndarray:
        """
        Returns a 1D float32 numpy vector.
        Downstream scripts typically call np.asarray(out) and write to npz.
        """
        ch_paths = self._resolve_channel_paths(sample_meta)
        if len(ch_paths) != self.expected_channels:
            raise RuntimeError(f"Channel mismatch: got {len(ch_paths)}, expected {self.expected_channels}")

        feats: List[torch.Tensor] = []
        for p in ch_paths:
            sig, sr = self._load_any_signal(p, sample_meta)     # [1, T] CPU tensor
            f = self._extract_single_channel_features(sig, sr)  # [D] GPU tensor
            feats.append(f)

        feats_t = torch.stack(feats, dim=0)  # [C, D] on GPU

        if self.multi_channel_strategy == "concatenate":
            out = feats_t.reshape(-1)
        elif self.multi_channel_strategy == "average":
            out = feats_t.mean(dim=0)
        elif self.multi_channel_strategy == "first":
            out = feats_t[0]
        elif self.multi_channel_strategy == "last":
            out = feats_t[-1]
        else:
            raise ValueError(f"Unknown multi_channel_strategy: {self.multi_channel_strategy}")

        # Only once: move GPU -> CPU -> numpy
        return out.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def extract_features_from_signal(self, signal: torch.Tensor, sample_rate: int) -> np.ndarray:
        out = self._extract_single_channel_features(signal, sample_rate)
        return out.detach().cpu().numpy().astype(np.float32).reshape(-1)

    # ---------------- single-channel forward ----------------
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Input:
          signal_tensor: [1, T] CPU tensor
        Output:
          feature: [D] GPU tensor
        """
        x = signal_tensor.to(torch.float32)
        x = x - x.mean()

        if int(sample_rate) != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(int(sample_rate), self.target_sample_rate)
            x = resampler(x)

        # Segment into <= 10s windows
        T = x.shape[-1]
        segs: List[torch.Tensor] = []
        if T <= self.max_samples:
            segs.append(x)
        else:
            n_full = T // self.max_samples
            for i in range(n_full):
                beg = i * self.max_samples
                end = beg + self.max_samples
                segs.append(x[..., beg:end])
            if n_full * self.max_samples < T:
                segs.append(x[..., -self.max_samples:])

        outs: List[torch.Tensor] = []
        with torch.no_grad():
            for s in segs:
                s = s.to(self.device)  # move to GPU once per segment

                # Dasheng forward: try extract_features() API first
                try:
                    padding_mask = torch.zeros(s.shape[0], s.shape[1], dtype=torch.bool, device=self.device)
                    y = self.model.extract_features(s, padding_mask=padding_mask)
                    if isinstance(y, (list, tuple)):
                        y = y[0]
                except Exception:
                    y = self.model(s)

                # Pool over time if needed
                if y.dim() == 3:
                    y = y.mean(dim=1)  # [B, D]
                elif y.dim() == 2:
                    pass  # [B, D]
                else:
                    raise RuntimeError(f"Unexpected Dasheng output shape: {tuple(y.shape)}")

                outs.append(y.squeeze(0).detach())  # keep on GPU (no .cpu() here)

        # Mean across segments on GPU
        return torch.stack(outs, dim=0).mean(dim=0)  # [D] on GPU

    # ---------------- data loading ----------------
    def _resolve_channel_paths(self, sample_meta: Dict[str, Any]) -> List[str]:
        if "paths" in sample_meta:
            return _resolve_paths_field(sample_meta["paths"])
        raise ValueError("sample_meta must contain 'paths' (list or JSON string).")

    def _load_any_signal(self, path: str, sample_meta: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """
        Audio:
          .wav/.flac/.m4a/.mp3/.mp4 (mp4 decoding may require ffmpeg depending on backend)
        Vibration:
          .txt/.csv/.tsv (header supported; ::col=K supported)

        Returns:
          waveform [1, T] float32 CPU tensor, sample_rate int
        """
        path = _clean_path(path)

        # Allow ::col=K for vibration
        base_path = path
        m = _COL_RE.match(path)
        if m:
            base_path = _clean_path(m.group(1))

        ext = os.path.splitext(base_path)[1].lower()

        # ---- audio ----
        if ext in [".wav", ".flac", ".m4a", ".mp3", ".mp4"]:
            try:
                wav, sr = torchaudio.load(base_path)
            except Exception as e:
                # Fallback: StreamReader (may require ffmpeg)
                try:
                    streamer = torchaudio.io.StreamReader(base_path)
                    streamer.add_audio_stream(frames_per_chunk=0)
                    chunks = []
                    for (chunk,) in streamer.stream():
                        chunks.append(chunk)
                    if not chunks:
                        raise RuntimeError("No audio chunks decoded via StreamReader")
                    wav = torch.cat(chunks, dim=0).t().unsqueeze(0)  # [1, T]
                    sr = int(streamer.get_src_stream_info(0).sample_rate)
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to read audio: {base_path}\n"
                        f"torchaudio.load error: {e}\n"
                        f"StreamReader error: {e2}"
                    )

            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)
            return wav.to(torch.float32), int(sr)

        # ---- vibration ----
        if ext in [".txt", ".csv", ".tsv"]:
            sig = _read_vibration_text(path)  # uses original path with ::col if present
            wav = torch.from_numpy(sig).view(1, -1).to(torch.float32)

            # Sampling rate priority: meta vibration_sr > vib_default_sr > error
            sr = sample_meta.get("vibration_sr", None)
            if sr is not None and sr != "" and not (isinstance(sr, float) and np.isnan(sr)):
                sr_i = int(sr)
            elif self.vib_default_sr is not None:
                sr_i = int(self.vib_default_sr)
            else:
                raise ValueError(f"Vibration input provided but sampling rate is missing for: {path}")
            return wav, sr_i

        raise ValueError(f"Unsupported file type: {path}")

    # ---------------- model loading ----------------
    def _load_dasheng_model(self, model_size: str):
        if model_size == "dasheng_base":
            return dasheng_base()
        if model_size == "dasheng_06B":
            return dasheng_06B()
        if model_size == "dasheng_12B":
            return dasheng_12B()
        raise ValueError(f"Unsupported model_size: {model_size}")


if __name__ == "__main__":
    ext = FeatureExtractor(
        multi_channel_strategy="concatenate",
        model_size="dasheng_06B",
        expect_channels=7,
        vib_default_sr=100000,
        dtype=None,
    )
    print("feature_dim =", ext.feature_dim)
    print("Extractor ready.")