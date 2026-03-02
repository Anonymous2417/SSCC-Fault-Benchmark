# beats_extractor.py
# -*- coding: utf-8 -*-
"""
BEATs Multichannel Feature Extractor (Audio + Vibration)
-------------------------------------------------------

This extractor treats three audio recordings and four vibration channels as
seven parallel input channels:

  - recorder audio (wav)
  - iOS audio (mp4)
  - Android audio (mp4)
  - vibration channel 1..4 (loaded from a CSV/TXT/TSV file)

Key behaviors:
  - BEATs expects 16 kHz audio. All inputs are resampled to 16 kHz.
  - Each channel is segmented into 10-second windows (160000 samples @ 16 kHz).
  - For each window: BEATs -> temporal mean pooling.
  - For a file with multiple windows: features are averaged across windows.
  - Multichannel aggregation is controlled by multi_channel_strategy:
      "concatenate" | "average" | "first" | "last"

Vibration file format:
  - A vibration file is expected to contain:
      timestamp, ch1, ch2, ...
  - The first column is treated as timestamp and ignored.
  - Channel selection can be specified by suffix:
      file.csv::col=1   or  file.csv|col=2  or  file.csv#col=3

Dependencies:
  torch, torchaudio, numpy, SIREN (BaseFeatureExtractor)

Notes:
  - Decoding mp4/m4a may require ffmpeg depending on the torchaudio backend.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple

import torch
import torchaudio
import numpy as np

from siren.core.base_extractor import BaseFeatureExtractor

# Add SIREN model directory (BEATs)
sys.path.append("/DATA1/chenzhang/SIREN/models")
from BEATs.BEATs import BEATs, BEATsConfig


class FeatureExtractor(BaseFeatureExtractor):
    """
    BEATs multichannel feature extractor (audio + vibration as high-SR waveforms).

    Default channel order (expected_channels=7):
      [recorder, ios, android, vib_ch1, vib_ch2, vib_ch3, vib_ch4]
    """

    def __init__(
        self,
        multi_channel_strategy: str = "concatenate",
        beats_ckpt: str = "/DATA1/chenzhang/SIREN/models/BEATs/checkpoints/BEATs_iter3.pt",
        expect_channels: int = 7,
        vib_default_sr: Optional[int] = 100000,
    ):
        super().__init__(multi_channel_strategy)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.expected_channels = expect_channels
        self.target_sample_rate = 16000
        self.max_samples = 160000  # 10s @ 16k
        self.vib_default_sr = vib_default_sr

        if not os.path.exists(beats_ckpt):
            raise FileNotFoundError(f"BEATs checkpoint not found: {beats_ckpt}")

        ckpt = torch.load(beats_ckpt, map_location=self.device)
        cfg = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        print(f"[beats_extractor] loaded: {beats_ckpt} on {self.device}")
        print(f"[beats_extractor] target_sr={self.target_sample_rate}, vib_default_sr={self.vib_default_sr}")
        print(f"[beats_extractor] expected_channels={self.expected_channels}, strategy={self.multi_channel_strategy}")

    @property
    def feature_dim(self) -> int:
        d = self._get_single_channel_feature_dim()
        if self.multi_channel_strategy == "concatenate":
            return d * self.expected_channels
        if self.multi_channel_strategy in ("average", "first", "last"):
            return d
        raise ValueError(f"Unknown multi_channel_strategy: {self.multi_channel_strategy}")

    def extract_features(self, sample_meta: Dict[str, Any]) -> torch.Tensor:
        """
        sample_meta["paths"]: list[str] or JSON string
          Expected length: self.expected_channels
        """
        ch_paths = self._resolve_channel_paths(sample_meta)
        if len(ch_paths) != self.expected_channels:
            raise RuntimeError(f"Channel mismatch: got {len(ch_paths)}, expected {self.expected_channels}")

        feats = []
        for idx, p in enumerate(ch_paths):
            sig, sr = self._load_any_signal(p, sample_meta)
            f = self._extract_single_channel_features(sig, sr)  # [768]
            if torch.isnan(f).any():
                print(f"[WARN] NaN in BEATs features for channel {idx}, path={p}")
                f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(f)

        feats = torch.stack(feats, dim=0)  # [C, 768]

        if self.multi_channel_strategy == "concatenate":
            return feats.reshape(-1)
        if self.multi_channel_strategy == "average":
            return feats.mean(dim=0)
        if self.multi_channel_strategy == "first":
            return feats[0]
        if self.multi_channel_strategy == "last":
            return feats[-1]
        raise ValueError(self.multi_channel_strategy)

    def extract_features_from_signal(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self._extract_single_channel_features(signal, sample_rate)

    # ---------------- Path resolution ----------------
    def _resolve_channel_paths(self, sample_meta: Dict[str, Any]) -> List[str]:
        """
        Contract:
          sample_meta["paths"] is either a JSON list string or a Python list[str].
        """
        if "paths" not in sample_meta:
            raise ValueError("sample_meta must contain 'paths'")
        paths = sample_meta["paths"]
        if isinstance(paths, str):
            paths = json.loads(paths)
        return list(paths)

    # ---------------- Unified signal loading (audio + vibration) ----------------
    def _load_any_signal(self, path: str, sample_meta: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """
        Supported inputs:
          - Audio: .wav/.flac/.m4a/.mp3/.mp4
          - Vibration: .txt/.csv/.tsv with (timestamp + N channels)

        Vibration channel selection suffix (1-based):
          xxx.csv::col=1   or xxx.csv|col=2   or xxx.csv#col=3

        Returns:
          waveform: [1, T] float32
          sample_rate: int
        """
        import re

        orig_path = path
        path = str(path).strip().strip('"').strip("'").replace("\ufeff", "").replace("\u200b", "")
        path = re.sub(r"[：:；;，,\s]+$", "", path)

        # Parse channel marker
        col = None
        m = re.search(r"(?::col=|\|col=|#col=)(\d+)$", path)
        if m:
            col = int(m.group(1))
            path = re.sub(r"(?::col=|\|col=|#col=)\d+$", "", path)
            path = re.sub(r"[：:；;，,\s]+$", "", path)

        path = re.sub(r"(\.(?:txt|csv|tsv))\s*[:：；;，,]+$", r"\1", path, flags=re.IGNORECASE)

        ext = os.path.splitext(path)[1].lower()

        # ===== Audio =====
        if ext in [".wav", ".flac", ".m4a", ".mp3", ".mp4"]:
            try:
                wav, sr = torchaudio.load(path)
            except Exception as e:
                # Fallback: StreamReader decode (may require ffmpeg)
                try:
                    streamer = torchaudio.io.StreamReader(path)
                    streamer.add_audio_stream(frames_per_chunk=0)
                    chunks = []
                    for (chunk,) in streamer.stream():
                        chunks.append(chunk)
                    if not chunks:
                        raise RuntimeError("No audio chunks decoded via StreamReader")
                    wav = torch.cat(chunks, dim=0).t().unsqueeze(0)  # [1, T]
                    sr = int(streamer.get_src_stream_info(0).sample_rate)
                except Exception as e2:
                    raise RuntimeError(f"Failed to read audio: {path}\nload_err={e}\nstreamreader_err={e2}")

            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)

            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            return wav.to(torch.float32), int(sr)

        # ===== Vibration (timestamp + channels) =====
        if ext in [".txt", ".csv", ".tsv"]:
            try:
                arr = np.genfromtxt(
                    path,
                    dtype=np.float32,
                    delimiter=",",
                    comments="#",
                    skip_header=1,  # skip header line, e.g., "timestamp_s, ch1, ch2, ..."
                )
            except Exception as e:
                raise RuntimeError(f"Failed to read vibration file: {path}\n{e}")

            if arr.ndim == 1 or arr.shape[1] < 2:
                raise ValueError(f"{path} must contain at least 2 columns: timestamp + >=1 channel")

            chmat = arr[:, 1:]  # drop timestamp

            if col is None:
                if chmat.shape[1] == 1:
                    sig = chmat[:, 0]
                else:
                    raise ValueError(
                        f"{path} contains {chmat.shape[1]} channels. "
                        f"Append '::col=1..{chmat.shape[1]}' to select one channel."
                    )
            else:
                if not (1 <= col <= chmat.shape[1]):
                    raise ValueError(f"{path} col={col} out of range (available={chmat.shape[1]})")
                sig = chmat[:, col - 1]

            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            wav = torch.from_numpy(sig).view(1, -1)

            # Vibration sampling rate: meta > default > error
            if "vibration_sr" in sample_meta and sample_meta["vibration_sr"] is not None:
                sr = int(sample_meta["vibration_sr"])
            elif self.vib_default_sr is not None:
                sr = int(self.vib_default_sr)
            else:
                raise ValueError(f"{orig_path} missing vibration_sr and vib_default_sr is not set")

            return wav.to(torch.float32), sr

        raise ValueError(f"Unsupported file type: {orig_path}")

    # ---------------- BEATs forward pass (single channel) ----------------
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Args:
          signal_tensor: [1, T]
          sample_rate: int

        Steps:
          - DC removal
          - Resample to 16 kHz
          - Chunk into 10s windows
          - For each window: BEATs -> temporal mean pooling
          - Average across windows

        Returns:
          torch.Tensor: [768]
        """
        x = signal_tensor - signal_tensor.mean()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if int(sample_rate) != self.target_sample_rate:
            try:
                resampler = torchaudio.transforms.Resample(int(sample_rate), self.target_sample_rate)
                x = resampler(x)
            except Exception as e:
                print(f"[WARN] Resample failed ({sample_rate} -> {self.target_sample_rate}); using original. err={e}")

        T = x.shape[-1]
        segs: List[torch.Tensor] = []
        if T <= self.max_samples:
            segs.append(x[..., -T:])
        else:
            n_full = T // self.max_samples
            for i in range(n_full):
                beg = i * self.max_samples
                end = beg + self.max_samples
                segs.append(x[..., beg:end])
            if n_full * self.max_samples < T:
                segs.append(x[..., -self.max_samples:])

        outs: List[torch.Tensor] = []
        for s in segs:
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).to(self.device)

            # BEATs uses padding_mask: [B, T]
            padding_mask = torch.zeros(s.shape[0], s.shape[1], dtype=torch.bool, device=self.device)

            with torch.no_grad():
                h, _ = self.model.extract_features(s, padding_mask=padding_mask)
                if h.dim() == 3:
                    h = h.mean(1)  # [B, D]
                elif h.dim() == 2:
                    pass
                else:
                    h = h.reshape(h.shape[0], -1)

            h = h.squeeze(0).detach().cpu()
            h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            outs.append(h)

        feats = torch.stack(outs, dim=0).mean(dim=0)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats

    def _get_single_channel_feature_dim(self) -> int:
        return 768


if __name__ == "__main__":
    ext = FeatureExtractor(multi_channel_strategy="concatenate")
    print("final feature_dim =", ext.feature_dim)
    print("Extractor ready.")