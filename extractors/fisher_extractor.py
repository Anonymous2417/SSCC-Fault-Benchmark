# -*- coding: utf-8 -*-
"""
FISHER multi-channel feature extractor (SIREN-compatible)

- Per-channel export: always returns List[Tensor] (one per channel), allowing variable dim across channels.
- No resampling.
- Frequency padding to band_width (right-pad on the frequency axis only).
- Segment-level mean pooling ONLY (no token/time pooling inside model outputs).


"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoModel

from siren.core.base_extractor import BaseFeatureExtractor

LOCAL_DIR = "/DATA1/chenzhang/models/FISHER-mini-0723"


class FeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        multi_channel_strategy: str = "concatenate",  # interface compatibility
        expect_channels: int = 7,
        vib_default_sr: Optional[int] = None,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        seg_frames: int = 1024,
        band_width: int = 100,  # frequency padding target (bins)
        norm_mean: float = 3.017344307886898,
        norm_std: float = 2.1531635155379805,
        local_dir: str = LOCAL_DIR,
        dtype: Optional[str] = None,  # "fp16" / "bf16" / None
    ):
        super().__init__(multi_channel_strategy)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.expected_channels = int(expect_channels)
        self.vib_default_sr = vib_default_sr

        self.frame_length_ms = float(frame_length_ms)
        self.frame_shift_ms = float(frame_shift_ms)
        self.seg_frames = int(seg_frames)
        self.band_width = int(band_width)

        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)

        self._torch_dtype = None
        if dtype:
            d = dtype.lower()
            if d == "fp16":
                self._torch_dtype = torch.float16
            elif d == "bf16":
                self._torch_dtype = torch.bfloat16

        if not os.path.isdir(local_dir):
            raise FileNotFoundError(
                f"[fisher_av_extractor] local_dir not found: {local_dir}\n"
                f"Download the HF repo to this folder first."
            )

        self.model = AutoModel.from_pretrained(
            local_dir, trust_remote_code=True, local_files_only=True
        )
        if self._torch_dtype is not None:
            self.model = self.model.to(self._torch_dtype)
        self.model = self.model.to(self.device).eval()

        # Informational only
        cfg = getattr(self.model, "config", None)
        embed_dim = getattr(cfg, "hidden_size", None) or getattr(cfg, "embed_dim", None)
        self._embed_dim_ref = int(embed_dim) if embed_dim is not None else -1

        print(
            f"[fisher_av_extractor] device={self.device} expected_channels={self.expected_channels} "
            f"seg_frames={self.seg_frames} band_width={self.band_width} (NO resample, PER-CHANNEL)"
        )
        print(f"[fisher_av_extractor] model loaded (local_dir): {local_dir}")

    @property
    def feature_dim(self) -> int:
        # Informational only for per-channel variable-dim mode
        return int(self._embed_dim_ref)

    def _get_single_channel_feature_dim(self) -> int:
        # Required by BaseFeatureExtractor
        return int(self._embed_dim_ref) if self._embed_dim_ref > 0 else 0

    # -----------------------------
    # Main API: always per-channel
    # -----------------------------
    def extract_features(self, sample_meta: Dict[str, Any]):
        """
        Always returns List[Tensor] (one per channel).
        Channel feature lengths may differ.
        """
        paths = self._resolve_channel_paths(sample_meta)
        if len(paths) != self.expected_channels:
            raise RuntimeError(
                f"channels mismatch: got {len(paths)}, expect {self.expected_channels}"
            )

        feats: List[torch.Tensor] = []
        for p in paths:
            sig, sr = self._load_any_signal(p, sample_meta)
            f = self._extract_single_channel_features(sig, sr)
            feats.append(f.reshape(-1).detach().cpu())
        return feats

    def extract_features_from_signal(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # Compatibility: single-channel forward
        return self._extract_single_channel_features(signal, sample_rate)

    # -----------------------------
    # Single-channel feature
    # -----------------------------
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Steps:
        - Ensure [1, T], float32
        - Remove DC
        - Spectrogram (center=False)
        - Log magnitude
        - Normalize: (spec + norm_mean) / (norm_std * 2)
        - Segment along time frames with seg_frames; include last tail segment [-seg_frames:]
        - Frequency pad each segment to band_width (right pad on freq axis) if needed
        - For each segment: forward -> collect model outputs (no token/time pooling here)
        - Segment-level mean pooling ONLY
        """
        x = signal_tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(torch.float32)
        x = x - x.mean()

        sr = int(sample_rate)
        n_fft = max(int(round(self.frame_length_ms * sr / 1000.0)), 16)
        hop = max(int(round(self.frame_shift_ms * sr / 1000.0)), 1)

        stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window_fn=torch.hann_window,
            power=1.0,
            center=False,
            pad=0,
        )

        spec = stft(x)  # [1, F, T]
        spec = torch.log(torch.abs(spec) + 1e-10)
        spec = spec.transpose(-2, -1)  # [1, T, F]
        spec = (spec + self.norm_mean) / (self.norm_std * 2.0)

        T_frames = int(spec.shape[1])
        seg_len = int(self.seg_frames)

        input_specs: List[torch.Tensor] = []
        num_segments = T_frames // seg_len

        for i in range(num_segments):
            cut_spec = spec[:, i * seg_len:(i + 1) * seg_len, :]  # [1, seg_len, F]
            if self.band_width > 0 and cut_spec.shape[-1] < self.band_width:
                cut_spec = F.pad(cut_spec, (0, self.band_width - cut_spec.shape[-1]))
            input_specs.append(cut_spec)

        if num_segments * seg_len < T_frames:
            cut_spec = spec[:, -seg_len:, :]  # [1, seg_len, F] (tail)
            if self.band_width > 0 and cut_spec.shape[-1] < self.band_width:
                cut_spec = F.pad(cut_spec, (0, self.band_width - cut_spec.shape[-1]))
            input_specs.append(cut_spec)

        outs: List[torch.Tensor] = []
        with torch.inference_mode():
            for segment_spec in input_specs:
                inp = segment_spec.unsqueeze(1).to(self.device)  # [1, 1, T, F]
                if self._torch_dtype is not None:
                    inp = inp.to(self._torch_dtype)

                if hasattr(self.model, "extract_features"):
                    y = self.model.extract_features(inp)
                else:
                    y = self.model(inp)

                if not torch.is_tensor(y):
                    if hasattr(y, "last_hidden_state"):
                        y = y.last_hidden_state
                    elif isinstance(y, (tuple, list)) and len(y) and torch.is_tensor(y[0]):
                        y = y[0]
                    else:
                        raise RuntimeError("Unexpected model output type.")

                outs.append(y.detach().cpu())

        if len(outs) == 0:
            return torch.zeros((0,), dtype=torch.float32)

        # Segment-level mean pooling only
        out = torch.stack(outs, dim=0).mean(dim=0)

        # Flatten to 1D feature vector (keeps variable length possible)
        return out.reshape(-1).to(torch.float32)

    # -----------------------------
    # Parse paths from index.csv
    # -----------------------------
    def _resolve_channel_paths(self, sample_meta: Dict[str, Any]) -> List[str]:
        """
        Accept:
        - list[str]
        - JSON string list
        - python-literal list string
        """
        if "paths" not in sample_meta:
            raise ValueError("sample_meta must contain 'paths' (from index.csv).")

        paths = sample_meta["paths"]
        if isinstance(paths, str):
            t = paths.strip()
            try:
                paths = json.loads(t)
            except Exception:
                import ast
                paths = ast.literal_eval(t)

        if isinstance(paths, dict):
            paths = list(paths.values())
        if not isinstance(paths, list):
            paths = [paths]
        return [str(p) for p in paths]

    # -----------------------------
    # Unified loader for audio/vibration
    # -----------------------------
    def _load_any_signal(self, path: str, sample_meta: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """
        Audio: .wav/.flac/.m4a/.mp3/.mp4 via torchaudio (fallback StreamReader for mp4)
        Vibration: .csv/.tsv/.txt with header row: timestamp, ch1, ch2, ...
        Select channel via ::col=K (1-based within channels, excluding timestamp)
        """
        orig = path
        path = str(path).strip().strip('"').strip("'").replace("\ufeff", "").replace("\u200b", "")
        path = re.sub(r"[：:；;，,\s]+$", "", path)

        col = None
        m = re.search(r"(?::col=|\|col=|#col=)(\d+)$", path)
        if m:
            col = int(m.group(1))
            path = re.sub(r"(?::col=|\|col=|#col=)\d+$", "", path)
            path = re.sub(r"[：:；;，,\s]+$", "", path)

        ext = os.path.splitext(path)[1].lower()

        # Audio
        if ext in [".wav", ".flac", ".m4a", ".mp3", ".mp4"]:
            try:
                wav, sr = torchaudio.load(path)
            except Exception as e:
                try:
                    streamer = torchaudio.io.StreamReader(path)
                    streamer.add_audio_stream(frames_per_chunk=0)
                    chunks = []
                    for (chunk,) in streamer.stream():
                        chunks.append(chunk)
                    if not chunks:
                        raise RuntimeError("No audio chunks decoded via StreamReader.")
                    wav = torch.cat(chunks, dim=0).t().unsqueeze(0)  # [1, T]
                    sr = int(streamer.get_src_stream_info(0).sample_rate)
                except Exception as e2:
                    raise RuntimeError(
                        f"Failed to read audio: {path}\nload_err={e}\nstreamreader_err={e2}"
                    )

            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)

            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            return wav.to(torch.float32), int(sr)

        # Vibration
        if ext in [".csv", ".tsv", ".txt"]:
            delimiter = "," if ext == ".csv" else ("\t" if ext == ".tsv" else ",")
            try:
                arr = np.genfromtxt(
                    path,
                    dtype=np.float32,
                    delimiter=delimiter,
                    comments="#",
                    skip_header=1,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to read vibration file: {path}\n{e}")

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 2:
                raise ValueError(f"{path} must have at least 2 columns: timestamp + >=1 channel")

            chmat = arr[:, 1:]  # drop timestamp

            if col is None:
                if chmat.shape[1] == 1:
                    sig = chmat[:, 0]
                else:
                    raise ValueError(
                        f"{path} has {chmat.shape[1]} channels, please append '::col=1..{chmat.shape[1]}'"
                    )
            else:
                if not (1 <= col <= chmat.shape[1]):
                    raise ValueError(f"{path} col={col} out of range (1..{chmat.shape[1]}).")
                sig = chmat[:, col - 1]

            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            wav = torch.from_numpy(sig).view(1, -1).to(torch.float32)

            if "vibration_sr" in sample_meta and sample_meta["vibration_sr"] is not None:
                sr = int(sample_meta["vibration_sr"])
            elif self.vib_default_sr is not None:
                sr = int(self.vib_default_sr)
            else:
                raise ValueError(
                    f"{orig} is vibration but no sampling rate provided (vibration_sr or vib_default_sr)."
                )

            return wav, sr

        raise ValueError(f"Unsupported file type: {orig}")


if __name__ == "__main__":
    extractor = FeatureExtractor(
        multi_channel_strategy="concatenate",
        expect_channels=7,
        vib_default_sr=None,
        local_dir=LOCAL_DIR,
        dtype=None,
    )
    print("feature_dim (informational) =", extractor.feature_dim)
    print("Extractor ready.")

