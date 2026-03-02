# -*- coding: utf-8 -*-
"""
CED AV extractor (AB3007, OFFLINE LOCAL, Design-B)

Design B:
- Per-channel: waveform -> feature_extractor -> CED model -> pooled embedding (D)
- Then multi-channel aggregation on embeddings:
    concatenate / average / first / last
- Output is a 1D vector (float32) for each sample.
  For CED-small + 7ch + concatenate => D = 384*7 = 2688.

Fixes:
1) vibration csv header like 'timestamp_s,...' -> robust reader
2) index.csv 'paths' is heavily-escaped list-string -> robust parser

Env (recommended):
  export HF_HOME=/DATA1/chenzhang/hf_cache
  export CED_LOCAL_DIR=/DATA1/chenzhang/SIREN/models/ced-small
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export CED_EXPECT_CH=7
  export CED_VIB_SR=100000   # optional fallback
"""

import os
import re
import json
import ast
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from siren.core.base_extractor import BaseFeatureExtractor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


# -----------------------------
# Robust parsing for index.csv "paths"
# -----------------------------
def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def parse_paths_field(paths_field: Any) -> List[str]:
    """
    index.csv paths may be:
      - list[str]
      - JSON list string
      - heavily escaped CSV string
      - python literal list string
    Return clean list[str].
    """
    if isinstance(paths_field, list):
        return [_strip_wrapping_quotes(str(x)) for x in paths_field]
    if paths_field is None:
        return []

    s = str(paths_field).strip()

    # Try JSON (sometimes double-encoded)
    for _ in range(2):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                out = []
                for x in obj:
                    x = str(x).replace('\\"', '"').strip()
                    x = _strip_wrapping_quotes(x)
                    out.append(x)
                return out
            if isinstance(obj, str):
                s = obj
                continue
        except Exception:
            break

    # Try python literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            out = []
            for x in obj:
                x = str(x).replace('\\"', '"').strip()
                x = _strip_wrapping_quotes(x)
                out.append(x)
            return out
    except Exception:
        pass

    # Last resort: bracket split
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        out = []
        for p in parts:
            p = p.replace('\\"', '"').strip()
            p = _strip_wrapping_quotes(p)
            out.append(p)
        return out

    return [_strip_wrapping_quotes(s)]


# -----------------------------
# Vibration CSV reader: header + ::col=k
# -----------------------------
_COL_RE = re.compile(r"^(.*?)(?:::col=(\d+))?$")


def _read_vibration_csv(path_with_opt: str) -> np.ndarray:
    """
    Reads vibration csv robustly:
      - supports /path/file.csv::col=k
      - skips 1-line header like 'timestamp_s,...'
      - returns float32 vector (T,)
    Column convention:
      - CSV columns: [timestamp, ch1, ch2, ch3, ch4, ...]
      - ::col=1 means ch1 (i.e., usecols=1), ::col=4 means ch4 (usecols=4)
    """
    m = _COL_RE.match(path_with_opt)
    real_path = m.group(1) if m else path_with_opt
    col = int(m.group(2)) if (m and m.group(2) is not None) else None

    # detect header
    with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    has_header = any(ch.isalpha() for ch in first)

    # default: take ch1 (column 1), because column 0 is timestamp
    usecols = [col] if col is not None else [1]

    arr = np.genfromtxt(
        real_path,
        delimiter=",",
        skip_header=1 if has_header else 0,
        usecols=usecols,
        dtype=np.float32,
        invalid_raise=False,
    )
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"Empty vibration signal after reading: {path_with_opt}")
    return arr


# -----------------------------
# Audio loading
# -----------------------------
def _load_audio_any(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio from .wav/.flac/.m4a/.mp3/.mp4 via torchaudio.
    For mp4 decoding, StreamReader is attempted if load fails.
    Return mono [1,T] float32 + sr.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        wav, sr = torchaudio.load(path)
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        elif wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav.to(torch.float32), int(sr)
    except Exception as e1:
        # try StreamReader for mp4/m4a
        if ext in [".mp4", ".m4a", ".mov", ".m4v", ".mp3"]:
            try:
                streamer = torchaudio.io.StreamReader(path)
                streamer.add_audio_stream(frames_per_chunk=0)
                chunks = []
                for (chunk,) in streamer.stream():
                    chunks.append(chunk)
                if not chunks:
                    raise RuntimeError("no audio chunks decoded via StreamReader")
                # chunk: [frames, channels] or [frames]? depends on backend
                x = torch.cat(chunks, dim=0)
                if x.ndim == 2:
                    x = x.mean(dim=1)  # mono
                wav = x.unsqueeze(0)
                info = streamer.get_src_stream_info(0)
                sr = int(getattr(info, "sample_rate", 16000))
                return wav.to(torch.float32), sr
            except Exception as e2:
                raise RuntimeError(f"Fail to read audio: {path}\nload_err={e1}\nstreamreader_err={e2}")
        raise RuntimeError(f"Fail to read audio: {path}\nload_err={e1}")


def _resample_to(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, sr, target_sr)


# -----------------------------
# FeatureExtractor (Design B)
# -----------------------------
class FeatureExtractor(BaseFeatureExtractor):
    """
    Offline-local CED extractor (Design B).
    """

    def __init__(
        self,
        multi_channel_strategy: str = "concatenate",
        expect_channels: Optional[int] = None,
        vib_default_sr: Optional[int] = None,
        model_local_dir: Optional[str] = None,
        target_sr: int = 16000,
        max_samples: int = 160000,  # 10s @ 16k
        **kwargs,
    ):
        super().__init__(multi_channel_strategy)

        # Force GPU only
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This extractor is configured to NOT use CPU.")
        self.device = "cuda"

        self.target_sr = int(target_sr)
        self.max_samples = int(max_samples)

        # channels
        if expect_channels is None:
            expect_channels = int(os.environ.get("CED_EXPECT_CH", "7"))
        self.expected_channels = int(expect_channels)

        # vibration sr fallback
        if vib_default_sr is None:
            env_v = os.environ.get("CED_VIB_SR", "").strip()
            vib_default_sr = int(env_v) if env_v else None
        self.vib_default_sr = vib_default_sr

        # local model dir
        model_local_dir = model_local_dir or os.environ.get("CED_LOCAL_DIR", "").strip()
        if not model_local_dir:
            raise ValueError("CED_LOCAL_DIR is empty. Please export CED_LOCAL_DIR=/path/to/local/ced model.")
        if not os.path.isdir(model_local_dir):
            raise FileNotFoundError(f"CED_LOCAL_DIR not found: {model_local_dir}")
        self.model_local_dir = model_local_dir

        # load feature_extractor + model from LOCAL ONLY
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_local_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.model_local_dir,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device).eval()

        # determine single-channel dim (prefer config.hidden_size)
        hidden = getattr(getattr(self.model, "config", None), "hidden_size", None)
        self._single_dim: Optional[int] = int(hidden) if isinstance(hidden, int) and hidden > 0 else None

        print(
            f"[ced_av_extractor] init: local_dir={self.model_local_dir} device={self.device} "
            f"target_sr={self.target_sr} max_samples={self.max_samples} "
            f"expected_channels={self.expected_channels} vib_default_sr={self.vib_default_sr} "
            f"strategy={multi_channel_strategy}"
        )

    # --- required abstract methods ---
    def _get_single_channel_feature_dim(self) -> int:
        if self._single_dim is not None:
            return self._single_dim
        # infer once
        dummy = torch.zeros(1, int(self.target_sr * 1.0), dtype=torch.float32)  # 1s
        vec = self._extract_single_channel_features(dummy, self.target_sr)
        self._single_dim = int(vec.numel())
        return self._single_dim

    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Per-channel CED embedding (Design B):
        waveform -> feature_extractor -> model -> last_hidden mean-pool over time -> [D]
        Return: 1D tensor [D] on CPU.
        """
        # ---- ensure [1, T] float32 ----
        if signal_tensor.ndim == 1:
            waveform = signal_tensor.unsqueeze(0)
        else:
            waveform = signal_tensor
        waveform = waveform.to(torch.float32)

        # remove DC
        waveform = waveform - waveform.mean()

        # resample to 16k
        if int(sample_rate) != int(self.target_sr):
            waveform = _resample_to(waveform, int(sample_rate), int(self.target_sr))

        T = waveform.shape[-1]

        # ---- chunking: 10s windows + last tail window ----
        segments: List[torch.Tensor] = []
        if T <= self.max_samples:
            segments = [waveform]
        else:
            n_full = T // self.max_samples
            for i in range(n_full):
                s = i * self.max_samples
                e = s + self.max_samples
                segments.append(waveform[..., s:e])
            if n_full * self.max_samples < T:
                segments.append(waveform[..., (T - self.max_samples):])

        feats: List[torch.Tensor] = []

        def _pool_to_1d(x: torch.Tensor) -> torch.Tensor:
            """
            Convert various possible hidden-state shapes to a 1D [D] vector by pooling over time/seq dim.
            Expected common cases:
            - [B, S, D] -> mean over S -> [D]
            - [S, D]    -> mean over S -> [D]
            - [B, D]    -> squeeze B -> [D]
            - [B, 1, S, D] or [B, H, S, D] -> reduce extra dims then mean over S -> [D]
            """
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"hidden state is not a Tensor: {type(x)}")

            # Ensure float for pooling stability
            x = x.to(dtype=torch.float32)

            if x.dim() == 4:
                # e.g., [B, H, S, D] or [B, 1, S, D]
                # Average over "H" (or the 2nd dim), keep [B, S, D]
                x = x.mean(dim=1)

            if x.dim() == 3:
                # [B, S, D] -> [D]
                x = x.mean(dim=1).squeeze(0)
            elif x.dim() == 2:
                # Could be [S, D] or [B, D]
                # Heuristic: if first dim is 1, treat as [B, D]
                if x.shape[0] == 1:
                    x = x.squeeze(0)  # [D]
                else:
                    # treat as [S, D]
                    x = x.mean(dim=0)  # [D]
            elif x.dim() == 1:
                # already [D]
                pass
            else:
                raise RuntimeError(f"Unexpected hidden state dim={x.dim()} shape={tuple(x.shape)}")

            # Final safety: ensure 1D
            if x.dim() != 1:
                raise RuntimeError(f"Pooling failed, got dim={x.dim()} shape={tuple(x.shape)}")
            return x

        with torch.no_grad():
            for seg in segments:
                seg_np = seg.squeeze(0).detach().cpu().numpy()

                inputs = self.feature_extractor(
                    seg_np,
                    sampling_rate=int(self.target_sr),
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # request hidden states
                try:
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                except TypeError:
                    outputs = self.model(**inputs)

                seg_feat = None

                # Prefer hidden states
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    last = outputs.hidden_states[-1]
                    seg_feat = _pool_to_1d(last)

                # Fallback: logits (already [num_labels] typically)
                if seg_feat is None and hasattr(outputs, "logits"):
                    logits = outputs.logits
                    if isinstance(logits, torch.Tensor):
                        if logits.dim() == 2 and logits.shape[0] == 1:
                            seg_feat = logits.squeeze(0).to(torch.float32)
                        elif logits.dim() == 1:
                            seg_feat = logits.to(torch.float32)

                if seg_feat is None:
                    raise RuntimeError("Cannot obtain features: neither hidden_states nor logits available.")

                # Debug print once
                if not hasattr(self, "_dbg_shape_printed"):
                    self._dbg_shape_printed = True
                    hs = outputs.hidden_states[-1] if (hasattr(outputs, "hidden_states") and outputs.hidden_states is not None) else None
                    if isinstance(hs, torch.Tensor):
                        print(f"[ced_offline][debug] last_hidden shape={tuple(hs.shape)} -> pooled seg_feat shape={tuple(seg_feat.shape)}")
                    else:
                        print(f"[ced_offline][debug] pooled seg_feat shape={tuple(seg_feat.shape)} (no hidden_states tensor)")

                feats.append(seg_feat.detach().cpu())

        # segment mean -> [D]
        feat = torch.stack(feats, dim=0).mean(dim=0)

        # cache single-channel dim
        if getattr(self, "_single_dim", None) is None:
            self._single_dim = int(feat.numel())

        return feat


    # --- load one channel signal from path ---
    def _load_any_signal(self, path: str, sample_meta: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        path = str(path).strip()

        # vibration CSV (may contain ::col=k)
        if path.endswith(".csv") or ".csv::col=" in path:
            arr = _read_vibration_csv(path)
            sig = torch.from_numpy(arr).float().unsqueeze(0)  # [1,T]

            sr = sample_meta.get("vibration_sr", None)
            if sr is None or sr == "" or (isinstance(sr, float) and np.isnan(sr)):
                sr = self.vib_default_sr
            if sr is None:
                raise ValueError(f"Missing vibration_sr for id={sample_meta.get('id')} and no vib_default_sr.")
            return sig, int(sr)

        # audio / video
        return _load_audio_any(path)

    # --- main entrypoint ---
    def extract_features(self, sample_meta: Dict[str, Any]) -> torch.Tensor:
        """
        Design B: per-channel embedding -> aggregate on embeddings.
        Return 1D torch.Tensor (will be converted to numpy by extract_features.py).
        """
        raw_paths = sample_meta.get("paths", [])
        ch_paths = parse_paths_field(raw_paths)

        if len(ch_paths) == 0:
            raise ValueError(f"Empty paths for id={sample_meta.get('id')}")

        if len(ch_paths) != self.expected_channels:
            raise RuntimeError(f"channels mismatch: got {len(ch_paths)}, expect {self.expected_channels}")

        # per-channel embeddings
        embs: List[torch.Tensor] = []
        for p in ch_paths:
            sig, sr = self._load_any_signal(p, sample_meta)
            emb = self._extract_single_channel_features(sig, sr)  # [D]
            embs.append(emb)

        feats = torch.stack(embs, dim=0)  # [C, D]

        strat = getattr(self, "multi_channel_strategy", "concatenate")
        if strat == "first":
            out = feats[0]
        elif strat == "last":
            out = feats[-1]
        elif strat == "average":
            out = feats.mean(dim=0)
        elif strat == "concatenate":
            out = feats.reshape(-1)  # [C*D]
        else:
            raise ValueError(f"Unknown multi_channel_strategy: {strat}")

        return out.to(torch.float32)


