# echo_extractor.py
# -*- coding: utf-8 -*-
"""
ECHO-Small Multi-channel Feature Extractor (SIREN compatible)

Design goals (as requested):
1) No resampling:
   - Always compute STFT using the original sampling rate of each channel.
2) No padding:
   - No padding on frequency axis (do NOT pad to band_width multiple).
   - No padding on time axis (short segments remain short; no zero-pad).
3) Per-channel export (variable dimension allowed):
   - If sample_meta["return_per_channel"] == True, return List[Tensor],
     one 1D embedding per channel, each may have different length.

Pipeline (single channel):
- waveform -> remove DC
- STFT (25ms window / 10ms hop, center=False)
- log magnitude -> normalization (official constants)
- time segmentation by frames (seg_frames, include last tail segment)
- model.extract_features per segment -> segment mean (utterance-level) -> average across segments
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio

from siren.core.base_extractor import BandSplitFeatureExtractor

# Local HF repo directory (downloaded by: huggingface-cli download yucongzh/echo-small-0824 --local-dir <DIR>)
LOCAL_DIR = "/DATA1/chenzhang/models/echo-small-0824"


class FeatureExtractor(BandSplitFeatureExtractor):
    """
    Args:
        multi_channel_strategy: "concatenate" | "average" | "first" | "last"
            Note: Only used when NOT returning per-channel features.
            If channel dims differ, aggregation will raise unless return_per_channel=True.
        expect_channels: expected number of channels per sample (e.g., AB3007 = 7)
        vib_default_sr: fallback sampling rate for vibration text files if meta has no vibration_sr
        frame_length_ms / frame_shift_ms: STFT parameters (ms)
        seg_frames: number of STFT frames per segment (time slicing)
        norm_mean / norm_std: normalization constants (official implementation)
        band_width: ECHO band width (32). Kept for reference; we do NOT pad to it.
        local_dir: local directory containing AudioMAEWithBand definition + model.safetensors
        dtype: "fp16" | "bf16" | None
    """

    def __init__(
        self,
        multi_channel_strategy: str = "concatenate",
        expect_channels: int = 7,
        vib_default_sr: Optional[int] = None,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        seg_frames: int = 2000,
        norm_mean: float = -5.874158,
        norm_std: float = 5.223174,
        band_width: int = 32,
        local_dir: str = LOCAL_DIR,
        dtype: Optional[str] = None,
    ):
        super().__init__(multi_channel_strategy, base_feature_dim=384, band_width=int(band_width))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.expected_channels = int(expect_channels)
        self.vib_default_sr = vib_default_sr

        self.frame_length_ms = float(frame_length_ms)
        self.frame_shift_ms = float(frame_shift_ms)
        self.seg_frames = int(seg_frames)

        self.norm_mean = float(norm_mean)
        self.norm_std = float(norm_std)

        # dtype control
        self._torch_dtype = None
        if dtype:
            d = dtype.lower()
            if d == "fp16":
                self._torch_dtype = torch.float16
            elif d == "bf16":
                self._torch_dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown dtype: {dtype} (use fp16/bf16/None)")

        # ---- import model definition from local_dir
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(
                f"[echo_av_extractor] local_dir not found: {local_dir}\n"
                f"Download first:\n  huggingface-cli download yucongzh/echo-small-0824 --local-dir {local_dir}"
            )
        import sys
        if local_dir not in sys.path:
            sys.path.insert(0, local_dir)

        try:
            from audioMAE_band_upgrade import AudioMAEWithBand  # type: ignore
        except Exception as e:
            raise ImportError(f"[echo_av_extractor] Failed to import AudioMAEWithBand from {local_dir}: {e}")

        weights_path = os.path.join(local_dir, "model.safetensors")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"[echo_av_extractor] Missing weights: {weights_path}")

        # ---- build model
        model_cfg = {
            "spec_len": self.seg_frames,
            "shift_size": 16,
            "in_chans": 1,
            "embed_dim": 384,
            "encoder_depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "norm_layer": lambda x: torch.nn.LayerNorm(x, eps=1e-6),
            "fix_pos_emb": True,
            "band_width": int(band_width),
            "mask_ratio": 0.75,
            "freq_pos_emb_dim": 384,
        }
        self.model = AudioMAEWithBand(**model_cfg)

        # ---- load weights
        try:
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        except Exception as e:
            raise RuntimeError(
                f"[echo_av_extractor] Failed to load safetensors: {e}\n"
                f"Install: pip install safetensors"
            )

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) + len(unexpected) > 0:
            print(f"[warn] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

        if self._torch_dtype is not None:
            self.model = self.model.to(self._torch_dtype)

        self.model = self.model.to(self.device).eval()

        print(
            f"[echo_av_extractor] device={self.device} expected_channels={self.expected_channels} "
            f"seg_frames={self.seg_frames} (NO resample, NO padding)"
        )
        print(f"[echo_av_extractor] model loaded from: {local_dir}")

    # ---------------------------------------------------------------------
    # NOTE:
    # feature_dim is only a reference (since per-channel dim varies with sr).
    # ---------------------------------------------------------------------
    @property
    def feature_dim(self) -> int:
        # Reference: compute bands under 16kHz for printing only.
        d_single_ref = 384 * self._calc_num_bands_for_sr(16000)
        if self.multi_channel_strategy == "concatenate":
            return d_single_ref * self.expected_channels
        if self.multi_channel_strategy in ("average", "first", "last"):
            return d_single_ref
        raise ValueError(f"Unknown multi_channel_strategy: {self.multi_channel_strategy}")

    def _get_single_channel_feature_dim(self) -> int:
        # Reference only
        return int(384 * self._calc_num_bands_for_sr(16000))

    def _calc_num_bands_for_sr(self, sr: int) -> int:
        n_fft = int(round(self.frame_length_ms * sr / 1000.0))
        freq_bins = n_fft // 2 + 1
        bands = (freq_bins + self.band_width - 1) // self.band_width
        return int(max(1, bands))

    # ---------------------------------------------------------------------
    # main entry for SIREN
    # ---------------------------------------------------------------------
    def extract_features(self, sample_meta: Dict[str, Any]):
        """
        If sample_meta["return_per_channel"] == True:
            Return List[Tensor], length = expected_channels.
            Each Tensor is 1D embedding, variable length allowed.
        Else:
            Aggregate by multi_channel_strategy (requires same dim across channels).
        """
        ch_paths = self._resolve_channel_paths(sample_meta)
        if len(ch_paths) != self.expected_channels:
            raise RuntimeError(f"Channel mismatch: got {len(ch_paths)}, expected {self.expected_channels}")

        feats: List[torch.Tensor] = []
        for p in ch_paths:
            sig, sr = self._load_any_signal(p, sample_meta)
            f = self._extract_single_channel_features(sig, sr)
            if f.dim() > 1:
                f = f.reshape(-1)
            feats.append(f.detach().cpu())

        # --- per-channel export (variable dim allowed)
        if bool(sample_meta.get("return_per_channel", False)):
            return feats  # List[Tensor], no stacking, no padding, keep each channel's own length

        # --- otherwise, require same dim to aggregate
        sizes = [int(t.numel()) for t in feats]
        if not all(s == sizes[0] for s in sizes):
            raise RuntimeError(
                "multi_channel_strategy requires same feature dim for all channels, but got: "
                f"{sizes}. If you want variable-length per-channel features, set "
                "sample_meta['return_per_channel']=True (extract_features.py --per_channel)."
            )

        x = torch.stack(feats, dim=0)  # [C, d]
        if self.multi_channel_strategy == "concatenate":
            return x.reshape(-1)
        if self.multi_channel_strategy == "average":
            return x.mean(dim=0)
        if self.multi_channel_strategy == "first":
            return x[0]
        if self.multi_channel_strategy == "last":
            return x[-1]
        raise ValueError(f"Unknown multi_channel_strategy: {self.multi_channel_strategy}")

    def extract_features_from_signal(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self._extract_single_channel_features(signal, sample_rate)

    # ---------------------------------------------------------------------
    # single-channel forward (NO resample, NO padding)
    # ---------------------------------------------------------------------
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        NO resample:
            - sr_used = sample_rate (original)
        NO padding:
            - STFT center=False, pad=0
            - No frequency/time padding anywhere
        """
        x = signal_tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, T]
        x = x.to(torch.float32)
        x = x - x.mean()

        sr_used = int(sample_rate)

        # STFT params from original sr (NO resample)
        n_fft = int(round(self.frame_length_ms * sr_used / 1000.0))
        hop = int(round(self.frame_shift_ms * sr_used / 1000.0))
        n_fft = max(n_fft, 16)
        hop = max(hop, 1)

        spec_tf = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window_fn=torch.hann_window,
            power=1.0,
            center=False,
            pad=0,
        )

        # spec: [1, F, T]
        spec = spec_tf(x)
        spec = torch.log(spec + 1e-9)

        # remove batch/channel axis -> [F, T]
        spec = spec.squeeze(0)
        if spec.dim() != 2:
            # defensive
            spec = spec.reshape(spec.shape[-2], spec.shape[-1])

        # normalize
        spec = (spec - self.norm_mean) / (self.norm_std * 2.0)

        T = int(spec.shape[1])
        segs: List[torch.Tensor] = []

        # time slicing (NO pad; include last tail segment)
        if T <= self.seg_frames:
            segs.append(spec[:, :T])  # keep as-is
        else:
            n_full = T // self.seg_frames
            for i in range(n_full):
                b = i * self.seg_frames
                e = b + self.seg_frames
                segs.append(spec[:, b:e])
            if n_full * self.seg_frames < T:
                segs.append(spec[:, -self.seg_frames:])

        outs: List[torch.Tensor] = []
        with torch.inference_mode():
            for s in segs:
                # Many ECHO implementations accept [F, T]. If fails, try [1, T, F].
                out_utt = None
                try:
                    utt, _ = self.model.extract_features(s.to(self.device), sr_used)
                    out_utt = utt
                except Exception:
                    st = s.transpose(0, 1).unsqueeze(0).to(self.device)  # [1, T, F]
                    utt, _ = self.model.extract_features(st, sr_used)
                    out_utt = utt

                if self._torch_dtype is not None:
                    out_utt = out_utt.to(torch.float32)
                outs.append(out_utt.detach().cpu().reshape(-1))

        # average across segments -> 1D embedding
        return torch.stack(outs, dim=0).mean(dim=0)

    # ---------------------------------------------------------------------
    # paths parsing (index.csv "paths" column)
    # ---------------------------------------------------------------------
    def _safe_parse_paths_str(self, s: str):
        t = (s or "").strip()
        if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
            t = t[1:-1]
        t = t.replace('""', '"').replace('\\"', '"').replace("\\'", "'")
        t = t.replace("\ufeff", "").strip()
        # strip trailing punctuations/spaces
        t = re.sub(r'[：:；;，,\s]+$', '', t)
        if t.startswith("[") or t.startswith("{"):
            try:
                return json.loads(t)
            except Exception:
                import ast
                return ast.literal_eval(t)
        return [t]

    def _normalize_paths_list(self, raw) -> List[str]:
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        elif isinstance(raw, dict):
            raw = list(raw.values())
        elif isinstance(raw, (tuple, set)):
            raw = list(raw)
        elif not isinstance(raw, list):
            raw = [raw]

        norm: List[str] = []
        for p in raw:
            if isinstance(p, dict):
                path = p.get("path") or p.get("file") or p.get("filepath") or p.get("src") or p.get("p")
                col = p.get("col") or p.get("channel") or p.get("ch")
                if path is None:
                    raise ValueError(f"Bad paths entry (no 'path'): {p}")
                if col is not None:
                    path = f"{path}::col={int(col)}"
                norm.append(str(path))
            else:
                norm.append(str(p))
        return norm

    def _resolve_channel_paths(self, sample_meta: Dict[str, Any]) -> List[str]:
        if "paths" not in sample_meta:
            raise ValueError("sample_meta must contain 'paths' (list/JSON string).")

        paths = sample_meta["paths"]
        if isinstance(paths, str):
            paths = self._safe_parse_paths_str(paths)
        elif isinstance(paths, dict):
            paths = list(paths.values())

        return self._normalize_paths_list(paths)

    # ---------------------------------------------------------------------
    # unified loader for audio + vibration
    # vibration parsing aligned to your BEATs extractor:
    #   - default: delimiter="," and skip_header=1 (timestamp in the first column)
    # ---------------------------------------------------------------------
    def _load_any_signal(self, path: str, sample_meta: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        path = str(path).strip().strip('"').strip("'").replace("\ufeff", "").replace("\u200b", "")
        path = re.sub(r'[：:；;，,\s]+$', '', path)

        # parse ::col=K / |col=K / #col=K
        m = re.search(r'(?::col=|\|col=|#col=)(\d+)$', path)
        col = int(m.group(1)) if m else None
        if m:
            path = re.sub(r'(?::col=|\|col=|#col=)\d+$', '', path)
            path = re.sub(r'[：:；;，,\s]+$', '', path)

        ext = os.path.splitext(path)[1].lower()

        # -------- audio --------
        if ext in [".wav", ".flac", ".m4a", ".mp3", ".mp4"]:
            try:
                wav, sr = torchaudio.load(path)
            except Exception as e:
                # fallback via StreamReader (ffmpeg)
                try:
                    streamer = torchaudio.io.StreamReader(path)
                    streamer.add_audio_stream(frames_per_chunk=0)
                    chunks = []
                    for (chunk,) in streamer.stream():
                        chunks.append(chunk)
                    if not chunks:
                        raise RuntimeError("no audio chunks decoded via StreamReader")
                    wav = torch.cat(chunks, dim=0).t().unsqueeze(0)  # [1, T]
                    sr = int(streamer.get_src_stream_info(0).sample_rate)
                except Exception as e2:
                    raise RuntimeError(f"Fail to read audio: {path}\nload_err={e}\nstreamreader_err={e2}")

            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
            return wav.to(torch.float32), int(sr)

        # -------- vibration text/csv/tsv --------
        if ext in [".txt", ".csv", ".tsv"]:
            arr = None
            last_err = None

            # 1) Preferred: align to BEATs version (comma + skip header)
            try:
                arr = np.genfromtxt(
                    path,
                    dtype=np.float32,
                    delimiter=",",
                    comments="#",
                    skip_header=1,   # first row is header: timestamp, ch1, ch2, ...
                )
            except Exception as e:
                last_err = e
                arr = None

            # 2) Fallbacks: try tab / auto delimiter
            if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
                for delim in ["\t", None]:
                    try:
                        arr = np.genfromtxt(
                            path,
                            dtype=np.float32,
                            delimiter=delim,
                            comments="#",
                            skip_header=1,
                        )
                        if isinstance(arr, np.ndarray) and arr.size > 0:
                            break
                    except Exception as e:
                        last_err = e
                        arr = None

            # 3) Final fallback: try without skipping header (in case no header exists)
            if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
                for delim in [",", "\t", None]:
                    try:
                        arr = np.genfromtxt(
                            path,
                            dtype=np.float32,
                            delimiter=delim,
                            comments="#",
                            skip_header=0,
                        )
                        if isinstance(arr, np.ndarray) and arr.size > 0:
                            break
                    except Exception as e:
                        last_err = e
                        arr = None

            if arr is None or (not isinstance(arr, np.ndarray)) or arr.size == 0:
                raise RuntimeError(f"Failed to read vibration file: {path}\nlast_err={last_err}")

            if arr.ndim == 1:
                # could happen if file has a single numeric column; treat as 1-channel (no timestamp)
                chmat = arr.reshape(-1, 1)
            else:
                # Decide whether first column is timestamp:
                # If there are >=2 columns, and the first column looks monotonic, treat it as timestamp.
                if arr.shape[1] >= 2:
                    first = arr[:, 0]
                    dif = np.diff(first)
                    is_mono = np.all(dif >= 0) or np.all(dif <= 0)
                    if is_mono:
                        chmat = arr[:, 1:]  # drop timestamp
                    else:
                        # Not monotonic -> treat all columns as channels
                        chmat = arr
                else:
                    chmat = arr

            if chmat.shape[1] < 1:
                raise ValueError(f"{path}: no valid vibration channels found after parsing.")

            # select channel
            if col is None:
                if chmat.shape[1] == 1:
                    sig = chmat[:, 0]
                else:
                    raise ValueError(
                        f"{path} has {chmat.shape[1]} channels, please specify '::col=1..{chmat.shape[1]}'"
                    )
            else:
                if not (1 <= col <= chmat.shape[1]):
                    raise ValueError(f"{path} col={col} out of range (1..{chmat.shape[1]}).")
                sig = chmat[:, col - 1]

            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            wav = torch.from_numpy(sig).view(1, -1).to(torch.float32)

            # sampling rate (vibration)
            if "vibration_sr" in sample_meta and sample_meta["vibration_sr"] is not None:
                sr = int(sample_meta["vibration_sr"])
            elif self.vib_default_sr is not None:
                sr = int(self.vib_default_sr)
            else:
                raise ValueError(
                    f"Vibration file given but no sampling rate provided: {path}. "
                    f"Provide meta['vibration_sr'] in index.csv or set vib_default_sr in extractor."
                )

            return wav, sr

        raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    ext = FeatureExtractor(
        multi_channel_strategy="concatenate",
        expect_channels=7,
        vib_default_sr=None,
        local_dir=LOCAL_DIR,
        dtype=None,
    )
    print("[echo_av_extractor] feature_dim (ref @16k) =", ext.feature_dim)
    print("[echo_av_extractor] Ready.")
