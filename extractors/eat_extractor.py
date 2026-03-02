# /DATA1/chenzhang/SIREN/examples/eat_av_extractor.py
# -*- coding: utf-8 -*-
"""
EAT 多通道特征提取器（本地离线版，兼容 AB3007 7ch）
- 继承 BaseFeatureExtractor
- 16kHz -> Kaldi fbank(128 mel, 25ms/10ms)，全局归一化
- 按 10s 切片；片内取 CLS（若无则时间均值），片间再均值
- 兼容多通道：concatenate / average / first / last
- 统一读取音频（含 mp4）与振动 txt/csv/tsv(::col=K)，支持 header: timestamp_s
- 不使用 AutoModel / trust_remote_code，不联网；从本地目录导入 EAT
- 推理与聚合尽量在 GPU；仅最终 return 前做一次 .cpu()（npz 写盘所需）
"""

import os
import sys
import json
import re
import inspect
import types
import importlib
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio

from siren.core.base_extractor import BaseFeatureExtractor


_COL_RE = re.compile(r"^(.*?)(?:::col=(\d+))?$", re.IGNORECASE)


def _clean_path(p: str) -> str:
    p = str(p).strip().strip('"').strip("'")
    p = p.replace("\ufeff", "").replace("\u200b", "")
    p = re.sub(r"[：:；;，,\s]+$", "", p)
    return p


def _resolve_paths_field(paths_field: Any) -> List[str]:
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
      - supports xxx.csv::col=K (1-based for value columns after timestamp)
      - supports header line like 'timestamp_s,...'
      - supports delimiter: csv(',', tsv('\\t'), txt(whitespace or comma)
    Returns float32 vector (T,)
    """
    m = _COL_RE.match(path_with_opt.strip())
    real_path = m.group(1) if m else path_with_opt
    col = int(m.group(2)) if (m and m.group(2) is not None) else None
    real_path = _clean_path(real_path)

    ext = os.path.splitext(real_path)[1].lower()

    # detect header
    with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
    has_header = any(ch.isalpha() for ch in first)  # catches timestamp_s

    # choose delimiter
    if ext == ".tsv":
        delimiter = "\t"
    elif ext == ".csv":
        delimiter = ","
    else:
        delimiter = None  # whitespace default for txt

    def _try(delim):
        return np.genfromtxt(
            real_path,
            delimiter=delim,
            skip_header=1 if has_header else 0,
            dtype=np.float32,
            invalid_raise=False,
        )

    arr = _try(delimiter)
    arr = np.asarray(arr, dtype=np.float32)

    # retry if parsing failed badly (common for .txt with comma)
    if arr.ndim < 2 or arr.shape[0] == 0 or (not np.isfinite(arr).any()):
        if delimiter is None:
            arr = np.asarray(_try(","), dtype=np.float32)
        else:
            arr = np.asarray(_try(None), dtype=np.float32)

    if arr.ndim == 1 or arr.shape[1] < 2:
        raise ValueError(f"{real_path} 至少需要两列（timestamp + 1 通道）")

    chmat = arr[:, 1:]  # drop timestamp
    if col is None:
        if chmat.shape[1] == 1:
            sig = chmat[:, 0]
        else:
            raise ValueError(
                f"{real_path} 含 {chmat.shape[1]} 个通道，请在路径后添加 '::col=1..{chmat.shape[1]}'"
            )
    else:
        if not (1 <= col <= chmat.shape[1]):
            raise ValueError(f"{real_path} 指定 col={col} 越界（1..{chmat.shape[1]})")
        sig = chmat[:, col - 1]

    sig = np.asarray(sig, dtype=np.float32)
    sig = sig[np.isfinite(sig)]
    if sig.size == 0:
        raise ValueError(f"{real_path} 振动信号为空或全为 NaN")
    return sig


class FeatureExtractor(BaseFeatureExtractor):
    """
    EAT feature extractor (local offline load).

    Args:
        multi_channel_strategy: "concatenate" | "average" | "first" | "last"
        model_name: 本地模型目录（包含 config.json、权重文件、以及 EAT 相关 *.py）
        expect_channels: 样本通道数（AB3007 通常 7）
        vib_default_sr: index.csv 没 vibration_sr 时振动采样率兜底
        dtype: 可选 "fp16"/"bf16"
    """

    def __init__(
        self,
        multi_channel_strategy: str = "concatenate",
        model_name: str = "/DATA1/chenzhang/models/EAT-base_epoch30_pretrain",
        expect_channels: int = 7,
        vib_default_sr: Optional[int] = None,
        dtype: Optional[str] = None,
    ):
        super().__init__(multi_channel_strategy)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.expected_channels = int(expect_channels)
        self.target_sample_rate = 16000
        self.max_samples = 160000  # 10s * 16k

        # EAT settings
        self.num_mel_bins = 128
        self.frame_length_ms = 25.0
        self.frame_shift_ms = 10.0
        self.target_length = 1024
        self.norm_mean = -4.268
        self.norm_std = 4.569

        self.vib_default_sr = vib_default_sr

        self._torch_dtype = None
        if dtype:
            if dtype.lower() == "fp16":
                self._torch_dtype = torch.float16
            elif dtype.lower() == "bf16":
                self._torch_dtype = torch.bfloat16

        self.model, self._embed_dim = self._load_eat_local(model_name)
        if self._torch_dtype is not None:
            self.model = self.model.to(self._torch_dtype)
        self.model = self.model.to(self.device).eval()

        print(f"[eat_av_extractor] loaded local EAT from: {model_name} on {self.device} (expected_channels={self.expected_channels})")

    # ---------------- required interface ----------------
    @property
    def feature_dim(self) -> int:
        d_single = self._get_single_channel_feature_dim()
        if self.multi_channel_strategy == "concatenate":
            return d_single * self.expected_channels
        return d_single

    def extract_features(self, sample_meta: Dict[str, Any]) -> np.ndarray:
        ch_paths = self._resolve_channel_paths(sample_meta)
        if len(ch_paths) != self.expected_channels:
            raise RuntimeError(f"channels mismatch: got {len(ch_paths)}, expect {self.expected_channels}")

        feats: List[torch.Tensor] = []
        for p in ch_paths:
            sig, sr = self._load_any_signal(p, sample_meta)           # CPU tensor [1,T]
            f = self._extract_single_channel_features(sig, sr)        # GPU tensor [D]
            feats.append(f)

        feats_t = torch.stack(feats, dim=0)  # [C,D] on GPU

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

        # 仅此一次：GPU->CPU->numpy（extract_features.py 写 npz 需要）
        return out.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def extract_features_from_signal(self, signal: torch.Tensor, sample_rate: int) -> np.ndarray:
        out = self._extract_single_channel_features(signal, sample_rate)
        return out.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def _get_single_channel_feature_dim(self) -> int:
        return int(self._embed_dim)

    # ---------------- single-channel forward ----------------
    def _extract_single_channel_features(self, signal_tensor: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        waveform = signal_tensor
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(torch.float32)
        waveform = waveform - waveform.mean()

        if int(sample_rate) != self.target_sample_rate:
            waveform = torchaudio.transforms.Resample(int(sample_rate), self.target_sample_rate)(waveform)

        segments = self._segment_waveform(waveform)

        seg_feats: List[torch.Tensor] = []
        with torch.inference_mode():
            for seg in segments:
                fb = self._waveform_to_fbank(seg)  # [1,1,T,F] on CPU
                fb = fb.to(self.device)
                if self._torch_dtype is not None:
                    fb = fb.to(self._torch_dtype)

                # 期望 outputs: [B, T', D] 或 [B, D] 或 dict-like
                outputs = self.model.extract_features(fb)

                if isinstance(outputs, dict):
                    # 常见键：'x'/'feat'/'features'
                    for k in ("x", "feat", "features", "last_hidden_state"):
                        if k in outputs:
                            outputs = outputs[k]
                            break

                if outputs.dim() == 3:
                    # 取 CLS（若你的实现没有 CLS 语义，可改为 outputs.mean(dim=1)）
                    vec = outputs[:, 0, :]
                elif outputs.dim() == 2:
                    vec = outputs
                else:
                    raise RuntimeError(f"Unexpected EAT output dim={outputs.dim()} shape={tuple(outputs.shape)}")

                seg_feats.append(vec.squeeze(0).detach())  # keep on GPU

        return torch.stack(seg_feats, dim=0).mean(dim=0)  # [D] on GPU

    # ---------------- segment / fbank ----------------
    def _segment_waveform(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        T = waveform.shape[-1]
        segs: List[torch.Tensor] = []
        if T <= self.max_samples:
            segs.append(waveform[..., -T:])
        else:
            n_full = T // self.max_samples
            for i in range(n_full):
                b = i * self.max_samples
                e = b + self.max_samples
                segs.append(waveform[..., b:e])
            if n_full * self.max_samples < T:
                segs.append(waveform[..., -self.max_samples:])
        return segs

    def _waveform_to_fbank(self, waveform: torch.Tensor) -> torch.Tensor:
        w = waveform.squeeze(0)  # [T]
        mel = torchaudio.compliance.kaldi.fbank(
            w.unsqueeze(0),
            htk_compat=True,
            sample_frequency=self.target_sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
        )  # [T', F]

        n = mel.shape[0]
        if n < self.target_length:
            pad = torch.zeros(self.target_length - n, mel.shape[1], dtype=mel.dtype)
            mel = torch.cat([mel, pad], dim=0)
        else:
            mel = mel[: self.target_length, :]

        mel = (mel - self.norm_mean) / self.norm_std
        return mel.unsqueeze(0).unsqueeze(0)  # [1,1,T,F]

    # ---------------- paths & signal I/O ----------------
    def _resolve_channel_paths(self, sample_meta: Dict[str, Any]) -> List[str]:
        if "paths" in sample_meta:
            return _resolve_paths_field(sample_meta["paths"])
        root = sample_meta["root"]
        stem = sample_meta["stem"]
        return [
            os.path.join(root, "audio", f"{stem}_recorder.wav"),
            os.path.join(root, "audio", f"{stem}_ios.wav"),
            os.path.join(root, "audio", f"{stem}_android.wav"),
            os.path.join(root, "vibration", f"{stem}_vibration.csv::col=1"),
            os.path.join(root, "vibration", f"{stem}_vibration.csv::col=2"),
            os.path.join(root, "vibration", f"{stem}_vibration.csv::col=3"),
            os.path.join(root, "vibration", f"{stem}_vibration.csv::col=4"),
        ]

    def _load_any_signal(self, path: str, sample_meta: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        path = _clean_path(path)
        base_path = path
        m = _COL_RE.match(path)
        if m:
            base_path = _clean_path(m.group(1))
        ext = os.path.splitext(base_path)[1].lower()

        # audio
        if ext in [".wav", ".flac", ".m4a", ".mp3", ".mp4"]:
            try:
                wav, sr = torchaudio.load(base_path)
            except Exception as e:
                try:
                    streamer = torchaudio.io.StreamReader(base_path)
                    streamer.add_audio_stream(frames_per_chunk=0)
                    chunks = []
                    for (chunk,) in streamer.stream():
                        chunks.append(chunk)
                    if not chunks:
                        raise RuntimeError("no audio chunks decoded via StreamReader")
                    wav = torch.cat(chunks, dim=0).t().unsqueeze(0)
                    sr = int(streamer.get_src_stream_info(0).sample_rate)
                except Exception as e2:
                    raise RuntimeError(f"Fail to read audio: {base_path}\nload_err={e}\nstreamreader_err={e2}")

            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)
            return wav.to(torch.float32), int(sr)

        # vibration
        if ext in [".txt", ".csv", ".tsv"]:
            sig = _read_vibration_text(path)  # includes ::col parsing + header skip
            wav = torch.from_numpy(sig).view(1, -1).to(torch.float32)

            sr = sample_meta.get("vibration_sr", None)
            if sr is not None and sr != "" and not (isinstance(sr, float) and np.isnan(sr)):
                sr_i = int(sr)
            elif self.vib_default_sr is not None:
                sr_i = int(self.vib_default_sr)
            else:
                raise ValueError(f"vibration file given but no sampling rate for: {path}")
            return wav, sr_i

        raise ValueError(f"Unsupported file type: {path}")

    # ---------------- local EAT loader ----------------
    def _load_eat_local(self, local_dir: str):
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"EAT local dir not found: {local_dir}")

        pkg_name = "eat_local_pkg"
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [local_dir]
        sys.modules[pkg_name] = pkg

        try:
            cfg_mod = importlib.import_module(f"{pkg_name}.configuration_eat")
            EATConfig = getattr(cfg_mod, "EATConfig")
        except Exception as e:
            raise ImportError(f"未找到 configuration_eat.py 或 EATConfig，目录: {local_dir}") from e

        cfg = EATConfig.from_pretrained(local_dir)

        candidate_modules = ["model_core", "eat_model", "modeling_eat"]
        Model = None
        last_err = None

        from transformers import PreTrainedModel

        for m in candidate_modules:
            try:
                mod = importlib.import_module(f"{pkg_name}.{m}")
                if hasattr(mod, "EATModel"):
                    Model = getattr(mod, "EATModel")
                    break
                for name, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, PreTrainedModel) and obj is not PreTrainedModel:
                        Model = obj
                        break
                if Model is not None:
                    break
            except Exception as e:
                last_err = e
                continue

        if Model is None:
            raise RuntimeError(
                f"未能在 {local_dir} 找到可用的 EAT 模型类；请检查目录内 *.py 的类名。\n最后一次导入错误: {last_err}"
            )

        model = Model.from_pretrained(local_dir, config=cfg)

        embed_dim = getattr(getattr(model, "config", None), "hidden_size", None)
        if embed_dim is None:
            # fallback guess
            embed_dim = 768
        return model, int(embed_dim)


if __name__ == "__main__":
    ext = FeatureExtractor(
        multi_channel_strategy="concatenate",
        model_name="/DATA1/chenzhang/models/EAT-base_epoch30_pretrain",
        expect_channels=7,
        vib_default_sr=100000,
        dtype=None,
    )
    print("final feature_dim =", ext.feature_dim)
    print("Extractor ready.")
