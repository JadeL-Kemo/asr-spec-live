# -*- coding: utf-8 -*-
"""SenseVoiceSmall ONNX 识别器（CPU / onnxruntime）。

选型依据见 REPORT.md §9：中文干净/噪声条件下优于 Whisper，CPU 延迟 71ms 均值、
1s 累计切片流式更新 36–80ms，且 partial 文本前缀稳定（35/35）。

依赖：torch / torchaudio（仅用于 Kaldi fbank）、onnxruntime、numpy/onnxruntime。
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi

from .config import SENSEVOICE_DIR
from .interfaces import ASRProvider, SAMPLE_RATE

DEFAULT_MODEL_DIR = SENSEVOICE_DIR


def _fbank(wave: torch.Tensor) -> torch.Tensor:
    """Kaldi fbank 80 维（与 FunASR WavFrontend 默认参数一致）。"""
    return kaldi.fbank(
        wave.unsqueeze(0) * 32768.0,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        snip_edges=True,
    )


def _lfr(feat: torch.Tensor, m: int = 7, n: int = 6) -> torch.Tensor:
    """Low Frame Rate：每 n 帧堆叠 m 帧（左补齐），得到 80*m 维。"""
    left = (m - 1) // 2
    feat = torch.cat([feat[0:1].repeat(left, 1), feat], dim=0)
    t_lfr = int(np.ceil(feat.shape[0] / n))
    rows = []
    for i in range(t_lfr):
        s = i * n
        e = min(s + m, feat.shape[0])
        chunk = feat[s:e]
        if e - s < m:
            chunk = torch.cat([chunk, chunk[-1:].repeat(m - (e - s), 1)], dim=0)
        rows.append(chunk.reshape(-1))
    return torch.stack(rows)


def _load_cmvn(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    txt = open(path, encoding="utf-8").read()
    vals: dict[str, torch.Tensor] = {}
    for key in ("AddShift", "Rescale"):
        seg = txt[txt.index("<%s>" % key):]
        lb, rb = seg.index("["), seg.index("]")
        vals[key] = torch.tensor([float(x) for x in seg[lb + 1:rb].split()])
    return vals["AddShift"], vals["Rescale"]


class SenseVoiceASR(ASRProvider):
    name = "sensevoice-small-onnx"

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL_DIR,
        *,
        num_threads: int = 8,
        language: int = 3,   # 3 = zh
        textnorm: int = 14,  # 14 = withitn（数字/时间归一化）
        use_cmvn: bool = True,
        hotwords: list[str] | None = None,
    ):
        self.model_dir = model_dir
        self.language = language
        self.textnorm = textnorm
        self.use_cmvn = use_cmvn
        self.hotwords = list(hotwords or [])
        self._tokens: list[str] = json.load(
            open(os.path.join(model_dir, "tokens.json"), encoding="utf-8"))
        self._shift, self._scale = _load_cmvn(os.path.join(model_dir, "am.mvn"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(
            os.path.join(model_dir, "model_quant.onnx"), so,
            providers=["CPUExecutionProvider"],
        )

    def set_hotwords(self, hotwords: list[str]) -> None:
        # SenseVoice 是 CTC 贪心解码，无原生热词偏置；保留接口，后处理纠错在其它项目做。
        self.hotwords = list(hotwords)

    def _decode_ids(self, ids: np.ndarray, valid_len: int) -> str:
        out: list[str] = []
        prev = -1
        for i in ids[:valid_len]:
            i = int(i)
            if i != prev and i != 0:
                tok = self._tokens[i]
                if not tok.startswith("<|"):
                    out.append(tok)
            prev = i
        return "".join(out).replace("▁", " ").strip()

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        wave = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        t0 = time.perf_counter()
        feat = _lfr(_fbank(wave))
        if self.use_cmvn:
            feat = (feat + self._shift) * self._scale
        sp = feat.unsqueeze(0).numpy().astype(np.float32)
        feed = {
            "speech": sp,
            "speech_lengths": np.array([sp.shape[1]], dtype=np.int32),
            "language": np.array([self.language], dtype=np.int32),
            "textnorm": np.array([self.textnorm], dtype=np.int32),
        }
        logits, lens = self._sess.run(None, feed)
        text = self._decode_ids(logits[0].argmax(-1), int(lens[0]))
        return text, (time.perf_counter() - t0) * 1000.0


class MockASR(ASRProvider):
    """测试用：按脚本逐次返回文本（最后一次作为 final）。"""

    name = "mock-asr"

    def __init__(self, script: list[str] | None = None, default_text: str = "测试文本"):
        self.script = list(script or [])
        self.default_text = default_text
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        self.calls += 1
        text = self.script.pop(0) if self.script else self.default_text
        return text, 3.0
