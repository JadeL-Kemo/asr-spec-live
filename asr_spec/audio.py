# -*- coding: utf-8 -*-
"""音频输入：文件（可实时回放）与麦克风（ffmpeg DirectShow 采集）。

为什么用 ffmpeg 采集而不是 sounddevice/pyaudio：
  本机环境没有安装这两个包（pip 走 HTTPS 不可用），而 ffmpeg 已就绪，
  `-f dshow -i audio=<device>` + `-f s16le` 可以零依赖拿到 16k mono PCM。
"""
from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.io import wavfile

from .config import FFMPEG  # noqa: F401  (再导出，供 demo 等模块使用)
from .interfaces import SAMPLE_RATE, AudioSource


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def load_wav(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """读取 wav -> float32 mono [-1,1]。只支持 PCM wav（实验语料即如此）。"""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if sr != target_sr:
        # 线性重采样（实验用；生产建议 ffmpeg/sox 预转换）
        n = int(round(len(data) * target_sr / sr))
        data = np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)
    return data


# --------------------------------------------------------------------------
# 音频源
# --------------------------------------------------------------------------
class FileAudioSource(AudioSource):
    """wav 文件 -> 音频块流。realtime=True 时按真实时间节奏产出（用于实时演示）。"""

    def __init__(self, path: str, *, realtime: bool = False, speed: float = 1.0):
        self.audio = load_wav(path)
        self.realtime = realtime
        self.speed = max(0.001, speed)

    def chunks(self, chunk_sec: float = 0.2) -> Iterator[tuple[float, np.ndarray]]:
        n = int(chunk_sec * SAMPLE_RATE)
        t0 = time.perf_counter()
        for start in range(0, len(self.audio), n):
            chunk = self.audio[start:start + n]
            t_end = (start + len(chunk)) / SAMPLE_RATE
            if self.realtime:
                target = t_end / self.speed
                dt = target - (time.perf_counter() - t0)
                if dt > 0:
                    time.sleep(dt)
            yield t_end, chunk


def list_input_devices() -> str:
    """列出 DirectShow 输入设备（用于选择麦克风）。"""
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, text=True, timeout=60,
    )
    return proc.stderr


class FfmpegMicSource(AudioSource):
    """麦克风实时采集：ffmpeg dshow -> stdout raw PCM s16le。

    注意：设备名必须与 `list_input_devices()` 输出中的 "(audio)" 条目一致。
    """

    def __init__(self, device: str = "Microphone (High Definition Audio Device)",
                 *, sample_rate: int = SAMPLE_RATE, read_timeout_s: float = 10.0):
        self.device = device
        self.sample_rate = sample_rate
        self.read_timeout_s = read_timeout_s
        self._proc: subprocess.Popen | None = None
        self._stderr_lines: deque[str] = deque(maxlen=40)

    def start(self) -> None:
        if self._proc is not None:
            return
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-f", "dshow", "-i", f"audio={self.device}",
            "-ac", "1", "-ar", str(self.sample_rate),
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def drain() -> None:
            assert self._proc and self._proc.stderr
            for raw in self._proc.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self._stderr_lines.append(line)

        threading.Thread(target=drain, daemon=True).start()

    def chunks(self, chunk_sec: float = 0.2) -> Iterator[tuple[float, np.ndarray]]:
        self.start()
        assert self._proc and self._proc.stdout
        bytes_per_chunk = int(chunk_sec * self.sample_rate) * 2
        t0 = time.perf_counter()
        leftover = b""
        while True:
            data = self._proc.stdout.read(bytes_per_chunk)  # blocking
            if not data:
                err = "\n".join(self._stderr_lines) or "(no stderr)"
                raise RuntimeError(f"ffmpeg 采集结束（设备不可用或被打断）：\n{err}")
            buf = leftover + data
            usefull = len(buf) // 2 * 2
            leftover, buf = buf[usefull:], buf[:usefull]
            pcm = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
            t_end = time.perf_counter() - t0
            yield t_end, pcm

    def close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:  # pragma: no cover
                self._proc.kill()
            self._proc = None


# --------------------------------------------------------------------------
# 语音端点检测（能量 VAD，块级）
# --------------------------------------------------------------------------
@dataclass
class SegmentEvent:
    kind: str           # "start" | "audio" | "end"
    audio: np.ndarray   # start: 预卷+当前块；audio: 当前块；end: 整段语音
    t_end: float        # 音频时钟（相对源开始）


class SpeechSegmenter:
    """能量阈值 + 连续块计数的轻量端点检测。

    参数以"块"为单位（chunk_sec 由调用方决定，默认 0.2s）：
    - start_speech_chunks: 连续多少个"有能量"块算说话开始（2 -> 0.4s）
    - end_silence_chunks: 连续多少个"无能量"块算说完（3 -> 0.6s）
    - pre_roll_chunks: 开始前保留多少块音频（2 -> 0.4s，避免吞掉首字）
    - min_speech_chunks: 少于该块数视为噪声丢弃（3 -> 0.6s）
    - max_utterance_sec: 强制切分上限
    """

    def __init__(
        self,
        *,
        energy_threshold: float = 0.010,
        start_speech_chunks: int = 2,
        end_silence_chunks: int = 3,
        pre_roll_chunks: int = 2,
        min_speech_chunks: int = 3,
        max_utterance_sec: float = 30.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.energy_threshold = energy_threshold
        self.start_speech_chunks = start_speech_chunks
        self.end_silence_chunks = end_silence_chunks
        self.pre_roll_chunks = pre_roll_chunks
        self.min_speech_chunks = min_speech_chunks
        self.max_utterance_sec = max_utterance_sec
        self.sample_rate = sample_rate
        self._t = 0.0
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._utt: list[np.ndarray] = []
        self._utt_sec = 0.0
        self._speech_chunks = 0
        self._pre: deque[np.ndarray] = deque(maxlen=max(1, pre_roll_chunks))

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((chunk.astype(np.float32)) ** 2)))

    def feed(self, chunk: np.ndarray) -> list[SegmentEvent]:
        events: list[SegmentEvent] = []
        if chunk.size == 0:
            return events
        self._t += len(chunk) / self.sample_rate
        energy = self._rms(chunk)
        has_speech = energy >= self.energy_threshold

        if not self._in_speech:
            self._pre.append(chunk)
            self._speech_run = self._speech_run + 1 if has_speech else 0
            if self._speech_run >= self.start_speech_chunks:
                self._in_speech = True
                self._silence_run = 0
                self._utt = list(self._pre)
                self._utt_sec = sum(len(c) for c in self._utt) / self.sample_rate
                self._speech_chunks = self._speech_run
                events.append(SegmentEvent("start", np.concatenate(self._utt), self._t))
            return events

        # 说话中
        self._utt.append(chunk)
        self._utt_sec += len(chunk) / self.sample_rate
        if has_speech:
            self._speech_chunks += 1
            self._silence_run = 0
        else:
            self._silence_run += 1

        force_end = self._utt_sec >= self.max_utterance_sec
        if self._silence_run >= self.end_silence_chunks or force_end:
            audio = np.concatenate(self._utt) if self._utt else np.zeros(0, dtype=np.float32)
            if self._silence_run >= self.end_silence_chunks:
                # 去掉尾部静音块（保留 pre-roll 的语音部分）
                keep = len(self._utt) - self._silence_run
                audio = np.concatenate(self._utt[: max(0, keep)]) if keep > 0 else audio
            valid = self._speech_chunks >= self.min_speech_chunks
            self._in_speech = False
            self._speech_run = 0
            self._silence_run = 0
            self._utt = []
            self._utt_sec = 0.0
            self._speech_chunks = 0
            self._pre.clear()
            if valid:
                events.append(SegmentEvent("end", audio, self._t))
        else:
            events.append(SegmentEvent("audio", chunk, self._t))
        return events
