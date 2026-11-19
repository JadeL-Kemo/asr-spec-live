# -*- coding: utf-8 -*-
"""本体核心接口。

设计原则（与可行性报告 §4/§9/§10 一致）：
1. 本体只依赖接口，不依赖具体实现；换 ASR / LLM / 个性化 / TTS 不改流水线。
2. **个性化能力是独立接口**（PersonalizationProvider）：本仓库只消费它，
   具体实现（声纹、画像、LoRA、热词挖掘）在其它项目中单独验证与实现。
3. 所有回调携带时间戳（音频时钟），便于离线回放与线上观测共用同一套指标。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------
# 数据对象
# --------------------------------------------------------------------------
@dataclass
class TranscriptUpdate:
    """一次流式识别更新。t_s 为相对当前 utterance 开始的音频时钟。"""

    seq: int
    t_s: float
    text: str
    latency_ms: float
    audio_sec: float


@dataclass
class LLMResult:
    text: str = ""
    first_ms: float = 0.0
    total_ms: float = 0.0
    prompt_ms: float | None = None
    prompt_n: int | None = None
    cache_n: int | None = None
    predicted_n: int | None = None
    #: 从请求发出到"第一个完整断句"（句末标点）出现的耗时；用于字幕/首句播报口径。
    first_sentence_ms: float | None = None
    first_sentence_text: str = ""
    ok: bool = True
    error: str | None = None


@dataclass
class TriggerInfo:
    """投机触发点：在音频时钟 t_s 处用 text 触发一次投机生成。"""

    text: str
    t_s: float
    seq: int


@dataclass
class TurnResult:
    """一轮完整对话的可观测结果（评估与线上观测共用）。"""

    audio_sec: float
    partials: list[TranscriptUpdate] = field(default_factory=list)
    trigger: TriggerInfo | None = None
    spec: LLMResult | None = None
    final_text: str = ""
    final_asr_ms: float = 0.0
    accepted: bool = False
    answer: str = ""
    events: list[dict] = field(default_factory=list)
    baseline: LLMResult | None = None
    metrics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 音频源
# --------------------------------------------------------------------------
class AudioSource(ABC):
    """音频源：产出 (t_end_s, float32 mono 16k chunk)。"""

    sample_rate = SAMPLE_RATE

    @abstractmethod
    def chunks(self, chunk_sec: float = 0.2) -> Iterator[tuple[float, np.ndarray]]:
        ...

    def close(self) -> None:  # pragma: no cover - 默认无资源
        pass


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------
class ASRProvider(ABC):
    """流式识别接口：每次传入"截止目前的整段音频"，返回文本。

    约定：实现无状态（不依赖调用顺序），这样累计重转写 / 滑窗重转写
    两种策略都可以复用它。
    """

    name: str = "asr"

    def warmup(self) -> None:
        self.transcribe(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """返回 (text, latency_ms)。audio 为 float32 [-1,1], 16k mono。"""

    def set_hotwords(self, hotwords: Sequence[str]) -> None:
        """可选：热词/词表偏置。默认不支持（实现方自行覆盖）。"""


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
class LLMProvider(ABC):
    """LLM 接口：prefill 预热前缀缓存，complete 生成。

    slot 概念保留给支持多 slot 的后端；单 slot 后端忽略即可。
    """

    name: str = "llm"

    @abstractmethod
    def prefill(self, prompt: str, slot: int = 0) -> LLMResult:
        """只做预填充（不生成内容），让后续以该 prompt 为前缀的请求命中缓存。"""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        n_predict: int = 48,
        slot: int = 0,
        cache_prompt: bool = True,
    ) -> LLMResult:
        ...

    def warmup(self) -> None:  # pragma: no cover - 默认无需
        pass

    def close(self) -> None:  # pragma: no cover - 默认无需
        pass


# --------------------------------------------------------------------------
# 个性化（独立接口：实现放在其它项目）
# --------------------------------------------------------------------------
class PersonalizationProvider(ABC):
    """个性化能力契约。本体只依赖这里的方法；不关心它怎么算出结果。

    约束（保证缓存命中，见 docs/ARCHITECTURE.md）：
    - persona_card / lora_adapter 必须**会话级稳定**：同一用户同一会话内不变化。
    - 返回内容只能在"稳定前缀"（system）与训练数据侧使用，不得在轮次之间
      改写历史文本或在前缀中插入动态内容。
    """

    @abstractmethod
    def hotwords(self) -> Sequence[str]:
        """本用户专有词（人名/地名/APP/术语），供 ASR 偏置或后处理纠错。"""

    def persona_card(self) -> str:
        """会话级人物画像卡（≤1–2KB），用于 system 稳定前缀。默认无。"""
        return ""

    def lora_adapter(self) -> str | None:
        """会话级加载的 LoRA adapter 路径（GGUF）。默认无。

        约定：会话开始时加载一次；会话内不得热切换（切换会使 prompt cache 失效）。
        """
        return None

    def session_start(self, user_id: str) -> None:  # pragma: no cover
        pass

    def session_end(self) -> None:  # pragma: no cover
        pass

    def rewrite_partial(self, text: str) -> str:
        """可选：对 partial 做个性化改写（专名纠正等）。默认原样返回。"""
        return text


class NullPersonalization(PersonalizationProvider):
    def hotwords(self) -> Sequence[str]:
        return []


class StaticPersonalization(PersonalizationProvider):
    """测试/演示用：显式给一组静态配置。"""

    def __init__(self, *, hotwords: Sequence[str] = (), persona_card: str = "", lora_adapter: str | None = None):
        self._hotwords = list(hotwords)
        self._persona_card = persona_card
        self._lora_adapter = lora_adapter

    def hotwords(self) -> Sequence[str]:
        return self._hotwords

    def persona_card(self) -> str:
        return self._persona_card

    def lora_adapter(self) -> str | None:
        return self._lora_adapter


# --------------------------------------------------------------------------
# TTS（阶段 3，占位接口）
# --------------------------------------------------------------------------
class TTSProvider(ABC):
    name: str = "tts"

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """返回音频字节（实现方决定编码；本体只负责送达）。"""

    def close(self) -> None:  # pragma: no cover
        pass


class NullTTS(TTSProvider):
    def synthesize(self, text: str) -> bytes:
        return b""
