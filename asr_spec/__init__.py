# -*- coding: utf-8 -*-
"""ASR + tiny 投机预填充 实时闭环（本体）。

阶段 1 实现：SenseVoice ONNX 流式识别 + 9B llama-server 前缀缓存投机 + 接受/回滚。
个性化能力通过 `PersonalizationProvider` 独立接口接入（实现在其它项目）。
"""
from .interfaces import (
    ASRProvider, AudioSource, LLMProvider, LLMResult, NullPersonalization, NullTTS,
    PersonalizationProvider, SAMPLE_RATE, StaticPersonalization, TranscriptUpdate,
    TriggerInfo, TTSProvider, TurnResult,
)
from .pipeline import PipelineConfig, SessionContext, SpeculativePipeline, merge_window
from .prompts import NO_THINK, SYS, build_hist, build_system, prompt_final, prompt_prefix
from .gate import is_accept, normalize

__all__ = [
    "ASRProvider", "AudioSource", "LLMProvider", "LLMResult", "PersonalizationProvider",
    "TTSProvider", "NullPersonalization", "NullTTS", "StaticPersonalization",
    "SAMPLE_RATE", "TranscriptUpdate", "TriggerInfo", "TurnResult",
    "PipelineConfig", "SessionContext", "SpeculativePipeline", "merge_window",
    "SYS", "NO_THINK", "build_hist", "build_system", "prompt_final", "prompt_prefix",
    "is_accept", "normalize",
]
