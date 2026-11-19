# -*- coding: utf-8 -*-
"""本体流水线：流式 ASR -> 触发投机 -> 安全预填充 -> 接受/回滚。

两种运行模式共用同一套核心（`_run_core` / `_gate_and_answer`）：

- `run_utterance(audio, final_text=None, ...)`：离线/评估模式。
  给出整段音频（可选真值 final_text 用作 oracle 触发），按音频时钟模拟
  "边说边识别"，输出与实时一致的 TurnResult（时间线用音频时钟表示）。
- `run_live(source, ...)`：实时模式。音频源按真实时间进块，能量 VAD 切分
  utterance，边说边更新 partial，检测到停顿后 final flush。

缓存策略（与 REPORT.md §10.2 一致）：
  prefill(prefix) 在说话开始时并行写入 slot 缓存；trigger 时 spec prompt 是
  prefix 的严格扩展 -> 命中；final 通常是一段新后缀 -> 由接受/回滚决定后续。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from .audio import SpeechSegmenter
from .gate import is_accept, normalize
from .interfaces import (
    ASRProvider, AudioSource, LLMProvider, LLMResult, NullPersonalization,
    PersonalizationProvider, SAMPLE_RATE, TranscriptUpdate, TriggerInfo, TurnResult,
)
from .prompts import build_hist, build_system, prompt_final, prompt_prefix

_ = None  # 占位，保持模块级常量区结构（历史常量已全部内联）


# --------------------------------------------------------------------------
# 配置与策略
# --------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    partial_interval_sec: float = 1.0     # 每 1s 音频触发一次 partial 更新
    max_partial_audio_sec: float = 8.0    # 超过则改用滑窗+合并（长句）
    n_predict: int = 48
    trigger_min_chars: int = 6
    trigger_ratio: float | None = None    # 评估模式（oracle）：partial 覆盖 final 的比例
    measure_baseline: bool = True         # 是否测冷基线（线上模式建议 False）
    prefill_slot: int = 0
    spec_slot: int = 0
    max_utterance_sec: float = 30.0


@dataclass
class SessionContext:
    """会话历史（只追加）。assistant 文本在 build_hist 中带 no-think 缓冲。"""

    history: list[str] = field(default_factory=list)

    def append_turn(self, user_text: str, answer: str) -> None:
        self.history.append("用户：" + user_text)
        self.history.append("助手：" + answer)


def merge_window(stable: str, hyp: str) -> str:
    """长句滑窗合并：保留已确定前缀，吸收窗口内新内容。

    与可行性实验一致：只在匹配块长度 >= 3 时拼接，避免 difflib 在短文本上
    把句首错误地覆盖掉（实验里 L2 win6 acc=0.0 就是这个问题）。
    """
    from difflib import SequenceMatcher

    if not stable:
        return hyp
    sm = SequenceMatcher(None, stable, hyp, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 3 and b.a > 0]
    if not blocks:
        return hyp if len(hyp) > len(stable) else stable
    b = max(blocks, key=lambda x: x.a + x.size)
    return stable[: b.a + b.size] + hyp[b.b + b.size:]


# --------------------------------------------------------------------------
# 流水线
# --------------------------------------------------------------------------
class SpeculativePipeline:
    def __init__(
        self,
        asr: ASRProvider,
        llm: LLMProvider,
        *,
        personalization: PersonalizationProvider | None = None,
        config: PipelineConfig | None = None,
    ):
        self.asr = asr
        self.llm = llm
        self.pers = personalization or NullPersonalization()
        self.cfg = config or PipelineConfig()
        self.system = build_system(self.pers.persona_card())
        self.hotwords = list(self.pers.hotwords())
        if self.hotwords:
            try:
                self.asr.set_hotwords(self.hotwords)
            except Exception:
                pass

    # ---------------- 内部工具 ----------------
    def _prefill_async(self, prefix: str, box: dict) -> threading.Thread:
        def run() -> None:
            try:
                box["result"] = self.llm.prefill(prefix, slot=self.cfg.prefill_slot)
            except Exception as e:  # pragma: no cover
                box["error"] = repr(e)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th

    def _speculate(self, prefix: str, trigger_text: str) -> LLMResult:
        prompt = prompt_final(prefix, trigger_text)
        return self.llm.complete(prompt, n_predict=self.cfg.n_predict, slot=self.cfg.spec_slot)

    def _should_trigger(self, text: str, prev_text: str, final_norm_len: int | None,
                        audio_sec: float, paused: bool = False) -> bool:
        nt = len(normalize(text))
        if nt < self.cfg.trigger_min_chars:
            return False
        if final_norm_len is not None:
            ratio = self.cfg.trigger_ratio if self.cfg.trigger_ratio is not None else 0.6
            return nt >= ratio * final_norm_len
        # 在线模式：只认"用户停下来了"的信号，避免 ASR 分块自带的句号造成过早触发。
        if paused:
            return True
        if prev_text and normalize(text) and normalize(text) == normalize(prev_text):
            return True
        return False

    @staticmethod
    def _recent_rms(audio: np.ndarray, win_sec: float = 0.6) -> float:
        n = max(1, int(win_sec * SAMPLE_RATE))
        seg = audio[-n:]
        return float(np.sqrt(np.mean(seg.astype(np.float32) ** 2))) if len(seg) else 0.0

    def _gate_and_answer(
        self,
        *,
        prefix: str,
        audio_sec: float,
        final_text: str,
        final_asr_ms: float,
        spec: LLMResult | None,
        trigger: TriggerInfo | None,
        last_text: str,
        events: list[dict],
        measure_baseline: bool,
    ) -> tuple[bool, str, LLMResult | None, LLMResult | None, dict]:
        trigger_text = trigger.text if trigger else last_text
        accepted = bool(trigger and spec and spec.ok and is_accept(final_text, trigger_text))
        regen: LLMResult | None = None
        first_sentence_ready_s: float
        if accepted:
            assert spec is not None and trigger is not None
            answer = spec.text
            answer_ready_s = trigger.t_s + spec.total_ms / 1000.0
            first_sentence_ready_s = trigger.t_s + (
                spec.first_sentence_ms if spec.first_sentence_ms is not None else spec.total_ms
            ) / 1000.0
            events.append({
                "type": "accept",
                "trigger_text": trigger_text,
                "final_text": final_text,
                "spec_total_ms": round(spec.total_ms, 1),
                "spec_first_ms": round(spec.first_ms, 1),
                "spec_first_sentence_ms": None if spec.first_sentence_ms is None else round(spec.first_sentence_ms, 1),
                "spec_first_sentence": spec.first_sentence_text,
                "cache_n": spec.cache_n,
                "prompt_n": spec.prompt_n,
            })
        else:
            regen = self.llm.complete(
                prompt_final(prefix, final_text), n_predict=self.cfg.n_predict,
                slot=self.cfg.spec_slot, cache_prompt=True,
            )
            answer = regen.text
            answer_ready_s = audio_sec + final_asr_ms / 1000.0 + regen.total_ms / 1000.0
            first_sentence_ready_s = audio_sec + final_asr_ms / 1000.0 + (
                regen.first_sentence_ms if regen.first_sentence_ms is not None else regen.total_ms
            ) / 1000.0
            events.append({
                "type": "reject",
                "trigger_text": trigger_text,
                "final_text": final_text,
                "reason": "no_trigger" if not trigger else ("spec_failed" if not spec or not spec.ok else "gate_mismatch"),
                "regen_total_ms": round(regen.total_ms, 1),
                "regen_first_sentence_ms": None if regen.first_sentence_ms is None else round(regen.first_sentence_ms, 1),
                "cache_n": regen.cache_n,
                "prompt_n": regen.prompt_n,
            })

        baseline: LLMResult | None = None
        baseline_ready_s: float | None = None
        baseline_first_sentence_ready_s: float | None = None
        if measure_baseline:
            baseline = self.llm.complete(
                prompt_final(prefix, final_text), n_predict=self.cfg.n_predict,
                slot=self.cfg.spec_slot, cache_prompt=False,
            )
            baseline_ready_s = audio_sec + final_asr_ms / 1000.0 + baseline.total_ms / 1000.0
            baseline_first_sentence_ready_s = audio_sec + final_asr_ms / 1000.0 + (
                baseline.first_sentence_ms if baseline.first_sentence_ms is not None else baseline.total_ms
            ) / 1000.0
            events.append({
                "type": "baseline",
                "total_ms": round(baseline.total_ms, 1),
                "first_sentence_ms": None if baseline.first_sentence_ms is None else round(baseline.first_sentence_ms, 1),
                "prompt_n": baseline.prompt_n,
                "cache_n": baseline.cache_n,
            })

        metrics = {
            "audio_sec": round(audio_sec, 3),
            "answer_ready_after_audio_ms": round((answer_ready_s - audio_sec) * 1000.0, 1),
            "baseline_after_audio_ms": None if baseline_ready_s is None
            else round((baseline_ready_s - audio_sec) * 1000.0, 1),
            "saved_ms": None if baseline_ready_s is None
            else round((baseline_ready_s - answer_ready_s) * 1000.0, 1),
            "first_sentence_ready_after_audio_ms": round((first_sentence_ready_s - audio_sec) * 1000.0, 1),
            "spec_first_sentence_ms": None if not (spec is not None) or spec.first_sentence_ms is None
            else round(spec.first_sentence_ms, 1),
            "spec_first_sentence": spec.first_sentence_text if spec is not None else "",
            "baseline_first_sentence_after_audio_ms": None if baseline_first_sentence_ready_s is None
            else round((baseline_first_sentence_ready_s - audio_sec) * 1000.0, 1),
            "first_sentence_saved_ms": None if baseline_first_sentence_ready_s is None
            else round((baseline_first_sentence_ready_s - first_sentence_ready_s) * 1000.0, 1),
            "final_asr_ms": round(final_asr_ms, 1),
        }
        return accepted, answer, regen, baseline, metrics

    # ---------------- 离线/评估模式 ----------------
    def run_utterance(
        self,
        audio: np.ndarray,
        *,
        context: Sequence[str] = (),
        final_text: str | None = None,
        turn_id: int = 0,
    ) -> TurnResult:
        audio = np.asarray(audio, dtype=np.float32)
        dur = len(audio) / SAMPLE_RATE
        hist_text = build_hist(context)
        prefix = prompt_prefix(self.system, hist_text)

        events: list[dict] = [{"type": "turn_start", "turn_id": turn_id, "audio_sec": round(dur, 3)}]
        partials: list[TranscriptUpdate] = []
        spec: LLMResult | None = None
        trigger: TriggerInfo | None = None
        final_norm_len = len(normalize(final_text)) if final_text else None
        box: dict = {}
        th = self._prefill_async(prefix, box)

        last_text = ""
        seq = 0
        t = 0.0
        while t < dur - 1e-9:
            t = min(dur, t + self.cfg.partial_interval_sec)
            text, lat = self.asr.transcribe(audio[: int(t * SAMPLE_RATE)])
            text = self.pers.rewrite_partial(text)
            seq += 1
            partials.append(TranscriptUpdate(seq=seq, t_s=round(t, 3), text=text, latency_ms=lat, audio_sec=round(t, 3)))
            events.append({"type": "partial", "t_s": round(t, 3), "text": text, "latency_ms": round(lat, 1)})
            if trigger is None and self._should_trigger(text, last_text, final_norm_len, t):
                trigger = TriggerInfo(text=text, t_s=round(t, 3), seq=seq)
                events.append({"type": "trigger", "t_s": round(t, 3), "text": text})
                th.join(timeout=120)
                spec = self._speculate(prefix, text)
                events.append({
                    "type": "spec_done",
                    "t_s": round(t + spec.total_ms / 1000.0, 3),
                    "text": spec.text,
                    "total_ms": round(spec.total_ms, 1),
                    "first_ms": round(spec.first_ms, 1),
                    "prompt_n": spec.prompt_n,
                    "cache_n": spec.cache_n,
                })
            last_text = text
        th.join(timeout=120)
        pre = box.get("result")
        if pre is not None:
            events.append({
                "type": "prefill_done",
                "total_ms": round(pre.total_ms, 1),
                "prompt_n": pre.prompt_n,
                "cache_n": pre.cache_n,
            })

        final_text_measured, final_lat = self.asr.transcribe(audio)
        events.append({"type": "final", "text": final_text_measured, "asr_ms": round(final_lat, 1)})
        accepted, answer, regen, baseline, metrics = self._gate_and_answer(
            prefix=prefix, audio_sec=dur, final_text=final_text_measured, final_asr_ms=final_lat,
            spec=spec, trigger=trigger, last_text=last_text, events=events,
            measure_baseline=self.cfg.measure_baseline,
        )
        return TurnResult(
            audio_sec=round(dur, 3), partials=partials, trigger=trigger, spec=spec,
            final_text=final_text_measured, final_asr_ms=round(final_lat, 1),
            accepted=accepted, answer=answer, events=events, baseline=baseline,
            metrics=metrics,
        )

    # ---------------- 实时模式 ----------------
    def run_live(
        self,
        source: AudioSource,
        *,
        context: Sequence[str] = (),
        user_id: str = "local",
        energy_threshold: float = 0.010,
        end_silence_chunks: int = 3,
        partial_interval: float | None = None,
        on_event=None,
    ) -> Iterator[TurnResult | dict]:
        """实时流：yield 事件 dict（partial/trigger/accept/reject/turn）或 TurnResult。"""
        self.pers.session_start(user_id)
        hist_lines = list(context)
        pi = partial_interval if partial_interval is not None else self.cfg.partial_interval_sec
        quiet_rms = max(energy_threshold * 0.5, 1e-4)
        segmenter = SpeechSegmenter(energy_threshold=energy_threshold,
                                    max_utterance_sec=self.cfg.max_utterance_sec,
                                    end_silence_chunks=end_silence_chunks)
        parts: list[np.ndarray] = []
        events: list[dict] = []
        partials: list[TranscriptUpdate] = []
        spec: LLMResult | None = None
        trigger: TriggerInfo | None = None
        prefix = ""
        box: dict = {}
        th: threading.Thread | None = None
        last_text = ""
        seq = 0
        last_partial_audio = 0.0
        t_turn0 = 0.0
        t_trig_wall: float | None = None
        t_end_wall: float | None = None

        def emit(ev: dict) -> dict:
            if on_event:
                on_event(ev)
            return ev

        try:
            for t_end, chunk in source.chunks(0.2):
                for ev in segmenter.feed(chunk):
                    if ev.kind == "start":
                        parts = [ev.audio]
                        events = [{"type": "speech_start", "t_audio": round(ev.t_end, 3)}]
                        partials = []
                        spec, trigger, last_text, seq = None, None, "", 0
                        last_partial_audio = 0.0
                        prefix = prompt_prefix(self.system, build_hist(hist_lines))
                        box = {}
                        th = self._prefill_async(prefix, box)
                        t_turn0, t_trig_wall, t_end_wall = time.perf_counter(), None, None
                        yield emit({"type": "speech_start", "t_audio": round(ev.t_end, 3)})
                    elif ev.kind == "audio":
                        parts.append(ev.audio)
                        audio_so_far = np.concatenate(parts) if len(parts) > 1 else parts[0]
                        so_far_sec = len(audio_so_far) / SAMPLE_RATE
                        if so_far_sec - last_partial_audio >= pi:
                            last_partial_audio = so_far_sec
                            text, lat = self.asr.transcribe(audio_so_far)
                            text = self.pers.rewrite_partial(text)
                            seq += 1
                            partials.append(TranscriptUpdate(seq=seq, t_s=round(so_far_sec, 3), text=text,
                                                             latency_ms=lat, audio_sec=round(so_far_sec, 3)))
                            yield emit({"type": "partial", "t_s": round(so_far_sec, 3), "text": text,
                                        "latency_ms": round(lat, 1)})
                            if self._should_trigger(
                                    text, last_text, None, so_far_sec,
                                    paused=self._recent_rms(audio_so_far) < quiet_rms):
                                # 允许"重触发"：先前的触发若只是句中停顿（文本随后变长），
                                # 在下一个停顿用更长文本重新投机，旧结果自然作废。
                                if trigger is not None and len(text) <= len(trigger.text) + 2:
                                    last_text = text
                                    continue
                                trigger = TriggerInfo(text=text, t_s=round(so_far_sec, 3), seq=seq)
                                t_trig_wall = time.perf_counter() - t_turn0
                                yield emit({"type": "trigger", "t_s": round(so_far_sec, 3),
                                            "text": text, "retrigger": spec is not None})
                                if th is not None:
                                    th.join(timeout=120)
                                spec = self._speculate(prefix, text)
                                yield emit({"type": "spec_done", "text": spec.text,
                                            "total_ms": round(spec.total_ms, 1), "cache_n": spec.cache_n,
                                            "prompt_n": spec.prompt_n})
                            last_text = text
                    else:  # end
                        audio = ev.audio
                        dur = len(audio) / SAMPLE_RATE
                        t_end_wall = time.perf_counter() - t_turn0
                        if th is not None:
                            th.join(timeout=120)
                        pre = box.get("result")
                        final_text, final_lat = self.asr.transcribe(audio)
                        accepted, answer, regen, baseline, metrics = self._gate_and_answer(
                            prefix=prefix, audio_sec=dur, final_text=final_text, final_asr_ms=final_lat,
                            spec=spec, trigger=trigger, last_text=last_text, events=events,
                            measure_baseline=self.cfg.measure_baseline,
                        )
                        # 实时模式用墙钟修正指标：以"用户停说"为原点（VAD 停顿终点 - 判定静音时长）。
                        silence_s = end_silence_chunks * 0.2
                        t_stop = max(0.0, t_end_wall - silence_s)
                        metrics["vad_end_wall_s"] = round(t_end_wall, 3)
                        metrics["stop_wall_s"] = round(t_stop, 3)
                        if t_trig_wall is not None:
                            metrics["trigger_wall_s"] = round(t_trig_wall, 3)
                        if trigger is not None and spec is not None and t_trig_wall is not None:
                            ready_after_stop_ms = (t_trig_wall + spec.total_ms / 1000.0 - t_stop) * 1000.0
                            metrics["answer_ready_after_stop_ms"] = round(ready_after_stop_ms, 1)
                            if spec.first_sentence_ms is not None:
                                fs_after_stop = (t_trig_wall + spec.first_sentence_ms / 1000.0 - t_stop) * 1000.0
                                metrics["first_sentence_after_stop_ms"] = round(fs_after_stop, 1)
                            baseline_after_stop_ms = None
                            if baseline is not None:
                                baseline_after_stop_ms = (silence_s + final_lat / 1000.0
                                                          + baseline.total_ms / 1000.0) * 1000.0
                                metrics["baseline_after_stop_ms"] = round(baseline_after_stop_ms, 1)
                                metrics["saved_ms"] = round(baseline_after_stop_ms - ready_after_stop_ms, 1)
                                if baseline.first_sentence_ms is not None:
                                    base_fs = (silence_s + final_lat / 1000.0
                                               + baseline.first_sentence_ms / 1000.0) * 1000.0
                                    metrics["baseline_first_sentence_after_stop_ms"] = round(base_fs, 1)
                                    metrics["first_sentence_saved_ms"] = round(
                                        base_fs - metrics.get("first_sentence_after_stop_ms", base_fs), 1)
                        result = TurnResult(
                            audio_sec=round(dur, 3), partials=partials, trigger=trigger, spec=spec,
                            final_text=final_text, final_asr_ms=round(final_lat, 1),
                            accepted=accepted, answer=answer, events=events, baseline=baseline,
                            metrics=metrics,
                        )
                        if accepted or regen is not None:
                            hist_lines.append("用户：" + final_text)
                            hist_lines.append("助手：" + answer)
                        yield emit({"type": "turn_done", "accepted": accepted, "answer": answer,
                                    "metrics": metrics, "prefill": None if pre is None else {
                                        "total_ms": round(pre.total_ms, 1), "cache_n": pre.cache_n}})
                        yield result
        finally:
            try:
                self.pers.session_end()
            except Exception:  # pragma: no cover
                pass
