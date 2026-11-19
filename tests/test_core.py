# -*- coding: utf-8 -*-
"""本体单元测试（无 GPU、无网络、无 LLM 服务）。

运行：cd work/asr-spec-live && python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_spec.asr_sensevoice import MockASR
from asr_spec.audio import FileAudioSource, SpeechSegmenter
from asr_spec.gate import is_accept, normalize
from asr_spec.interfaces import (
    NullPersonalization, SAMPLE_RATE, StaticPersonalization,
)
from asr_spec.llm_llama import MockLLM
from asr_spec.personalization import JsonPersonalization
from asr_spec.pipeline import PipelineConfig, SpeculativePipeline, merge_window
from asr_spec.prompts import NO_THINK, build_hist, build_system, prompt_final, prompt_prefix


class TestGate(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("你好，世界！ 123 Abc"), "你好世界123abc")

    def test_accept_exact(self):
        self.assertTrue(is_accept("今天天气如何？", "今天天气如何"))

    def test_accept_prefix_extra(self):
        self.assertTrue(is_accept("今天天气怎么样啊", "今天天气怎么"))
        self.assertFalse(is_accept("今天天气怎么样啊", "今天"))  # 差别太大 -> 拒绝
        self.assertFalse(is_accept("今天天气如何？", "今天节目怎么样"))  # 错词 -> 拒绝

    def test_reject_quantity_change(self):
        # 多出来的尾巴含数量/单位词 -> 语义可能变了，拒绝
        self.assertFalse(is_accept("算了，改成7点半吧。", "算了，改成7点。"))
        self.assertFalse(is_accept("把空调调到26度", "把空调调到2"))
        self.assertTrue(is_accept("那后天呢会降温吗？", "那后天呢会降温。"))  # 纯语气词 -> 接受

    def test_reject_short_trigger(self):
        self.assertFalse(is_accept("你好", "你"))


class TestPrompts(unittest.TestCase):
    def test_prefix_ends_with_user_marker(self):
        p = prompt_prefix(build_system(), build_hist([]))
        self.assertTrue(p.endswith("<|im_start|>user\n"))

    def test_final_extends_prefix(self):
        p = prompt_prefix(build_system(), build_hist(["用户：你好", "助手：你好呀"]))
        f = prompt_final(p, "今天天气")
        self.assertTrue(f.startswith(p))
        self.assertTrue(f.endswith(NO_THINK))
        self.assertIn("今天天气<|im_end|>", f)

    def test_hist_assistant_uses_no_think(self):
        h = build_hist(["用户：你好", "助手：你好呀"])
        self.assertIn("<|im_start|>user\n你好<|im_end|>", h)
        self.assertIn(NO_THINK + "你好呀<|im_end|>", h)

    def test_persona_in_system(self):
        s = build_system("用户喜欢简洁回答")
        self.assertIn("用户画像", s)
        self.assertNotIn("用户画像", build_system(""))


class TestSegmenter(unittest.TestCase):
    def _tone(self, sec: float, amp: float = 0.2) -> np.ndarray:
        t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
        return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_detects_utterance(self):
        seg = SpeechSegmenter(energy_threshold=0.01)
        silence = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
        tail = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)  # > end_silence_chunks*0.2
        audio = np.concatenate([silence, self._tone(2.0), tail])
        chunk = int(0.2 * SAMPLE_RATE)
        kinds = []
        utt = None
        for i in range(0, len(audio), chunk):
            for ev in seg.feed(audio[i:i + chunk]):
                kinds.append(ev.kind)
                if ev.kind == "end":
                    utt = ev.audio
        self.assertIn("start", kinds)
        self.assertIn("end", kinds)
        self.assertIsNotNone(utt)
        self.assertGreater(len(utt) / SAMPLE_RATE, 1.6)

    def test_ignores_short_noise(self):
        seg = SpeechSegmenter(energy_threshold=0.01, min_speech_chunks=3)
        audio = np.concatenate([self._tone(0.2), np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)])
        chunk = int(0.2 * SAMPLE_RATE)
        ends = []
        for i in range(0, len(audio), chunk):
            ends += [ev for ev in seg.feed(audio[i:i + chunk]) if ev.kind == "end"]
        self.assertEqual(ends, [])


class TestPipelineMock(unittest.TestCase):
    def _audio(self, sec: float = 3.0) -> np.ndarray:
        return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)

    def test_accept_path(self):
        # 3s 音频 @1s 间隔 -> 3 次 partial + 1 次 final，脚本需要 4 条
        asr = MockASR(["你好啊", "你好啊，能帮我看看", "你好啊，能帮我看看今天天气",
                       "你好啊，能帮我看看今天天气如何？"])
        llm = MockLLM(spec_text="北京今天晴，18 到 26 度。", final_text="北京今天晴。")
        pipe = SpeculativePipeline(asr, llm, config=PipelineConfig(measure_baseline=True, trigger_ratio=0.6))
        res = pipe.run_utterance(self._audio(), final_text="你好啊，能帮我看看今天天气如何？")
        self.assertTrue(res.accepted, res.events)
        self.assertEqual(res.answer, "北京今天晴，18 到 26 度。")
        self.assertIsNotNone(res.trigger)
        self.assertIsNotNone(res.metrics["saved_ms"])
        self.assertTrue(any(e["type"] == "accept" for e in res.events))

    def test_reject_path(self):
        asr = MockASR(["你好啊", "你好啊，能帮我看看", "你好啊，能帮我看看今天节目"])
        llm = MockLLM(spec_text="你想看什么节目？", final_text="北京今天晴。")
        pipe = SpeculativePipeline(asr, llm, config=PipelineConfig(measure_baseline=True, trigger_ratio=0.6))
        res = pipe.run_utterance(self._audio(), final_text="你好啊，能帮我看看今天天气如何？")
        self.assertFalse(res.accepted)
        self.assertEqual(res.answer, "北京今天晴。")
        self.assertTrue(any(e["type"] == "reject" for e in res.events))

    def test_no_trigger_still_answers(self):
        asr = MockASR(["嗯", "嗯嗯", "嗯嗯嗯"])  # 全都不足 6 字
        llm = MockLLM(spec_text="好的。", final_text="好的。")  # 无投机 -> 只走一次 complete
        pipe = SpeculativePipeline(asr, llm, config=PipelineConfig(measure_baseline=False, trigger_ratio=0.6))
        res = pipe.run_utterance(self._audio(2.0), final_text="嗯嗯嗯")
        self.assertFalse(res.accepted)
        self.assertEqual(res.answer, "好的。")

    def test_live_mode_yields_turn(self):
        # 合成音频：0.4s 静音 + 2.0s 正弦 + 1.0s 静音
        t = np.arange(int(2.0 * SAMPLE_RATE)) / SAMPLE_RATE
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.concatenate([np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32), tone,
                                np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)])
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            from scipy.io import wavfile
            wavfile.write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
            asr = MockASR(["你好", "你好啊，今天天气怎么", "你好啊，今天天气怎么样"])
            llm = MockLLM(spec_text="北京晴。", final_text="北京晴。")
            pipe = SpeculativePipeline(asr, llm, config=PipelineConfig(measure_baseline=False))
            source = FileAudioSource(tmp.name, realtime=False)
            turns = [x for x in pipe.run_live(source) if not isinstance(x, dict)]
            self.assertEqual(len(turns), 1)
            self.assertTrue(turns[0].answer)
        finally:
            os.unlink(tmp.name)

    def test_personalization_persona_changes_system(self):
        asr = MockASR(["你好啊", "你好啊今天天气"])
        llm = MockLLM(spec_text="晴。", final_text="晴。")
        pers = StaticPersonalization(hotwords=["小语", "豆包"], persona_card="用户偏好简短回答")
        pipe = SpeculativePipeline(asr, llm, personalization=pers,
                                   config=PipelineConfig(measure_baseline=False, trigger_ratio=0.9))
        self.assertIn("用户画像", pipe.system)
        self.assertEqual(pipe.hotwords, ["小语", "豆包"])
        res = pipe.run_utterance(self._audio(2.0), final_text="你好啊今天天气怎么样")
        self.assertIsNotNone(res)

    def test_json_personalization(self):
        cfg = {"hotwords": ["杭州", "浙大"], "persona_card": "住在杭州", "lora_adapter": None}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False)
            path = f.name
        try:
            pers = JsonPersonalization(path)
            self.assertEqual(list(pers.hotwords()), ["杭州", "浙大"])
            self.assertEqual(pers.persona_card(), "住在杭州")
            self.assertIsNone(pers.lora_adapter())
        finally:
            os.unlink(path)

    def test_null_personalization(self):
        pers = NullPersonalization()
        self.assertEqual(list(pers.hotwords()), [])
        self.assertEqual(pers.persona_card(), "")
        self.assertIsNone(pers.lora_adapter())


class WebDemoTest(unittest.TestCase):
    """可视化平台的纯函数（不启动 HTTP 服务、不加载模型）。"""

    def test_first_sentence(self):
        from asr_spec.webdemo import _first_sentence
        self.assertEqual(_first_sentence("你好啊，天气如何？"), "你好啊，天气如何？")
        self.assertEqual(_first_sentence("第一句。第二句。"), "第一句。")
        self.assertEqual(_first_sentence("没有标点"), "没有标点")

    def test_diff(self):
        from asr_spec.webdemo import _diff
        d = _diff("我也是时后会骑士团了。", "我也是时后回骑士团了。")
        self.assertEqual(d["prefix"], "我也是时后")
        self.assertEqual(d["removed"], "会")
        self.assertEqual(d["added"], "回")
        self.assertEqual(d["suffix"], "骑士团了。")
        same = _diff("完全一致", "完全一致")
        self.assertEqual((same["removed"], same["added"]), ("", ""))

    def test_stats_payload(self):
        from asr_spec.webdemo import stats_payload
        turns = [
            {"idx": 1, "accepted": True, "spec": {"total_ms": 500, "cache_n": 220},
             "saved_ms": 800, "fs_after_stop_ms": 1200, "baseline_fs_after_stop_ms": 2000},
            {"idx": 2, "accepted": False, "spec": {"total_ms": 520, "cache_n": 255},
             "saved_ms": None, "fs_after_stop_ms": 1500, "baseline_fs_after_stop_ms": 2100},
        ]
        s = stats_payload(turns)
        self.assertEqual(s["turns"], 2)
        self.assertEqual(s["accepted"], 1)
        self.assertAlmostEqual(s["hit_rate"], 0.5)
        self.assertEqual(s["avg_saved_ms"], 800.0)
        self.assertEqual(s["avg_spec_ms"], 510.0)
        self.assertEqual(s["avg_cache_n"], 238)
        self.assertEqual(s["avg_fs_after_stop_ms"], 1350.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
