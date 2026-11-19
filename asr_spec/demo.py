# -*- coding: utf-8 -*-
"""命令行入口：自测 / 离线回放评估 / 实时回放演示 / 麦克风演示。

用法（在项目根目录下）：

  python -m asr_spec.demo selftest
  python -m asr_spec.demo wav  --ids u01,u06,u11,u16 --ratio 0.75 --baseline
  python -m asr_spec.demo wav  --dir examples/wav --ratio 0.6 --json runs/x.json
  python -m asr_spec.demo multiturn
  python -m asr_spec.demo live --ids u01,u06 --gap 2.4
  python -m asr_spec.demo mic  --seconds 20
  python -m asr_spec.demo devices

模型与可执行文件路径见 asr_spec/config.py（可用环境变量覆盖）。
LLM 服务：默认自动拉起 llama-server（--server auto），也可 --server external 复用已有服务，
或 --server none 用 MockLLM 做纯逻辑自测。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

from .asr_sensevoice import MockASR, SenseVoiceASR
from .audio import FFMPEG, FfmpegMicSource, FileAudioSource, list_input_devices, load_wav
from .config import EXAMPLES_WAV_DIR, RUNS_DIR as RUNS_DIR_CFG
from .interfaces import SAMPLE_RATE
from .llm_llama import LlamaServerProvider, MockLLM
from .personalization import JsonPersonalization
from .pipeline import PipelineConfig, SessionContext, SpeculativePipeline
from .prompts import build_hist, build_system, prompt_prefix

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WAV_DIR = EXAMPLES_WAV_DIR
STREAM_JSON = os.path.join(HERE, "examples", "records.json")
RUNS_DIR = RUNS_DIR_CFG


# --------------------------------------------------------------------------
# 公共构造
# --------------------------------------------------------------------------
def make_llm(args):
    if args.server == "none":
        return MockLLM(), False
    prov = LlamaServerProvider(
        base_url=args.base_url,
        auto_start=(args.server == "auto"),
        ctx_size=args.ctx_size,
        parallel=1,
    )
    if args.server == "auto":
        t0 = time.perf_counter()
        prov.ensure_running()
        print(f"[llm] server ready in {time.perf_counter() - t0:.1f}s -> {args.base_url}", flush=True)
    elif not prov.health():
        raise SystemExit(f"[llm] {args.base_url} 未运行；用 --server auto 自动拉起")
    return prov, args.server != "none"


def make_pipeline(llm, args) -> SpeculativePipeline:
    pers = JsonPersonalization(args.persona)
    asr = SenseVoiceASR(num_threads=args.threads)
    print("[asr] warming up SenseVoice ...", flush=True)
    t0 = time.perf_counter()
    asr.warmup()
    print(f"[asr] ready in {time.perf_counter() - t0:.2f}s", flush=True)
    cfg = PipelineConfig(
        trigger_ratio=getattr(args, "ratio", None),
        measure_baseline=getattr(args, "baseline", False),
        n_predict=args.n_predict,
    )
    return SpeculativePipeline(asr, llm, personalization=pers, config=cfg)


def load_records() -> dict:
    if not os.path.exists(STREAM_JSON):
        return {}
    with open(STREAM_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {r["id"]: r for r in data.get("records", [])}


# --------------------------------------------------------------------------
# selftest（无 GPU / 无网络）
# --------------------------------------------------------------------------
def cmd_selftest(args) -> int:
    from .gate import is_accept
    from .pipeline import merge_window

    cfg = PipelineConfig(measure_baseline=True, trigger_ratio=0.6)
    audio = np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32)

    # 1) 接受路径（3s 音频 @1s 间隔 -> 3 次 partial + 1 次 final）
    asr = MockASR(["你好啊", "你好啊，能帮我看看", "你好啊，能帮我看看今天天气",
                   "你好啊，能帮我看看今天天气如何？"])
    llm = MockLLM(spec_text="今天北京晴，气温 18 到 26 度。", final_text="今天北京晴。")
    pipe = SpeculativePipeline(asr, llm, config=cfg)
    r_ok = pipe.run_utterance(audio, final_text="你好啊，能帮我看看今天天气如何？")
    assert r_ok.accepted, r_ok.events
    assert r_ok.answer == "今天北京晴，气温 18 到 26 度。", r_ok.answer
    assert r_ok.metrics["saved_ms"] is not None

    # 2) 拒绝路径
    asr2 = MockASR(["你好啊", "你好啊，能帮我看看", "你好啊，能帮我看看今天节目"])
    llm2 = MockLLM(spec_text="你想看什么节目？", final_text="今天北京晴。")
    pipe2 = SpeculativePipeline(asr2, llm2, config=cfg)
    r_bad = pipe2.run_utterance(audio, final_text="你好啊，能帮我看看今天天气如何？")
    assert not r_bad.accepted, r_bad.events
    assert r_bad.answer == "今天北京晴。"

    # 3) gate / prompt / merge 单点
    assert is_accept("今天天气如何？", "今天天气如何")
    assert is_accept("今天天气如何", "今天天气如")
    assert not is_accept("今天天气如何？", "今天节目怎么样")
    assert merge_window("你好啊", "你好啊，今天天气") == "你好啊，今天天气"

    print("selftest OK: accept/reject/gate/merge 全部通过")
    return 0


# --------------------------------------------------------------------------
# 离线回放评估（真实 ASR + 真实 9B）
# --------------------------------------------------------------------------
def cmd_wav(args) -> int:
    records = load_records()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    llm, owned = make_llm(args)
    try:
        pipe = make_pipeline(llm, args)
        rows = []
        for rid in ids:
            wav_path = os.path.join(args.dir, rid + ".wav")
            if not os.path.exists(wav_path):
                print(f"[wav] missing {wav_path}", flush=True)
                continue
            rec = records.get(rid, {})
            audio = load_wav(wav_path)
            ctx = rec.get("context") or []
            oracle_final = rec.get("full_text") or None
            t0 = time.perf_counter()
            res = pipe.run_utterance(audio, context=ctx, final_text=oracle_final)
            wall = time.perf_counter() - t0
            rows.append({"id": rid, "res": res, "wall_s": round(wall, 2)})
            ev = {e["type"]: e for e in res.events}
            trig = repr(res.trigger.text) if res.trigger else None
            spec_done = ev.get("spec_done", {})
            print(
                f"[{rid}] {res.audio_sec:.2f}s accept={res.accepted} "
                f"trig={trig} "
                f"spec_total={spec_done.get('total_ms')}ms "
                f"cache_n={spec_done.get('cache_n')} "
                f"answer_ready={res.metrics['answer_ready_after_audio_ms']:+.0f}ms "
                f"baseline={res.metrics['baseline_after_audio_ms']}ms "
                f"saved={res.metrics['saved_ms']}ms",
                flush=True,
            )
            print(f"    final={res.final_text!r}", flush=True)
            print(f"    answer={res.answer!r}", flush=True)
        if not rows:
            print("[wav] no records processed")
            return 1

        acc = [r["res"] for r in rows if r["res"].accepted]
        rej = [r["res"] for r in rows if not r["res"].accepted]
        saved = [res.metrics["saved_ms"] for res in acc if res.metrics["saved_ms"] is not None]
        print("\n== summary ==", flush=True)
        print(f"accepted {len(acc)}/{len(rows)}")
        if saved:
            print(f"accepted avg saved: {statistics.mean(saved):.0f}ms")
        if rej:
            deltas = [res.metrics["answer_ready_after_audio_ms"] for res in rej]
            base = [res.metrics["baseline_after_audio_ms"] for res in rej if res.metrics["baseline_after_audio_ms"] is not None]
            line = f"rejected avg answer_ready: {statistics.mean(deltas):.0f}ms"
            if base:
                line += f" (baseline {statistics.mean(base):.0f}ms)"
            print(line)
        spec = [e for r in rows for e in r["res"].events if e["type"] == "spec_done"]
        if spec:
            print(f"spec: mean total {statistics.mean([x['total_ms'] for x in spec]):.0f}ms, "
                  f"mean cache_n {statistics.mean([x['cache_n'] or 0 for x in spec]):.0f}")

        os.makedirs(RUNS_DIR, exist_ok=True)
        out = args.json or os.path.join(RUNS_DIR, "wav_eval.json")
        payload = {
            "config": {"ratio": args.ratio, "baseline": args.baseline, "n_predict": args.n_predict},
            "rows": [{"id": r["id"], "accepted": r["res"].accepted, "metrics": r["res"].metrics,
                      "events": r["res"].events, "final": r["res"].final_text,
                      "trigger": r["res"].trigger.text if r["res"].trigger else None,
                      "answer": r["res"].answer} for r in rows],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"saved {out}")
        return 0
    finally:
        if owned:
            llm.close()


# --------------------------------------------------------------------------
# 跨轮缓存验证：turn2 的 prefill 应复用 turn1 全部 token
# --------------------------------------------------------------------------
def cmd_multiturn(args) -> int:
    llm, owned = make_llm(args)
    try:
        pipe = make_pipeline(llm, args)
        audio1 = load_wav(os.path.join(DEFAULT_WAV_DIR, "u06.wav"))
        audio2 = load_wav(os.path.join(DEFAULT_WAV_DIR, "u31.wav"))
        r1 = pipe.run_utterance(audio1, context=[], final_text=None)
        print(f"[turn1] final={r1.final_text!r} answer={r1.answer!r}", flush=True)
        ctx = ["用户：" + r1.final_text, "助手：" + r1.answer]
        r2 = pipe.run_utterance(audio2, context=ctx, final_text=None)
        ev2 = {e["type"]: e for e in r2.events}
        pre2 = ev2.get("prefill_done", {})
        print(f"[turn2] cache_n={pre2.get('cache_n')} prompt_n={pre2.get('prompt_n')} "
              f"prefill_total={pre2.get('total_ms')}ms", flush=True)
        print(f"[turn2] spec cache_n={ev2.get('spec_done', {}).get('cache_n')}", flush=True)
        if (pre2.get("cache_n") or 0) < 10:
            print("[turn2] WARN: 跨轮缓存未命中，检查 build_hist 的 assistant 模板是否一致", flush=True)
            return 2
        return 0
    finally:
        if owned:
            llm.close()


# --------------------------------------------------------------------------
# 实时模式离线回放（VAD + 流式，1:1 音频时钟；不需要人工说话）
# --------------------------------------------------------------------------
def concat_ids(ids, gap_sec: float = 0.8, out_path: str | None = None,
               head_sec: float = 0.5, tail_sec: float = 2.0) -> np.ndarray:
    parts: list[np.ndarray] = [np.zeros(int(head_sec * SAMPLE_RATE), dtype=np.float32)]
    for i, rid in enumerate(ids):
        a = load_wav(os.path.join(DEFAULT_WAV_DIR, rid + ".wav"))
        if i:
            parts.append(np.zeros(int(gap_sec * SAMPLE_RATE), dtype=np.float32))
        parts.append(a)
    parts.append(np.zeros(int(tail_sec * SAMPLE_RATE), dtype=np.float32))
    audio = np.concatenate(parts)
    if out_path:
        from scipy.io import wavfile
        wavfile.write(out_path, SAMPLE_RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return audio


def cmd_live(args) -> int:
    llm, owned = make_llm(args)
    try:
        os.makedirs(RUNS_DIR, exist_ok=True)
        input_wav = os.path.join(RUNS_DIR, "live_demo_input.wav")
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        audio = concat_ids(ids, args.gap, input_wav, args.head, args.tail)
        pipe = make_pipeline(llm, args)
        src = FileAudioSource(input_wav, realtime=True, speed=args.speed)
        print(f"[live] ids={ids} total={len(audio) / SAMPLE_RATE:.2f}s realtime=x{args.speed} "
              f"energy={args.energy} partial={args.partial}s end_silence={args.end_silence * 0.2:.1f}s",
              flush=True)
        t0 = time.perf_counter()
        turns = 0
        out_path = os.path.join(RUNS_DIR, "live_session_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
        with open(out_path, "w", encoding="utf-8") as log:
            for ev in pipe.run_live(src, energy_threshold=args.energy,
                                    end_silence_chunks=args.end_silence,
                                    partial_interval=args.partial):
                now = time.perf_counter() - t0
                if isinstance(ev, dict):
                    kind = ev.get("type")
                    if kind == "partial":
                        print(f"  {now:5.2f}s partial: {ev['text']!r}", flush=True)
                    elif kind == "trigger":
                        print(f"  {now:5.2f}s TRIGGER: {ev['text']!r}", flush=True)
                    elif kind == "spec_done":
                        print(f"  {now:5.2f}s spec: {ev['text']!r} ({ev['total_ms']}ms, cache_n={ev['cache_n']})",
                              flush=True)
                    elif kind == "speech_start":
                        print(f"  {now:5.2f}s speech_start (t_audio={ev.get('t_audio')})", flush=True)
                    log.write(json.dumps({"t": round(now, 3), **ev}, ensure_ascii=False) + "\n")
                else:
                    turns += 1
                    m = ev.metrics
                    ready = m.get("answer_ready_after_stop_ms", m.get("answer_ready_after_audio_ms"))
                    print(f"  {now:5.2f}s TURN accept={ev.accepted} final={ev.final_text!r} "
                          f"answer={ev.answer!r} saved={m.get('saved_ms')}ms "
                          f"ready_after_stop={ready}ms "
                          f"baseline_after_stop={m.get('baseline_after_stop_ms')}ms", flush=True)
                    log.write(json.dumps({"t": round(now, 3), "turn": {
                        "audio_sec": ev.audio_sec, "final": ev.final_text, "answer": ev.answer,
                        "accepted": ev.accepted, "metrics": ev.metrics, "events": ev.events,
                    }}, ensure_ascii=False) + "\n")
        print(f"[live] session log: {out_path} (turns={turns})", flush=True)
        return 0
    finally:
        if owned:
            llm.close()


# --------------------------------------------------------------------------
# 实时麦克风
# --------------------------------------------------------------------------
def cmd_mic(args) -> int:
    llm, owned = make_llm(args)
    src = FfmpegMicSource(device=args.device)
    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = os.path.join(RUNS_DIR, "mic_session_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S"))
    t_start = time.perf_counter()
    turns = 0
    try:
        pipe = make_pipeline(llm, args)
        print(f"[mic] listening on {args.device!r} for {args.seconds}s ... (说话即可)", flush=True)
        with open(out_path, "w", encoding="utf-8") as log:
            for ev in pipe.run_live(src, energy_threshold=args.energy,
                                    end_silence_chunks=args.end_silence):
                now = time.perf_counter() - t_start
                if isinstance(ev, dict):
                    kind = ev.get("type")
                    if kind == "partial":
                        print(f"  {now:5.2f}s partial: {ev['text']!r}", flush=True)
                    elif kind == "trigger":
                        print(f"  {now:5.2f}s TRIGGER: {ev['text']!r}", flush=True)
                    elif kind == "spec_done":
                        print(f"  {now:5.2f}s spec done: {ev['text']!r} ({ev['total_ms']}ms)", flush=True)
                    elif kind == "turn_done":
                        print(f"  {now:5.2f}s TURN accept={ev['accepted']} answer={ev['answer']!r}", flush=True)
                    log.write(json.dumps({"t": round(now, 3), **ev}, ensure_ascii=False) + "\n")
                else:  # TurnResult
                    turns += 1
                    log.write(json.dumps({"t": round(time.perf_counter() - t_start, 3), "turn": {
                        "audio_sec": ev.audio_sec, "final": ev.final_text, "answer": ev.answer,
                        "accepted": ev.accepted, "metrics": ev.metrics, "events": ev.events,
                    }}, ensure_ascii=False) + "\n")
                if time.perf_counter() - t_start > args.seconds:
                    break
    except KeyboardInterrupt:
        print("\n[mic] interrupted", flush=True)
    finally:
        src.close()
        if owned:
            llm.close()
    print(f"[mic] session log: {out_path} (turns={turns})", flush=True)
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ASR + tiny 投机预填充 实时闭环（本体）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--server", choices=["auto", "external", "none"], default="auto",
                        help="auto=自动拉起 8402 llama-server；external=用已运行服务；none=MockLLM")
        sp.add_argument("--base-url", default="http://127.0.0.1:8402")
        sp.add_argument("--ctx-size", type=int, default=8192)
        sp.add_argument("--threads", type=int, default=8, help="SenseVoice CPU 线程数")
        sp.add_argument("--n-predict", type=int, default=48)
        sp.add_argument("--baseline", action="store_true", help="额外测冷基线（更慢）")
        sp.add_argument("--persona", default=None, help="个性化 JSON 配置路径（hotwords/persona_card/lora_adapter）")

    sp = sub.add_parser("selftest", help="无 GPU 自测")
    add_common(sp)
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("wav", help="离线回放评估（真实组件）")
    add_common(sp)
    sp.add_argument("--dir", default=DEFAULT_WAV_DIR)
    sp.add_argument("--ids", default="u01,u06,u11,u16,u26,u29,u31,u34")
    sp.add_argument("--ratio", type=float, default=0.6, help="oracle 触发比例（评估模式）")
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_wav)

    sp = sub.add_parser("multiturn", help="验证跨轮 prefix 缓存")
    add_common(sp)
    sp.set_defaults(func=cmd_multiturn)

    sp = sub.add_parser("live", help="实时模式回放（VAD+流式，1:1 音频时钟，不需要说话）")
    add_common(sp)
    sp.add_argument("--ids", default="u01,u06")
    sp.add_argument("--gap", type=float, default=0.8, help="句间静音秒数")
    sp.add_argument("--speed", type=float, default=1.0, help="回放倍速（1.0=真实时间）")
    sp.add_argument("--energy", type=float, default=0.010)
    sp.add_argument("--partial", type=float, default=0.5, help="partial 更新间隔秒")
    sp.add_argument("--end-silence", type=int, default=8, help="停顿判定（0.2s 块数，8=1.6s）")
    sp.add_argument("--head", type=float, default=0.5, help="回放音频头部静音秒")
    sp.add_argument("--tail", type=float, default=2.0, help="回放音频尾部静音秒（供 VAD 收尾）")
    sp.set_defaults(func=cmd_live)

    sp = sub.add_parser("mic", help="实时麦克风演示")
    add_common(sp)
    sp.add_argument("--seconds", type=float, default=20.0)
    sp.add_argument("--device", default="Microphone (High Definition Audio Device)")
    sp.add_argument("--energy", type=float, default=0.010)
    sp.add_argument("--end-silence", type=int, default=3, help="停顿判定（0.2s 块数，3=0.6s）")
    sp.set_defaults(func=cmd_mic)

    sp = sub.add_parser("devices", help="列出 DirectShow 音频输入设备")
    sp.set_defaults(func=lambda args: (print(list_input_devices()), 0)[1])

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
