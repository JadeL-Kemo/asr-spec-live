"""asr_spec.webdemo — 可视化实时演示平台（浏览器 + 本地 HTTP 服务）。

零额外依赖：stdlib http.server 提供轮询 JSON API，前端是单页 HTML。
两种模式：
  - replay：把若干条语料拼成一段 wav，在浏览器里 1:1 播放（同时后端按真实时钟喂 VAD/ASR），
            适合无麦克风环境演示"边说边预测"。
  - mic：浏览器采集麦克风 → 16k PCM → 后端队列（可选，需要浏览器授权）。
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .audio import FileAudioSource, SAMPLE_RATE
from .asr_sensevoice import SenseVoiceASR
from .interfaces import AudioSource as _AudioSource
from .llm_llama import LlamaServerProvider
from .pipeline import PipelineConfig, SpeculativePipeline

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
PROJECT_DIR = HERE.parent
RUNS_WEB = PROJECT_DIR / "runs" / "web"
DEFAULT_IDS = ["u01", "u06", "u11", "u16", "u26", "u29", "u31", "u34"]


def _num(cfg: dict, key: str, default: float) -> float:
    """容忍 None / 空串 / NaN / 非数字，统一回退到默认值（前端可能传 null）。"""
    v = cfg.get(key, default)
    if isinstance(v, bool):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _int(cfg: dict, key: str, default: int) -> int:
    return int(_num(cfg, key, float(default)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# 音频源：麦克风队列
# ---------------------------------------------------------------------------
class QueueAudioSource(_AudioSource):
    """把浏览器送来的 16k float32 块重新切分成固定 0.2s 块。"""

    def __init__(self, chunk_sec: float = 0.2):
        self.chunk_sec = chunk_sec
        self._q: queue.Queue = queue.Queue()
        self.closed = False

    def feed(self, pcm: np.ndarray) -> None:
        if not self.closed:
            self._q.put(pcm)

    def close(self) -> None:
        self.closed = True
        self._q.put(None)

    def chunks(self, chunk_sec: float = 0.2):  # noqa: D401
        n = int(chunk_sec * SAMPLE_RATE)
        buf = np.zeros(0, dtype=np.float32)
        t = 0.0
        while True:
            try:
                item = self._q.get(timeout=60)
            except queue.Empty:
                return
            if item is None:
                if len(buf):
                    yield round(t + len(buf) / SAMPLE_RATE, 3), buf
                return
            buf = np.concatenate([buf, item]) if len(buf) else item
            while len(buf) >= n:
                chunk, buf = buf[:n], buf[n:]
                t += n / SAMPLE_RATE
                yield round(t, 3), chunk


# ---------------------------------------------------------------------------
# 全局共享模型（跨会话复用，避免每次重启 llama-server）
# ---------------------------------------------------------------------------
class Runtime:
    def __init__(self, base_url: str, ctx_size: int, threads: int):
        self.base_url = base_url
        self.ctx_size = ctx_size
        self.threads = threads
        self._asr: SenseVoiceASR | None = None
        self._llm: LlamaServerProvider | None = None
        self._lock = threading.Lock()

    def asr(self) -> SenseVoiceASR:
        with self._lock:
            if self._asr is None:
                a = SenseVoiceASR(num_threads=self.threads)
                a.warmup()
                self._asr = a
            return self._asr

    def llm(self) -> LlamaServerProvider:
        with self._lock:
            if self._llm is None:
                p = LlamaServerProvider(base_url=self.base_url, auto_start=True,
                                        ctx_size=self.ctx_size, parallel=1)
                p.ensure_running()
                self._llm = p
            return self._llm


# ---------------------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------------------
def _first_sentence(text: str) -> str:
    for i, ch in enumerate(text):
        if ch in "。！？!?；;":
            return text[: i + 1]
    return text


def _diff(trigger: str, final: str) -> dict:
    """trigger vs final 的字符级差异（用公共前缀/后缀表示），用于展示"修正"。"""
    a, b = trigger or "", final or ""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    j = 0
    while j < len(a) - i and j < len(b) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    return {
        "prefix": a[:i],
        "removed": a[i: len(a) - j],
        "added": b[i: len(b) - j],
        "suffix": a[len(a) - j:] if j else "",
    }


def turn_payload(idx: int, res) -> dict:
    m = dict(res.metrics or {})
    spec = res.spec
    spec_answer = spec.text.strip() if spec else ""
    first = (spec.first_sentence_text or _first_sentence(spec_answer)) if spec else ""
    payload = {
        "idx": idx,
        "audio_sec": round(res.audio_sec, 2),
        "trigger": {"text": res.trigger.text, "t_s": res.trigger.t_s} if res.trigger else None,
        "spec": None if spec is None else {
            "answer": spec_answer,
            "first_sentence": first,
            "total_ms": round(spec.total_ms, 1),
            "first_sentence_ms": None if spec.first_sentence_ms is None else round(spec.first_sentence_ms, 1),
            "cache_n": spec.cache_n,
            "prompt_n": spec.prompt_n,
        },
        "final_text": res.final_text,
        "accepted": bool(res.accepted),
        "answer": res.answer,
        "correction": _diff(res.trigger.text, res.final_text) if res.trigger else None,
        "metrics": m,
        "saved_ms": m.get("first_sentence_saved_ms", m.get("saved_ms")),
        "fs_after_stop_ms": m.get("first_sentence_after_stop_ms"),
        "baseline_fs_after_stop_ms": m.get("baseline_first_sentence_after_stop_ms"),
    }
    return payload


def stats_payload(turns: list[dict]) -> dict:
    n = len(turns)
    acc = [t for t in turns if t["accepted"]]
    saved = [t["saved_ms"] for t in acc if t.get("saved_ms") is not None]
    spec_ms = [t["spec"]["total_ms"] for t in turns if t.get("spec")]
    cache = [t["spec"]["cache_n"] for t in turns if t.get("spec") and t["spec"].get("cache_n")]
    fs = [t["fs_after_stop_ms"] for t in turns if t.get("fs_after_stop_ms") is not None]
    base = [t["baseline_fs_after_stop_ms"] for t in turns
            if t.get("baseline_fs_after_stop_ms") is not None]
    return {
        "turns": n,
        "accepted": len(acc),
        "hit_rate": round(len(acc) / n, 3) if n else None,
        "avg_saved_ms": round(sum(saved) / len(saved), 1) if saved else None,
        "avg_spec_ms": round(sum(spec_ms) / len(spec_ms), 1) if spec_ms else None,
        "avg_cache_n": round(sum(cache) / len(cache)) if cache else None,
        "avg_fs_after_stop_ms": round(sum(fs) / len(fs), 1) if fs else None,
        "avg_baseline_fs_after_stop_ms": round(sum(base) / len(base), 1) if base else None,
    }


class Session:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            if getattr(self, "_mic", None) is not None:
                try:
                    self._mic.close()
                except Exception:
                    pass
            self._gen = getattr(self, "_gen", 0) + 1
            self.status = "idle"
            self.mode = "replay"
            self.message = ""
            self.started_at = None
            self.audio_url = None
            self.config: dict = {}
            self.live = {"partial": "", "partial_t": None, "partial_latency": None,
                         "trigger": None, "spec": None, "phase": "idle"}
            self.turns: list[dict] = []
            self.events: list[dict] = []
            self.stats: dict = {}
            self._stop = threading.Event()
            self._thread: threading.Thread | None = None
            self._mic: QueueAudioSource | None = None

    # -- 统一状态快照 --------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "mode": self.mode,
                "message": self.message,
                "started_at": self.started_at,
                "audio_url": self.audio_url,
                "config": dict(self.config),
                "live": dict(self.live),
                "turns": list(self.turns),
                "stats": dict(self.stats),
                "events": self.events[-80:],
            }

    def add_event(self, ev: dict) -> None:
        with self.lock:
            ev = dict(ev)
            ev["_t"] = round(time.time(), 3)
            self.events.append(ev)
            if ev["type"] == "partial":
                self.live["partial"] = ev.get("text", "")
                self.live["partial_t"] = ev.get("t_s")
                self.live["partial_latency"] = ev.get("latency_ms")
                self.live["phase"] = "listening"
            elif ev["type"] == "trigger":
                self.live["trigger"] = {"text": ev.get("text"), "t_s": ev.get("t_s"),
                                        "retrigger": ev.get("retrigger", False)}
                self.live["spec"] = None
                self.live["phase"] = "speculating"
            elif ev["type"] == "spec_done":
                self.live["spec"] = {"answer": ev.get("text", ""),
                                     "total_ms": ev.get("total_ms"),
                                     "cache_n": ev.get("cache_n"),
                                     "prompt_n": ev.get("prompt_n")}
            elif ev["type"] == "speech_start":
                self.live.update({"partial": "", "partial_t": None, "trigger": None,
                                  "spec": None, "phase": "listening"})
            elif ev["type"] == "turn_done":
                self.live["phase"] = "gated"
            elif ev["type"] in ("partial_error", "spec_error"):
                self.live["phase"] = "error"

    def consume(self, items, gen: int) -> None:
        idx = 0
        try:
            for item in items:
                if gen != self._gen or self.status == "stopped":
                    break
                if isinstance(item, dict):
                    self.add_event(item)
                else:  # TurnResult
                    idx += 1
                    payload = turn_payload(idx, item)
                    with self.lock:
                        if gen != self._gen or self.status == "stopped":
                            break
                        self.turns.append(payload)
                        self.stats = stats_payload(self.turns)
                        self.live["phase"] = "turn_done"
            with self.lock:
                if gen == self._gen and self.status != "stopped":
                    self.status = "finished"
                    self.live["phase"] = "finished"
        except Exception as e:  # pragma: no cover
            with self.lock:
                self.status = "error"
                self.message = f"{type(e).__name__}: {e}"
                self.live["phase"] = "error"

    # -- replay --------------------------------------------------------
    def start_replay(self, cfg: dict) -> None:
        self.reset()
        gen = self._gen
        with self.lock:
            self.status = "preparing"
            self.mode = "replay"
            self.config = cfg
            self.message = "构建回放音频…"
        t = threading.Thread(target=self._run_replay, args=(cfg, gen), daemon=True)
        self._thread = t
        t.start()

    def _run_replay(self, cfg: dict, gen: int) -> None:
        try:
            from .demo import concat_ids

            ids = [str(x) for x in (cfg.get("ids") or DEFAULT_IDS[:2]) if str(x).strip()]
            RUNS_WEB.mkdir(parents=True, exist_ok=True)
            name = f"web_{int(time.time())}.wav"
            wav_path = str(RUNS_WEB / name)
            gap = _clamp(_num(cfg, "gap", 2.4), 0.2, 10.0)
            concat_ids(ids, gap, wav_path, head_sec=0.5, tail_sec=2.5)

            asr = self.rt.asr()
            llm = self.rt.llm()
            pipe = SpeculativePipeline(
                asr, llm, config=PipelineConfig(
                    trigger_ratio=None,
                    measure_baseline=bool(cfg.get("baseline", True)),
                    n_predict=_int(cfg, "n_predict", 48),
                ),
            )
            src = FileAudioSource(wav_path, realtime=True,
                                  speed=_clamp(_num(cfg, "speed", 1.0), 0.25, 4.0))
            endpoint = _clamp(_num(cfg, "endpoint", 1.6), 0.2, 5.0)
            partial = _clamp(_num(cfg, "partial", 0.5), 0.2, 3.0)
            with self.lock:
                self.status = "running"
                self.audio_url = f"/audio/{name}"
                self.message = f"回放 {len(ids)} 条，{os.path.getsize(wav_path) / 32000:.1f}s 音频"
            time.sleep(0.8)  # 给浏览器 audio 元素留出起播时间
            self.consume(pipe.run_live(
                src, end_silence_chunks=max(1, int(round(endpoint / 0.2))),
                partial_interval=partial), gen)
        except Exception as e:  # pragma: no cover
            with self.lock:
                self.status = "error"
                self.message = f"{type(e).__name__}: {e}"

    # -- mic -----------------------------------------------------------
    def start_mic(self, cfg: dict) -> None:
        self.reset()
        gen = self._gen
        self._mic = QueueAudioSource()
        with self.lock:
            self.status = "running"
            self.mode = "mic"
            self.config = cfg
            self.message = "等待浏览器麦克风数据…"
        t = threading.Thread(target=self._run_mic, args=(cfg, gen), daemon=True)
        self._thread = t
        t.start()

    def _run_mic(self, cfg: dict, gen: int) -> None:
        try:
            asr = self.rt.asr()
            llm = self.rt.llm()
            pipe = SpeculativePipeline(
                asr, llm, config=PipelineConfig(
                    trigger_ratio=None,
                    measure_baseline=bool(cfg.get("baseline", False)),
                    n_predict=_int(cfg, "n_predict", 48),
                ),
            )
            endpoint = _clamp(_num(cfg, "endpoint", 1.0), 0.2, 5.0)
            partial = _clamp(_num(cfg, "partial", 0.5), 0.2, 3.0)
            self.consume(pipe.run_live(
                self._mic, end_silence_chunks=max(1, int(round(endpoint / 0.2))),
                partial_interval=partial,
                energy_threshold=_clamp(_num(cfg, "energy", 0.010), 1e-5, 0.5)), gen)
        except Exception as e:  # pragma: no cover
            with self.lock:
                self.status = "error"
                self.message = f"{type(e).__name__}: {e}"

    def stop(self) -> None:
        with self.lock:
            if self.status in ("running", "preparing"):
                self.status = "stopped"
                self._gen += 1
                self.message = "已停止"
        if self._mic is not None:
            self._mic.close()

    def mic_feed(self, pcm_i16: bytes) -> None:
        if self._mic is None:
            return
        a = np.frombuffer(pcm_i16, dtype=np.int16).astype(np.float32) / 32768.0
        if len(a):
            self._mic.feed(a)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "asr-spec-webdemo/1.0"
    session: Session
    rt: Runtime

    def log_message(self, fmt, *args):  # 静默
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(self.session.snapshot())
        elif self.path.startswith("/audio/"):
            name = os.path.basename(self.path)
            p = RUNS_WEB / name
            if not p.exists():
                self._json({"error": "not found"}, 404)
                return
            self._send(200, p.read_bytes(), "audio/wav")
        elif self.path == "/api/health":
            self._json({"ok": True, "status": self.session.status})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            if self.path == "/api/start":
                try:
                    cfg = json.loads(raw or b"{}")
                except (ValueError, UnicodeDecodeError) as e:
                    self._json({"error": f"bad json: {e}"}, 400)
                    return
                if not isinstance(cfg, dict):
                    self._json({"error": "config must be a JSON object"}, 400)
                    return
                if self.session.status in ("running", "preparing"):
                    self._json({"error": "已有会话在运行"}, 409)
                    return
                mode = cfg.get("mode", "replay")
                if mode == "mic":
                    self.session.start_mic(cfg)
                else:
                    self.session.start_replay(cfg)
                self._json({"ok": True, "mode": mode})
            elif self.path == "/api/stop":
                self.session.stop()
                self._json({"ok": True})
            elif self.path == "/api/reset":
                self.session.reset()
                self._json({"ok": True})
            elif self.path == "/api/mic":
                self.session.mic_feed(raw)
                self._json({"ok": True, "bytes": len(raw)})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def make_server(port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    rt = Runtime(base_url="http://127.0.0.1:8402", ctx_size=8192, threads=8)
    session = Session(rt)
    cls = type("BoundHandler", (Handler,), {"session": session, "rt": rt})
    httpd = ThreadingHTTPServer((host, port), cls)
    httpd.daemon_threads = True
    return httpd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="asr-spec-live 可视化实时演示平台")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    httpd = make_server(args.port, args.host)
    url = f"http://{args.host}:{args.port}/"
    print(f"[web] 打开 {url}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
