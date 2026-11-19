# -*- coding: utf-8 -*-
"""llama.cpp server 客户端：prefix 预填充 + 生成 + 生命周期管理。

关键缓存规则（实测，见 REPORT.md §3）：
- 请求 B 的 prompt 必须以请求 A 的**完整 prompt**（token 级）为前缀，才能命中 A 的缓存；
- 单 slot 足够：先 prefill 稳定前缀（system+历史+user 标记），投机请求在其后追加，
  自然构成严格前缀扩展；
- 生成过 token 的请求会让下一次扩展匹配变差，因此 prefill 用 n_predict=1（只为写入缓存）。

服务生命周期：本类可自起 llama-server（8402，ctx 8192，parallel 1），close() 时终止。
"""
from __future__ import annotations

import json
import os
import subprocess
import time

import requests

from .config import LLAMA_EXE, LLAMA_LOG_DIR, LLAMA_MODEL
from .interfaces import LLMProvider, LLMResult

DEFAULT_LLAMA_EXE = LLAMA_EXE
DEFAULT_MODEL = LLAMA_MODEL
DEFAULT_BASE = "http://127.0.0.1:8402"
DEFAULT_LOG_DIR = LLAMA_LOG_DIR


class LlamaServerProvider(LLMProvider):
    name = "llama-server"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE,
        *,
        llama_exe: str = DEFAULT_LLAMA_EXE,
        model_path: str = DEFAULT_MODEL,
        ctx_size: int = 8192,
        n_gpu_layers: int = 99,
        threads: int = 8,
        parallel: int = 1,
        auto_start: bool = True,
        log_dir: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.llama_exe = llama_exe
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.threads = threads
        self.parallel = parallel
        self.auto_start = auto_start
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self._proc: subprocess.Popen | None = None
        self._spawned = False
        self._log_fh = None

    # ---------------- 生命周期 ----------------
    def health(self) -> bool:
        try:
            return requests.get(self.base_url + "/health", timeout=1.5).status_code == 200
        except Exception:
            return False

    def ensure_running(self, timeout_s: float = 180.0) -> None:
        if self.health():
            return
        if not self.auto_start:
            raise RuntimeError(f"LLM server 未运行且 auto_start=False: {self.base_url}")
        self._spawn()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if self.health():
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"llama-server 启动后退出（code={self._proc.returncode}），见日志 {self._log_path}")
            time.sleep(0.5)
        raise RuntimeError(f"llama-server 启动超时（{timeout_s}s），见日志 {self._log_path}")

    def _spawn(self) -> None:
        port = self.base_url.rsplit(":", 1)[-1]
        self._log_path = os.path.join(self.log_dir, f"llama_{port}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        self._log_fh = open(self._log_path, "w", encoding="utf-8")
        cmd = [
            self.llama_exe, "serve",
            "--model", self.model_path,
            "--host", "127.0.0.1",
            "--port", port,
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--threads", str(self.threads),
            "--parallel", str(self.parallel),
            "--no-warmup",
        ]
        self._proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT)
        self._spawned = True

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=15)
            except Exception:  # pragma: no cover
                self._proc.kill()
        self._proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    # ---------------- 请求 ----------------
    def _request(self, body: dict, timeout_s: float = 300.0) -> LLMResult:
        t0 = time.perf_counter()
        first_ms: float | None = None
        first_sentence_ms: float | None = None
        first_sentence_text = ""
        text_parts: list[str] = []
        timings: dict = {}
        try:
            r = requests.post(self.base_url + "/completion", json=body, stream=True, timeout=timeout_s)
            if r.status_code != 200:
                return LLMResult(ok=False, error=f"HTTP {r.status_code}: {r.text[:300]}",
                                 total_ms=(time.perf_counter() - t0) * 1000)
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                s = line[6:] if line.startswith("data: ") else line
                if s.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                c = obj.get("content") or ""
                if c:
                    if first_ms is None:
                        first_ms = (time.perf_counter() - t0) * 1000
                    text_parts.append(c)
                    if first_sentence_ms is None:
                        joined = "".join(text_parts)
                        for idx, ch in enumerate(joined):
                            if ch in "。！？!?；;":
                                first_sentence_ms = (time.perf_counter() - t0) * 1000
                                first_sentence_text = joined[: idx + 1]
                                break
                if "timings" in obj:
                    timings = obj["timings"]
                if obj.get("stop") and not obj.get("stopped_limit"):
                    break
        except Exception as e:  # pragma: no cover
            return LLMResult(ok=False, error=repr(e), total_ms=(time.perf_counter() - t0) * 1000)
        text = "".join(text_parts).strip()
        if first_sentence_ms is None:
            # 没有句末标点：整段答案视为一句（greedy 输出常省略标点）。
            first_sentence_ms = (time.perf_counter() - t0) * 1000
            first_sentence_text = text
        return LLMResult(
            text=text,
            first_ms=first_ms if first_ms is not None else 0.0,
            total_ms=(time.perf_counter() - t0) * 1000,
            prompt_ms=timings.get("prompt_ms"),
            prompt_n=timings.get("prompt_n"),
            cache_n=timings.get("cache_n"),
            predicted_n=timings.get("predicted_n"),
            first_sentence_ms=first_sentence_ms,
            first_sentence_text=first_sentence_text,
        )

    def prefill(self, prompt: str, slot: int = 0) -> LLMResult:
        body = {
            "prompt": prompt,
            "n_predict": 1,           # 只为把 prompt 写入 slot 缓存
            "temperature": 0.0,
            "top_p": 1.0,
            "cache_prompt": True,
            "id_slot": slot,
            "stop": [],
        }
        return self._request(body)

    def complete(self, prompt: str, *, n_predict: int = 48, slot: int = 0, cache_prompt: bool = True) -> LLMResult:
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "top_p": 1.0,
            "cache_prompt": cache_prompt,
            "id_slot": slot,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        return self._request(body)


class MockLLM(LLMProvider):
    """测试用：不依赖服务，返回固定文本并模拟缓存计数。"""

    name = "mock-llm"

    def __init__(self, spec_text: str = "好的，我帮你看一下。", final_text: str | None = None):
        self.spec_text = spec_text
        self.final_text = final_text if final_text is not None else spec_text
        self.prefill_calls = 0
        self.complete_calls = 0

    def prefill(self, prompt: str, slot: int = 0) -> LLMResult:
        self.prefill_calls += 1
        n = len(prompt) // 2
        return LLMResult(text="", first_ms=1.0, total_ms=8.0, prompt_ms=8.0, prompt_n=n, cache_n=0)

    def complete(self, prompt: str, *, n_predict: int = 48, slot: int = 0, cache_prompt: bool = True) -> LLMResult:
        self.complete_calls += 1
        n = len(prompt) // 2
        if self.complete_calls == 1:
            text = self.spec_text
        else:
            text = self.final_text
        return LLMResult(text=text, first_ms=5.0, total_ms=20.0, prompt_ms=15.0, prompt_n=n,
                         cache_n=max(0, n - 12) if cache_prompt else 0, predicted_n=12)
