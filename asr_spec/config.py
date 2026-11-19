# -*- coding: utf-8 -*-
"""集中配置：模型路径、llama.cpp 可执行文件、ffmpeg、示例音频。

所有路径都可以通过环境变量覆盖，方便在没有 GPU / 不同目录的机器上运行：

  ASR_SPEC_MODELS          模型根目录（默认 <project>/models）
  ASR_SPEC_SENSEVOICE_DIR  SenseVoiceSmall ONNX 目录（默认 <models>/SenseVoiceSmall-onnx）
  ASR_SPEC_LLAMA_EXE       llama-server / llama 可执行文件（默认在 PATH 中查找）
  ASR_SPEC_LLAMA_MODEL     Qwen3.5-9B GGUF 文件（默认 <models>/gguf/Qwen3.5-9B-Q4_K_M.gguf）
  ASR_SPEC_FFMPEG          ffmpeg 可执行文件（默认 PATH 中的 ffmpeg）
  ASR_SPEC_WAV_DIR         示例音频目录（默认 <project>/examples/wav）
  ASR_SPEC_RUNS            运行输出目录（默认 <project>/runs）
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else None


def find_llama_exe() -> str:
    """按 环境变量 -> PATH -> 常见安装位置 查找 llama.cpp 可执行文件。"""
    v = _env("ASR_SPEC_LLAMA_EXE")
    if v:
        return v
    for name in ("llama-server.exe", "llama-server", "llama.exe", "llama"):
        p = shutil.which(name)
        if p:
            return p
    # 也支持把 llama.cpp 解压到项目内 tools/llama/ 下
    for cand in (
        PROJECT_ROOT / "tools" / "llama" / "llama-server.exe",
        PROJECT_ROOT / "tools" / "llama" / "llama.exe",
    ):
        if cand.exists():
            return str(cand)
    return "llama-server"  # 让子进程报错时给出可读提示


MODELS_DIR = Path(_env("ASR_SPEC_MODELS") or (PROJECT_ROOT / "models")).expanduser()
SENSEVOICE_DIR = _env("ASR_SPEC_SENSEVOICE_DIR") or str(MODELS_DIR / "SenseVoiceSmall-onnx")
LLAMA_EXE = find_llama_exe()
LLAMA_MODEL = _env("ASR_SPEC_LLAMA_MODEL") or str(
    MODELS_DIR / "gguf" / "Qwen3.5-9B-Q4_K_M.gguf"
)
FFMPEG = _env("ASR_SPEC_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
EXAMPLES_WAV_DIR = _env("ASR_SPEC_WAV_DIR") or str(PROJECT_ROOT / "examples" / "wav")
RUNS_DIR = _env("ASR_SPEC_RUNS") or str(PROJECT_ROOT / "runs")
LLAMA_LOG_DIR = os.path.join(RUNS_DIR, "logs")
