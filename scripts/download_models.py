#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 ASR-Spec-Live 运行所需的模型权重（不包含在 Git 仓库中）。

用法:
    python scripts/download_models.py                 # 下载 ASR + LLM
    python scripts/download_models.py --what asr      # 只下载 SenseVoice ONNX
    python scripts/download_models.py --what llm      # 只下载 Qwen3.5-9B GGUF
    python scripts/download_models.py --models-dir models
    python scripts/download_models.py --force         # 重新下载

默认从 ModelScope（国内直连较快）下载；可用 --mirror 换源。
llama.cpp 可执行文件需要单独安装（见 README），本脚本不下载二进制。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_MIRROR = "https://www.modelscope.cn/models"

SENSEVOICE_REPO = "iic/SenseVoiceSmall-onnx"
SENSEVOICE_FILES = [
    ("model_quant.onnx", 241_216_270),
    ("tokens.json", 352_064),
    ("am.mvn", 11_203),
    ("config.yaml", 1_855),
]

LLM_REPO = "unsloth/Qwen3.5-9B-GGUF"
LLM_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
LLM_SIZE = 5_680_522_464
LLM_SUBDIR = "gguf"


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def download(url: str, dest: Path, expected: int | None, force: bool = False) -> bool:
    if dest.exists() and not force:
        got = dest.stat().st_size
        if expected is None or got == expected:
            print(f"[skip] {dest}  ({_fmt(got)})")
            return True
        if got > expected:
            print(f"[warn] {dest} 大小异常，重新下载")
        else:
            print(f"[resume] {dest} 已有 {_fmt(got)}，继续下载")

    dest.parent.mkdir(parents=True, exist_ok=True)
    pos = dest.stat().st_size if dest.exists() and not force else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if pos and r.status_code == 200:  # 服务器不支持 Range，从头来
            pos = 0
        r.raise_for_status()
        total = expected or (pos + int(r.headers.get("Content-Length", 0)))
        mode = "ab" if pos else "wb"
        t0 = last = time.time()
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                pos += len(chunk)
                if time.time() - last > 2:
                    pct = f"{pos * 100.0 / total:5.1f}%" if total else "?"
                    print(f"\r[download] {dest.name}: {_fmt(pos)} / {_fmt(total)} ({pct})",
                          end="", flush=True)
                    last = time.time()
        print(f"\r[ok] {dest.name}: {_fmt(pos)} / {_fmt(total)}"
              f"  用时 {time.time() - t0:.0f}s" + " " * 10)
    if expected and dest.stat().st_size != expected:
        print(f"[error] 大小不匹配: {dest} = {dest.stat().st_size}, 期望 {expected}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 ASR-Spec-Live 模型权重")
    ap.add_argument("--what", choices=["all", "asr", "llm"], default="all")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--mirror", default=DEFAULT_MIRROR, help="ModelScope models 根地址")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.models_dir).expanduser().resolve()
    ok = True
    if args.what in ("all", "asr"):
        d = root / "SenseVoiceSmall-onnx"
        print(f"== SenseVoiceSmall ONNX -> {d}")
        for name, size in SENSEVOICE_FILES:
            url = f"{args.mirror}/{SENSEVOICE_REPO}/resolve/master/{name}"
            ok = download(url, d / name, size, args.force) and ok
    if args.what in ("all", "llm"):
        url = f"{args.mirror}/{LLM_REPO}/resolve/master/{LLM_FILE}"
        d = root / LLM_SUBDIR
        print(f"== {LLM_REPO}/{LLM_FILE} (~5.3GB) -> {d}")
        ok = download(url, d / LLM_FILE, LLM_SIZE, args.force) and ok

    print()
    print("模型目录:", root)
    if args.what in ("all", "asr"):
        print("设置:  ASR_SPEC_SENSEVOICE_DIR=" + str(root / "SenseVoiceSmall-onnx"))
    if args.what in ("all", "llm"):
        print("设置:  ASR_SPEC_LLAMA_MODEL=" + str(root / LLM_SUBDIR / LLM_FILE))
    print("llama.cpp: 请从 https://github.com/ggml-org/llama.cpp/releases 下载构建，")
    print("           并把 llama-server 所在目录加入 PATH，或设置 ASR_SPEC_LLAMA_EXE。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
