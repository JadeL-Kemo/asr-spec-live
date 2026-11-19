# ASR-Spec-Live

**在用户说话的同时，让大模型提前把答案生成出来。**

流式 ASR 每 0.5–1 秒吐一次 partial 文本；本项目把这些文本实时预填充进大模型（llama.cpp）的 KV cache，
并在用户停顿/接近说完时触发**投机生成**。等用户真正说完、最终转写到达时，答案往往已经就绪：
命中则直接交付（省掉整个"端点等待 + 最终转写 + 首 token 等待"），不命中则安全回滚、用最终文本重新生成——
而由于前缀缓存已热，重生成也不比标准链路慢。

> 适合：ASR → LLM → TTS 的实时语音助手、语音对话、字幕/同传等"说完就要答案"的场景。

---

## 它解决什么问题

标准链路：`用户说完 → 端点静音检测(0.6–1.6s) → 最终 ASR → LLM 首 token →（TTS）`
每一段都在串行等待。

本方案把"ASR 识别"和"LLM 预填充/生成"从串行改为**重叠**：

```
用户还在说话
  ├── 流式 ASR partial（SenseVoiceSmall ONNX，CPU，~70ms/次）
  ├── 前缀预填充：system + 历史 + "<|im_start|>user\n" 提前灌进 llama.cpp slot
  └── 停顿/触发 → 投机生成（带 ASR 前缀的完整 prompt）
用户说完
  ├── 端点后 final ASR（~100ms）
  ├── 门控：触发文本与最终文本一致 / 仅差 ≤2 字（且不含数字量词变化）
  └── 命中 → 直接用投机答案；否则 → 回滚，在热缓存上用最终 prompt 重生成
```

## 核心设计

- **稳定前缀缓存**：`system + 画像卡 + 对话历史` 放在 prompt 最前面且 append-only；动态内容只追加在末尾。
  换 LoRA adapter / 换 system 会清空缓存，因此这两者被视为"会话级"配置。
- **两种触发**：
  - `pause`（推荐）：检测到用户停顿且 partial 已稳定 → 用当前完整前缀投机。零语义风险（最终文本与触发文本相同）。
  - `ratio`（研究用）：partial 长度达到预测最终长度的某个比例 → 更早触发，但需要更严的门控。
- **门控回滚**：`normalize(trigger) == normalize(final)`，或互为前缀且多出 ≤2 字；若多出的尾巴含
  数字/量词/单位（如 `7点` vs `7点半`）则强制回滚，避免"槽位值不同答案却相似"的误接受。
- **命中与回滚都记账**：每轮记录 `spec` 首句耗时、`cache_n`、接受/回滚、相对冷基线的节省，便于调参与回归。

## 目录结构

```
asr_spec/
  interfaces.py       抽象接口：AudioSource / ASRProvider / LLMProvider / TTSProvider / PersonalizationProvider
  prompts.py          chat 模板与 prompt 构造（稳定前缀 + 动态尾部）
  gate.py             门控：接受 / 回滚判定
  audio.py            音频读取、ffmpeg 麦克风采集、能量 VAD 切句
  asr_sensevoice.py   SenseVoiceSmall ONNX 推理（自写 Kaldi fbank/LFR/CMVN 前端）
  llm_llama.py        llama-server OpenAI 兼容接口（含自动拉起、streaming、timings）
  pipeline.py         投机流水线：预填充线程 / partial 循环 / 触发 / 投机 / 门控 / 回滚 / 跨轮历史
  personalization.py  个性化接口的 JSON 实现示例（热词 / 画像卡 / LoRA adapter）
  demo.py             CLI：selftest / wav / live / multiturn / mic / devices
  webdemo.py          可视化演示服务器（stdlib，无额外依赖）
  web/index.html      单页控制台：实时 partial、触发、投机答案、命中率、修正 diff
tests/test_core.py    单元测试（无 GPU / 无网络 / 无模型即可跑）
scripts/download_models.py  模型权重下载器（Git 仓库不含权重）
examples/             画像配置示例 + 两段合成测试音频（可自行替换）
```

## 快速开始

依赖：Python ≥3.10、ffmpeg、[llama.cpp](https://github.com/ggml-org/llama.cpp/releases) 构建（`llama-server`）、
磁盘约 6.4GB（模型权重）。

```bash
# 1) 安装 Python 依赖（torch/torchaudio 请装与你的 CUDA/CPU 匹配的版本）
pip install -r requirements.txt

# 2) 下载模型权重（约 5.9GB；从 ModelScope，支持断点续传）
python scripts/download_models.py

# 3) 让程序找到 llama.cpp（二选一）
#    a. 把 llama-server 加入 PATH；或
#    b. export ASR_SPEC_LLAMA_EXE=/path/to/llama-server
#    （Windows PowerShell: $env:ASR_SPEC_LLAMA_EXE="D:\llama.cpp\llama-server.exe"）

# 4) 纯逻辑自测（不需要模型 / GPU）
python -m asr_spec.demo selftest

# 5) 离线回放评估（自动拉起 llama-server，加载约 5.7GB 模型）
python -m asr_spec.demo wav --ids u01,u06 --ratio 0.75 --baseline

# 6) 实时回放（1:1 播放示例音频，模拟真实用户节奏）
python -m asr_spec.demo live --ids u01,u06 --gap 2.4 --baseline

# 7) 可视化控制台
python -m asr_spec.webdemo --port 8788   # 打开 http://127.0.0.1:8788

# 8) 麦克风实时演示
python -m asr_spec.demo devices          # 列出 ffmpeg DirectShow 输入设备
python -m asr_spec.demo mic --seconds 20 # 说话，观察 partial / 触发 / 投机 / 答案
```

## 配置

路径全部通过环境变量覆盖（见 `asr_spec/config.py`）：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ASR_SPEC_MODELS` | `<项目>/models` | 模型根目录 |
| `ASR_SPEC_SENSEVOICE_DIR` | `<models>/SenseVoiceSmall-onnx` | SenseVoiceSmall ONNX 目录 |
| `ASR_SPEC_LLAMA_EXE` | `PATH` 中的 `llama-server` | llama.cpp 可执行文件 |
| `ASR_SPEC_LLAMA_MODEL` | `<models>/gguf/Qwen3.5-9B-Q4_K_M.gguf` | 主 LLM 权重 |
| `ASR_SPEC_FFMPEG` | `PATH` 中的 `ffmpeg` | 麦克风采集用 |
| `ASR_SPEC_WAV_DIR` | `<项目>/examples/wav` | 回放示例音频目录 |
| `ASR_SPEC_RUNS` | `<项目>/runs` | 运行日志/结果输出 |

llama-server 由程序按需自动拉起（默认端口 8402，ctx 8192，全 GPU 层）；
也可用 `--server external` 复用已在运行的服务，`--server none` 用 MockLLM 跑纯逻辑测试。

## 指标口径

- `answer_ready_after_stop_ms`：从"用户停说"到首个**可播报断句**（LLM 输出第一句以 `。！？；` 结尾的文本）就绪的时延。
- `baseline_*`：同一问题上标准链路（端点 + final ASR + 冷/热 prompt LLM）的同口径时延。
- `cache_n` / `prompt_n`：llama.cpp 的缓存命中 token 数 / 实际重算 token 数。

## 实测摘要（本机 RTX 2080 Ti，9B Q4_K_M + SenseVoiceSmall CPU）

| 场景 | 结果 |
|---|---|
| 实时回放 2 轮（端点 1.6s） | 2/2 命中，每轮省 **784 / 789ms** |
| 跨轮对话 | 第 2 轮只重算 **5 token**，prefix 缓存 `cache_n=260` |
| 音频回放评估 ratio 0.75（8 条） | 命中 5/8，命中平均省 **915ms**，回滚平均 400ms（< 冷基线 467ms） |
| 240 条真实中文配音（1.6–9.0s，多角色） | pause 模式：端点 1.0s 每轮省 **254ms**、1.6s 省 **457ms**；答案与标准链路**逐字一致**（零语义风险） |

结论：收益取决于端点静音长度（越短越难提前跑完）与答案长度；回滚路径因为前缀缓存已热，通常不比标准链路慢。

## 个性化接口（可选）

`interfaces.PersonalizationProvider` 预留了 `hotwords()` / `persona_card()` / `lora_adapter()` /
`session_start()` / `session_end()` / `rewrite_partial()`；`personalization.JsonPersonalization` 是静态 JSON 示例。
画像卡会进入稳定前缀（不影响缓存命中），热词可传给 ASR（`ASRProvider.set_hotwords`），
LoRA adapter 由外部提供（llama.cpp 支持 `--lora` 与运行时切换，但切换会使 prompt 缓存失效，需在会话边界做）。

## 已知限制

- SenseVoiceSmall 是**非流式**模型，本项目用"每 0.5s 累计重转写"模拟流式，长音频成本线性上涨；生产可换原生流式 ASR（接口不变）。
- 端点静音 <0.6s 时投机来不及，`pause` 模式收益为负；建议端点 ≥1.0s。
- 中文 ASR 对同音字/专有名词仍会出错（本机评估 CER 约 2–8%），门控回滚只保证"不交付错答案"，不修正识别错误。
- 未包含 TTS：`TTSProvider` 接口已预留，答案是文本。

## 第三方组件与版权

本项目代码以 MIT 许可发布（见 [LICENSE](LICENSE)）。运行依赖的模型与工具不随仓库分发，版权归各自作者：

| 组件 | 来源 | 许可 |
|---|---|---|
| SenseVoiceSmall（ONNX 版） | ModelScope `iic/SenseVoiceSmall-onnx`（阿里巴巴通义实验室） | 见模型页 Model License |
| Qwen3.5-9B（GGUF 量化） | ModelScope `unsloth/Qwen3.5-9B-GGUF`，基座 Qwen3.5-9B（阿里巴巴） | Apache-2.0（以 Qwen 官方声明为准） |
| llama.cpp / llama-server | https://github.com/ggml-org/llama.cpp | MIT |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | MIT |
| PyTorch / TorchAudio | https://pytorch.org | BSD-3-Clause |
| FFmpeg | https://ffmpeg.org | LGPL/GPL（取决于构建） |

`examples/wav/` 下两段示例音频由公开 TTS 服务合成的通用中文语句，仅用于演示与回归测试，可自由替换或删除。
本仓库不包含任何模型权重；`scripts/download_models.py` 仅从上述公开源下载。
