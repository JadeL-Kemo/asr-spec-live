# ASR-Spec-Live

**Let the LLM start answering while the user is still speaking.**

A streaming ASR emits partial transcripts every 0.5–1 s. This project feeds those partials into the
LLM's (llama.cpp) KV cache ahead of time and starts a **speculative generation** as soon as the user
pauses or is close to finishing. By the time the final transcript arrives, the answer is usually ready:
if it matches, it is delivered immediately (skipping the whole "endpoint wait + final ASR + first-token
wait" chain); if not, the pipeline rolls back safely and regenerates from the final transcript — which is
no slower than the standard path because the prefix cache is already warm.

> For real-time voice assistants, speech chat, live subtitles, etc. — anything where "answer right after
> the user stops talking" matters.

---

## The problem

Standard pipeline: `user stops → endpoint silence (0.6–1.6 s) → final ASR → first LLM token → (TTS)`,
strictly serial.

This project overlaps ASR with LLM prefill/generation:

```
while the user is speaking
  ├── streaming ASR partials (SenseVoiceSmall ONNX, CPU, ~70 ms each)
  ├── prefix prefill: system + history + "<|im_start|>user\n" pushed into a llama.cpp slot
  └── pause / trigger → speculative generation (full prompt ending mid-user-text)
after the user stops
  ├── final ASR after endpoint (~100 ms)
  ├── gate: trigger text equals final text, or differs by ≤2 chars (no numeric/unit changes)
  └── hit → deliver the speculative answer; miss → roll back and regenerate on the warm cache
```

## Core design

- **Stable prefix cache**: `system + persona card + conversation history` lives at the front and only
  grows by appending; dynamic content goes at the tail. Changing the LoRA adapter or system prompt
  invalidates the cache, so those are session-level settings.
- **Two triggers**:
  - `pause` (recommended): once the user pauses and the partial is stable, speculate on the full prefix.
    Zero semantic risk — the trigger text eventually equals the final text.
  - `ratio` (research): trigger when the partial reaches a ratio of the predicted final length — earlier,
    but needs a stricter gate.
- **Gate / rollback**: accept if `normalize(trigger) == normalize(final)` or one is a prefix of the other
  with ≤2 extra chars; always roll back when the extra tail contains digits/quantifiers/units
  (e.g. `7点` vs `7点半`) to avoid "same-looking answer, wrong slot value".
- **Every turn is accounted for**: speculative first-sentence latency, `cache_n`, accept/rollback, and the
  saving vs. a cold baseline are recorded for tuning and regression.

## Layout

```
asr_spec/
  interfaces.py       abstract audio / ASR / LLM / TTS / personalization interfaces
  prompts.py          chat-template and prompt builders (stable prefix + dynamic tail)
  gate.py             accept / rollback gate
  audio.py            wav loading, ffmpeg microphone capture, energy VAD
  asr_sensevoice.py   SenseVoiceSmall ONNX frontend/inference (hand-written Kaldi fbank/LFR/CMVN)
  llm_llama.py        llama-server client (auto-spawn, streaming, timings)
  pipeline.py         speculative loop: prefill thread / partials / trigger / speculate / gate / rollback
  personalization.py  JSON-based example of the personalization interface
  demo.py             CLI: selftest / wav / live / multiturn / mic / devices
  webdemo.py          visualization server (stdlib only)
  web/index.html      single-page console: partials, trigger, speculative answer, hit-rate, diffs
tests/test_core.py    unit tests (no GPU / network / models needed)
scripts/download_models.py  model downloader (weights are not in the repo)
examples/             persona example + two synthetic test clips (replace freely)
```

## Quick start

Requirements: Python ≥3.10, ffmpeg, a [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) build
(`llama-server`), and ~6.4 GB of disk for weights.

```bash
# 1) Python deps (install a torch/torchaudio build matching your CUDA/CPU)
pip install -r requirements.txt

# 2) Download weights (~5.9 GB from ModelScope, resumable)
python scripts/download_models.py

# 3) Point the project at llama.cpp (either add it to PATH or set the env var)
export ASR_SPEC_LLAMA_EXE=/path/to/llama-server

# 4) Pure-logic self test (no models / GPU)
python -m asr_spec.demo selftest

# 5) Offline replay evaluation (auto-spawns llama-server, ~5.7 GB model)
python -m asr_spec.demo wav --ids u01,u06 --ratio 0.75 --baseline

# 6) Real-time replay (1:1 playback of example audio, realistic user pacing)
python -m asr_spec.demo live --ids u01,u06 --gap 2.4 --baseline

# 7) Visualization console
python -m asr_spec.webdemo --port 8788   # open http://127.0.0.1:8788

# 8) Live microphone
python -m asr_spec.demo devices
python -m asr_spec.demo mic --seconds 20
```

## Configuration

All paths are overridable via environment variables (see `asr_spec/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `ASR_SPEC_MODELS` | `<project>/models` | models root |
| `ASR_SPEC_SENSEVOICE_DIR` | `<models>/SenseVoiceSmall-onnx` | SenseVoice ONNX directory |
| `ASR_SPEC_LLAMA_EXE` | `llama-server` on PATH | llama.cpp binary |
| `ASR_SPEC_LLAMA_MODEL` | `<models>/gguf/Qwen3.5-9B-Q4_K_M.gguf` | main LLM weights |
| `ASR_SPEC_FFMPEG` | `ffmpeg` on PATH | microphone capture |
| `ASR_SPEC_WAV_DIR` | `<project>/examples/wav` | replay audio directory |
| `ASR_SPEC_RUNS` | `<project>/runs` | logs and results |

`llama-server` is auto-spawned on demand (port 8402, ctx 8192, all layers on GPU). Use
`--server external` to reuse a running server, or `--server none` to run with MockLLM only.

## Metrics

- `answer_ready_after_stop_ms`: latency from "user stopped speaking" to the first **speakable sentence**
  (first LLM sentence ending with `。！？；`) being ready.
- `baseline_*`: the same metric for the standard pipeline (endpoint + final ASR + LLM with cold/warm prompt).
- `cache_n` / `prompt_n`: reused vs. re-evaluated tokens reported by llama.cpp.

## Measured summary (local RTX 2080 Ti, 9B Q4_K_M + SenseVoiceSmall on CPU)

| Scenario | Result |
|---|---|
| Real-time replay, 2 turns (1.6 s endpoint) | 2/2 hits, **784 / 789 ms** saved per turn |
| Multi-turn dialogue | turn 2 re-evaluates only **5 tokens** (`cache_n=260`) |
| Offline replay, ratio 0.75 (8 clips) | 5/8 hits, **915 ms** saved on hits; rollback 400 ms (< 467 ms cold baseline) |
| 240 real Chinese voice clips (1.6–9.0 s, multi-speaker) | pause mode: **254 ms** saved/turn at 1.0 s endpoint, **457 ms** at 1.6 s; answers **identical** to the standard path (zero semantic risk) |

The gain scales with endpoint silence and answer length. The rollback path is generally not slower than
the standard pipeline because the prefix cache is already warm.

## Personalization interface (optional)

`interfaces.PersonalizationProvider` exposes `hotwords()`, `persona_card()`, `lora_adapter()`,
`session_start()`, `session_end()`, `rewrite_partial()`; `personalization.JsonPersonalization` is a static
JSON example. The persona card lives in the stable prefix (cache-friendly), hotwords can be passed to the
ASR (`ASRProvider.set_hotwords`), and LoRA adapters are provided externally (llama.cpp supports `--lora`
and runtime switching, but switching invalidates the prompt cache — do it at session boundaries).

## Known limitations

- SenseVoiceSmall is **not** a streaming model; this project simulates streaming by re-transcribing the
  accumulated audio every 0.5 s, so cost grows linearly with utterance length. Swap in a native streaming
  ASR behind the same interface for production.
- If endpoint silence < 0.6 s, speculation cannot finish in time and `pause` mode loses; use endpoints ≥1.0 s.
- Chinese ASR still makes homophone/proper-noun errors (CER ≈2–8% in our local evaluation); the gate only
  guarantees "never deliver a wrong answer", it does not fix recognition errors.
- TTS is not included: `TTSProvider` is an interface placeholder; answers are text.

## Third-party components and licensing

This project's code is MIT-licensed (see [LICENSE](LICENSE)). Models and tools it depends on are **not**
bundled and remain under their own licenses:

| Component | Source | License |
|---|---|---|
| SenseVoiceSmall (ONNX) | ModelScope `iic/SenseVoiceSmall-onnx` (Alibaba Tongyi Lab) | see the model card's Model License |
| Qwen3.5-9B (GGUF quantization) | ModelScope `unsloth/Qwen3.5-9B-GGUF`; base Qwen3.5-9B (Alibaba) | Apache-2.0 (per Qwen's official statements) |
| llama.cpp / llama-server | https://github.com/ggml-org/llama.cpp | MIT |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | MIT |
| PyTorch / TorchAudio | https://pytorch.org | BSD-3-Clause |
| FFmpeg | https://ffmpeg.org | LGPL/GPL (depending on build) |

The two clips under `examples/wav/` are generic Chinese sentences synthesized with a public TTS service,
used only for demo and regression testing; replace or delete them freely.

The repository contains no model weights; `scripts/download_models.py` only downloads from the public
sources listed above.
