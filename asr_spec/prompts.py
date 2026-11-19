# -*- coding: utf-8 -*-
"""Prompt 构建：所有 prompt 都从同一个"稳定前缀"派生。

缓存规则（llama.cpp 实测）：已缓存 prompt 必须是新 prompt 的严格 token 前缀才命中。
因此：
  prefix = system(含画像卡) + 历史(user/assistant 回合，assistant 回合带 no-think 缓冲)
  投机 prompt = prefix + partial + "<|im_end|>\\n" + NO_THINK      # 扩展 prefix
  最终 prompt = prefix + final    + "<|im_end|>\\n" + NO_THINK
历史里的 assistant 回合必须写成 NO_THINK + answer + "<|im_end|>\\n"，
这样上一轮生成的 prompt 才能成为下一轮 prefix 的前缀（跨轮缓存成立）。
"""
from __future__ import annotations

from typing import Sequence

# 本轮 PoC 沿用可行性实验里验证过的本地助手 system prompt
SYS = (
    "你是“小智”，一个运行在本地设备上的中文语音助手。用户通过语音与你交流，你的回答会被转化为语音播放。\n"
    "要求：\n"
    "1. 回答简洁明了，一般不超过三句话，避免使用列表、代码块和复杂格式。\n"
    "2. 涉及时间、天气、日程、音乐、家居控制等请求时，先确认关键信息再执行。\n"
    "3. 语气自然友好，像真人助手一样。\n"
    "4. 如果用户的问题信息不足，用一句话追问。\n"
    "5. 不要重复用户的话，不要输出多余寒暄。\n"
    "6. 你能处理：日程提醒、天气查询、音乐播放、智能家居控制、闲聊。\n"
    "7. 你只能通过语音回答，不能显示图片或表格。\n"
    "8. 如果用户说了半句话或者听不清，可以礼貌地请对方再说一遍。\n"
    "9. 每次回答只围绕用户最新的请求，不要提及系统的限制。\n"
    "10. 保持回答口语化，适合 TTS 朗读。"
)

# 抑制 thinking 的 assistant 前缀（Qwen3.5 chat template enable_thinking=false 时的前缀）
NO_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_system(persona_card: str = "") -> str:
    """system 文本 = 基础 SYS + 会话级画像卡（稳定前缀的一部分）。"""
    persona_card = (persona_card or "").strip()
    if not persona_card:
        return SYS
    return SYS + "\n\n用户画像（会话级，勿在回答中复述）：\n" + persona_card


def build_hist(context: Sequence[str]) -> str:
    """把 ["用户：…", "助手：…"] 历史行渲染成 chat 模板片段。

    assistant 回合带 NO_THINK，保证跨轮 token 前缀一致。
    """
    out: list[str] = []
    for line in context:
        line = (line or "").strip()
        if not line:
            continue
        if line.startswith("用户："):
            out.append("<|im_start|>user\n" + line[3:] + "<|im_end|>\n")
        elif line.startswith("助手："):
            out.append(NO_THINK + line[3:] + "<|im_end|>\n")
    return "".join(out)


def prompt_prefix(system_text: str, hist_text: str) -> str:
    """稳定前缀，以 "<|im_start|>user\\n" 结尾（等待用户文本）。"""
    return (
        "<|im_start|>system\n" + system_text + "<|im_end|>\n"
        + hist_text
        + "<|im_start|>user\n"
    )


def prompt_final(prefix: str, user_text: str) -> str:
    """完整 prompt：前缀 + user 文本 + assistant 前缀。"""
    return prefix + user_text + "<|im_end|>\n" + NO_THINK


def prompt_prefill(prefix: str) -> str:
    """安全预填充 prompt：恰好等于最终 prompt 的前缀（不含用户文本）。"""
    return prefix
