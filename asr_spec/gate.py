# -*- coding: utf-8 -*-
"""接受/回滚门控：判断"基于 trigger partial 的投机结果"能否直接交付。

规则（与可行性实验一致，宁严勿松）：
1. 规范化（去标点/空白、保留中英文数字）后完全相等 -> 接受。
2. final 以 trigger 为前缀且多出 ≤ max_extra_chars 个字符 -> 接受（ASR 补了尾字）。
3. trigger 比 final 多 ≤ max_extra_chars 个字符且 final 是 trigger 前缀 -> 接受（多识别了语气词）。
其余 -> 拒绝，回滚重生成。

注意：本模块只做**文本级**判断。语义级判断（bge 相似度）在实验中证明会放过
槽位值错误（如 七点 vs 七点半），因此不作为主判据，只建议用于日志告警。
"""
from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff]")

# 数量/单位类字符：出现在"多出来的尾巴"里意味着数量语义变了，不能接受
# 例：trigger="…改成7点" / final="…改成7点半吧" -> 多出"半吧"含数量词 -> 拒绝
_QUANTITY_CHARS = set("0123456789零一二三四五六七八九十百千万亿点半号度元块层岁倍%拾佰仟")


def _has_quantity_change(suffix: str) -> bool:
    return any(ch in _QUANTITY_CHARS for ch in suffix)


def normalize(text: str) -> str:
    """去空白与标点，只保留中英文与数字，统一小写。"""
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        if ch.isalnum() or _CJK.match(ch):
            out.append(ch.lower())
    return "".join(out)


def is_accept(
    final_text: str,
    trigger_text: str,
    *,
    max_extra_chars: int = 2,
    min_trigger_chars: int = 3,
) -> bool:
    nf, nt = normalize(final_text), normalize(trigger_text)
    if len(nt) < min_trigger_chars or not nf:
        return False
    if nf == nt:
        return True
    if nf.startswith(nt) and len(nf) - len(nt) <= max_extra_chars:
        return not _has_quantity_change(nf[len(nt):])
    if nt.startswith(nf) and len(nt) - len(nf) <= max_extra_chars:
        return not _has_quantity_change(nt[len(nf):])
    return False
