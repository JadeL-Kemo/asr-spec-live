# -*- coding: utf-8 -*-
"""个性化实现示例（本体只消费 `interfaces.PersonalizationProvider` 接口）。

契约与边界见 docs/ARCHITECTURE.md「个性化接口」一节：
- 本仓库**不实现**声纹分离、画像学习、热词挖掘、LoRA 训练；这些属于独立项目，
  通过同一个接口接入（避免与本体耦合，也便于单独验证与灰度）。
- 本文件提供两个可直接用的实现，用于联调与演示：
  * `JsonPersonalization`：从 JSON 文件读静态配置（hotwords / persona_card / lora_adapter）。
  * `NullPersonalization` / `StaticPersonalization`（在 interfaces 中）。
"""
from __future__ import annotations

import json
import os

from .interfaces import PersonalizationProvider


class JsonPersonalization(PersonalizationProvider):
    """静态个性化配置：{"hotwords": [...], "persona_card": "...", "lora_adapter": "..."}"""

    def __init__(self, path: str | None = None):
        self._hotwords: list[str] = []
        self._persona_card = ""
        self._lora_adapter: str | None = None
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            self._hotwords = [str(x) for x in cfg.get("hotwords") or []]
            self._persona_card = str(cfg.get("persona_card") or "")
            self._lora_adapter = cfg.get("lora_adapter") or None

    def hotwords(self) -> list[str]:
        return self._hotwords

    def persona_card(self) -> str:
        return self._persona_card

    def lora_adapter(self) -> str | None:
        return self._lora_adapter
