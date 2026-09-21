"""OpenAI-compatible 执行器的配置来源：只读环境变量。

公开仓库不绑定任何人的本机配置。用法：

    export AZFLS_API_BASE=https://api.deepseek.com/v1
    export AZFLS_API_KEY=sk-...
    export AZFLS_MODEL=deepseek-chat
    .venv/bin/python -m bench.candidate_api_probe

密钥只在请求头里使用，不会被打进日志、异常或返回值。
"""

from __future__ import annotations

import os

from codejev.api_engine import APIConfig

DEFAULT_MODEL = "deepseek-chat"


def load_provider(model: str | None = None, *, reasoning_effort: str | None = "none") -> APIConfig:
    """从环境变量组装 APIConfig；缺项就给明确的报错，不猜默认值。"""
    base_url = os.environ.get("AZFLS_API_BASE", "").rstrip("/")
    api_key = os.environ.get("AZFLS_API_KEY", "")
    chosen = model or os.environ.get("AZFLS_MODEL") or DEFAULT_MODEL
    if not base_url:
        raise RuntimeError("缺少环境变量 AZFLS_API_BASE（例如 https://api.deepseek.com/v1）")
    if not api_key:
        raise RuntimeError("缺少环境变量 AZFLS_API_KEY")
    return APIConfig(
        base_url=base_url,
        api_key=api_key,
        model=chosen,
        timeout_seconds=90.0,
        reasoning_effort=reasoning_effort,
    )
