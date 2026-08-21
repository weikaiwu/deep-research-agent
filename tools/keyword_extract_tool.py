"""
关键词抽取工具：调用项目二微调模型（vLLM 服务）
================================================================
项目二（qwen2.5-7b-keyword-extraction-sft）微调模型部署为 vLLM 服务后，
本工具通过 OpenAI 兼容接口调用它做关键词抽取。

一份核心逻辑（keyword_extract_impl），两个入口：
  - keyword_extract      ：langchain 工具（供主 Agent 直接调用）
  - mcp_rag/server.py    ：FastMCP 工具（供 Claude Desktop / Cursor 消费）

配置（.env）：
  VLLM_BASE_URL=http://localhost:8000
  VLLM_MODEL=kw_r8
"""
import os
import re
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv
from pathlib import Path

# 直接定位项目根目录的 .env，避免 find_dotenv() 在无文件场景下找不到
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)


def keyword_extract_impl(text: str) -> str:
    """调用项目二微调模型的 vLLM 服务，从文本中抽取关键词。"""
    vllm_url = os.getenv("VLLM_BASE_URL")
    model_name = os.getenv("VLLM_MODEL", "kw_r8")
    if not vllm_url:
        return (
            "未配置项目二微调模型服务（环境变量 VLLM_BASE_URL 为空）。"
            "在 AutoDL 上部署 vLLM 后设置 VLLM_BASE_URL 即可启用。"
        )
    if not text.strip():
        return "错误：text 不能为空"

    try:
        r = requests.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "你是关键词抽取器。从用户文本中抽取最重要的关键词，只输出关键词，用逗号分隔。"},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.0,
                "max_tokens": 128,
            },
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        # 轻量后处理：按中英文分隔符拆分 → 去重 → 过滤空项/过短项 → 截断
        parts = re.split(r"[;；,，、\n]+", raw)
        seen, cleaned = set(), []
        for p in parts:
            p = p.strip().strip("。，,；;!！?？")
            if len(p) >= 2 and p not in seen:
                seen.add(p)
                cleaned.append(p)
        return "；".join(cleaned[:20])
    except Exception as e:
        return f"关键词抽取失败: {e}"


@tool
def keyword_extract(text: str) -> str:
    """
    从文本中抽取关键词/短语（调用项目二微调关键词抽取模型的 vLLM 服务）。
    适合：需要从长文本/调研材料中快速提炼核心关键词的场景。

    Args:
        text: 待抽取关键词的文本
    Returns:
        抽取出的关键词列表（分号/逗号分隔）
    """
    return keyword_extract_impl(text)
