"""
MCP Server：把项目一 RAG pipeline + 项目二微调抽取包成 MCP 工具
================================================================
工具列表：
  1. rag_search(query, top_k) —— 完整 RAG 检索链（HyDE → BGE-M3 双向量 →
     Milvus 混合检索 → RRF → BGE-reranker → 带引用返回）
     复用 tools/rag_retrieval_tool.py 的 rag_retrieve（与项目一完全同源的 pipeline）
  2. keyword_extract(text) —— 关键词抽取（调用项目二微调模型的 vLLM 端点，
     需配置 VLLM_BASE_URL；未配置时返回说明）

启动（stdio 模式，供 Claude Desktop / Cursor 消费）：
    python mcp/server.py

连接调试（MCP Inspector）：
    npx @modelcontextprotocol/inspector python mcp/server.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env（复用项目的模型/数据库配置）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastmcp import FastMCP
import requests

# ============ 工具 1：rag_search（复用项目一 RAG pipeline）============
# rag_retrieve 内部有模型懒加载（_create_hyde_doc 延迟导入 LLM），直接复用即可
from tools.rag_retrieval_tool import rag_retrieve


mcp = FastMCP("RAG Agent 检索服务")


@mcp.tool
def rag_search(query: str, top_k: int = 5) -> str:
    """
    从技术文档知识库（LangChain/Agent/Python 文档，Milvus 混合检索）中检索与 query 最相关的文档片段。

    Args:
        query: 检索问题或关键词
        top_k: 返回结果条数（默认 5，最多 10）
    Returns:
        带引用来源的检索结果（JSON 文本：content/title/source）
    """
    if not query.strip():
        return "错误：query 不能为空"

    top_k = max(1, min(int(top_k), 10))
    result = rag_retrieve(query, max_results=top_k)

    if not result.get("results"):
        return "未检索到相关文档片段。"

    import json
    return json.dumps(result, ensure_ascii=False, indent=2)


# ============ 工具 2：keyword_extract（项目二微调抽取，vLLM 端点）============
@mcp.tool
def keyword_extract(text: str) -> str:
    """
    从文本中抽取关键词/短语（调用项目二微调模型的 vLLM 服务）。

    Args:
        text: 待抽取关键词的文本
    Returns:
        抽取出的关键词列表（JSON）
    """
    vllm_url = os.getenv("VLLM_BASE_URL")
    model_name = os.getenv("VLLM_MODEL", "qwen2.5-7b-instruct-sft")
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
        keywords = r.json()["choices"][0]["message"]["content"].strip()
        return keywords
    except Exception as e:
        return f"关键词抽取失败: {e}"


if __name__ == "__main__":
    mcp.run()
