# 定义一个网络搜索的工具！
# ======================== 导入核心依赖 ========================
from typing import Literal
from langchain_core.tools import tool

# 系统/第三方依赖
import os  # 系统路径/环境变量处理
import requests  # 直连 dashscope 兼容接口，绕过 langchain 对 enable_search 的参数拦截
from dotenv import load_dotenv  # 加载 .env 文件中的环境变量
from pathlib import Path

# 自定义模块：工具调用埋点监控（需确保 api 模块可导入）
from api.monitor import monitor

# ======================== 初始化配置 ========================
# 直接定位项目根目录的 .env，避免 find_dotenv() 在无文件场景（如 python -c）下找不到 .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

# 搜索后端切换（在 .env 里配置 SEARCH_BACKEND 即可）：
#   - qwen（默认）：使用通义千问 qwen-max 的联网搜索（enable_search）。
#       百炼是国内服务，本机直连、无需额外 Key，是国内网络环境最稳的方案。
#       注意：langchain 的 ChatOpenAI 会把 enable_search 当成非法参数拦截，
#       因此必须用 requests 直接打 dashscope 兼容接口，不能走 langchain 封装。
#   - duckduckgo：免 key 免代理，但依赖海外网络，国内/沙箱通常连不上。
#   - tavily：官方 API，需有效 TAVILY_API_KEY。
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "qwen").lower()

BASE_URL = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("LLM_QWEN_MAX", "qwen-max")

# 懒加载客户端，避免模块导入阶段就初始化 Tavily（其 key 当前可能已失效）
_dd_search = None
_tavily_client = None


def _get_dd_wrapper():
    """懒加载 DuckDuckGo 搜索封装，返回 list[dict]{title,link,body}。"""
    global _dd_search
    if _dd_search is None:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        _dd_search = DuckDuckGoSearchAPIWrapper()
    return _dd_search


def _get_tavily_client():
    """懒加载 Tavily 客户端。"""
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily_client


def _ddg_to_tavily_style(query: str, max_results: int) -> dict:
    """把 DuckDuckGo 结果包成与 Tavily `search()` 兼容的返回结构，下游无需改动。"""
    try:
        raw = _get_dd_wrapper().results(query, max_results=max_results)
    except Exception as e:  # DuckDuckGo 偶发限流，降级为单条提示，不影响主链路
        return {"results": [{"content": f"[搜索后端暂不可用: {e}]", "url": "", "title": "", "score": 0.0}],
                "query": query}
    items = []
    for r in raw:
        items.append({
            # DuckDuckGoSearchAPIWrapper.results() 返回键为 snippet/title/link；body 兜底兼容旧版
            "content": r.get("snippet") or r.get("body") or "",
            "url": r.get("link", ""),
            "title": r.get("title", ""),
            "score": 1.0,
        })
    return {"results": items, "query": query}


def _qwen_web_search(query: str, max_results: int) -> dict:
    """使用 qwen-max 的联网搜索（enable_search）。

    关键点：langchain 的 ChatOpenAI 会把 enable_search 当成未知参数直接抛 TypeError，
    因此必须用 requests 直接打 dashscope 兼容接口（国内可达）。

    返回结构与 Tavily 兼容：{"results":[{content,url,title,score}],"query"}。
    qwen 返回的是综合检索内容，这里要求模型按要点列出事实并附来源 URL，
    以便下游 agent 引用、支撑"引用准确率"KPI。
    """
    try:
        prompt = (
            f"请联网检索关于「{query}」的权威最新资料。\n"
            f"要求：\n"
            f"1. 用要点列出检索到的客观事实；\n"
            f"2. 每条要点后必须附上真实来源网页 URL（若确实无链接则标注「无链接」）；\n"
            f"3. 不要展开主观分析，只列事实与来源。\n"
            f"检索问题：{query}"
        )
        r = requests.post(
            BASE_URL + "/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "enable_search": True,
            },
            timeout=90,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # 包装成与 Tavily 兼容的单条结果（content 含要点+链接）
        return {"results": [{"content": content, "url": "", "title": query, "score": 1.0}], "query": query}
    except Exception as e:
        return {"results": [{"content": f"[qwen 联网搜索暂不可用: {e}]", "url": "", "title": "", "score": 0.0}],
                "query": query}


# ======================== 步骤：定义一个网络搜索工具 ========================
@tool
def internet_search(
        query: str,
        topic: Literal["news", "finance", "general"] = "general",
        max_results: int = 5,
        include_raw_content: bool = False
):
    """
    根据用户问题，进行网络信息收集！
    注意：主要搜索公开的网络信息！如果指定查询数据库或者rag不能使用此工具！
    :param query: 用户的查询信息
    :param topic: 查询的类型（qwen 后端会忽略此参数）
    :param max_results: 返回的最大条数
    :param include_raw_content: 是否返回原内容（qwen 后端会忽略此参数）
    :return:
    """
    # 每次调用工具，都会向前端推进调用进度！
    monitor.report_tool(tool_name="网络搜索工具",
                        args={"query": query, "topic": topic, "max_results": max_results,
                              "include_raw_content": include_raw_content})

    if SEARCH_BACKEND == "tavily":
        client = _get_tavily_client()
        return client.search(query=query, topic=topic,
                             max_results=max_results, include_raw_content=include_raw_content)
    if SEARCH_BACKEND == "duckduckgo":
        return _ddg_to_tavily_style(query, max_results)
    # 默认走 qwen 联网搜索（国内直连，最稳）
    return _qwen_web_search(query, max_results)
