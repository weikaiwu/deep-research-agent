# ============================================================
# RAG 检索链工具：HyDE → BGE-M3 双向量 → Milvus 混合检索 → RRF → BGE-rerank → 动态 TopK
# 复用项目一（Milvus 混合检索 RAG）的完整 pipeline，作为 deepagents 的本地知识库数据源。
# 返回结构与 internet_search 兼容：{"results": [{content, title, url, score}], "query"}
# ============================================================
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from api.monitor import monitor
from rag.embedding_utils import generate_embeddings
from rag.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from rag.rag_config import rag_config
from rag.reranker_utils import get_reranker_model

# ---------------- RRF / 动态 TopK 常量（与项目一保持一致） ----------------
RRF_K: int = 60            # 经典 RRF 常数
RERANK_MAX_TOPK: int = 10  # 重排后最多保留条数
RERANK_MIN_TOPK: int = 1   # 最少保留条数
RERANK_GAP_RATIO: float = 0.25  # 断崖相对阈值
RERANK_GAP_ABS: float = 0.5     # 断崖绝对阈值


# ---------------- 1. HyDE：LLM 生成假设文档 ----------------
def _create_hyde_doc(query: str) -> str:
    """让 LLM 先写一段假设答案，再拿答案去检索，提升召回率。"""
    from agent.llm import model  # 延迟导入，避免工具初始化时加载 LLM
    hyde_prompt = (
        "请针对下面这个技术选型调研问题，写一段 300 字左右的假设性调研答案，"
        "尽可能覆盖：候选技术、关键对比维度、可能的数据出处。只输出答案正文，不要任何前缀。\n\n"
        f"问题：{query}"
    )
    resp = model.invoke([HumanMessage(content=hyde_prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ---------------- 2. 单路混合检索 ----------------
def _search_one(query_str: str, limit: int = 5):
    """BGE-M3 双向量 -> Milvus 混合检索，返回 hit 列表（含 entity）。"""
    embeddings = generate_embeddings([query_str])
    reqs = create_hybrid_search_requests(
        dense_vector=embeddings["dense"][0],
        sparse_vector=embeddings["sparse"][0],
        expr=None,  # 技术文档库不按 item_name 过滤，全库语义检索
        limit=limit,
    )
    client = get_milvus_client()
    if client is None:
        return []
    resp = hybrid_search(
        client=client,
        collection_name=rag_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=limit,
        output_fields=["chunk_id", "content", "title", "parent_title", "item_name"],
    )
    return resp[0] if resp else []


# ---------------- 3. RRF 加权融合 ----------------
def _reciprocal_rank_fusion(source_with_weight, top_k: int = 5):
    score_dict, chunk_dict = {}, {}
    for source, weight in source_with_weight:
        for rank, chunk in enumerate(source, start=1):
            chunk_id = chunk.get("id") or chunk.get("entity", {}).get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (RRF_K + rank)) * weight
            chunk_dict.setdefault(chunk_id, chunk)
    merged = [(chunk_dict[cid], score) for cid, score in score_dict.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in merged[:top_k]]


# ---------------- 4. BGE-rerank + 动态 TopK ----------------
def _rerank(doc_list, query: str):
    """Cross-Encoder 打分排序 + 断崖截断（与项目一 node_rerank 一致）。"""
    if not doc_list:
        return []
    rerank = get_reranker_model()
    text_list = [d.get("entity", {}).get("content") or d.get("content", "") for d in doc_list]
    pairs = [[query, t] for t in text_list]
    scores = rerank.compute_score(pairs, normalize=True)

    scored = sorted(
        [{"doc": doc, "score": float(s)} for doc, s in zip(doc_list, scores)],
        key=lambda x: x["score"], reverse=True,
    )
    # 动态 TopK：硬上下限 + 断崖检测
    topk = min(RERANK_MAX_TOPK, len(scored))
    if topk > RERANK_MIN_TOPK:
        for i in range(RERANK_MIN_TOPK - 1, topk - 1):
            s1, s2 = scored[i]["score"], scored[i + 1]["score"]
            gap = s1 - s2
            rel = gap / (abs(s1) + 1e-6)
            if gap >= RERANK_GAP_ABS or rel >= RERANK_GAP_RATIO:
                topk = i + 1
                break
    return scored[:topk]


# ---------------- 5. 对外主函数 ----------------
def rag_retrieve(query: str, max_results: int = 5) -> dict:
    """
    完整 RAG 检索链：HyDE → 双路混合检索 → RRF → BGE 重排 → 动态 TopK。
    返回 {"results": [{content, title, url, score}], "query"}，供报告引用。
    """
    # 埋点：通知前端知识库检索工具开始执行（与其它工具保持一致的 Trace 粒度）
    monitor.report_tool("知识库检索工具",
                        {"query": query, "max_results": max_results})

    # 双路召回：普通向量 + HyDE 增强（与项目一 node_search_embedding / _hyde 对齐）
    normal_chunks = _search_one(query, limit=max_results + 3)
    try:
        hyde_doc = _create_hyde_doc(query)
        hyde_chunks = _search_one(query + hyde_doc, limit=max_results + 3)
    except Exception as e:
        print(f"[RAG] HyDE 生成失败，仅用普通向量路: {e}")
        hyde_chunks = []

    # RRF 融合两路
    rrf_chunks = _reciprocal_rank_fusion(
        [(normal_chunks, 1.0), (hyde_chunks, 1.0)], top_k=10,
    )
    if not rrf_chunks:
        monitor.report_tool_end("知识库检索工具", "未检索到匹配文档片段（置信度 0.00），已触发降级")
        return {
            "results": [], "query": query, "confidence": 0.0,
            "degradation_hint": (
                "知识库未检索到匹配文档片段（confidence=0.0）。"
                "请改用网络搜索助手补充检索该问题；若仍无可靠依据，"
                "结论必须标注「无法确认（推断）」，不得编造。"
            ),
        }

    # BGE 重排 + 动态 TopK
    reranked = _rerank(rrf_chunks, query)
    results = []
    for item in reranked:
        entity = item["doc"].get("entity", {})
        results.append({
            "content": entity.get("content") or item["doc"].get("content", ""),
            "title": entity.get("title") or item["doc"].get("title", "（本地知识库）"),
            "url": "",  # 本地文档无 URL，报告引用用 title 标注来源
            "score": round(item["score"], 4),
        })

    # confidence gate：以重排后最高分作为本次检索的置信度（供调度层降级判断）
    top_score = results[0]["score"] if results else 0.0
    confidence = round(top_score, 4)

    # B2 硬门控（代码层强制）：置信度不足时自动附加降级指令，不依赖 LLM 自觉
    degradation_hint = None
    if confidence < 0.5:
        degradation_hint = (
            f"知识库检索置信度不足（confidence={confidence} < 0.5），"
            "知识库中没有足够相关的文档片段。请改用网络搜索助手补充检索该问题；"
            "若仍无可靠依据，结论必须标注「无法确认（推断）」，不得编造。"
        )
        monitor.report_tool_end(
            "知识库检索工具",
            f"检索到 {len(results)} 条文档片段，但置信度不足（{confidence:.2f}），已触发降级",
        )
    else:
        monitor.report_tool_end(
            "知识库检索工具",
            f"检索到 {len(results)} 条文档片段（HyDE+双路混合检索→RRF→BGE重排），置信度 {confidence:.2f}",
        )

    return {
        "results": results[:max_results],
        "query": query,
        "confidence": confidence,
        "degradation_hint": degradation_hint,   # None 或降级指令（主 Agent 必须遵循）
    }


class RagRetrievalInput(BaseModel):
    query: str = Field(description="要从本地知识库（技术文档库）检索的问题")
    max_results: int = Field(default=5, ge=1, le=10, description="最多返回的结果条数")


rag_retrieval = StructuredTool.from_function(
    func=rag_retrieve,
    name="rag_retrieval",
    description=(
        "从本地技术文档知识库（LangChain/Agent/Python 等技术文档，Milvus 混合检索）检索与问题最相关的文档片段。"
        "当需要基于团队内部技术文档、官方文档原文回答时使用；返回结果带来源标题，可直接作为报告引用。"
    ),
    args_schema=RagRetrievalInput,
)
