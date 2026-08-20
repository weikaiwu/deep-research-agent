# ============================================================
# Milvus 混合检索客户端（稠密 + 稀疏加权融合）
# 与项目一保持同一实现：WeightedRanker 融合，BGE-M3 双向量
# ============================================================
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from rag.rag_config import rag_config

_milvus_client = None


def get_milvus_client():
    """Milvus 客户端单例，连接失败返回 None（由调用方降级处理）。"""
    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client
    try:
        if not rag_config.milvus_url:
            print("[RAG] Milvus 连接失败：缺少 MILVUS_URL 配置")
            return None
        _milvus_client = MilvusClient(uri=rag_config.milvus_url)
        print(f"[RAG] Milvus 客户端连接成功: {rag_config.milvus_url}")
        return _milvus_client
    except Exception as e:
        print(f"[RAG] Milvus 客户端连接异常: {e}")
        return None


def create_hybrid_search_requests(dense_vector, sparse_vector, expr=None, limit=5,
                                  dense_params=None, sparse_params=None):
    """
    构建稠密 + 稀疏两个 AnnSearchRequest，供后续混合搜索融合。
    稠密默认 COSINE（建库一致），稀疏默认 IP（BGE-M3 稀疏向量）。
    """
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector], anns_field="dense_vector", param=dense_params,
        expr=expr, limit=limit,
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector], anns_field="sparse_vector", param=sparse_params,
        expr=expr, limit=limit,
    )
    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.9, 0.1),
                  norm_score=True, limit=5, output_fields=None):
    """
    执行 Milvus 稠密 + 稀疏混合搜索（WeightedRanker 加权融合）。
    默认权重 (0.9, 0.1)：稠密主导 + 稀疏兜底，与项目一建库/查询参数一致。
    """
    try:
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        if output_fields is None:
            output_fields = ["chunk_id", "content", "title", "parent_title", "item_name"]

        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
        )
        print(f"[RAG] Milvus 混合搜索完成，集合[{collection_name}]返回 {len(res[0])} 条")
        return res
    except Exception as e:
        print(f"[RAG] Milvus 混合搜索失败，集合[{collection_name}]: {e}")
        return None
