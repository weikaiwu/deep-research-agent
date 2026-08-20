# ============================================================
# BGE-M3 双向量（稠密 + 稀疏）Embedding 模块
# 与项目一保持同一实现：原生 L2 归一化，适配 Milvus IP 内积检索
# ============================================================
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from rag.rag_config import rag_config

_bge_m3_ef = None


def get_bge_m3_ef():
    """BGE-M3 模型单例：首次调用时初始化，之后复用。"""
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=rag_config.bge_m3_path,
        device=rag_config.bge_device,
        use_fp16=rag_config.bge_fp16,
        normalize_embeddings=True,  # 模型原生对稠密+稀疏向量做 L2 归一化
    )
    print(f"[RAG] BGE-M3 模型初始化成功: {rag_config.bge_m3_path} (device={rag_config.bge_device})")
    return _bge_m3_ef


def generate_embeddings(texts):
    """
    为文本列表生成稠密 + 稀疏混合向量。
    :param texts: 文本列表（单文本也要包成列表）
    :return: {"dense": [list...], "sparse": [{token: weight}...]}，与 texts 一一对应
    """
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("参数 texts 必须是包含文本的非空列表")

    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)

    # 解析稀疏向量：CSR 格式 -> 字典列表（适配 Milvus 与 JSON 序列化）
    processed_sparse = []
    for i in range(len(texts)):
        sparse_indices = embeddings["sparse"].indices[
            embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
        ].tolist()
        sparse_data = embeddings["sparse"].data[
            embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
        ].tolist()
        processed_sparse.append({k: v for k, v in zip(sparse_indices, sparse_data)})

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse,
    }
