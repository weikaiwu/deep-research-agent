# ============================================================
# RAG 检索配置：从 deepagents/.env 读取（与项目一共享同一套 Milvus/模型）
# ============================================================
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# 定位项目根目录（rag/ 的上级目录），直接加载 .env，不依赖调用栈猜测
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class RagConfig:
    milvus_url: str            # Milvus 服务端连接地址
    chunks_collection: str     # 文档切片集合名
    embedding_dim: int         # 稠密向量维度（BGE-M3 = 1024）
    bge_m3_path: str           # BGE-M3 模型本地路径（无则自动下载 BAAI/bge-m3）
    bge_device: str            # BGE-M3 运行设备（cpu / cuda:0）
    bge_fp16: bool             # BGE-M3 是否半精度
    bge_reranker_large: str    # BGE-reranker-large 模型本地路径
    bge_reranker_device: str   # reranker 运行设备
    bge_reranker_fp16: bool    # reranker 是否半精度


def _to_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


rag_config = RagConfig(
    milvus_url=os.getenv("MILVUS_URL", "http://localhost:19530"),
    chunks_collection=os.getenv("CHUNKS_COLLECTION", "kb_chunks"),
    embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
    bge_m3_path=os.getenv("BGE_M3_PATH") or "BAAI/bge-m3",
    bge_device=os.getenv("BGE_DEVICE", "cpu"),
    bge_fp16=_to_bool(os.getenv("BGE_FP16", "0")),
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE") or "BAAI/bge-reranker-large",
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE", "cpu"),
    bge_reranker_fp16=_to_bool(os.getenv("BGE_RERANKER_FP16", "0")),
)
