# ============================================================
# BGE-reranker 重排模块（Cross-Encoder 精确打分）
# 与项目一保持同一实现：FlagReranker + 单例
#
# 关键 monkey patch：torch 2.13 + transformers 4.57.6 下，
# transformers 内部 from_pretrained 默认走 meta 中转（low_cpu_mem_usage=True），
# 然后 model.to('cpu') 触发 "Cannot copy out of meta tensor"。
#
# 兜底：除了 patch from_pretrained，还 monkey patch torch.nn.Module.to，
# 让跨 meta→cpu 的 to() 失败时自动用 to_empty + load 解决。
# ============================================================
import torch
import torch.nn as nn
import transformers
from FlagEmbedding import FlagReranker

from rag.rag_config import rag_config


# === 1. Patch from_pretrained（强制 low_cpu_mem_usage=False + float32 cpu 加载）===
_orig_from_pretrained_fns = {}


def _patch_class(cls):
    """强制 cls.from_pretrained 用 low_cpu_mem_usage=False + dtype=float32"""
    orig = cls.from_pretrained.__func__
    _orig_from_pretrained_fns[cls.__name__] = orig

    def wrapper(*args, **kwargs):
        kwargs["low_cpu_mem_usage"] = False
        if kwargs.get("dtype") is None and kwargs.get("torch_dtype") is None:
            kwargs["dtype"] = torch.float32
        return orig(cls, *args, **kwargs)

    cls.from_pretrained = classmethod(lambda cls_, *a, **kw: wrapper(*a, **kw))


for name in ("AutoModel", "AutoModelForSequenceClassification", "AutoModelForCausalLM"):
    cls = getattr(transformers, name, None)
    if cls is not None:
        _patch_class(cls)
print("[RAG] transformers low_cpu_mem_usage meta-patch installed "
      f"({list(_orig_from_pretrained_fns.keys())})")


# === 2. 兜底：patch nn.Module.to 让 meta→cpu 失败时用 to_empty ===
_orig_module_to = nn.Module.to


def _safe_module_to(self, *args, **kwargs):
    try:
        return _orig_module_to(self, *args, **kwargs)
    except (NotImplementedError, RuntimeError) as e:
        msg = str(e)
        if "meta" in msg.lower() or "Cannot copy out" in msg:
            device = args[0] if args else kwargs.get("device", "cpu")
            return self.to_empty(device=device)
        raise


nn.Module.to = _safe_module_to
print("[RAG] torch.nn.Module.to meta-safe-patch installed")


_reranker_model = None


def get_reranker_model():
    """BGE-reranker 单例。"""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = FlagReranker(
            model_name_or_path=rag_config.bge_reranker_large,
            device=rag_config.bge_reranker_device,
            use_fp16=rag_config.bge_reranker_fp16,
        )
        print(f"[RAG] BGE-reranker 初始化成功: {rag_config.bge_reranker_large} "
              f"(device={rag_config.bge_reranker_device})")
    return _reranker_model