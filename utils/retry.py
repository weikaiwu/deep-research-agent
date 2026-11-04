"""
通用指数退避重试装饰器
================================================================
用于外部 API（网络搜索 / 数据库 / 微调模型服务）的瞬时失败重试：
  - 默认重试 2 次（共 3 次尝试），间隔 2s / 4s 指数退避
  - 重试耗尽后异常继续向上抛，由调用方决定降级策略
  - 不引入 tenacity 依赖，~20 行自实现，重试逻辑集中在一处便于排查
"""
import functools
import logging
import time

logger = logging.getLogger(__name__)


def retry_with_backoff(
    retries: int = 2,
    base_delay: float = 2.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    指数退避重试装饰器。

    Args:
        retries: 重试次数（不含首次尝试，默认 2 次 → 共 3 次尝试）
        base_delay: 首次重试前等待秒数
        backoff_factor: 退避倍率（间隔 = base_delay * factor^(attempt-1)）
        exceptions: 需要重试的异常类型（默认所有异常）

    Returns:
        装饰器，包装函数具备重试能力；重试耗尽后抛出最后一次异常。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt > retries:
                        # 重试耗尽：把最后一次异常抛给调用方，由调用方降级
                        logger.error(f"[Retry] {func.__name__} 重试 {retries} 次后仍失败: {e}")
                        raise
                    delay = base_delay * (backoff_factor ** (attempt - 1))
                    logger.warning(
                        f"[Retry] {func.__name__} 第 {attempt} 次调用失败 ({e})，"
                        f"{delay:.1f}s 后重试（共重试 {retries} 次）"
                    )
                    time.sleep(delay)
        return wrapper
    return decorator
