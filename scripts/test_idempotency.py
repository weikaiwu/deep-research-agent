"""幂等键单元测试：验证"重复提交只跑一次"的判定逻辑。

运行：.venv\\Scripts\\python.exe scripts\\test_idempotency.py
或：  pytest scripts/test_idempotency.py -v
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from engine.idempotency import (
    IdempotencyState,
    IdempotencyStore,
    derive_key,
)


def test_inflight_blocks_duplicate():
    """任务运行中再提交同键 → 判定 inflight，不开第二个任务。"""
    store = IdempotencyStore()
    first = store.begin("k1", thread_id="t1", ttl_seconds=60)
    second = store.begin("k1", thread_id="t1", ttl_seconds=60)

    assert first.state is IdempotencyState.STARTED
    assert second.state is IdempotencyState.INFLIGHT
    assert second.record.thread_id == "t1"
    assert second.record.hits == 1, f"命中次数应累计为 1，实际 {second.record.hits}"
    print("[PASS] 任务运行中重复提交被拦截（inflight）")


def test_explicit_key_replays_after_done():
    """显式 Idempotency-Key + 已完成 → 重放，不重跑。"""
    store = IdempotencyStore()
    store.begin("k2", thread_id="t2", ttl_seconds=1800)
    store.complete("k2")

    result = store.begin("k2", thread_id="t2", ttl_seconds=1800, allow_replay=True)
    assert result.state is IdempotencyState.REPLAY, f"应重放，实际 {result.state}"
    assert result.record.state == "done"
    print("[PASS] 显式幂等键完成后重放（replay）")


def test_auto_key_allows_rerun_after_done():
    """自动派生键只拦并发重复：任务已完成后再次提交视为新意图，允许重跑。"""
    store = IdempotencyStore()
    store.begin("k3", thread_id="t3", ttl_seconds=60)
    store.complete("k3")

    result = store.begin("k3", thread_id="t3", ttl_seconds=60, allow_replay=False)
    assert result.state is IdempotencyState.STARTED, f"完成后应允许重跑，实际 {result.state}"
    print("[PASS] 自动键不拦截已完成任务的重跑（只拦并发重复）")


def test_failed_task_allows_retry():
    """任务失败立刻释放键，允许重试。"""
    store = IdempotencyStore()
    store.begin("k4", thread_id="t4", ttl_seconds=60)
    store.fail("k4")

    result = store.begin("k4", thread_id="t4", ttl_seconds=60)
    assert result.state is IdempotencyState.STARTED, f"失败后应允许重跑，实际 {result.state}"
    print("[PASS] 失败的键立刻释放，可重试")


def test_expired_record_allows_new_run():
    """完成记录超过 TTL 后允许重跑（同一问题隔一段时间再问是正常需求）。"""
    store = IdempotencyStore()
    store.begin("k5", thread_id="t5", ttl_seconds=0.01)
    store.complete("k5")
    time.sleep(0.03)

    result = store.begin("k5", thread_id="t5", ttl_seconds=60)
    assert result.state is IdempotencyState.STARTED, f"过期后应重跑，实际 {result.state}"
    print("[PASS] TTL 过期后允许重新执行")


def test_stale_running_released():
    """运行太久（僵尸任务）应释放键，避免永久锁死。"""
    store = IdempotencyStore(stale_seconds=0.01)
    store.begin("k6", thread_id="t6", ttl_seconds=60)
    time.sleep(0.03)

    result = store.begin("k6", thread_id="t6", ttl_seconds=60)
    assert result.state is IdempotencyState.STARTED, f"僵尸任务应被释放，实际 {result.state}"
    print("[PASS] 僵尸任务超时释放键")


def test_derive_key():
    """派生键：同会话同问题稳定，换会话或换问题必须不同。"""
    a = derive_key("thread-1", "对比 LangGraph 和 CrewAI")
    b = derive_key("thread-1", "对比 LangGraph 和 CrewAI")
    c = derive_key("thread-1", "对比 LangGraph 和 AutoGen")
    d = derive_key("thread-2", "对比 LangGraph 和 CrewAI")

    assert a == b, "同会话同问题应得到同一个键"
    assert len({a, c, d}) == 3, "问题或会话变化必须换键，否则会误判重复"
    assert a.startswith("auto:thread-1:")
    print("[PASS] 派生键稳定且能区分会话/问题")


if __name__ == "__main__":
    print("=== 幂等键测试开始 ===")
    test_inflight_blocks_duplicate()
    test_explicit_key_replays_after_done()
    test_auto_key_allows_rerun_after_done()
    test_failed_task_allows_retry()
    test_expired_record_allows_new_run()
    test_stale_running_released()
    test_derive_key()
    print("=== 全部 7 项通过 ✅ ===")
