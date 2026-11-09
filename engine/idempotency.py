"""写接口幂等：同一个任务重复提交只执行一次。

/api/task 是「提交即异步执行」，前端双击、网络重试、页面刷新重发都会再起
一整轮调研（一次几十万 token）。在入口判重比在 Agent 内部检测便宜得多，
也不会污染研究流程。

键的来源：
  - 客户端显式带 Idempotency-Key → 按客户端语义去重，完成后一段时间内重放
  - 没带 → 用 (thread_id, 问题内容) 派生，任务运行期间视为重复提交（防连点/重试）；
    任务已完成后再提交视为新意图，允许重跑
  - 连 thread_id 都没有 → 无法判定是否同一会话，不做判重（宁可放行也不误杀）

存储：进程内字典 + TTL。单 worker 部署够用；多 worker / 多实例时把这个类
换成 Redis 实现即可（接口不变）。
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class IdempotencyState(str, Enum):
    STARTED = "started"      # 首次提交，调用方可以开跑
    INFLIGHT = "inflight"    # 同键任务正在跑（或刚跑完在窗口内），拒绝重复执行
    REPLAY = "replay"        # 同键任务已完成且允许重放，直接返回上次结果引用


@dataclass
class IdempotencyRecord:
    key: str
    thread_id: str
    state: str = "running"   # running | done | failed
    ttl_seconds: float = 60.0
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    expires_at: float = 0.0
    hits: int = 0

    def __post_init__(self) -> None:
        self.expires_at = self.created_at + self.ttl_seconds


@dataclass
class BeginResult:
    state: IdempotencyState
    record: IdempotencyRecord


class IdempotencyStore:
    """按 key 记录任务执行状态，判定重复提交。线程安全。"""

    def __init__(self, stale_seconds: float = 900.0):
        # stale_seconds 要大于单任务最长执行时间，否则长任务中途会被当成僵尸重跑
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = threading.Lock()
        self._stale_seconds = stale_seconds

    def begin(
        self,
        key: str,
        thread_id: str,
        ttl_seconds: float = 60.0,
        allow_replay: bool = False,
    ) -> BeginResult:
        """登记一次提交，返回本次该怎么处理。"""
        now = time.time()
        with self._lock:
            self._gc(now)
            record = self._records.get(key)
            if record is not None:
                record.hits += 1
                if record.state == "running" and now - record.created_at < self._stale_seconds:
                    return BeginResult(IdempotencyState.INFLIGHT, record)
                if record.state == "done" and allow_replay and now < record.expires_at:
                    return BeginResult(IdempotencyState.REPLAY, record)
                # 已完成但不在重放语义内 / 失败 / 跑太久（僵尸）→ 覆盖重跑

            record = IdempotencyRecord(
                key=key, thread_id=thread_id, ttl_seconds=ttl_seconds, created_at=now
            )
            self._records[key] = record
            return BeginResult(IdempotencyState.STARTED, record)

    def complete(self, key: str) -> None:
        """任务正常结束：标记完成，并按 TTL 决定重放窗口。"""
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return
            record.state = "done"
            record.finished_at = time.time()
            record.expires_at = record.finished_at + record.ttl_seconds

    def fail(self, key: str) -> None:
        """任务异常结束：立刻允许重试，不占着键。"""
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return
            record.state = "failed"
            record.finished_at = time.time()
            record.expires_at = record.finished_at

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._gc(time.time())
            return {
                "records": len(self._records),
                "running": sum(1 for r in self._records.values() if r.state == "running"),
            }

    def _gc(self, now: float) -> None:
        expired = [
            key
            for key, record in self._records.items()
            if (record.state != "running" and now >= record.expires_at)
            or (record.state == "running" and now - record.created_at >= self._stale_seconds)
        ]
        for key in expired:
            self._records.pop(key, None)


def derive_key(thread_id: str, query: str) -> str:
    """无显式 Idempotency-Key 时，用 (会话, 问题) 派生去重键。"""
    payload = f"{thread_id}\x00{query.strip()}".encode()
    return f"auto:{thread_id}:{hashlib.sha1(payload).hexdigest()[:16]}"
