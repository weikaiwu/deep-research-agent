"""执行期预算护栏：全链路工具计数、token 累计预算、重复调用检测。

为什么不能只靠 recursion_limit / timeout：
  1. deepagents 的子智能体是在 `task` 工具里嵌套 `ainvoke` 跑的
     （见 deepagents/middleware/subagents.py 的 atask），它自己的循环步数
     不计入主图，recursion_limit 管不到；
  2. 原来的 MAX_TOOL_CALLS 只数顶层 `tools` 节点的 chunk，子智能体内部的
     搜索 / 查库调用一次都没算进来；
  3. 轮数和时长都管不住「同一个工具用同样参数反复调用」这种绕圈。

所以把计数挂到 callback 上：handler 通过顶层 config 传入，langgraph 的
ensure_config 会把它传播到子智能体内部（deepagents 源码注释已明确这点），
on_tool_start / on_llm_end 因此能覆盖整条调用链。

硬上限触发时抛 BudgetExceededError，配合 handler.raise_error = True，
异常会穿透回调层在工具/模型真正执行前中断，而不是等这一轮跑完。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class BudgetExceededError(RuntimeError):
    """执行预算耗尽（token / 工具调用次数 / 子智能体派发 / 重复调用）。"""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"执行预算护栏触发[{reason}]：{detail}")


@dataclass
class GuardrailLimits:
    """单次任务的预算上限。"""

    max_tool_calls: int = 60          # 全链路工具调用总数（含子智能体内部）
    max_delegations: int = 20         # task 派发子智能体的次数
    max_total_tokens: int = 300_000   # 全链路 token 累计（输入+输出）
    max_duplicate_calls: int = 3      # 同一工具 + 同一参数允许重复的次数
    warn_token_ratio: float = 0.8     # token 用量预警线（只告警不中断）


def _normalize_args(inputs: Any) -> str:
    """把工具入参归一化成稳定字符串：dict 键排序，忽略格式化差异。"""
    try:
        return json.dumps(inputs, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(inputs)


@dataclass
class TaskBudget:
    """一次任务的预算账本。计数可能来自不同线程（回调），内部加锁。"""

    session_id: str
    limits: GuardrailLimits = field(default_factory=GuardrailLimits)

    tool_calls: int = 0
    delegations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    violation: tuple[str, str] | None = None

    _call_counts: dict[str, int] = field(default_factory=dict)
    _call_labels: dict[str, str] = field(default_factory=dict)
    _warnings: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _token_warned: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    # ---------- 记录 ----------

    def record_tool_call(self, tool_name: str, inputs: Any) -> None:
        """记录一次工具调用；重复/超量时抛 BudgetExceededError。"""
        args = _normalize_args(inputs)
        signature = f"{tool_name}|{hashlib.sha1(args.encode('utf-8')).hexdigest()[:12]}"

        with self._lock:
            self.tool_calls += 1
            if tool_name == "task":
                self.delegations += 1
            seen = self._call_counts.get(signature, 0) + 1
            self._call_counts[signature] = seen
            if signature not in self._call_labels:
                preview = args if len(args) <= 60 else args[:57] + "..."
                self._call_labels[signature] = f"{tool_name}({preview})"
            tool_calls, delegations = self.tool_calls, self.delegations

        if seen > self.limits.max_duplicate_calls:
            self._raise("重复调用",
                        f"{self._call_labels[signature]} 用相同参数已被调用 {seen} 次"
                        f"（上限 {self.limits.max_duplicate_calls} 次），疑似陷入循环")
        if delegations > self.limits.max_delegations:
            self._raise("子智能体派发上限",
                        f"已派发 {delegations} 次子智能体（上限 {self.limits.max_delegations} 次）")
        if tool_calls > self.limits.max_tool_calls:
            self._raise("工具调用上限",
                        f"全链路工具调用 {tool_calls} 次（上限 {self.limits.max_tool_calls} 次）")

    def record_llm_call(self, input_tokens: int, output_tokens: int) -> None:
        """记录一次大模型调用的 token 消耗；超预算时抛 BudgetExceededError。"""
        input_tokens = max(int(input_tokens or 0), 0)
        output_tokens = max(int(output_tokens or 0), 0)

        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            total = self.input_tokens + self.output_tokens
            should_warn = (
                not self._token_warned
                and self.limits.max_total_tokens > 0
                and total >= self.limits.max_total_tokens * self.limits.warn_token_ratio
                and total <= self.limits.max_total_tokens
            )
            if should_warn:
                self._token_warned = True
                self._warnings.append(
                    f"token 用量已达 {total}/{self.limits.max_total_tokens}"
                    f"（预警线 {int(self.limits.warn_token_ratio * 100)}%）"
                )

        if self.limits.max_total_tokens > 0 and total > self.limits.max_total_tokens:
            self._raise("token 预算",
                        f"累计消耗 {total} tokens，超出预算 {self.limits.max_total_tokens}")

    def _raise(self, reason: str, detail: str) -> None:
        self.violation = (reason, detail)
        raise BudgetExceededError(reason, detail)

    # ---------- 读取 ----------

    def take_warnings(self) -> list[str]:
        """取走待推送的告警（取完即清空，避免前端重复刷）。"""
        with self._lock:
            warnings, self._warnings = self._warnings, []
        return warnings

    def summary(self) -> dict[str, Any]:
        with self._lock:
            repeated = [
                {"call": self._call_labels.get(sig, sig), "count": count}
                for sig, count in self._call_counts.items()
                if count > 1
            ]
            repeated.sort(key=lambda item: item["count"], reverse=True)
            payload = {
                "session_id": self.session_id,
                "tool_calls": self.tool_calls,
                "delegations": self.delegations,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "token_budget": self.limits.max_total_tokens,
                "repeated_calls": repeated,
                "limits": {
                    "max_tool_calls": self.limits.max_tool_calls,
                    "max_delegations": self.limits.max_delegations,
                    "max_total_tokens": self.limits.max_total_tokens,
                    "max_duplicate_calls": self.limits.max_duplicate_calls,
                },
            }
        if self.violation:
            payload["violation"] = {"reason": self.violation[0], "detail": self.violation[1]}
        return payload


def _extract_usage(response: Any) -> tuple[int, int]:
    """从 LLMResult 里取 token 用量，兼容 usage_metadata 与旧版 token_usage。"""
    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


class BudgetCallbackHandler(BaseCallbackHandler):
    """把全链路工具调用与 token 消耗记进 TaskBudget。

    raise_error = True 让 BudgetExceededError 穿透回调层真正中断执行；
    但计数本身出错（字段缺失等）不该毁掉用户的任务，所以业务异常一律吞掉。
    """

    raise_error = True

    def __init__(self, budget: TaskBudget):
        super().__init__()
        self.budget = budget

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._guard(self.budget.record_tool_call, tool_name, inputs if inputs is not None else input_str)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        input_tokens, output_tokens = _extract_usage(response)
        self._guard(self.budget.record_llm_call, input_tokens, output_tokens)

    @staticmethod
    def _guard(func: Any, *args: Any) -> None:
        try:
            func(*args)
        except BudgetExceededError:
            raise  # 预算超限必须中断，不能被吞
        except Exception:  # 计数本身出错不该毁掉用户任务
            logger.warning("预算计数失败，已忽略该次记录", exc_info=True)
