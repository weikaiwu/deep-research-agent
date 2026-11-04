"""
故障注入测试：验证容错 / 护栏机制真实生效
================================================================
运行：.venv\\Scripts\\python.exe scripts\\test_fault_injection.py

覆盖（故障注入测试）：
  1. 指数退避重试：瞬时失败自动重试后成功
  2. 重试耗尽：异常向上抛，交由调用方降级（不中断主链路）
  3. 主 Agent 护栏接线：循环上限 / 工具次数 / 子智能体派发 / token 预算 / 超时
  4. 网络搜索降级：请求持续失败 → 返回降级提示，主链路不中断
  5. 重试先于降级：首次失败重试后成功 → 返回正常结果而非降级
  6. 预算护栏行为：重复调用检测 / token 预算 / 工具与派发上限
  7. 回调不吞预算异常：BudgetExceededError 能穿透回调层中断执行
"""
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).parents[1].resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def test_retry_success():
    """测试 1：瞬时失败（前 2 次抛异常）应自动重试，第 3 次成功。"""
    from utils.retry import retry_with_backoff
    calls = {"n": 0}

    @retry_with_backoff(retries=2, base_delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("瞬时失败")
        return "ok"

    assert flaky() == "ok", "瞬时失败应重试后成功"
    assert calls["n"] == 3, f"应尝试 3 次，实际 {calls['n']}"
    print("[PASS] 测试1: 瞬时失败自动重试成功（共尝试 3 次）")


def test_retry_exhausted():
    """测试 2：重试耗尽后异常向上抛（由调用方决定降级）。"""
    from utils.retry import retry_with_backoff

    @retry_with_backoff(retries=2, base_delay=0.01)
    def always_fail():
        raise ConnectionError("持续失败")

    try:
        always_fail()
        raise AssertionError("重试耗尽后应抛出异常")
    except ConnectionError:
        pass
    print("[PASS] 测试2: 重试耗尽后异常向上抛（交由调用方降级）")


def test_guardrail_wiring():
    """测试 3：主 Agent 护栏参数与接线（读源文件静态检查，避免 import 副作用）。"""
    src = (PROJECT_ROOT / "agent" / "main_agent.py").read_text(encoding="utf-8")
    checks = {
        "循环上限常量": "MAX_RECURSION_LIMIT = 40" in src,
        "工具次数上限常量": "MAX_TOOL_CALLS = 60" in src,
        "子智能体派发上限常量": "MAX_DELEGATIONS = 20" in src,
        "重复调用上限常量": "MAX_DUPLICATE_CALLS = 3" in src,
        "token 预算常量": "MAX_TOTAL_TOKENS = int(os.getenv" in src,
        "长任务超时常量": "TASK_TIMEOUT_SECONDS = 600" in src,
        "recursion_limit 接入 config": '"recursion_limit": MAX_RECURSION_LIMIT' in src,
        "超时护栏接线": "asyncio.wait_for(_consume_stream()" in src,
        "预算回调接入 config": '"callbacks": [BudgetCallbackHandler(budget)]' in src,
        "预算异常单独处理": "except BudgetExceededError as e:" in src,
        "预算摘要上报": "monitor.report_budget(budget_summary)" in src,
    }
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, f"未接线: {failed}"
    print("[PASS] 测试3: 主 Agent 护栏接线完整（轮数/工具/派发/token/超时）")


def test_budget_duplicate_detection():
    """测试 6：同一工具 + 同一参数重复调用应被识别，参数不同不算重复。"""
    from engine.guardrails import BudgetExceededError, GuardrailLimits, TaskBudget

    budget = TaskBudget("t-dup", GuardrailLimits(max_duplicate_calls=3))
    same_args = {"query": "LangGraph vs CrewAI"}
    for _ in range(3):
        budget.record_tool_call("network_search", same_args)  # 前 3 次放行
    try:
        budget.record_tool_call("network_search", same_args)
        raise AssertionError("第 4 次相同调用应触发重复调用护栏")
    except BudgetExceededError as exc:
        assert exc.reason == "重复调用", f"原因应为重复调用，实际 {exc.reason}"

    other = TaskBudget("t-dup2", GuardrailLimits(max_duplicate_calls=3))
    for index in range(5):
        other.record_tool_call("network_search", {"query": f"不同查询 {index}"})
    print("[PASS] 测试6a: 重复调用检测生效（相同参数第 4 次中断，不同参数不误判）")


def test_budget_token_budget():
    """测试 6：token 累计到 80% 预警、超出预算中断。"""
    from engine.guardrails import BudgetExceededError, GuardrailLimits, TaskBudget

    budget = TaskBudget("t-token", GuardrailLimits(max_total_tokens=1000))
    budget.record_llm_call(400, 100)
    assert budget.take_warnings() == [], "未到预警线不应告警"

    budget.record_llm_call(400, 100)          # 累计 1000，正好到预算
    assert budget.total_tokens == 1000
    warnings = budget.take_warnings()
    assert warnings and "预警" in warnings[0], f"到达 80% 应告警，实际 {warnings}"
    assert budget.take_warnings() == [], "预警只推一次"

    try:
        budget.record_llm_call(1, 0)
        raise AssertionError("超出 token 预算应中断")
    except BudgetExceededError as exc:
        assert exc.reason == "token 预算", f"原因应为 token 预算，实际 {exc.reason}"

    summary = budget.summary()
    assert summary["total_tokens"] == 1001 and summary["token_budget"] == 1000
    assert summary["violation"]["reason"] == "token 预算"
    print("[PASS] 测试6b: token 预算生效（80% 预警一次，超预算中断并留档）")


def test_budget_tool_and_delegation_caps():
    """测试 6：全链路工具调用上限与子智能体派发上限各自生效。"""
    from engine.guardrails import BudgetExceededError, GuardrailLimits, TaskBudget

    tools_budget = TaskBudget("t-tools", GuardrailLimits(max_tool_calls=3, max_delegations=10))
    for index in range(3):
        tools_budget.record_tool_call("milvus_search", {"query": f"q{index}"})
    try:
        tools_budget.record_tool_call("milvus_search", {"query": "q99"})
        raise AssertionError("超过工具调用上限应中断")
    except BudgetExceededError as exc:
        assert exc.reason == "工具调用上限", f"原因应为工具调用上限，实际 {exc.reason}"

    deleg_budget = TaskBudget("t-deleg", GuardrailLimits(max_tool_calls=100, max_delegations=2))
    for index in range(2):
        deleg_budget.record_tool_call("task", {"description": f"调研方向 {index}",
                                               "subagent_type": "network_search_agent"})
    try:
        deleg_budget.record_tool_call("task", {"description": "第三个方向",
                                               "subagent_type": "network_search_agent"})
        raise AssertionError("超过子智能体派发上限应中断")
    except BudgetExceededError as exc:
        assert exc.reason == "子智能体派发上限", f"原因应为派发上限，实际 {exc.reason}"

    assert deleg_budget.summary()["delegations"] == 3
    print("[PASS] 测试6c: 工具调用上限 / 子智能体派发上限分别生效")


def test_budget_callback_propagation():
    """测试 7：handler 读得到 token 用量，且预算异常不被回调层吞掉。"""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from engine.guardrails import (
        BudgetCallbackHandler,
        BudgetExceededError,
        GuardrailLimits,
        TaskBudget,
    )

    budget = TaskBudget("t-cb", GuardrailLimits(max_duplicate_calls=1, max_total_tokens=10_000))
    handler = BudgetCallbackHandler(budget)
    assert handler.raise_error is True, "预算异常必须能穿透回调层，否则中断无效"

    message = AIMessage(
        content="调研完成",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert budget.total_tokens == 150, f"应累计 150 token，实际 {budget.total_tokens}"

    handler.on_tool_start({"name": "network_search"}, "", inputs={"query": "x"})
    try:
        handler.on_tool_start({"name": "network_search"}, "", inputs={"query": "x"})
        raise AssertionError("重复调用应抛 BudgetExceededError")
    except BudgetExceededError:
        pass
    print("[PASS] 测试7: 回调取到 token 用量且预算异常可穿透")



def test_qwen_degrade():
    """测试 4：网络搜索持续失败 → 重试 2 次耗尽 → 返回降级提示，不中断主链路。"""
    from tools.tavily_tool import _qwen_web_search

    with mock.patch("tools.tavily_tool.requests.post",
                    side_effect=Exception("网络不可达")), \
         mock.patch("utils.retry.time.sleep"):  # 加速重试等待（不真睡 2s/4s）
        result = _qwen_web_search("测试查询", 5)

    content = result["results"][0]["content"]
    assert "暂不可用" in content, f"应返回降级提示，实际: {content[:60]}"
    print("[PASS] 测试4: 网络搜索重试 2 次耗尽后降级（主链路不中断）")


def test_retry_before_degrade():
    """测试 5：首次失败应重试并成功（mock 网络第 1 次失败、第 2 次成功）。"""
    from tools.tavily_tool import _qwen_web_search
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("第一次请求失败")

        class FakeResp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "正常检索结果"}}]}

        return FakeResp()

    with mock.patch("tools.tavily_tool.requests.post", side_effect=fake_post), \
         mock.patch("utils.retry.time.sleep"):
        result = _qwen_web_search("测试查询", 5)

    assert result["results"][0]["content"] == "正常检索结果", f"重试成功后应返回正常结果，实际: {result['results'][0]['content'][:60]}"
    assert calls["n"] == 2, f"应尝试 2 次，实际 {calls['n']}"
    print("[PASS] 测试5: 重试先于降级（第 1 次失败 → 重试 → 返回正常结果）")


if __name__ == "__main__":
    print("=== 故障注入测试开始 ===")
    test_retry_success()
    test_retry_exhausted()
    test_guardrail_wiring()
    test_qwen_degrade()
    test_retry_before_degrade()
    test_budget_duplicate_detection()
    test_budget_token_budget()
    test_budget_tool_and_delegation_caps()
    test_budget_callback_propagation()
    print("=== 全部 9 项通过 ✅ ===")
