"""
故障注入测试：验证容错 / 护栏机制真实生效
================================================================
运行：.venv\\Scripts\\python.exe scripts\\test_fault_injection.py

覆盖（面试可讲的"故障注入测试"）：
  1. 指数退避重试：瞬时失败自动重试后成功
  2. 重试耗尽：异常向上抛，交由调用方降级（不中断主链路）
  3. 主 Agent 护栏接线：循环上限 / 工具次数上限 / 长任务超时
  4. 网络搜索降级：请求持续失败 → 返回降级提示，主链路不中断
  5. 重试先于降级：首次失败重试后成功 → 返回正常结果而非降级
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
        "工具次数上限常量": "MAX_TOOL_CALLS = 30" in src,
        "长任务超时常量": "TASK_TIMEOUT_SECONDS = 600" in src,
        "recursion_limit 接入 config": '"recursion_limit": MAX_RECURSION_LIMIT' in src,
        "超时护栏接线": "asyncio.wait_for(_consume_stream()" in src,
        "工具计数护栏接线": "tool_call_count > MAX_TOOL_CALLS" in src,
    }
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, f"未接线: {failed}"
    print("[PASS] 测试3: 主 Agent 护栏接线完整（循环上限/工具上限/超时）")


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
    print("=== 全部 5 项通过 ✅ ===")
