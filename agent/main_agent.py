from agent.subagents.database_query_agent import database_query_agent
from agent.subagents.network_search_agent import network_search_agent
from agent.subagents.knowledge_base_agent import knowledge_base_agent
from langgraph.checkpoint.memory import InMemorySaver

# main_agent tool导入
from tools.markdown_tools import generate_markdown
from tools.pdf_tools import convert_md_to_pdf
from tools.upload_file_read_tool import read_file_content
from tools.keyword_extract_tool import keyword_extract  # 项目二微调模型（vLLM 服务）

from deepagents import create_deep_agent

from agent.llm import model
from agent.prompts import main_agent_content

from api.monitor import monitor
from engine.research_frame import research_frame_manager
from engine.guardrails import (
    BudgetCallbackHandler,
    BudgetExceededError,
    GuardrailLimits,
    TaskBudget,
)
import asyncio
import os
import time
import uuid
import shutil
from pathlib import Path

from api.context import set_session_context, reset_session_context, set_thread_context

from langchain_core.messages import AIMessage

main_agent = create_deep_agent(
   model = model,
   system_prompt=main_agent_content['system_prompt'],
   tools= [generate_markdown,convert_md_to_pdf,read_file_content,keyword_extract],
   checkpointer=InMemorySaver(),
   subagents=[
       database_query_agent,
       network_search_agent,
       knowledge_base_agent
   ]
)

# 执行
"""
  1. 执行主智能体 一定选异步，原因：对应多个客户端
  2. 什么时候触发我们智能体的调用或者执行？？？
  3. 客户端 -》 api/task -> fastapi 接口 -》 异步执行 -》 main_agent的运行 （异步方法）
  4. main_agent执行stream流式处理 -》 调用工具 -》 已经埋好了点  
                                   调用子智能体 -》 结果解析 -》 name = task -> monitor -> 发送子智能体
                                   调用最终结果 -》 结果 -》 monitor -> 发送结果的方法
                                   开启调用以后 -》 当前会话 -》 文件夹地址 -》 推送到前端
"""



project_root_path = Path(__file__).parents[1].resolve() # 绝对 解析路径标识以及软连接
# project_root_path = Path(__file__).parents[1].absolute() # 绝对
# main_agent.invoke()
# main_agent.stream()
# main_agent.astream() [选他]

# 记录最近一次调度的子智能体名（tools 节点返回时用于标记"哪个子智能体执行完成"）
_last_subagent_name = {"name": ""}

# ============ 护栏参数（防无限搜索烧钱 / 防长任务挂死）============
# LangGraph 最大递归步数：限制主 Agent 循环轮次上限（默认 25，这里放宽到 40 给足多轮调研空间，但仍封顶）
MAX_RECURSION_LIMIT = 40
# 全链路工具调用次数上限：含子智能体内部的搜索/查库调用（子智能体是嵌套 ainvoke，
# 不计入主图步数，只数顶层 tools 节点会漏掉绝大部分调用）
MAX_TOOL_CALLS = 60
# 子智能体派发次数上限：每次 task 调用背后都是一轮完整的子 Agent 循环
MAX_DELEGATIONS = 20
# token 累计预算：输入+输出合计。默认给到 1M，是为了只兜"失控"而不是误杀正常调研
# （主 Agent 40 轮 + 3 个子智能体各约 20 轮，正常一次调研就可能烧到几十万 token）；
# 可用环境变量按模型单价调整
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "1000000"))
# 同一工具 + 同一参数允许重复的次数，超过判定为绕圈
MAX_DUPLICATE_CALLS = 3
# 单任务最长执行秒数：超时强制终止（长任务护栏）
TASK_TIMEOUT_SECONDS = 600


async def run_deep_agent(task_query,session_id):
    """
    定义流式+异步执行主智能体！！
    执行过程中，返回  会话文件化返回  调用子智能体  调用最终结果 （monitor）
    task_query: 前端提问的问题
    session_id: 每个前端会话对应的标识 （1.存储session_id ContextVars 2.session_id 给他创建对应的output输出地址）
    """
    print(f"当前会话的main_agent开始执行了！ 会话id:{session_id}")
    # 准备工作 【1. session_dir（前端） 2. relative_session_dir (大模型) 3. 上传的文件拼接上传文件专属提示词】
    # project_root_path / output / session_session_id(uuid)
    # 当前会话存储生成文件的专属文件夹
    session_dir = project_root_path / "output" / f"session_{session_id}"
    # 文件夹可能没有，第一次请求要创建
    session_dir.mkdir(parents=True, exist_ok=True)
    # \  \n \t -> /
    session_dir_str = str(session_dir).replace("\\","/")
    # 获取相对文件夹
    # session_dir : project_root_path / output / session_session_id(uuid)
    # project_root_path : project_root_path
    # relative_session_dir_str: / output / session_session_id(uuid)
    relative_session_dir_str = str(session_dir.relative_to(project_root_path)).replace("\\","/")

    #处理上传文件 （updated / session_session_id）
    updated_dir_path = project_root_path / "updated" / f"session_{session_id}"
    updated_info_prompt = "" # 有上传文件，拼接上传文件专属解析位置的提示词
    if updated_dir_path.exists():
        # 有
        files = [ f.name  for f in updated_dir_path.iterdir()  if f.is_file()]
        # 将上传文件统一赋值到 output_dir 方便前端统一读取 session_dir
        if files:
            for filename in files:
                # 将原文件 -》 复制 -》 目标文件中  （copy2 保留原文件修改时间和权限等元数据）
                shutil.copy2(updated_dir_path / filename, session_dir / filename)
            # 构建提示词！告诉大模型，有上传文件，你要读取上传文件！！
            updated_info_prompt = (f"\n    [已上传文件] 已加载到工作目录:\n" +
                             "\n".join([f"    - {f}" for f in files]) +
                             "\n    请优先使用工具（read_file_content）读取并参考这些文件。")

    # 继续准备 1. 当前会话的对应的session_id session_dir 存储到contextVars [后续工具获取，socket -> 推送消息] 2.调用monitor给前端推送session_dir信息
    session_dir_token = set_session_context(session_dir_str)  # 存储的当前会话对应的文件夹地址
    session_id_token = set_thread_context(session_id)  #获取当前会话的session_id对应socket

    monitor.report_session_dir(session_dir_str)  # 当前会话对应的文件夹地址推送给起前端！

    # 执行main_agent
    # 本次任务的预算账本：token 累计 / 全链路工具调用 / 重复调用检测。
    # 计数挂在 callback 上，langgraph 的 ensure_config 会把它带进子智能体内部，
    # 所以子智能体自己的搜索、查库调用也会被算进来。
    budget = TaskBudget(
        session_id=session_id,
        limits=GuardrailLimits(
            max_tool_calls=MAX_TOOL_CALLS,
            max_delegations=MAX_DELEGATIONS,
            max_total_tokens=MAX_TOTAL_TOKENS,
            max_duplicate_calls=MAX_DUPLICATE_CALLS,
        ),
    )
    config = {
        "configurable":{
            "thread_id":session_id
        },
        # 循环轮次上限：LangGraph 达到 recursion_limit 会抛 GraphRecursionError，
        # 由外层 try/except 捕获后向前端报告，避免"无限搜索烧钱"
        "recursion_limit": MAX_RECURSION_LIMIT,
        # 预算护栏回调（BudgetExceededError 会穿透回调层中断执行）
        "callbacks": [BudgetCallbackHandler(budget)],
    }

    # 构建提示词
    # ---- ResearchFrame：同主题检测（断点续跑，借鉴 DialogueStack 思想）----
    frame_hint = ""
    similar_frame = research_frame_manager.find_similar(session_id, task_query)
    if similar_frame and similar_frame.status == "completed":
        # 命中已完成帧 → 复用已收集证据，不重头搜索
        monitor._emit("info", 
            f"命中历史调研帧 {similar_frame.frame_id}（同主题），复用上次调研结果，从断点继续"
        )
        frame_hint = (
            "\n    [历史调研上下文] 你之前已调研过相似主题，以下是已收集证据（直接复用，不要重复搜索同一主题）：\n"
            f"{similar_frame.summary()}"
        )
        current_frame = similar_frame
        current_frame.status = "active"
    else:
        # 新调研任务 → 压栈新帧
        current_frame = research_frame_manager.new_frame(
            session_id, task_query[:80], task_query, thread_id=session_id,
        )
        monitor._emit("info", f"新建调研帧 {current_frame.frame_id}（主题：{task_query[:40]}）")

    path_instruction = f"""
    【工作环境指令】
    工作目录: {relative_session_dir_str}
    {updated_info_prompt}{frame_hint}

    规则：
    1. 新生成文件必须保存到工作目录：'{relative_session_dir_str}/filename'
    2. 读取已上传的文件时，请直接将文件名（例如：'开篇.txt'）作为 filename 参数传入（read_file_content）读取工具，不要带上任何目录前缀。
    3. 使用相对路径，禁止使用绝对路径
    4. 若存在上传文件，请先分析内容
    """
    # 反馈结果
    start_time = time.time()

    async def _consume_stream():
        """消费主 Agent 流式输出。

        工具次数 / token 预算 / 重复调用的判定都在 BudgetCallbackHandler 里做，
        这里只负责把告警推给前端；超限由回调抛异常中断，落到下面的 except 分支。
        """
        async for chunk in main_agent.astream({
            "messages":[
                {
                    "role":"user","content":task_query+path_instruction
                }
            ]
        },config=config):
            # 预算预警（如 token 用到 80%）先推给前端，任务继续跑
            for warning in budget.take_warnings():
                monitor._emit("warning", warning)
            # {"model [大模型决定调用工具 子智能体  最终结果] / tools" : {messages:[xxx...]}}
            for node_name,state in chunk.items():
                if not state or "messages" not in state: continue
                messages = state["messages"]
                if messages and isinstance(messages,list):
                    last_msg = messages[-1]
                    if node_name == 'model':
                        if last_msg.tool_calls:
                            # 工具和子智能体
                            for tool_call in last_msg.tool_calls:
                                """
                                  tool_call = {
                                      name: task
                                      args:{
                                          subagent_type:子智能体的名字
                                          description:子智能体的描述
                                      }
                                  }                                
                                """
                                if tool_call['name'] == 'task':
                                    # 调用某个子智能体
                                    _last_subagent_name["name"] = tool_call['args']['subagent_type']
                                    monitor.report_assistant(tool_call['args']['subagent_type'],{'description':tool_call['args']['description']})
                        elif last_msg.content:
                            # 最终结果
                            print(f"主智能体执行结果，最终结果：{last_msg.content[:100]}")
                            monitor.report_task_result(last_msg.content, session_dir_str)
                            # ResearchFrame：标记当前帧完成 + 记录结论摘要
                            try:
                                research_frame_manager.mark_completed(
                                    session_id, current_frame.frame_id,
                                    conclusion=last_msg.content[:200],
                                )
                                monitor._emit("info", f"调研帧 {current_frame.frame_id} 已完成")
                            except Exception as e:
                                print(f"[Frame] 标记完成失败: {e}")
                    elif node_name == 'tools':
                        # 工具/子智能体执行完后的返回消息（ToolMessage）
                        # 子智能体 = task 工具：返回内容即子智能体的最终输出
                        if getattr(last_msg, 'name', None) == 'task':
                            content = getattr(last_msg, 'content', '') or ''
                            summary = content[:200] if isinstance(content, str) else str(content)[:200]
                            monitor.report_assistant_end(_last_subagent_name["name"] or "子智能体", summary)

    try:
        # 长任务护栏：整体超时强制终止（防止任务无限挂起）
        await asyncio.wait_for(_consume_stream(), timeout=TASK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        # 超时终止（长任务护栏触发）
        monitor._emit("error", f"任务执行超过 {TASK_TIMEOUT_SECONDS}s，触发长任务护栏，任务终止")
    except BudgetExceededError as e:
        # 预算护栏触发（token / 工具次数 / 子智能体派发 / 重复调用）
        monitor._emit("error", f"任务终止：{e}")
    except Exception as e :
        if budget.violation:
            # 预算异常可能被 langgraph 包装成别的异常类型，以账本记录为准
            reason, detail = budget.violation
            monitor._emit("error", f"任务终止：执行预算护栏触发[{reason}]：{detail}")
        else:
            # 报错推送错误信息给前端（含 recursion_limit 触发的 GraphRecursionError）
            monitor._emit("error",f"执行主智能发生异常信息：{str(e)}")
    finally:
        # 耗时统计（省时对比 KPI 的 Agent 侧真实数据）
        elapsed = time.time() - start_time
        print(f"[耗时] 会话 {session_id} 端到端耗时: {elapsed:.1f}s")
        monitor._emit("info", f"任务耗时 {elapsed:.1f} 秒")
        # 预算用量上报（token / 工具调用 / 重复调用），供前端与成本分析用
        budget_summary = budget.summary()
        monitor.report_budget(budget_summary)
        print(f"[预算] 工具调用 {budget_summary['tool_calls']} 次 / "
              f"子智能体 {budget_summary['delegations']} 次 / "
              f"token {budget_summary['total_tokens']}（预算 {budget_summary['token_budget']}）")
        # 释放存储的地址和session_id
        reset_session_context(session_dir_token, session_id_token)

