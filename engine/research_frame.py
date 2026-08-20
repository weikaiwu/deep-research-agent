"""
ResearchFrame 栈 —— 调研任务状态管理（借鉴 DialogueStack 的 LIFO 栈思想）
========================================================================
每个调研任务建模为一个 ResearchFrame（帧），多个帧组成 LIFO 栈。
核心能力：
  1. 同主题检测：新任务进来，先检查是否已有同主题帧 → 有则从断点续跑（复用已收集证据），不重头搜索
  2. 断点续跑：帧保存 thread_id（LangGraph checkpoint 会话标识），同一 thread 继续执行可复用中间状态
  3. 帧状态流转：active（进行中）→ completed（完成）；中断的帧保留在栈中供恢复

设计思想（栈式状态管理，借 LIFO 栈思想，非代码复用）：
  task_stack        → ResearchFrameStack（LIFO 栈，天然支持多任务切换与恢复）
  task_interrupted  → frame 标记 paused 保留在栈中
  task_resumed      → find_similar 命中 → 复用帧续跑
"""
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ResearchFrame:
    """单个调研任务的状态帧"""
    topic: str                          # 调研主题（从用户问题提取）
    goal: str                           # 用户原始需求
    frame_id: str = ""                  # 帧唯一标识（push 时由栈分配）
    sub_questions: List[str] = field(default_factory=list)   # 已拆解的调研子问题
    evidence: List[Dict] = field(default_factory=list)       # 已收集证据（子 agent 返回摘要）
    status: str = "active"              # active / paused / completed
    thread_id: str = ""                 # LangGraph checkpoint 会话标识（断点续跑用）
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_evidence(self, source: str, summary: str):
        """追加一条证据摘要"""
        self.evidence.append({"source": source, "summary": summary[:500]})
        self.updated_at = time.time()

    def summary(self, max_len: int = 400) -> str:
        """帧摘要：主题 + 已收集证据，供断点续跑时拼进 prompt"""
        parts = [f"调研主题：{self.topic}"]
        if self.evidence:
            parts.append("已收集证据：")
            for e in self.evidence[-5:]:
                parts.append(f"- [{e['source']}] {e['summary'][:max_len]}")
        return "\n".join(parts)


class ResearchFrameStack:
    """LIFO 调研帧栈（一个会话一个栈）"""

    def __init__(self):
        self._frames: List[ResearchFrame] = []
        self._next_id = 1

    # ---------- 栈操作 ----------
    def push(self, frame: ResearchFrame):
        frame.frame_id = f"frame_{self._next_id}"
        self._next_id += 1
        self._frames.append(frame)

    def pop(self) -> Optional[ResearchFrame]:
        return self._frames.pop() if self._frames else None

    def top(self) -> Optional[ResearchFrame]:
        return self._frames[-1] if self._frames else None

    def all_frames(self) -> List[ResearchFrame]:
        return list(self._frames)

    # ---------- 同主题检测 ----------
    @staticmethod
    def _extract_keywords(text: str, max_kw: int = 8) -> set:
        """粗粒度关键词提取：英文单词 + 中文词（按长度切），去停用词"""
        text = text.lower()
        # 英文词
        en_words = re.findall(r"[a-z][a-z0-9\-\.]+", text)
        # 中文词：按 2-4 字滑窗（粗粒度）
        cn_seg = re.findall(r"[\u4e00-\u9fff]{2,4}", text)
        stop = {"对比", "区别", "介绍", "什么", "如何", "怎么", "进行", "一个", "这个",
                "那个", "为什么", "哪个", "我们", "他们", "可以", "需要", "相关", "请问"}
        kws = set(en_words) | set(cn_seg)
        filtered = [k for k in kws if k not in stop and len(k) >= 2]
        return set(filtered[:max_kw])

    def find_similar(self, query: str, min_overlap: int = 2) -> Optional[ResearchFrame]:
        """
        同主题检测：新问题与栈中帧做关键词重叠，重叠数 >= min_overlap 视为同主题。
        返回可复用的帧（优先 completed 的，其次最近 active 的）。
        """
        if not self._frames:
            return None
        q_kws = self._extract_keywords(query)
        if not q_kws:
            return None
        best_frame, best_score = None, 0
        for frame in self._frames:
            f_kws = self._extract_keywords(frame.topic + " " + frame.goal)
            overlap = len(q_kws & f_kws)
            if overlap >= min_overlap and overlap > best_score:
                best_frame, best_score = frame, overlap
        return best_frame


class ResearchFrameManager:
    """会话级管理器：session_id -> ResearchFrameStack（进程内存态，单例）"""

    _instance: Optional["ResearchFrameManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._stacks: Dict[str, ResearchFrameStack] = {}
            return cls._instance

    def get_stack(self, session_id: str) -> ResearchFrameStack:
        if session_id not in self._stacks:
            self._stacks[session_id] = ResearchFrameStack()
        return self._stacks[session_id]

    def new_frame(self, session_id: str, topic: str, goal: str, thread_id: str = "") -> ResearchFrame:
        stack = self.get_stack(session_id)
        frame = ResearchFrame(topic=topic, goal=goal, thread_id=thread_id)
        stack.push(frame)
        return frame

    def find_similar(self, session_id: str, query: str) -> Optional[ResearchFrame]:
        return self.get_stack(session_id).find_similar(query)

    def mark_completed(self, session_id: str, frame_id: str, conclusion: str = ""):
        for frame in self.get_stack(session_id).all_frames():
            if frame.frame_id == frame_id:
                frame.status = "completed"
                frame.updated_at = time.time()
                if conclusion:
                    frame.add_evidence("结论", conclusion)
                return frame
        return None


# 全局单例
research_frame_manager = ResearchFrameManager()
