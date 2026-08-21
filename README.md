# DeepAgents 深度搜索

基于 LangGraph 的多智能体深度调研系统。用户提出一个调研任务（如「对比 LangGraph / CrewAI / AutoGen 的 GitHub 活跃度，出一份带引用的选型报告」），系统自动编排 **1 个主 Agent + 3 个专业子 Agent**，经历「搜索 → 阅读 → 反思 → 再搜索」的多轮循环，最终产出带逐句引用来源的 Markdown / PDF 报告。

![Python](https://img.shields.io/badge/Python-3.12-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-brightgreen) ![Milvus](https://img.shields.io/badge/Milvus-Hybrid%20Search-teal) ![Docker](https://img.shields.io/badge/Docker-ready-blueviolet)

## 特性

- **主 / 子多智能体编排**：主 Agent 负责任务拆解、工具调度与反思循环；子 Agent 分工协作（网络搜索 / 数据库查询 / 知识库检索），基于 `deepagents` + LangGraph 实现
- **强制数据溯源**：涉及量化对比（star、issue、release 时间、更新频率）时**必须**先查真实数据库，禁止凭网络快照编造数字；查不到的仓库如实标注
- **逐句级引用标注**：报告中每个论断句内联标注出处（数据库 / 网络链接 / 知识库文档），文末附数据来源与采集日期说明
- **置信度降级（confidence gate）**：知识库检索返回 BGE-reranker 重排置信度，低于阈值自动切换网络搜索补充，杜绝低置信度硬写结论
- **调研中断恢复（ResearchFrame）**：每个任务建模为状态帧（LIFO 栈），同主题新任务自动复用已收集证据、从断点续跑，不重头搜索
- **真实可验证的评估**：引用准确率（断言句带出处占比）、维度覆盖度（选型必答 checklist）、省时对比（人工 vs Agent 耗时）三项指标脚本化测量
- **一键 Docker 部署**：应用容器化，外部依赖（MySQL / Milvus / 模型）按部署机配置挂载，密钥不打入镜像

## 系统架构

```
浏览器 (UI + Trace 可视化)
      │ WebSocket
      ▼
:8000  FastAPI + SSE/WS 事件流
      │
      ▼
  ┌─────────────────────────────────────┐
  │   Main Agent（主调度，反思循环）       │
  │   · 任务拆解 todo-list               │
  │   · 强制调度规则（量化对比→数据库）    │
  │   · 置信度降级决策（confidence gate） │
  └───────┬──────────────┬──────────────┘
          │ 调度          │
  ┌───────▼─────┐ ┌──────▼──────┐ ┌──────▼──────────┐
  │ 网络搜索助手  │ │ 数据库查询助手 │ │ 知识库检索助手    │
  │ qwen 联网     │ │ MySQL        │ │ Milvus 混合检索  │
  │ / Tavily     │ │ github_repos │ │ BGE-M3+RRF+重排  │
  └─────────────┘ └─────────────┘ └─────────────────┘
          │              │              │
     互联网公开资料   GitHub API 采集   团队技术文档库
                    （14 个开源仓库）  （LangChain/Agent/Python）
```

调研流程：**拆解子问题 → 多源检索（网络/数据库/知识库）→ 反思是否足够 → 不足则补充检索 → 生成带引用报告**。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 智能体编排 | LangGraph、deepagents、LangChain |
| LLM | Qwen 系列（百炼 API，联网搜索内置）、OpenAI 兼容接口 |
| 检索 | BGE-M3 稠密+稀疏双向量、Milvus 混合检索、RRF 融合、BGE-reranker |
| 数据 | MySQL（GitHub 仓库元数据，GitHub API 采集）、Tavily |
| 服务 | FastAPI、WebSocket 流式、前端 Trace 可视化 |
| 部署 | Docker Compose、requirements-lock.txt 锁定依赖 |

## 快速开始（本地开发）

### 1. 克隆并安装

```bash
git clone <your-repo-url>
cd deepagents
python -m venv .venv
# Windows: .venv\Scripts\activate   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写：

```bash
# LLM（阿里云百炼，或任意 OpenAI 兼容接口）
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-xxx
LLM_QWEN_MAX=qwen-max

# 搜索后端：qwen（默认，走百炼联网）/ duckduckgo / tavily
SEARCH_BACKEND=qwen
TAVILY_API_KEY=          # 若用 tavily 时填写

# MySQL（GitHub 仓库元数据，脚本 scripts/fetch_github_repos.py 写入）
MYSQL_USER=root
MYSQL_PASSWORD=xxx
MYSQL_DATABASE=tech_research
MYSQL_HOST=localhost
MYSQL_PORT=3306

# Milvus 混合检索（BGE-M3 双向量）
MILVUS_URL=localhost:19530
CHUNKS_COLLECTION=tech_docs
EMBEDDING_DIM=1024
BGE_M3_PATH=/path/to/bge-m3
BGE_RERANKER_LARGE=/path/to/bge-reranker-large
```

### 3. 准备数据（可选，用于量化对比）

用 GitHub API 采集候选开源仓库的元数据（star / issue / release / 更新频率）入库：

```bash
python scripts/fetch_github_repos.py
```

### 4. 启动

```bash
python -m api.server
```

浏览器打开 `http://localhost:8000`，输入调研问题（如「对比 LangGraph / CrewAI / AutoGen 的 GitHub 活跃度」），左侧可实时查看 Agent 调度与工具调用 Trace，等待报告生成。

## Docker 部署

详见 [`docker/DEPLOY.md`](docker/DEPLOY.md)。要点：

```bash
# 部署机准备环境配置（密钥不打入镜像）
mkdir -p env models
cp .env env/.env          # 按部署机修改 MILVUS_URL / MYSQL_HOST 等

# 构建并启动
docker compose up -d --build
```

外部依赖（需部署机可达）：MySQL（`tech_research.github_repos`）、Milvus（可选，缺失时知识库子 Agent 自动降级为网络搜索）、BGE 模型目录（可选挂载）。

## 评估指标（真实可验证）

`scripts/evaluate_report.py` 对生成报告做三项脚本化评估。实测数据（全量 15 份正式报告、181 条论断句）：

1. **引用准确率**：报告中断言句（含数字/结论/对比/推荐）中带内联出处标注的占比——**实施强制数据溯源规则前 8 份报告平均 15.4%，实施后 7 份报告平均 36.4%（提升 2.4 倍），单份最高 55.6%**
2. **维度覆盖度**：技术选型必答维度 checklist（性能/生态/学习成本/社区活跃度/维护状态…）逐项核对——全维度选型问题覆盖 60%~100%；聚焦单维（如仅活跃度对比）的报告天然覆盖偏低，属评估设计边界
3. **省时对比**：同一调研任务人工完成 vs Agent 完成的耗时差（自测真实数据）

## 目录结构

```
deepagents/
├── agent/                  # 主 Agent 与子 Agent
│   ├── main_agent.py       #   主调度（反思循环 + 强制调度规则）
│   ├── subagents/          #   网络搜索 / 数据库查询 / 知识库检索
│   └── prompts.py          #   提示词加载
├── api/                    # FastAPI 服务（REST + WebSocket 流式 + Trace）
├── engine/research_frame.py# 调研状态帧（断点续跑，LIFO 栈）
├── tools/                  # Agent 工具（数据库/检索/搜索/文件/PDF）
├── rag/                    # BGE-M3 混合检索（稠密+稀疏、RRF、重排）
├── prompt/prompts.yml      # 主 Agent 系统提示词（调度规则/降级规则/引用要求）
├── scripts/                # GitHub 数据采集 / 报告评估
├── ui/                     # 前端（Trace 可视化）
├── mcp_rag/                # RAG pipeline MCP Server（可选）
├── docker/                 # 部署文档
├── Dockerfile
└── docker-compose.yml
```

## License

MIT
