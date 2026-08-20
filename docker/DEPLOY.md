# DeepAgents 深度搜索 —— Docker 部署说明

## 架构

```
浏览器 → :8000 (FastAPI + 前端 + WebSocket)
              ├── 网络搜索子 Agent（qwen 自带 enable_search，走百炼 API）
              ├── 数据库查询子 Agent（MySQL：tech_research.github_repos）
              └── 知识库检索子 Agent（Milvus 混合检索：BGE-M3 + RRF + reranker）
```

外部依赖（不随容器起，需部署机可达）：
- Milvus（`env/.env` 中配置 `MILVUS_URL`，如 `host.docker.internal:19530`）
- MySQL（`tech_research` 库 + `github_repos` 表，`env/.env` 中配置 `MYSQL_HOST` 等）
- 模型目录（BGE-M3 / bge-reranker-large，挂载到 `/models`）
- 阿里云百炼 API Key（`env/.env`）

## 部署步骤（在目标机器上）

```bash
# 1. 拷贝项目到部署机，进入目录
cd deepagents

# 2. 准备环境配置（密钥不打入镜像）
mkdir -p env models
cp .env env/.env          # 按部署机改 MILVUS_URL / MYSQL_HOST 等

# 3. 模型目录（若部署机没有，可跳过——知识库子 Agent 会降级为网络搜索）
#    将 BGE-M3、bge-reranker-large 放到 models/ 下，或在 .env 中改 BGE_*_PATH=/models/...

# 4. 构建并启动
docker compose up -d --build

# 5. 验证
curl http://localhost:8000/docs        # API 文档
# 浏览器打开 http://localhost:8000     # 前端
```

## 已知约束

- `torch` 使用 CPU 版，镜像约 2-3GB；首次构建 10-20 分钟
- `pywin32` 为 Windows 专属，Linux 容器中 `convert_md_to_pdf` 不可用（PDF 导出仅限 Windows 环境）
- 若部署机无模型：知识库检索返回 `confidence=0`，按降级规则自动改用网络搜索（主链路不受影响）
- 若部署机无法访问外部 Milvus：可将 Milvus 一并容器化（etcd + minio + milvus 编排）或改用部署机可达的实例地址

## 目录结构（Docker 相关）

```
deepagents/
├── Dockerfile               # 应用镜像
├── docker-compose.yml       # 编排（应用 + 挂载）
├── requirements-lock.txt    # 锁定版依赖（由 .venv pip freeze 生成）
├── env/.env                 # 部署机环境配置（不打入镜像）
└── models/                  # BGE-M3 / reranker（可选挂载）
```
