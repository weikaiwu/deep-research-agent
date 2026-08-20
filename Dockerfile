# DeepAgents 深度搜索（AI 技术选型调研助手）应用镜像
# 基础镜像：python 3.12-slim（与开发 venv 的 cpython-312 一致）
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# 系统级依赖：torch CPU / pymilvus / FlagEmbedding 所需基础库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# 依赖安装（锁定版）
COPY requirements-lock.txt /tmp/requirements-lock.txt
# torch 用 CPU 版（镜像体积小）；其余走 pypi
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements-lock.txt

# 复制源码
COPY . .

# 运行目录
RUN mkdir -p /app/output /app/updated

EXPOSE 8000

# 启动 API 服务（含前端静态挂载 + WebSocket）
CMD ["python", "api/server.py"]
