# syntax=docker/dockerfile:1
# AssistGen 后端镜像（多阶段构建：构建阶段装依赖，运行阶段只带产物）
#
# 说明：本镜像刻意不包含 .env —— 密钥一律由运行期环境变量注入（compose 的 env_file /
# environment），这样镜像可以被安全地推送和分享。应用侧由 pydantic-settings 读取环境变量，
# 其优先级高于 .env 文件，因此容器内无需存在 .env。

# ---------- 阶段一：构建（只负责把依赖装进独立前缀）----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# faiss / torch / graphrag 生态中的部分包需要本地编译工具链
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

# ---------- 阶段二：运行 ----------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1

# curl 仅用于容器 HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 把构建阶段的依赖整目录搬过来（含 console scripts）
COPY --from=builder /install/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /install/bin /usr/local/bin

# 工作目录必须与本地一致：应用用相对路径写 uploads/、读 app/graphrag/data
WORKDIR /app/dashscope_agent/llm_backend

COPY main.py run.py requirements.txt ./
COPY app ./app
COPY scripts ./scripts
COPY static ./static

EXPOSE 8000

# 存活探针：只证明进程可响应；真正的依赖检查走 /ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# 生产启动参数：单 worker（索引任务跑在进程内线程池，多 worker 会并发写同一份磁盘索引）、
# 关闭热重载、信任反向代理头
CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
