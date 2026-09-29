# LangGraph 多智能体智能客服系统

基于 FastAPI、Vue 3 与 LangGraph 的智能客服系统。项目的模型能力统一由 DashScope 提供，包括通用对话、思考模式、图像理解和检索向量；不支持切换到其他模型提供商。

## 模型映射

| 用途 | 默认模型 | 环境变量 |
| --- | --- | --- |
| 通用问答与工具调用 | `qwen3.7-plus` | `DASHSCOPE_CHAT_MODEL` |
| 思考模式 | `qwen3.7-plus` | `DASHSCOPE_REASON_MODEL` |
| 图像理解 | `qwen3-vl-plus` | `DASHSCOPE_VISION_MODEL` |
| 文档检索向量 | `qwen3.7-text-embedding` | `DASHSCOPE_EMBEDDING_MODEL` |

## 快速启动

1. 创建并激活虚拟环境，然后在后端目录安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. 将后端项目中的 `.env.example` 复制为 `.env`。不要把真实密钥提交到仓库。以下是模型相关的最小配置：

```env
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
DASHSCOPE_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen3.7-plus
DASHSCOPE_REASON_MODEL=qwen3.7-plus
DASHSCOPE_VISION_MODEL=qwen3-vl-plus
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding
```

`.env.example` 还包含 MySQL、Redis、Neo4j 与 SerpAPI 的本地连接占位配置；启动前请按部署环境补齐这些必填项。

3. 进入后端目录并启动：

```bash
cd <后端目录>/llm_backend
python run.py
```

默认服务地址为 `http://localhost:8000`，API 文档为 `http://localhost:8000/docs`。

## 流式接口

`POST /api/chat`、`POST /api/reason` 与 LangGraph 流式接口返回 `text/event-stream`。每条 SSE 均仅使用 `data:` 行，负载为 JSON，例如 `data: {"type":"content","content":"你好"}`：

- `content`：回答正文增量。
- `reasoning`：仅思考模式可能出现的推理增量。
- `done`：流正常结束。
- `error`：上游认证、限流、超时或服务错误的安全提示，不泄露上游响应内容。

`/api/reason` 会同时产生 `reasoning` 和 `content`；普通聊天通常只产生 `content`、`done` 或 `error`。`/api/langgraph/query` 和 `/api/langgraph/resume` 同样发送 `content`、`done`、`error`，并可发送带 `conversation_id` 的 `interruption`。客户端应读取 JSON 的 `type` 字段并按类型累积内容，而不是依赖 SSE 的 `event:` 字段或假设每个事件都包含正文。

`POST /api/search` 也返回 SSE，所有负载同样位于 `data:` 行的 JSON 中，事件类型由 JSON 的 `type` 字段表示。未调用搜索工具时会先发送 `direct_answer`；调用搜索时依次发送 `search_start` 和 `search_results`（包含 `total`、`query` 与 `results`），随后会发送摘要流的 `content`、`done` 或 `error`。客户端应按 `type` 分别处理这些状态事件与正文事件。

## RAG 行为

`POST /chat-rag` 需要已配置且可用的检索适配器。若部署尚未接入检索适配器、没有可用上下文或检索失败，接口会以 SSE `error` 事件返回安全错误，而不会编造检索答案。

## 技术栈

- 后端：FastAPI、SQLAlchemy、MySQL、Redis、Neo4j、LangGraph / GraphRAG、DashScope
- 前端：Vue 3、Element Plus、TypeScript

## 部署（上线演练）

### 路径 A：只容器化依赖服务（最快）

```bash
cd dashscope_agent
docker compose up -d            # 仅启动 MySQL 8 + Redis 7

cd llm_backend
python scripts/init_db.py       # 仅空库首次执行；脚本含 drop_all，切勿对已有数据的库执行
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers
```

不要用 `python run.py` 上线：它是 `reload=True` 的开发入口。

> 端口冲突：如果本机已经跑着 MySQL / Redis，3306 与 6379 会被占用导致容器起不来。此时先停掉本机服务，或把 `docker-compose.yml` 里对应服务的 `ports` 映射改成宿主机的其他端口。

### 路径 B：完整容器化（应用也进镜像）

```bash
cd dashscope_agent
docker compose --profile full up -d --build
docker compose exec backend python scripts/init_db.py   # 仅空库首次执行

# 需要 Neo4j、以及一层反向代理时：
docker compose --profile full --profile proxy up -d --build
```

- 前端产物已随镜像发布：`llm_backend/static/dist/dist`，由 `main.py` 挂载到 `/`。
- 后端镜像**不包含 `.env`**，密钥一律由运行期环境变量注入（`env_file` / `environment`）。
- 上传文件与 GraphRAG 索引使用命名卷持久化，容器重建不丢数据。

### 容器内的配置覆盖

`docker-compose.yml` 的 `environment` 会把 `.env` 中面向宿主机的地址改成容器网络内的服务名（`DB_HOST=mysql`、`REDIS_HOST=redis`、`NEO4J_URL=bolt://neo4j:7687`），并把 `GRAPHRAG_PROJECT_DIR` 纠正为容器内的相对路径。

### 迁移到服务器

前置条件：Linux（Ubuntu 22.04+ 最省事）、已装 Docker 与 Compose、**内存建议 ≥8GB（再跑 Neo4j 建议 16GB）**、磁盘 ≥40GB（镜像本身就有数 GB）。

需要搬的东西共四类：代码（含前端编译产物）、`.env`（**在服务器上重新写，不要拷贝明文**）、MySQL 数据、运行期数据（`uploads/` 与 `app/graphrag/data/` 下的索引产物，不搬就得在服务器上重建索引）。

```bash
# 服务器：装 Docker
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER

# 本地：同步代码（排除 .env 与 .git）
rsync -av --exclude '.env' --exclude '.git' dashscope_agent/ user@server:/opt/assistgen/

# 服务器：在 /opt/assistgen/.env 里填好密钥，并把 DB_PASSWORD 换成强密码
cd /opt/assistgen && docker compose --profile full up -d --build
curl -fsS localhost:8000/ready
```

数据库迁移 —— **先导入数据，就不要再跑 `init_db.py`**（它含 `drop_all`，会清库）：

```bash
# 本地导出
mysqldump -uroot -p --single-transaction --routines assist_gen > assist_gen.sql

# 传到服务器后导入
docker compose exec -T mysql mysql -uroot -p"你的DB_PASSWORD" assist_gen < assist_gen.sql
```

服务器网络差、拉不动 PyPI / Docker Hub 时，改为本地构建再把镜像搬过去：

```bash
# 本地
docker compose --profile full build
docker save assistgen-backend | gzip > assistgen-backend.tar.gz
scp assistgen-backend.tar.gz user@server:/opt/assistgen/

# 服务器
gunzip -c assistgen-backend.tar.gz | docker load
docker compose --profile full up -d
```

注意事项：只对公网开 80/443（compose 已把 8000 绑在 `127.0.0.1`）；`docker compose down -v` 会删掉数据卷；首次启动要加载嵌入模型，冷启动较慢，`/ready` 的 `start_period` 需要留足；上线前把 `CORS_ORIGINS` 改成域名并配置 HTTPS。

### 已知的生产化缺口（需要时可以讲清楚，但当前未实现）

- `SECRET_KEY` 仍是默认示例值时只打警告、不阻止启动；`NEO4J_PASSWORD` 有弱默认值。
- 没有 Alembic 迁移，`scripts/init_db.py` 使用 `drop_all` + `create_all`，表结构变更无法回滚。
- `/api/token` 没有限流；部分接口把内部异常原文放进 500 响应。
- 索引任务运行在进程内线程池，因此 `--workers` 固定为 1，暂不支持横向扩展。
- 前端只有编译产物、没有源码；暗色主题令牌已就绪但没有界面切换入口。
- 镜像内以 root 运行，且因 `sentence_transformers` / `torch` / `graphrag` 体积偏大。

真实上线还需：强随机 `SECRET_KEY`、明确 `CORS_ORIGINS`、HTTPS 与 HSTS、登录接口限流、日志轮转、数据库定时备份。

## License

MIT
