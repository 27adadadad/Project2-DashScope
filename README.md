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

## 部署注意事项

- 在生产环境设置强随机 `SECRET_KEY`、明确的 CORS 来源与 HTTPS。
- 生产环境关闭 `reload=True`。
- 将 `.env` 保持在版本控制之外，并使用安全的密钥管理方式注入 `DASHSCOPE_API_KEY`。

## License

MIT
