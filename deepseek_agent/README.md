# AssistGen 智能客服系统

AssistGen 是一个基于 FastAPI 和 Vue 3 的智能客服应用，使用 DashScope 作为唯一的模型提供商。它覆盖通用对话、思考模式、图片描述、联网搜索和文档检索问答。

## 配置

将 `.env.example` 复制为 `.env`，再填写本地基础设施与密钥。密钥仅应保存在本地或部署环境的安全配置中，示例值不能用于生产。

```env
DASHSCOPE_API_KEY=your-dashscope-api-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
DASHSCOPE_COMPATIBLE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen3.7-plus
DASHSCOPE_REASON_MODEL=qwen3.7-plus
DASHSCOPE_VISION_MODEL=qwen3-vl-plus
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding
```

模型用途固定如下：通用问答和工具调用使用 `qwen3.7-plus`，思考模式使用 `qwen3.7-plus`，图片理解使用 `qwen3-vl-plus`，检索向量使用 `qwen3.7-text-embedding`。可以通过对应环境变量调整型号，但模型服务仍统一使用 DashScope。

还需根据部署环境配置 `SERPAPI_KEY`、MySQL、Redis 和 Neo4j；完整占位项见 `.env.example`。

## 启动

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
cd llm_backend
python run.py
```

服务默认运行在 `http://localhost:8000`，交互式 API 文档位于 `http://localhost:8000/docs`。

## SSE 协议

`POST /api/chat` 和 `POST /api/reason` 均返回 `text/event-stream`。事件类别为：

- `reasoning`：思考过程的增量，仅 `/api/reason` 可能发送；
- `content`：回复正文增量；
- `done`：请求正常完成；
- `error`：认证、限流、超时或上游异常的安全错误信息。

客户端应分别处理 `reasoning` 与 `content`，并在收到 `done` 后结束渲染。收到 `error` 时展示该安全错误，不应依赖或展示上游原始响应。

## RAG 行为

`POST /chat-rag` 依赖部署时注入的检索适配器。若未配置适配器、检索没有上下文或发生异常，服务返回 SSE `error` 事件并停止生成，以避免在没有检索依据时给出答案。

## 生产环境注意事项

- 使用强随机 `SECRET_KEY`，并限制 CORS 来源。
- 使用 HTTPS 且关闭 `reload=True`。
- 不提交 `.env`，通过受控的部署配置提供 `DASHSCOPE_API_KEY`。

## License

MIT
