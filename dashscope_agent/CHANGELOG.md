# 更新日志

## [v4.0] - DashScope 单一模型提供商迁移

### 变更

- 统一通用问答、思考模式、图像理解和检索向量的模型配置。
- 默认模型分别为 `qwen3.7-plus`、`qwen3-vl-plus` 与 `qwen3.7-text-embedding`。
- 流式接口规范为 `reasoning`、`content`、`done` 和 `error` 事件。
- 未配置检索适配器时，`/chat-rag` 返回安全的 SSE 错误事件。

### 配置

- 使用 `DASHSCOPE_API_KEY` 与 `DASHSCOPE_*_MODEL` 环境变量。
- `.env.example` 仅提供占位值；真实密钥不得写入版本控制。
