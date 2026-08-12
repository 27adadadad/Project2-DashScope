# DashScope API 全面迁移设计

## 目标

将项目内全部模型调用从 DeepSeek 与 Ollama 迁移至阿里云百炼 DashScope API，并清除旧 Provider 的配置、服务、依赖与文档。

迁移覆盖普通聊天、深度推理、联网搜索、LangGraph 图谱客服、图片问答，以及 RAG/GraphRAG 向量检索。除模型调用层外，不改变既有业务接口的用途和 Neo4j、MySQL、Redis、SerpAPI 的职责。

## 模型与配置

所有模型调用通过 DashScope 原生 API 适配层完成。配置项如下：

```env
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1

DASHSCOPE_CHAT_MODEL=qwen3.7-plus
DASHSCOPE_REASON_MODEL=qwen3.7-plus
DASHSCOPE_VISION_MODEL=qwen3-vl-plus
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding
```

`qwen3.7-plus` 用于聊天、联网搜索结果总结与 LangGraph；普通路径关闭思考模式，深度推理路径启用思考模式。视觉模型使用 `qwen3-vl-plus`。文本向量模型使用 `qwen3.7-text-embedding`，并在非对称检索中显式标记文档为 `document`、查询为 `query`。

## 组件边界

新增 DashScope 适配层，并由下列相互独立的组件构成：

- 聊天客户端：构造消息、发起普通或思考模式请求、将 DashScope 流转换为应用流。
- 工具调用客户端：处理 Qwen 的工具调用参数与工具结果，供联网搜索使用。
- 视觉客户端：负责图片消息的构造和 `qwen3-vl-plus` 响应解析。
- 嵌入客户端：负责批量向量生成及 `query`/`document` 参数传递。
- LangChain/LangGraph 适配器：为现有状态图提供统一模型对象，替代 `ChatDeepSeek` 与 `ChatOllama`。

API 层、会话服务、Neo4j 图谱工作流与 SerpAPI 搜索工具保持其业务边界；它们只依赖新的抽象接口，不直接读取 Provider 细节。

## 流式协议

普通聊天沿用 SSE 输出。深度推理不混合模型的思考内容和最终回答，事件载荷统一为：

```json
{"type":"reasoning","content":"推理片段"}
{"type":"content","content":"回答片段"}
{"type":"done"}
```

前端可将 `reasoning` 作为默认折叠的推理区显示，将 `content` 作为正式回答显示。若模型未返回思考片段，仍正常返回 `content` 和 `done`。

## 迁移范围

1. 用 DashScope 实现聊天、推理、搜索总结、视觉、嵌入及 LangGraph 模型调用。
2. 删除 DeepSeek/Ollama 的配置、工厂分支、服务文件、LangChain Provider 依赖和文档说明。
3. RAG 路径统一接入新嵌入客户端，并修复现有的缺失 RAG 服务引用与未初始化 `config_mapping`。
4. 保留 SerpAPI 作为搜索数据源，由 Qwen 进行工具调用与结果总结。
5. 更新 `.env.example`、依赖清单、README 与接口说明。

## 错误处理与安全

- 启动时校验 `DASHSCOPE_API_KEY` 与必要模型配置，并给出可定位的配置错误。
- 对网络超时、认证失败、限流和上游服务错误映射为统一应用异常并写入结构化日志；日志不得输出 API Key 或完整敏感请求体。
- 继续使用环境变量保存密钥，不在代码、测试夹具或文档中写入真实凭据。

## 验收与测试

- 普通流式聊天能正确生成内容事件。
- 思考模式能将 `reasoning_content` 与最终 `content` 分别转为 SSE 事件，并以 `done` 收尾。
- 联网搜索能够完成工具调用、执行 SerpAPI 查询并总结结果。
- 图片消息可被视觉客户端正确构造；向量客户端会传递正确的 `query` 或 `document` 类型。
- 缺失 API Key、上游超时、认证失败与限流均有可验证的失败路径。
- RAG 相关服务不再存在未定义引用或未初始化字段。

## 非目标

- 不迁移或替换 MySQL、Redis、Neo4j、SerpAPI。
- 不改变现有业务路由的功能语义；必要时仅扩展深度推理 SSE 的事件类型。
- 不在本次工作中重新设计前端视觉样式。
