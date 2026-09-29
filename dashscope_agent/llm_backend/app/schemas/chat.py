"""聊天相关接口的请求体模型。

集中在一处的好处：字段类型与校验规则只有一个定义，路由函数只关心业务逻辑；
前端 JSON 在进入路由前就会被 Pydantic 转换并校验，不合法直接返回 422。
"""

from typing import Dict, List

from pydantic import BaseModel


class ReasonRequest(BaseModel):
    messages: List[Dict[str, str]]
    user_id: int


class ChatMessage(BaseModel):
    messages: List[Dict[str, str]]
    user_id: int
    conversation_id: int


class RAGChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    index_id: str
    user_id: int


class CreateConversationRequest(BaseModel):
    user_id: int


class UpdateConversationNameRequest(BaseModel):
    name: str


class LangGraphResumeRequest(BaseModel):
    query: str
    user_id: int
    conversation_id: str
