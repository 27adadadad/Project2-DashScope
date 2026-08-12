"""基于 SerpAPI 和 DashScope Qwen 的联网搜索服务。"""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any, Optional

from app.core.logger import get_logger
from app.prompts.search_prompts import SEARCH_SUMMARY_PROMPT, SEARCH_SYSTEM_PROMPT
from app.services.dashscope_service import DashScopeService, DashScopeServiceError
from app.services.function_tools import FunctionTool, ToolRegistry
from app.tools.definitions import SEARCH_TOOL
from app.tools.search import SearchTool


logger = get_logger(service="search")


class SearchService:
    def __init__(self, model_service: Optional[DashScopeService] = None):
        self.model_service = model_service or DashScopeService()
        self.search_tool = SearchTool()
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(FunctionTool(**SEARCH_TOOL, handler=self._handle_search))
        self.tools_description = self._generate_tools_description()

    def _generate_tools_description(self) -> str:
        descriptions = []
        for tool in self.tool_registry.get_tools_definition():
            function = tool["function"]
            descriptions.append(f"{function['name']}，{function['description']}")
        return "你现在可用的工具有：\n\n" + "\n".join(descriptions)

    async def _handle_search(self, query: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.search_tool.search, query)

    async def generate_stream(
        self,
        query: str,
        user_id: Optional[int] = None,
        conversation_id: Optional[int] = None,
        on_complete: Optional[Callable[..., Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """执行模型工具决策、返回搜索结果事件并流式总结。"""
        messages = [
            {
                "role": "system",
                "content": SEARCH_SYSTEM_PROMPT.format(
                    tools_description=self.tools_description
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            choice = await self.model_service.generate_tool_response(
                messages, self.tool_registry.get_tools_definition()
            )
            tool_call = self._first_tool_call(choice)
            if tool_call is None:
                yield self._event("direct_answer")
                async for event in self.model_service.generate_search_summary_stream(
                    messages,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    on_complete=on_complete,
                ):
                    yield event
                return

            function = self._field(tool_call, "function")
            tool_name = self._field(function, "name")
            arguments = self._field(function, "arguments") or "{}"
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            search_results = await self.tool_registry.execute_tool(
                tool_name, json.dumps(parsed_arguments, ensure_ascii=False)
            )
            yield self._event("search_start")
            yield self._event(
                "search_results",
                total=len(search_results),
                query=parsed_arguments.get("query", query),
                results=[
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", ""),
                    }
                    for result in search_results
                ],
            )
            context = "\n---\n".join(
                "来源：{title}\n链接：{url}\n内容：{snippet}".format(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("snippet", ""),
                )
                for result in search_results
            ) or "未检索到可用的联网资料。"
            summary_messages = [
                {
                    "role": "system",
                    "content": SEARCH_SUMMARY_PROMPT.format(
                        context=context,
                        query=query,
                        cur_date=datetime.now().strftime("%Y年%m月%d日"),
                    ),
                }
            ]
            async for event in self.model_service.generate_search_summary_stream(
                summary_messages,
                user_id=user_id,
                conversation_id=conversation_id,
                on_complete=on_complete,
            ):
                yield event
        except DashScopeServiceError as error:
            logger.warning("DashScope search request failed: %s", error.category)
            yield self._event("error", message=str(error))
        except Exception:
            logger.exception("Search request failed")
            yield self._event("error", message="DashScope upstream request failed.")

    @staticmethod
    def _field(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    @classmethod
    def _first_tool_call(cls, choice: Any) -> Any:
        message = cls._field(choice, "message")
        tool_calls = cls._field(message, "tool_calls")
        if not tool_calls:
            return None
        return tool_calls[0]

    @staticmethod
    def _event(event_type: str, **payload: Any) -> str:
        return f"data: {json.dumps({'type': event_type, **payload}, ensure_ascii=False)}\n\n"
