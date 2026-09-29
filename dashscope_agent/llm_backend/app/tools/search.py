from typing import Dict, List, Optional

import requests

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(service="search_tool")


class SearchTool:
    def __init__(self):
        self.api_key = settings.SERPAPI_KEY
        if not self.api_key:
            raise ValueError("未设置SERPAPI_KEY环境变量")

    def search(self, query: str, num_results: Optional[int] = None) -> List[Dict]:
        """执行搜索并返回结构化结果；未显式指定条数时使用配置值。"""
        limit = num_results or settings.SEARCH_RESULT_COUNT
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": limit,
                "hl": "zh-CN",
                "gl": "cn"
            }

            response = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=15
            )
            response.raise_for_status()

            return self._parse_results(response.json(), limit)

        except Exception as exc:
            # 联网搜索只是可选增强：失败时返回空列表让上层继续用模型回答，
            # 但必须留下带堆栈的告警日志，否则“静默无结果”无法排查。
            logger.warning("SerpAPI 搜索失败: %s", exc, exc_info=True)
            return []

    def _parse_results(self, data: dict, limit: int) -> List[Dict]:
        results = []

        if "organic_results" in data:
            for item in data["organic_results"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })

        return results[:limit]
