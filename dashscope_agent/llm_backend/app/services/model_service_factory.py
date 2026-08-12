"""DashScope 模型服务工厂。"""

from app.services.dashscope_service import DashScopeService


class ModelServiceFactory:
    """聊天和推理均使用 DashScope 服务。"""

    @staticmethod
    def create_chat_service() -> DashScopeService:
        return DashScopeService()

    @staticmethod
    def create_reasoner_service() -> DashScopeService:
        return DashScopeService()
