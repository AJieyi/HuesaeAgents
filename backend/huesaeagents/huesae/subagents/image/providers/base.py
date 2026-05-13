"""图片生成提供者抽象基类"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """图片生成结果"""

    url: str
    provider: str
    prompt: str
    size: str | None = None


class ImageProvider(ABC):
    """图片生成提供者抽象基类

    所有生图Provider必须继承此类，实现 name 和 generate 方法。
    新增Provider无需修改已有代码，符合开闭原则。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称，用于注册和选择"""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        size: str = "2K",
        **kwargs,
    ) -> GenerationResult:
        """生成图片

        Args:
            prompt: 提示词（Danbooru标签或自然语言）
            size: 图片尺寸
            **kwargs: 额外参数

        Returns:
            GenerationResult: 生成结果
        """
        pass

    def supports_size(self, size: str) -> bool:
        """检查是否支持指定尺寸（默认支持所有）"""
        return True
