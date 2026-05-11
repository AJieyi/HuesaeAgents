"""图片生成提供者集合"""
from .base import ImageProvider, GenerationResult
from .doubao import DoubaoProvider
from .jimeng import JimengProvider

__all__ = [
    "ImageProvider",
    "GenerationResult",
    "DoubaoProvider",
    "JimengProvider",
]
