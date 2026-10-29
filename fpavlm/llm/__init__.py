"""LLM interfaces and model-specific adapters."""

from fpavlm.llm.embedding_adapter import LLMEmbeddingAdapter
from fpavlm.llm.interface import LLMInterface, LLMRegistry
from fpavlm.llm.internvl2_adapter import InternVL2Adapter
from fpavlm.llm.qwen2vl_adapter import Qwen2VLAdapter

__all__ = [
    "InternVL2Adapter",
    "LLMEmbeddingAdapter",
    "LLMInterface",
    "LLMRegistry",
    "Qwen2VLAdapter",
]
