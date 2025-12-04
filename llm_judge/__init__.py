"""
Interfaces and utilities for running large language model judges over CAD generations.

This package exposes a simple factory to construct judges as well as the
base protocol that custom judges should implement.
"""

from .base import BaseJudge, JudgeResponse, JudgeScoringConfig
from .gemini_judge import GeminiLLMJudge
from .gemini_models import GEMINI_FLASH_MODEL, GEMINI_MODEL_ALIASES, GEMINI_PRO_MODEL
from .hf_judge import HuggingFaceLLMJudge
from .openai_judge import OpenAILLMJudge

__all__ = [
    "BaseJudge",
    "JudgeResponse",
    "JudgeScoringConfig",
    "OpenAILLMJudge",
    "GeminiLLMJudge",
    "HuggingFaceLLMJudge",
    "GEMINI_MODEL_ALIASES",
    "GEMINI_FLASH_MODEL",
    "GEMINI_PRO_MODEL",
]
