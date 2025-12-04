"""OpenAI-backed CAD code generator."""

from __future__ import annotations

import os
from typing import Iterable, Optional

from evaluate.generator import BaseGenerator

from .llm_shared import DEFAULT_MODEL_NAME, SYSTEM_MESSAGE

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]


class OpenAIAPIGenerator(BaseGenerator):
    """Thin wrapper around the OpenAI Responses API for CAD generation."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: Optional[str] = None,
        extra_messages: Optional[Iterable[dict]] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self._model_name = model_name
        self._extra_messages = list(extra_messages or [])
        self._temperature = temperature

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is required for OpenAI-backed generation."
            )
        if OpenAI is None:
            raise ImportError(
                "The `openai` package is required for OpenAI-backed generation. "
                "Install it via `pip install openai`."
            )
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str) -> str:
        print(f"Generating CAD code with OpenAI model: {self._model_name}")
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": SYSTEM_MESSAGE,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            },
            *self._extra_messages,
        ]
        request_kwargs = {
            "model": self._model_name,
            "input": messages,
        }
        if self._temperature is not None:
            request_kwargs["temperature"] = self._temperature

        response = self._client.responses.create(**request_kwargs)
        return response.output_text or ""
