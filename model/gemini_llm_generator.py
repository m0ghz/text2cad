"""Gemini-backed CAD code generator."""

from __future__ import annotations

import os
from typing import Iterable, Optional

from evaluate.generator import BaseGenerator

from llm_judge.gemini_models import GEMINI_PRO_MODEL, normalise_gemini_model_name

from .llm_shared import SYSTEM_MESSAGE

try:
    import google.generativeai as genai  # type: ignore
    from google.generativeai import types as genai_types  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


def _extract_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text
    return ""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence (e.g. ``` or ```python)
        lines = lines[1:]
        stripped = "\n".join(lines).lstrip("\n")
    stripped = stripped.rstrip()
    if stripped.endswith("```"):
        stripped = stripped[: -3].rstrip()
    return stripped


class GeminiAPIGenerator(BaseGenerator):
    """Thin wrapper around the Gemini API for CAD generation."""

    def __init__(
        self,
        *,
        model_name: str = GEMINI_PRO_MODEL,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        extra_messages: Optional[Iterable[dict]] = None,
    ) -> None:
        if extra_messages:
            raise ValueError(
                "extra_messages are not supported when using the Gemini generator."
            )
        if genai is None:
            raise ImportError(
                "The `google-generativeai` package is required for Gemini-backed generation. "
                "Install it via `pip install google-generativeai`."
            )

        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is required for Gemini-backed generation."
            )

        genai.configure(api_key=resolved_key)
        resolved_model = normalise_gemini_model_name(model_name) or model_name
        self._model_name = resolved_model
        self._temperature = temperature
        self._client = genai.GenerativeModel(self._model_name)

    def generate(self, prompt: str) -> str:
        print(f"Generating CAD code with Gemini model: {self._model_name}")
        parts = [
            {"text": SYSTEM_MESSAGE},
            {"text": f"Prompt:\n{prompt}"},
            {
                "text": (
                    "IMPORTANT OUTPUT REQUIREMENT: return only the raw Python source code and do not wrap it in "
                    "markdown fences (no ``` or ```python). Begin immediately with the code itself."
                )
            },
        ]
        contents = [{"role": "user", "parts": parts}]

        generation_kwargs = {}
        if self._temperature is not None:
            generation_kwargs["temperature"] = self._temperature
        if genai_types is not None and hasattr(genai_types, "GenerationConfig"):
            generation_config = genai_types.GenerationConfig(**generation_kwargs)
        else:  # pragma: no cover - fallback for stripped installs
            generation_config = generation_kwargs or None

        response = self._client.generate_content(
            contents,
            generation_config=generation_config,
        )
        raw = _extract_text(response)
        return _strip_code_fences(raw)
