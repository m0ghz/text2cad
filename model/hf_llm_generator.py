"""CAD generator via Hugging Face's OpenAI-compatible router."""

from __future__ import annotations

import os
import re
from typing import Iterable, Optional

from evaluate.generator import BaseGenerator

from .llm_shared import SYSTEM_MESSAGE
from utils.code_cleaning import sanitize_cad_code

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HF_MODEL = "Qwen/Qwen3-32B"


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        stripped = "\n".join(lines).lstrip("\n")
    stripped = stripped.rstrip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].rstrip()
    return stripped


THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _remove_think_blocks(text: str) -> str:
    return THINK_BLOCK_PATTERN.sub("", text).strip()


def _ensure_cadquery_import(code: str) -> str:
    normalized = code.replace("\r", "")
    if "import cadquery as cq" in normalized:
        return normalized
    return f"import cadquery as cq\n\n{normalized.lstrip()}"


def _extract_text_from_payload(data: dict) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {}) or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    text_parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    return "\n".join(text_parts)


class HuggingFaceAPIGenerator(BaseGenerator):
    """Thin wrapper around Hugging Face's OpenAI-compatible inference router."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_HF_MODEL,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        extra_messages: Optional[Iterable[dict]] = None,
        timeout: float = 120.0,
        verify_tls: bool = True,
        max_tokens: Optional[int] = 16384,
    ) -> None:
        if requests is None:  # pragma: no cover - ensure dependency
            raise ImportError(
                "The `requests` package is required for Hugging Face-backed generation. "
                "Install it via `pip install requests`."
            )
        if extra_messages:
            raise ValueError(
                "extra_messages are not supported when using the Hugging Face generator."
            )
        resolved_key = (
            api_key
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("QWEN_API_KEY")
        )
        if not resolved_key:
            raise EnvironmentError(
                "Set HF_TOKEN (preferred) or HUGGINGFACEHUB_API_TOKEN to call the Hugging Face router."
            )

        self._endpoint = f"{API_BASE_URL.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
        self._model_name = model_name
        self._temperature = temperature
        self._timeout = timeout
        self._verify_tls = verify_tls
        self._max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        print(f"Generating CAD code with Hugging Face model: {self._model_name}")
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": (
                    "Return only raw cadquery Python code with no markdown fences.\n\n"
                    f"Prompt:\n{prompt}"
                ),
            },
        ]

        payload: dict = {
            "model": self._model_name,
            "messages": messages,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens

        response = requests.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
            verify=self._verify_tls,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Hugging Face generation failed ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Qwen generation returned non-JSON payload.") from exc

        raw_text = _extract_text_from_payload(data)
        return sanitize_cad_code(raw_text)
