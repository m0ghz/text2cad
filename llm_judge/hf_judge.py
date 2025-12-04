from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Optional

import requests

from .base import BaseJudge, JudgeResponse, JudgeScoringConfig
from .openai_judge import DEFAULT_RUBRIC

HF_API_BASE = "https://router.huggingface.co/v1"
DEFAULT_HF_JUDGE_MODEL = "Qwen/Qwen3-32B"


class HuggingFaceLLMJudge(BaseJudge):
    """LLM judge implementation backed by Hugging Face's OpenAI-compatible router."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_HF_JUDGE_MODEL,
        api_key: Optional[str] = None,
        default_config: Optional[JudgeScoringConfig] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 2048,
        timeout: float = 120.0,
        verify_tls: bool = True,
    ) -> None:
        resolved_key = (
            api_key
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("QWEN_API_KEY")
        )
        if not resolved_key:
            raise EnvironmentError(
                "Set HF_TOKEN (preferred) or HUGGINGFACEHUB_API_TOKEN for the Hugging Face judge."
            )
        self._endpoint = f"{HF_API_BASE.rstrip('/')}/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._verify_tls = verify_tls
        self._default_config = default_config or JudgeScoringConfig(
            rubric=DEFAULT_RUBRIC,
            max_score=5.0,
        )

    @staticmethod
    def _encode_images(rendering_paths: Optional[list[str]]) -> tuple[List[str], Optional[str]]:
        if not rendering_paths:
            return [], "No renderings were provided; treat as a compilation failure."
        encoded_images: List[str] = []
        missing_paths: List[str] = []
        for path_str in rendering_paths:
            path = Path(path_str)
            if not path.exists():
                missing_paths.append(path_str)
                continue
            with path.open("rb") as handle:
                encoded_images.append(base64.b64encode(handle.read()).decode("utf-8"))
        if missing_paths:
            reason = f"Missing renderings: {', '.join(missing_paths)}."
        else:
            reason = None
        return encoded_images, reason

    def _build_user_message(
        self,
        *,
        prompt: str,
        cad_code: str,
        encoded_images: List[str],
        issues: Optional[str],
        compilation_log: Optional[str],
    ) -> str:
        segments = [
            "You are evaluating a CAD generation.",
            f"Prompt:\n{prompt}",
            "CAD code:\n" + cad_code,
        ]
        if issues:
            segments.append(f"Rendering warnings: {issues}")
        if compilation_log:
            segments.append(f"Compilation log (for context only):\n{compilation_log}")
        if encoded_images:
            image_lines: List[str] = ["Renderings (base64-encoded PNG data URIs):"]
            for idx, data in enumerate(encoded_images, start=1):
                image_lines.append(f"[Image {idx}] data:image/png;base64,{data}")
            segments.append("\n".join(image_lines))
        return "\n\n".join(segments)

    def _post_completion(self, messages: list[dict]) -> dict:
        payload: dict = {
            "model": self._model,
            "messages": messages,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens

        print(f"Invoking Hugging Face judge model: {self._model}")
        response = requests.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
            verify=self._verify_tls,
        )
        print(f"Hugging Face judge response ({response.status_code}): {response.text}")
        if response.status_code >= 400:
            raise RuntimeError(
                f"Hugging Face judge failed ({response.status_code}): {response.text}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Hugging Face judge returned non-JSON payload.") from exc

    @staticmethod
    def _extract_content(payload: dict) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(parts)
        return str(content)

    def score(
        self,
        *,
        prompt: str,
        cad_code: str,
        rendering_paths: Optional[list[str]] = None,
        compilation_log: Optional[str] = None,
        config: Optional[JudgeScoringConfig] = None,
    ) -> JudgeResponse:
        scoring_config = config or self._default_config

        images_b64, issues = self._encode_images(rendering_paths)
        if not images_b64:
            reasoning = issues or "No renderings available."
            return JudgeResponse(score=0.0, reasoning=reasoning, raw_response=None)

        user_message = self._build_user_message(
            prompt=prompt,
            cad_code=cad_code,
            encoded_images=images_b64,
            issues=issues,
            compilation_log=compilation_log,
        )
        messages = [
            {"role": "system", "content": scoring_config.rubric},
            {"role": "user", "content": user_message},
        ]

        payload = self._post_completion(messages)
        raw_content = self._extract_content(payload).strip()

        def _extract_from_mapping(mapping: dict) -> tuple[float, str]:
            score_raw = mapping.get("score", 0.0)
            reason_raw = mapping.get("reasoning", "")
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0
            reason = str(reason_raw).strip()
            return score, reason

        score_value = 0.0
        reasoning = "Failed to parse judge response."

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            parsed = raw_content.strip()

        if isinstance(parsed, dict):
            score_value, reasoning = _extract_from_mapping(parsed)
        elif isinstance(parsed, (list, tuple)) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                score_value, reasoning = _extract_from_mapping(first)
            else:
                try:
                    score_value = float(first)
                    reasoning = "Judge response missing reasoning; received bare list value."
                except (TypeError, ValueError):
                    reasoning = f"Unrecognised judge response: {raw_content}"
        else:
            try:
                score_value = float(parsed)
                reasoning = "Judge response missing reasoning; received bare value."
            except (TypeError, ValueError):
                reasoning = f"Unrecognised judge response: {raw_content}"

        if issues:
            reasoning = f"{issues} | {reasoning}"

        if not reasoning:
            reasoning = (
                "Judge did not provide reasoning text. "
                f"Raw response: {raw_content}".strip()
            )

        return JudgeResponse(
            score=score_value,
            reasoning=reasoning,
            raw_response=raw_content,
        )
