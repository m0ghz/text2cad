from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

from .base import BaseJudge, JudgeResponse, JudgeScoringConfig
from .gemini_models import GEMINI_FLASH_MODEL, GEMINI_MODEL_ALIASES, GEMINI_PRO_MODEL
from .openai_judge import DEFAULT_RUBRIC

try:
    import google.generativeai as genai  # type: ignore
    from google.generativeai import types as genai_types  # type: ignore
except ImportError:  # pragma: no cover
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]


class GeminiLLMJudge(BaseJudge):
    """LLM judge implementation backed by the Gemini API."""

    def __init__(
        self,
        *,
        model: str = GEMINI_PRO_MODEL,
        api_key: Optional[str] = None,
        default_config: Optional[JudgeScoringConfig] = None,
        client: Optional["genai.GenerativeModel"] = None,
        temperature: Optional[float] = None,
    ) -> None:
        if genai is None and client is None:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install `google-generativeai>=0.7` or provide an initialised client."
            )

        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._model_name = model
        if client is not None:
            self._client = client
        else:
            if not self._api_key:
                raise ValueError(
                    "Gemini API key not provided. "
                    "Set GEMINI_API_KEY/GOOGLE_API_KEY or pass api_key/client explicitly."
                )
            assert genai is not None  # for mypy / static checkers
            genai.configure(api_key=self._api_key)
            self._client = genai.GenerativeModel(self._model_name)

        self._default_config = default_config or JudgeScoringConfig(
            rubric=DEFAULT_RUBRIC,
            max_score=5.0,
        )
        self._temperature = temperature

    @staticmethod
    def _load_images(rendering_paths: Optional[list[str]]) -> tuple[List[bytes], Optional[str]]:
        if not rendering_paths:
            return [], "No renderings were provided; treat as a compilation failure."

        image_payloads: List[bytes] = []
        missing_paths: List[str] = []
        for path_str in rendering_paths:
            path = Path(path_str)
            if not path.exists():
                missing_paths.append(path_str)
                continue
            with path.open("rb") as handle:
                image_payloads.append(handle.read())

        if missing_paths:
            reason = f"Missing renderings: {', '.join(missing_paths)}."
        else:
            reason = None
        return image_payloads, reason

    @staticmethod
    def _build_user_parts(
        *,
        prompt: str,
        cad_code: str,
        image_payloads: Sequence[bytes],
        issues: Optional[str],
        compilation_log: Optional[str],
    ) -> list[dict]:
        user_parts: list[dict] = [
            {
                "text": (
                    "You are evaluating a CAD generation.\n\n"
                    f"Prompt:\n{prompt}\n\n"
                    "CAD code:\n"
                    f"{cad_code}"
                )
            }
        ]
        if issues:
            user_parts.append({"text": f"Rendering warnings: {issues}"})
        if compilation_log:
            user_parts.append(
                {"text": f"Compilation log (for context only):\n{compilation_log}"}
            )
        for image_bytes in image_payloads:
            user_parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                }
            )
        return user_parts

    @staticmethod
    def _build_contents(
        *,
        rubric: str,
        user_parts: list[dict],
    ) -> list[dict]:
        """Gemini chat payload supporting only user/model roles."""

        parts = [{"text": rubric}]
        parts.extend(user_parts)
        return [
            {
                "role": "user",
                "parts": parts,
            }
        ]

    @staticmethod
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

        images, issues = self._load_images(rendering_paths)
        if not images:
            reasoning = issues or "No renderings available."
            return JudgeResponse(score=0.0, reasoning=reasoning, raw_response=None)

        user_parts = self._build_user_parts(
            prompt=prompt,
            cad_code=cad_code,
            image_payloads=images,
            issues=issues,
            compilation_log=compilation_log,
        )
        contents = self._build_contents(
            rubric=scoring_config.rubric,
            user_parts=user_parts,
        )

        generation_kwargs = {"response_mime_type": "application/json"}
        if self._temperature is not None:
            generation_kwargs["temperature"] = self._temperature
        if genai_types is not None and hasattr(genai_types, "GenerationConfig"):
            generation_config = genai_types.GenerationConfig(**generation_kwargs)
        else:  # pragma: no cover - fallback for stripped installs
            generation_config = generation_kwargs

        print(f"Invoking Gemini judge model: {self._model_name}")
        response = self._client.generate_content(
            contents,
            generation_config=generation_config,
        )
        print(f"Judge LLM response: {response}")
        raw_content = self._extract_text(response).strip()

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
