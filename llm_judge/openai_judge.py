from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel


from .base import BaseJudge, JudgeResponse, JudgeScoringConfig

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

DEFAULT_RUBRIC = """You are an expert mechanical CAD engineer tasked with judging CAD code.
You will be given:
- A natural language prompt describing a 3D object to model.
- The generated CAD code in Python or a CAD DSL.
- Multi-view renderings produced by executing the CAD code (assume the code compiled successfully).

Score the CAD code from 0 to 5 using the following rubric:
0: Geometry is unrelated to the prompt.
1: Geometry is clearly incorrect or missing critical features.
2: Geometry captures some high-level intent but misses key dimensions or shapes.
3: Geometry is mostly correct with minor inaccuracies or missing small features.
4: Geometry matches the prompt with only negligible issues.
5: Geometry is an excellent match for the prompt, faithful in both shape and proportions.

Guidance:
- Scrutinise EVERY rendering detail (shapes, proportions, fillets, holes, symmetry) and cross-check against the entire prompt text. Missing or incorrect minor features must influence the score.
- Use fine-grained floats (increments as small as 0.1) to reflect partial compliance rather than rounding to whole numbers.
- Before scoring, mentally form a short rubric that considers geometry accuracy, dimensional fidelity, and completeness. Reference this rubric explicitly in your reasoning (e.g., “Rubric: geometry accuracy, proportions, details. Deductions: ...”).
- In the reasoning, clearly state where points were deducted and why (cite the specific prompt requirement or visual deviation).
- Pay special attention to coordinate frames and attachment points: verify that parts line up, touch, or intersect exactly as described. Deduct points whenever components float, intersect incorrectly, or misalign due to bad origins/transforms.

You must respond with a valid JSON object using this exact schema (the key name must be exactly "reasoning"; do NOT use synonyms such as "rationale"):
{
  "score": <float between 0 and 5>,
  "reasoning": "<2-3 sentence justification describing key alignment or mismatches>"
}

If you cannot evaluate the CAD for any reason, set "score" to 0 and explain why in "reasoning".
Do not include extra keys or prose outside this JSON object.

ALWAYS INCLUDE a key literally named "reasoning" (not "rationale", "analysis", or other synonyms) with at least two full sentences summarising the judgement, the rubric applied, and the deductions made. ABSOLUTELY NEVER omit this field; the response is invalid without it.
NEVER return a bare number or any format other than the JSON object described above."""

class JudgeResponseSchema(BaseModel):
    score: float
    reasoning: str


class OpenAILLMJudge(BaseJudge):
    """LLM judge implementation backed by the OpenAI API."""

    def __init__(
        self,
        *,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        default_config: Optional[JudgeScoringConfig] = None,
        client: Optional["OpenAI"] = None,
        temperature: Optional[float] = None,
    ) -> None:
        if OpenAI is None and client is None:
            raise ImportError(
                "openai package not installed. Install `openai>=1.0` or provide a client."
            )
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if client is not None:
            self._client = client
        else:
            self._client = OpenAI(api_key=self._api_key)
        self._model = model
        self._default_config = default_config or JudgeScoringConfig(
            rubric=DEFAULT_RUBRIC,
            max_score=5.0,
        )
        self._temperature = temperature

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

    def _build_messages(
        self,
        *,
        prompt: str,
        cad_code: str,
        encoded_images: List[str],
        issues: Optional[str],
        compilation_log: Optional[str],
        rubric: str,
    ) -> list[dict]:
        user_content: list[dict] = [
            {
                "type": "input_text",
                "text": (
                    "You are evaluating a CAD generation.\n\n"
                    f"Prompt:\n{prompt}\n\n"
                    "CAD code:\n"
                    f"{cad_code}"
                ),
            }
        ]
        if issues:
            user_content.append(
                {
                    "type": "input_text",
                    "text": f"Rendering warnings: {issues}",
                }
            )
        if compilation_log:
            user_content.append(
                {
                    "type": "input_text",
                    "text": f"Compilation log (for context only):\n{compilation_log}",
                }
            )
        for encoded in encoded_images:
            data_uri = f"data:image/png;base64,{encoded}"
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": data_uri,
                }
            )

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": rubric,
                    }
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

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

        messages = self._build_messages(
            prompt=prompt,
            cad_code=cad_code,
            encoded_images=images_b64,
            issues=issues,
            compilation_log=compilation_log,
            rubric=scoring_config.rubric,
        )

        request_kwargs = {
            "model": self._model,
            "input": messages,
        }
        if self._temperature is not None:
            request_kwargs["temperature"] = self._temperature

        print(f"Invoking OpenAI judge model: {self._model}")
        response = self._client.responses.parse(**request_kwargs, text_format=JudgeResponseSchema)
        raw_content = response.output_text or ""

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
            parsed = raw_content.strip()  # fallback to raw text

        if isinstance(parsed, dict):
            score_value, reasoning = _extract_from_mapping(parsed)
        elif isinstance(parsed, (list, tuple)) and parsed:
            first = parsed[0]
            if isinstance(first, dict):
                score_value, reasoning = _extract_from_mapping(first)
            else:
                try:
                    score_value = float(first)
                    reasoning = (
                        "Judge response missing reasoning; received bare list value."
                    )
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
