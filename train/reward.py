"""Reward computation utilities that reuse the evaluation pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from evaluate.cad_executor import CADCompilationResult, CADExecutor
from llm_judge import (
    GEMINI_MODEL_ALIASES,
    BaseJudge,
    GeminiLLMJudge,
    HuggingFaceLLMJudge,
    JudgeResponse,
    JudgeScoringConfig,
    OpenAILLMJudge,
)


LOGGER = logging.getLogger(__name__)


def _build_judge(
    *,
    backend: str,
    model_name: str,
    scoring_config: JudgeScoringConfig,
) -> BaseJudge:
    backend_normalized = backend.strip().lower()
    if backend_normalized == "gemini":
        normalized = model_name.strip().lower()
        resolved_model = GEMINI_MODEL_ALIASES.get(normalized, model_name)
        return GeminiLLMJudge(model=resolved_model, default_config=scoring_config)
    if backend_normalized == "huggingface":
        return HuggingFaceLLMJudge(model=model_name, default_config=scoring_config)
    if backend_normalized == "openai":
        return OpenAILLMJudge(model=model_name, default_config=scoring_config)
    raise ValueError(
        f"Unsupported judge backend '{backend}'. Expected one of: openai, gemini, huggingface."
    )


def _blend_reward(
    *,
    judge_response: JudgeResponse,
    compilation_success: bool,
    judge_weight: float,
    scoring_config: JudgeScoringConfig,
) -> float:
    judge_score = judge_response.clipped_score(
        upper=scoring_config.max_score
    )
    normalised = judge_score / scoring_config.max_score
    compilation_score = 1.0 if compilation_success else 0.0
    return judge_weight * normalised + (1.0 - judge_weight) * compilation_score


@dataclass
class RewardComputation:
    """Container for artefacts produced when computing a reward."""

    prompt: str
    cad_code: str
    compilation: CADCompilationResult
    judge: JudgeResponse
    reward: float
    artefact_dir: Path


class RewardEvaluator:
    """Thin wrapper around the CAD executor + LLM judge to score candidates."""

    def __init__(
        self,
        *,
        output_root: Path,
        views: Optional[Iterable[str]] = None,
        judge_backend: str = "openai",
        judge_model_name: str = "gpt-5",
        judge_weight: float = 0.7,
        scoring_config: Optional[JudgeScoringConfig] = None,
    ) -> None:
        self._output_root = output_root
        self._executor = CADExecutor(views=views)
        self._scoring_config = scoring_config or JudgeScoringConfig(
            rubric=(
                "You are asked to evaluate whether CAD code produces a 3D object "
                "that matches the provided prompt. Score between 0 and 5."
            ),
            max_score=5.0,
        )
        self._judge = _build_judge(
            backend=judge_backend,
            model_name=judge_model_name,
            scoring_config=self._scoring_config,
        )
        self._judge_weight = judge_weight
        self._output_root.mkdir(parents=True, exist_ok=True)

    def score(
        self,
        *,
        prompt: str,
        cad_code: str,
        sample_id: str,
    ) -> RewardComputation:
        artefact_dir = self._output_root / sample_id
        artefact_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Sample %s: compiling CAD code", sample_id)
        compilation = self._executor.compile(
            cad_code=cad_code,
            artefact_dir=artefact_dir,
            sample_id=sample_id,
        )

        # Save compilation log
        compilation_log = "\n".join(
            filter(
                None,
                [
                    f"STDOUT:\n{compilation.stdout.strip()}",
                    f"STDERR:\n{compilation.stderr.strip()}",
                    f"ERROR:\n{compilation.error}" if compilation.error else None,
                ],
            )
        )
        (artefact_dir / "compilation.log").write_text(compilation_log)

        LOGGER.info("Sample %s: invoking LLM judge", sample_id)
        try:
            judge_response = self._judge.score(
                prompt=prompt,
                cad_code=cad_code,
                rendering_paths=compilation.rendering_paths,
                compilation_log="\n".join(
                    filter(
                        None,
                        [
                            compilation.stdout.strip(),
                            compilation.stderr.strip(),
                            compilation.error or "",
                        ],
                    )
                )
                or None,
                config=self._scoring_config,
            )
            # Save judge response to artefact_dir
            (artefact_dir / "judge_response.json").write_text(
                json.dumps(
                    {
                        "score": judge_response.score,
                        "reasoning": judge_response.reasoning,
                        "raw_response": judge_response.raw_response,
                    },
                    indent=2,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Sample %s: judge invocation failed", sample_id)
            judge_response = JudgeResponse(
                score=0.0,
                reasoning=f"Judge failure: {exc}",
                raw_response=None,
            )

        reward = _blend_reward(
            judge_response=judge_response,
            compilation_success=compilation.success,
            judge_weight=self._judge_weight,
            scoring_config=self._scoring_config,
        )
        LOGGER.info(
            "Sample %s: reward %.3f (compiled=%s, judge=%.2f)",
            sample_id,
            reward,
            "yes" if compilation.success else "no",
            judge_response.score,
        )
        return RewardComputation(
            prompt=prompt,
            cad_code=cad_code,
            compilation=compilation,
            judge=judge_response,
            reward=reward,
            artefact_dir=artefact_dir,
        )
