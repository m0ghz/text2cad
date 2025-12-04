from __future__ import annotations

import abc
import dataclasses
from typing import Optional


@dataclasses.dataclass(frozen=True)
class JudgeScoringConfig:
    """Configuration describing how the judge should score generations."""

    rubric: str
    max_score: float = 5.0


@dataclasses.dataclass(frozen=True)
class JudgeResponse:
    """Structured response returned by an LLM judge evaluation."""

    score: float
    reasoning: str
    raw_response: Optional[str] = None

    def clipped_score(self, upper: Optional[float] = None, lower: float = 0.0) -> float:
        """
        Return the score clipped to provided bounds.

        Many LLM responses are written free-form, so we standardise by clipping to
        sane defaults. Callers can override the bounds if they know the expected
        range for a given rubric.
        """
        if upper is None:
            return max(lower, self.score)
        return max(lower, min(upper, self.score))


class BaseJudge(abc.ABC):
    """Abstract base class implemented by all LLM judges."""

    @abc.abstractmethod
    def score(
        self,
        *,
        prompt: str,
        cad_code: str,
        rendering_paths: Optional[list[str]] = None,
        compilation_log: Optional[str] = None,
        config: Optional[JudgeScoringConfig] = None,
    ) -> JudgeResponse:
        """
        Score a CAD generation given its originating prompt and optional artefacts.

        Parameters
        ----------
        prompt:
            Input text prompt used to guide the CAD generator.
        cad_code:
            The string representation of the CAD program that was produced.
        rendering_paths:
            Optional list of file paths (local) pointing to supporting renderings,
            such as STL previews or multi-view PNGs.
        compilation_log:
            Raw stdout/stderr produced when compiling the CAD program. This is
            useful context for the judge to identify failure modes.
        config:
            Overrides for the rubric used to evaluate this sample. If omitted, an
            implementation-specific default rubric is used.

        Returns
        -------
        JudgeResponse
            A structured object containing the numeric score, reasoning, and the
            raw model response where available.
        """

