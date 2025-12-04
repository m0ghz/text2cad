from __future__ import annotations

import dataclasses
import html
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from llm_judge import (
    BaseJudge,
    GEMINI_MODEL_ALIASES,
    GeminiLLMJudge,
    HuggingFaceLLMJudge,
    JudgeResponse,
    JudgeScoringConfig,
    OpenAILLMJudge,
)

from .cad_executor import CADCompilationResult, CADExecutor
from .generator import BaseGenerator

from model.gemini_llm_generator import GeminiAPIGenerator
from model.hf_llm_generator import HuggingFaceAPIGenerator
from model.openai_llm_generator import OpenAIAPIGenerator


@dataclass
class EvaluationConfig:
    """Runtime configuration for the evaluation pipeline."""

    input_file: Path
    output_dir: Path
    generator_backend: str = "openai"
    generator_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    judge_model_name: str = "gpt-5"
    judge_backend: str = "openai"
    judge_weight: float = 0.7
    max_samples: Optional[int] = None
    views: Iterable[str] = dataclasses.field(
        default_factory=lambda: ("front", "side", "top", "iso")
    )
    scoring_config: JudgeScoringConfig = dataclasses.field(
        default_factory=lambda: JudgeScoringConfig(
            rubric=(
                "You are asked to evaluate whether CAD code produces a 3D object "
                "that matches the provided prompt. Score between 0 and 5."
            ),
            max_score=5.0,
        )
    )


@dataclass
class EvaluationSampleResult:
    """Complete artefacts for a single evaluation sample."""

    prompt: str
    cad_code: str
    compilation: CADCompilationResult
    judge: JudgeResponse
    final_score: float
    sample_id: str


@dataclass
class EvaluationResult:
    """Aggregate over the whole evaluation run."""

    samples: List[EvaluationSampleResult] = field(default_factory=list)

    @property
    def average_score(self) -> float:
        if not self.samples:
            return 0.0
        return sum(item.final_score for item in self.samples) / len(self.samples)

    def to_jsonl(self) -> str:
        """Return newline-delimited JSON for the run."""
        output_lines = []
        for sample in self.samples:
            payload = {
                "sample_id": sample.sample_id,
                "prompt": sample.prompt,
                "cad_code": sample.cad_code,
                "compilation": sample.compilation.as_dict(),
                "judge": {
                    "score": sample.judge.score,
                    "reasoning": sample.judge.reasoning,
                    "raw_response": sample.judge.raw_response,
                },
                "final_score": sample.final_score,
            }
            output_lines.append(json.dumps(payload))
        return "\n".join(output_lines)


def _build_generator(
    *,
    backend: str,
    kwargs: Dict[str, Any],
) -> BaseGenerator:
    backend_normalized = backend.strip().lower()
    if backend_normalized == "openai":
        return OpenAIAPIGenerator(**kwargs)
    if backend_normalized == "gemini":
        return GeminiAPIGenerator(**kwargs)
    if backend_normalized == "huggingface":
        return HuggingFaceAPIGenerator(**kwargs)
    raise ValueError(
        f"Unsupported generator backend '{backend}'. Expected one of: openai, gemini, huggingface."
    )


def _build_judge(
    *,
    backend: str,
    model_name: str,
    scoring_config: JudgeScoringConfig,
) -> BaseJudge:
    """Return the requested judge backend."""

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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class Evaluator:
    """Orchestrates CAD generation, compilation, and LLM judging."""

    def __init__(
        self,
        *,
        generator: BaseGenerator,
        judge: BaseJudge,
        executor: CADExecutor,
        config: EvaluationConfig,
    ) -> None:
        self._generator = generator
        self._judge = judge
        self._executor = executor
        self._config = config
        _ensure_dir(self._config.output_dir)
        self._logger = logging.getLogger(__name__)
        self._state_path = self._config.output_dir / "partial_state.jsonl"

    @classmethod
    def from_generator_class(
        cls,
        *,
        input_file: str | Path,
        output_dir: str | Path,
        generator_backend: str = "openai",
        generator_kwargs: Optional[Dict[str, Any]] = None,
        judge_model_name: str = "gpt-5",
        judge_backend: str = "openai",
        judge_weight: float = 0.7,
        max_samples: Optional[int] = None,
        views: Iterable[str] | None = None,
    ) -> "Evaluator":
        config = EvaluationConfig(
            input_file=Path(input_file),
            output_dir=Path(output_dir),
            generator_backend=generator_backend,
            generator_kwargs=generator_kwargs or {},
            judge_model_name=judge_model_name,
            judge_backend=judge_backend,
            judge_weight=judge_weight,
            max_samples=max_samples,
            views=views or ("front", "side", "top", "iso"),
        )
        generator = _build_generator(
            backend=config.generator_backend,
            kwargs=config.generator_kwargs,
        )
        judge = _build_judge(
            backend=config.judge_backend,
            model_name=config.judge_model_name,
            scoring_config=config.scoring_config,
        )
        executor = CADExecutor(views=config.views)
        return cls(generator=generator, judge=judge, executor=executor, config=config)

    def _read_prompts(self) -> List[str]:
        with self._config.input_file.open("r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
        if self._config.max_samples is not None:
            return prompts[: self._config.max_samples]
        return prompts

    def _compute_final_score(
        self,
        *,
        judge_response: JudgeResponse,
        compilation_success: bool,
    ) -> float:
        judge_score = judge_response.clipped_score(
            upper=self._config.scoring_config.max_score
        )
        normalised = judge_score / self._config.scoring_config.max_score
        compilation_score = 1.0 if compilation_success else 0.0
        return (
            self._config.judge_weight * normalised
            + (1.0 - self._config.judge_weight) * compilation_score
        )

    def run(self) -> EvaluationResult:
        artefact_root = self._config.output_dir / "artefacts"
        _ensure_dir(artefact_root)
        # Reset partial state file
        self._state_path.write_text("", encoding="utf-8")

        prompts = self._read_prompts()
        self._logger.info(
            "Starting evaluation: %d prompts, generator=%s, judge=%s",
            len(prompts),
            self._config.generator_backend,
            self._config.judge_model_name,
        )
        evaluation = EvaluationResult()

        for index, prompt in enumerate(prompts):
            sample_id = f"sample_{index:05d}"
            sample_dir = artefact_root / sample_id
            _ensure_dir(sample_dir)

            self._logger.info("Sample %s: generating CAD code", sample_id)
            gen_start = time.perf_counter()
            try:
                cad_code_raw = self._generator.generate(prompt)
                cad_code = str(cad_code_raw)
            except Exception as exc:  # pragma: no cover - defensive
                cad_code = ""
                compilation_result = CADCompilationResult(
                    success=False,
                    stl_path=None,
                    image_paths=[],
                    stdout="",
                    stderr="",
                    error=f"Model generation failed: {exc}",
                )
                self._logger.exception("Sample %s: generation failed", sample_id)
            else:
                gen_duration = time.perf_counter() - gen_start
                self._logger.info(
                    "Sample %s: generation completed in %.2fs (%d chars)",
                    sample_id,
                    gen_duration,
                    len(cad_code),
                )
                self._logger.info("Sample %s: compiling CAD code", sample_id)
                compile_start = time.perf_counter()
                compilation_result = self._executor.compile(
                    cad_code=cad_code,
                    artefact_dir=sample_dir,
                    sample_id=sample_id,
                )
                compile_duration = time.perf_counter() - compile_start
                self._logger.info(
                    "Sample %s: compilation %s in %.2fs (images=%d)",
                    sample_id,
                    "succeeded" if compilation_result.success else "failed",
                    compile_duration,
                    len(compilation_result.image_paths),
                )

            try:
                self._logger.info("Sample %s: invoking LLM judge", sample_id)
                judge_start = time.perf_counter()
                judge_response = self._judge.score(
                    prompt=prompt,
                    cad_code=cad_code,
                    rendering_paths=compilation_result.rendering_paths,
                    compilation_log="\n".join(
                        filter(
                            None,
                            [
                                compilation_result.stdout.strip(),
                                compilation_result.stderr.strip(),
                                compilation_result.error or "",
                            ],
                        )
                    )
                    or None,
                    config=self._config.scoring_config,
                )
            except Exception as exc:  # pragma: no cover - defensive
                judge_response = JudgeResponse(
                    score=0.0,
                    reasoning=f"Judge failure: {exc}",
                    raw_response=None,
                )
                judge_duration = time.perf_counter() - judge_start
                self._logger.info(
                    "Sample %s: judge failed in %.2fs", sample_id, judge_duration
                )
                self._logger.exception("Sample %s: judge invocation failed", sample_id)
            else:
                judge_duration = time.perf_counter() - judge_start
                self._logger.info(
                    "Sample %s: judge completed in %.2fs (score=%.2f)",
                    sample_id,
                    judge_duration,
                    judge_response.score,
                )
                self._logger.debug(
                    "Sample %s: judge response type=%s",
                    sample_id,
                    type(judge_response),
                )

            final_score = self._compute_final_score(
                judge_response=judge_response,
                compilation_success=compilation_result.success,
            )
            self._logger.info(
                "Sample %s: final blended score %.3f (compile=%s)",
                sample_id,
                final_score,
                "success" if compilation_result.success else "fail",
            )

            sample_result = EvaluationSampleResult(
                prompt=prompt,
                cad_code=cad_code,
                compilation=compilation_result,
                judge=judge_response,
                final_score=final_score,
                sample_id=sample_id,
            )
            evaluation.samples.append(sample_result)
            self._append_state_record(sample_result)

            processed = index + 1
            if processed % 50 == 0:
                self._logger.info(
                    "Checkpoint: writing partial outputs after %d samples", processed
                )
                self._write_outputs(evaluation)

        self._write_outputs(evaluation)
        self._logger.info(
            "Evaluation finished. Average score: %.3f over %d samples.",
            evaluation.average_score,
            len(evaluation.samples),
        )
        return evaluation

    def _write_outputs(self, evaluation: EvaluationResult) -> None:
        results_path = self._config.output_dir / "results.jsonl"
        metrics_path = self._config.output_dir / "metrics.json"

        results_path.write_text(evaluation.to_jsonl(), encoding="utf-8")
        metrics_payload = {
            "average_score": evaluation.average_score,
            "num_samples": len(evaluation.samples),
            "judge_weight": self._config.judge_weight,
            "scoring_max": self._config.scoring_config.max_score,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        self._write_html_report(evaluation)

    def _write_html_report(self, evaluation: EvaluationResult) -> None:
        report_path = self._config.output_dir / "report.html"

        rows: List[str] = []
        for sample in evaluation.samples:
            prompt_html = (
                "<div class='prompt'>"
                f"{html.escape(sample.prompt)}"
                "</div>"
            )
            code_html = (
                "<pre class='code-block'><code>"
                f"{html.escape(sample.cad_code)}"
                "</code></pre>"
            )

            compilation = sample.compilation
            if compilation.success:
                stl_text = (
                    html.escape(self._relativize_path(compilation.stl_path))
                    if compilation.stl_path
                    else "N/A"
                )
                compilation_details = f"<strong>Success</strong><br>STL: {stl_text}"
            else:
                error_text = html.escape(compilation.error or "Unknown error")
                if compilation.stderr:
                    stderr_excerpt = compilation.stderr.strip()
                    if len(stderr_excerpt) > 600:
                        stderr_excerpt = stderr_excerpt[:600] + "…"
                    stderr_text = html.escape(stderr_excerpt)
                else:
                    stderr_text = ""
                compilation_details = f"<strong>Failed</strong><br>{error_text}"
                if stderr_text:
                    compilation_details += f"<br><span class='log'>{stderr_text}</span>"

            image_tags: List[str] = []
            for idx, image_path in enumerate(compilation.image_paths):
                rel_path = self._relativize_path(image_path)
                if not rel_path:
                    continue
                image_tags.append(
                    f"<img src=\"{html.escape(rel_path)}\" "
                    f"alt=\"render {idx + 1} for {html.escape(sample.sample_id)}\" "
                    "class=\"render-img\" />"
                )
            images_html = "".join(image_tags) if image_tags else "<em>No renders</em>"

            reasoning_text = sample.judge.reasoning or "No reasoning provided."
            reasoning_html = html.escape(reasoning_text).replace("\n", "<br>")
            judge_html = (
                f"<strong>Score:</strong> {sample.judge.score:.2f}<br>"
                f"<span class='reason'>{reasoning_html}</span>"
            )
            if not compilation.success:
                judge_html = (
                    "<em>Compilation failed.</em><br>"
                    f"{judge_html}"
                )

            rows.append(
                "<tr>"
                f"<td>{prompt_html}</td>"
                f"<td>{code_html}</td>"
                f"<td>{compilation_details}</td>"
                f"<td>{images_html}</td>"
                f"<td>{judge_html}</td>"
                "</tr>"
            )

        table_rows = "\n".join(rows)
        generator_label = self._config.generator_kwargs.get("model_name") or self._config.generator_backend
        judge_label = self._config.judge_model_name

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CAD Evaluation Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 2rem;
      background: #f9fafb;
      color: #1f2933;
    }}
    h1 {{
      margin-bottom: 0.25rem;
    }}
    .summary {{
      margin-bottom: 1.5rem;
      color: #4b5563;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #ffffff;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.1);
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 0.75rem;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
      font-weight: 600;
    }}
    .prompt {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .code-block {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 0.75rem;
      overflow-x: auto;
      max-height: 260px;
      white-space: pre-wrap;
    }}
    .render-img {{
      max-width: 160px;
      max-height: 160px;
      margin: 0.3rem;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      background: #ffffff;
    }}
    .log {{
      display: block;
      margin-top: 0.5rem;
      font-size: 0.85rem;
      color: #b91c1c;
      white-space: pre-wrap;
    }}
    .reason {{
      display: block;
      margin-top: 0.35rem;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <h1>CAD Evaluation Report</h1>
  <div class="summary">
    <div><strong>Prompts evaluated:</strong> {len(evaluation.samples)}</div>
    <div><strong>Average blended score:</strong> {evaluation.average_score:.3f}</div>
    <div><strong>Judge weight:</strong> {self._config.judge_weight:.2f}</div>
    <div><strong>Generator:</strong> {html.escape(str(generator_label))}</div>
    <div><strong>Judge model:</strong> {html.escape(judge_label)}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Prompt</th>
        <th>Generated Code</th>
        <th>Compilation</th>
        <th>Renderings</th>
        <th>LLM Judge</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""
        report_path.write_text(html_content, encoding="utf-8")

    def _relativize_path(self, path: Optional[Path]) -> str:
        if path is None:
            return ""
        path_obj = Path(path)
        try:
            relative = path_obj.relative_to(self._config.output_dir)
        except ValueError:
            relative = path_obj
        return relative.as_posix()
    def _sample_to_dict(self, sample: EvaluationSampleResult) -> dict:
        compilation = sample.compilation.as_dict()
        return {
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "cad_code": sample.cad_code,
            "compilation": compilation,
            "judge": {
                "score": sample.judge.score,
                "reasoning": sample.judge.reasoning,
                "raw_response": sample.judge.raw_response,
            },
            "final_score": sample.final_score,
        }

    def _append_state_record(self, sample: EvaluationSampleResult) -> None:
        record = self._sample_to_dict(sample)
        with self._state_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
