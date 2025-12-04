"""
Evaluation pipeline orchestrating CAD compilation, rendering, and LLM judging.

Typical usage:

>>> from evaluate.evaluator import Evaluator
>>> evaluator = Evaluator.from_generator_class(
...     input_file="data/text.txt",
...     output_dir="outputs/eval",
...     generator_backend="openai",
... )
>>> evaluator.run()
"""

from .cad_executor import CADCompilationResult, CADExecutor
from .evaluator import EvaluationConfig, EvaluationResult, Evaluator
from .generator import BaseGenerator

__all__ = [
    "CADCompilationResult",
    "CADExecutor",
    "EvaluationConfig",
    "EvaluationResult",
    "Evaluator",
    "BaseGenerator",
]
