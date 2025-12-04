from __future__ import annotations

import abc


class BaseGenerator(abc.ABC):
    """Abstract base class representing a text-to-CAD generator."""

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Return CAD code corresponding to the provided natural language prompt."""
