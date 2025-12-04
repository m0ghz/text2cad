"""Shared constants and helpers for Gemini model aliases."""

from __future__ import annotations

from typing import Optional

GEMINI_FLASH_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_PRO_MODEL = "gemini-2.5-pro"

GEMINI_MODEL_ALIASES = {
    "gemini2.5-flash": GEMINI_FLASH_MODEL,
    "gemini2.5 flash": GEMINI_FLASH_MODEL,
    "gemini-2.5-flash": GEMINI_FLASH_MODEL,
    "gemini-2.5-flash-preview-09-2025": GEMINI_FLASH_MODEL,
    "gemini2.5-pro": GEMINI_PRO_MODEL,
    "gemini2.5 pro": GEMINI_PRO_MODEL,
    "gemini-2.5-pro": GEMINI_PRO_MODEL,
}


def normalise_gemini_model_name(model_name: str) -> Optional[str]:
    """Return the canonical Gemini model string or None if not recognised."""

    if not model_name:
        return None
    return GEMINI_MODEL_ALIASES.get(model_name.strip().lower())
