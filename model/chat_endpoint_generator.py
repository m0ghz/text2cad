"""Generator that calls a user-specified Chat Completions compatible endpoint."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from urllib.parse import urljoin

from evaluate.generator import BaseGenerator

from .llm_shared import DEFAULT_MODEL_NAME, SYSTEM_MESSAGE

try:  # pragma: no cover - optional dependency for ChatEndpointGenerator
    import requests
except ImportError:  # pragma: no cover - delay failure until the generator is instantiated
    requests = None  # type: ignore[assignment]


class ChatEndpointGenerator(BaseGenerator):
    """Call a user-specified Chat Completions-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str = DEFAULT_MODEL_NAME,
        api_key: Optional[str] = None,
        extra_messages: Optional[Iterable[dict]] = None,
        temperature: Optional[float] = None,
        timeout: float = 300.0,
        extra_headers: Optional[Mapping[str, str]] = None,
        endpoint_path: str = "/v1/chat/completions",
        verify_tls: bool = True,
    ) -> None:
        if requests is None:  # pragma: no cover - handled when dependency missing
            raise ImportError(
                "The `requests` package is required for ChatEndpointGenerator. Install it via `pip install requests`."
            )

        if not base_url:
            raise ValueError("base_url must be provided for ChatEndpointGenerator.")

        # Normalise the endpoint and construct the final URL used for requests.
        normalised_base = base_url.rstrip("/") + "/"
        self._endpoint_url = urljoin(normalised_base, endpoint_path.lstrip("/"))

        self._model_name = model_name
        self._temperature = temperature
        self._timeout = timeout
        self._verify_tls = verify_tls
        self._extra_messages = list(extra_messages or [])

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._headers = headers

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
            *self._extra_messages,
        ]

        payload = {
            "model": self._model_name,
            "messages": messages,
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature

        response = requests.post(
            self._endpoint_url,
            json=payload,
            headers=self._headers,
            timeout=self._timeout,
            verify=self._verify_tls,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "ChatEndpointGenerator received a non-JSON response from the endpoint."
            ) from exc

        try:
            choices = data["choices"]
            if not choices:
                raise KeyError
            message = choices[0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "ChatEndpointGenerator received an unexpected response payload from the endpoint."
            ) from exc

        return content or ""
