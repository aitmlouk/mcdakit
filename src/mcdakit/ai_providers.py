"""Ready-made ``ask`` callables for the common language-model providers.

:mod:`mcdakit.ai` takes any ``Callable[[str], str]``, which is deliberately the
smallest interface that can work and commits the package to no vendor. The
cost of that generality is boilerplate: every user writes the same adapter,
looks up where their SDK buries the reply text, and gets no retry when a rate
limit arrives mid-run.

These adapters remove that work without spending the package's independence.
**No provider SDK is required.** Each adapter speaks the vendor's HTTP API
directly through :mod:`urllib` from the standard library, so

    pip install mcdakit

is enough to reach OpenAI, Anthropic, Google or a local model. NumPy remains
the only runtime dependency. Where a vendor SDK *is* installed it is not used:
one code path is easier to trust than two that must agree.

.. code-block:: python

    from mcdakit.ai import propose_criteria
    from mcdakit.ai_providers import anthropic, ollama

    ask = anthropic()                 # reads ANTHROPIC_API_KEY
    ask = ollama("llama3")            # local, no key, nothing leaves the machine
    proposal = propose_criteria("choosing a supplier", ask=ask)

Every adapter returns a plain function, so it can be wrapped, logged, cached
or replaced by a fixture in a test exactly like a hand-written one.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping

from .types import McdaError

__all__ = [
    "ProviderError",
    "anthropic",
    "google",
    "ollama",
    "openai",
    "openai_compatible",
]

#: Status codes worth retrying: a rate limit, and the transient server errors.
#: A 400 or 401 is a mistake in the request and will fail identically however
#: many times it is sent.
_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3


class ProviderError(McdaError):
    """A provider could not be reached, or refused the request."""


def _require_key(explicit: str | None, variable: str, provider: str) -> str:
    """Resolve an API key, and say plainly when there is none.

    A missing key otherwise surfaces as a 401 from the vendor, which reads as
    though the key were wrong rather than absent.
    """
    key = explicit or os.environ.get(variable)
    if not key:
        raise ProviderError(
            f"No API key for {provider}. Set the {variable} environment "
            f"variable, or pass api_key= explicitly. For a local model that "
            f"needs no key, use ollama() or openai_compatible()."
        )
    return key


def _post(
    url: str,
    payload: Mapping,
    headers: Mapping[str, str],
    *,
    timeout: float,
    retries: int,
    provider: str,
) -> dict:
    """POST JSON and return the decoded reply, retrying what is worth retrying.

    Backoff is exponential and honours ``Retry-After`` when the provider sends
    one, since a server's own estimate beats a guess. The final failure is
    raised with the provider's message attached: debugging a rate limit
    without knowing it was a rate limit is needlessly hard.
    """
    if retries < 1:
        raise ProviderError(
            f"retries must be at least 1; got {retries}. A request that is "
            f"never attempted cannot succeed."
        )

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )

    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if exc.code not in _RETRYABLE or attempt == retries - 1:
                raise ProviderError(
                    f"{provider} returned HTTP {exc.code}: {detail}"
                ) from exc
            last = exc
            delay = _backoff(exc, attempt)
        except urllib.error.URLError as exc:
            if attempt == retries - 1:
                raise ProviderError(
                    f"Could not reach {provider}: {exc.reason}. For a local "
                    f"model, check the server is running."
                ) from exc
            last = exc
            delay = 2.0**attempt
        time.sleep(delay)

    # Unreachable: the loop runs at least once, and its final iteration either
    # returns or raises. Kept as a guard against a future edit to that logic.
    raise ProviderError(  # pragma: no cover
        f"{provider} failed after {retries} attempts: {last}"
    )


def _backoff(error: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait, preferring the provider's own Retry-After."""
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass
    return 2.0**attempt


def _dig(payload: dict, path: tuple, provider: str) -> str:
    """Walk a reply to the text, and report the shape when it is not there.

    Providers change response shapes between versions, and an ``IndexError``
    from deep inside an adapter tells the caller nothing about which provider
    returned what.
    """
    node = payload
    for step in path:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Unexpected reply shape from {provider}: could not read "
                f"{'.'.join(map(str, path))}. Got: {json.dumps(payload)[:300]}"
            ) from exc
    if not isinstance(node, str):
        raise ProviderError(
            f"Expected text from {provider}, got {type(node).__name__}."
        )
    return node


def openai(
    model: str = "gpt-4o",
    *,
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Callable[[str], str]:
    """An ``ask`` callable backed by OpenAI's chat completions API.

    Reads ``OPENAI_API_KEY`` unless ``api_key`` is given.

    ``temperature`` defaults to 0. The module asks a model the same question
    more than once to measure whether it is consistent, and a temperature
    above zero would make the replies differ for reasons that have nothing to
    do with the model's confidence.
    """
    key = _require_key(api_key, "OPENAI_API_KEY", "OpenAI")
    return openai_compatible(
        model=model,
        base_url=base_url,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        provider="OpenAI",
    )


def openai_compatible(
    model: str,
    base_url: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    provider: str = "OpenAI-compatible server",
) -> Callable[[str], str]:
    """An ``ask`` callable for any server exposing ``/chat/completions``.

    That interface has become the common denominator, so this one adapter
    reaches vLLM, LM Studio, llama.cpp's server, Together, Groq, OpenRouter,
    DeepSeek, Mistral and most others — including models running on your own
    machine, where ``api_key`` is usually unnecessary.

    .. code-block:: python

        ask = openai_compatible("mistral-7b", "http://localhost:8000/v1")
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def ask(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        reply = _post(
            url,
            payload,
            headers,
            timeout=timeout,
            retries=retries,
            provider=provider,
        )
        return _dig(reply, ("choices", 0, "message", "content"), provider)

    return ask


def anthropic(
    model: str = "claude-sonnet-5",
    *,
    api_key: str | None = None,
    base_url: str = "https://api.anthropic.com/v1",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Callable[[str], str]:
    """An ``ask`` callable backed by Anthropic's messages API.

    Reads ``ANTHROPIC_API_KEY`` unless ``api_key`` is given.
    """
    key = _require_key(api_key, "ANTHROPIC_API_KEY", "Anthropic")
    url = base_url.rstrip("/") + "/messages"
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}

    def ask(prompt: str) -> str:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        reply = _post(
            url,
            payload,
            headers,
            timeout=timeout,
            retries=retries,
            provider="Anthropic",
        )
        return _dig(reply, ("content", 0, "text"), "Anthropic")

    return ask


def google(
    model: str = "gemini-2.0-flash",
    *,
    api_key: str | None = None,
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> Callable[[str], str]:
    """An ``ask`` callable backed by Google's Gemini API.

    Reads ``GOOGLE_API_KEY``, then ``GEMINI_API_KEY``, unless ``api_key`` is
    given.
    """
    key = (
        api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    )
    key = _require_key(key, "GOOGLE_API_KEY", "Google")
    url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
    headers = {"x-goog-api-key": key}

    def ask(prompt: str) -> str:
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        reply = _post(
            url,
            payload,
            headers,
            timeout=timeout,
            retries=retries,
            provider="Google",
        )
        return _dig(
            reply,
            ("candidates", 0, "content", "parts", 0, "text"),
            "Google",
        )

    return ask


def ollama(
    model: str = "llama3",
    *,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.0,
    timeout: float = 120.0,
    retries: int = DEFAULT_RETRIES,
) -> Callable[[str], str]:
    """An ``ask`` callable backed by a local Ollama server.

    No API key and no network: the problem description, the criteria and the
    measurements stay on the machine. That matters for decisions involving
    commercially sensitive data, which is a large share of real MCDA work.

    The timeout is longer than the hosted default because a local model on
    CPU is markedly slower than an API call.

    .. code-block:: python

        ask = ollama("llama3")
        ask = ollama("mixtral", base_url="http://gpu-box:11434")
    """
    url = base_url.rstrip("/") + "/api/generate"

    def ask(prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        reply = _post(
            url,
            payload,
            {},
            timeout=timeout,
            retries=retries,
            provider="Ollama",
        )
        return _dig(reply, ("response",), "Ollama")

    return ask
