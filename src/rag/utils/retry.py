"""Exponential-backoff retry decorator for transient API failures.

Free-tier LLM endpoints (Gemini, OpenAI) routinely return ``429 Too Many Requests``
under bursty workloads. Wrapping every external call with this decorator turns
those into automatic, capped retries instead of crashes.

Usage::

    @retry_with_backoff(retries=5, base_delay=1.0)
    def generate(self, prompt: str) -> str:
        ...

Errors classified as transient (matched by class name OR HTTP status text):
    - 429 / "rate limit" / "quota" / "ResourceExhausted"
    - 500 / 502 / 503 / 504 / "Internal" / "Unavailable" / "Service Unavailable"
    - Network errors (ConnectionError, TimeoutError, httpx.TransportError)
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import TypeVar

from rag.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


_TRANSIENT_TOKENS = (
    "429",
    "rate limit",
    "ratelimit",
    "quota",
    "resourceexhausted",
    "resource_exhausted",
    "500",
    "502",
    "503",
    "504",
    "internal",
    "unavailable",
    "deadline",
    "timeout",
    "temporar",
)


def _looks_transient(exc: BaseException) -> bool:
    """Return True if the exception text suggests a retryable failure."""
    if isinstance(exc, ConnectionError | TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if "transport" in name or "timeout" in name or "connection" in name:
        return True
    msg = str(exc).lower()
    return any(token in msg for token in _TRANSIENT_TOKENS)


def retry_with_backoff(
    retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.25,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a decorator that retries the wrapped callable with exponential backoff.

    Args:
        retries: Maximum number of attempts (including the first). Must be >= 1.
        base_delay: Initial sleep, in seconds, before the second attempt.
        max_delay: Cap on the sleep between attempts.
        jitter: Fractional jitter (e.g. 0.25 = +/- 25%) applied to each delay.

    Returns:
        A decorator that wraps any callable and retries it on transient failures.
        Non-transient failures propagate immediately.
    """
    if retries < 1:
        raise ValueError("retries must be >= 1")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> T:
            attempt = 0
            while True:
                attempt += 1
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    if attempt >= retries or not _looks_transient(exc):
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay *= 1 + random.uniform(-jitter, jitter)
                    log.warning(
                        "Transient failure in %s (attempt %d/%d): %s. Sleeping %.2fs.",
                        fn.__qualname__,
                        attempt,
                        retries,
                        exc,
                        delay,
                    )
                    time.sleep(max(0.0, delay))

        return wrapper

    return decorator
