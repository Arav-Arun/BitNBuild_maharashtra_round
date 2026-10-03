"""Small async OpenAI-compatible client with rate-limit aware retries."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from blackbox.config import Settings

_DURATION_PART = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)")


def parse_duration(value: str | None) -> float:
    """Parse Groq-style reset durations such as ``2m59.56s``."""

    if not value:
        return 0.0
    total = 0.0
    factors = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    for match in _DURATION_PART.finditer(value.strip()):
        total += float(match.group("value")) * factors[match.group("unit")]
    if total == 0.0:
        try:
            return float(value)
        except ValueError:
            return 0.0
    return total


class HeaderRateLimiter:
    """Cooperatively waits when response headers report an exhausted quota."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._blocked_until = 0.0

    async def wait(self) -> None:
        async with self._lock:
            delay = self._blocked_until - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def update(self, headers: Mapping[str, str]) -> None:
        requests = headers.get("x-ratelimit-remaining-requests")
        tokens = headers.get("x-ratelimit-remaining-tokens")
        exhausted = requests == "0" or tokens == "0"
        if not exhausted:
            return
        reset = max(
            parse_duration(headers.get("x-ratelimit-reset-requests")),
            parse_duration(headers.get("x-ratelimit-reset-tokens")),
        )
        async with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + reset)


class AsyncLLMClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_retries = max_retries
        self.rate_limiter = HeaderRateLimiter()
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> AsyncLLMClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required for a live LLM call")
        payload: dict[str, Any] = {"messages": list(messages), "model": model, **params}
        if tools is not None:
            payload["tools"] = list(tools)
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        for attempt in range(self.max_retries + 1):
            await self.rate_limiter.wait()
            response = await self._client.post(
                "chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            await self.rate_limiter.update(response.headers)
            if response.status_code != 429:
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("LLM response must be a JSON object")
                return data
            if attempt == self.max_retries:
                response.raise_for_status()
            retry_after = parse_duration(response.headers.get("retry-after"))
            backoff = min(8.0, 0.25 * (2**attempt)) + random.random() * 0.1
            await asyncio.sleep(max(retry_after, backoff))
        raise AssertionError("retry loop exhausted")


async def _ping(settings: Settings) -> None:
    async with AsyncLLMClient(
        api_key=settings.groq_api_key,
        base_url=settings.llm_base_url,
    ) as client:
        response = await client.chat(
            model=settings.agent_model,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            temperature=0,
            max_tokens=8,
        )
    print(json.dumps(response, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ping"])
    args = parser.parse_args()
    if args.command == "ping":
        asyncio.run(_ping(Settings.load()))


if __name__ == "__main__":
    main()
