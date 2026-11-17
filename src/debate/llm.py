"""Model clients.

`OpenAICompatibleLLM` talks to any OpenAI-compatible API (Groq, Gemini, OpenRouter...).
Because free tiers are rate-limited, it:

* caches every response on disk, so re-running an experiment costs nothing and is
  reproducible;
* spaces requests out and retries with exponential backoff on rate-limit errors.

`FakeLLM` is a scripted stand-in used by the tests, so the whole pipeline can be checked
without an API key.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

Message = dict[str, str]


class LLM(Protocol):
    def complete(
        self, messages: list[Message], model: str, temperature: float = 0.4, max_tokens: int = 600
    ) -> str: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        cache_dir: str | Path = ".cache/llm",
        min_seconds_between_requests: float = 2.0,
        max_retries: int = 6,
    ):
        from openai import OpenAI  # imported here so tests don't need it configured

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set. "
                f"Get a free key from your provider and run: export {api_key_env}=..."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_seconds_between_requests
        self.max_retries = max_retries
        self._last_request = 0.0
        self.calls = 0
        self.cache_hits = 0

    def _cache_path(self, payload: dict) -> Path:
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def complete(
        self, messages: list[Message], model: str, temperature: float = 0.4, max_tokens: int = 600
    ) -> str:
        payload = {"messages": messages, "model": model, "temperature": temperature,
                   "max_tokens": max_tokens}
        path = self._cache_path(payload)
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())["content"]

        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()
            try:
                response = self.client.chat.completions.create(**payload)
                content = response.choices[0].message.content or ""
                break
            except Exception as err:  # rate limits, timeouts, server errors
                if attempt == self.max_retries:
                    raise
                backoff = min(60, 2**attempt) + random.random()
                print(f"  API error ({type(err).__name__}); retrying in {backoff:.0f}s")
                time.sleep(backoff)

        self.calls += 1
        path.write_text(json.dumps({"request": payload, "content": content}))
        return content


class FakeLLM:
    """Scripted model for tests: `responder(messages, model) -> str`."""

    def __init__(self, responder: Callable[[list[Message], str], str]):
        self.responder = responder
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(
        self, messages: list[Message], model: str, temperature: float = 0.4, max_tokens: int = 600
    ) -> str:
        self.calls.append((model, messages))
        return self.responder(messages, model)
