from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI


class AsyncLLMClient:
    def __init__(self, api_key: str, base_url: str, model_name: str | None = None, cache_path: str = "./cache.jsonl"):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.cache_path = cache_path
        self.cache: dict[tuple[Any, ...], str] = {}
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url) if api_key and base_url else None

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    datum = json.loads(line.strip())
                    key = datum["input"]
                    if isinstance(key, list):
                        key = tuple(tuple(x) if isinstance(x, list) else x for x in key)
                    self.cache[key] = datum["response"]

    def _make_cache_key(self, prompt: str, model: str, max_tokens: int, stop, logprobs: bool):
        if stop is None or isinstance(stop, str):
            return (prompt, model, max_tokens, stop, logprobs)
        return (prompt, model, max_tokens, tuple(stop), logprobs)

    async def request(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop=None,
        logprobs: bool = False,
        use_cache: bool = True,
    ):
        if self.async_client is None:
            raise ValueError("LLM client is not initialized. Set OPENAI_API_KEY and OPENAI_BASE_URL or pass them as args.")

        model = self.model_name or model or "gpt-4o-mini"
        cache_key = self._make_cache_key(prompt, model, max_tokens, stop, logprobs)
        if use_cache and temperature == 0 and cache_key in self.cache:
            result = self.cache[cache_key]
            if result and "error" not in result.lower():
                return result, True

        result = None
        for _ in range(5):
            try:
                response = await self.async_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    logprobs=logprobs,
                )
                result = response.choices[0].message.content
                break
            except Exception:
                await asyncio.sleep(0.5)

        if temperature == 0 and result is not None:
            self.cache[cache_key] = result
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"input": list(cache_key), "response": result}, ensure_ascii=False) + "\n")

        if result is None:
            return "Error", False
        return result, True
