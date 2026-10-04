"""Thin client for the Ollama HTTP API (embed + chat), with retries."""

from __future__ import annotations

import time

import requests

from .config import Config


class Ollama:
    def __init__(self, cfg: Config):
        self.base = cfg.ollama_url.rstrip("/")
        self.embed_timeout = cfg.embed_timeout
        self.chat_timeout = cfg.chat_timeout

    # ---- internal ---------------------------------------------------------
    def _post(self, url: str, json: dict, timeout: int, retries: int = 3):
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=json, timeout=timeout)
                resp.raise_for_status()
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exc = exc
                time.sleep(2 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    # ---- embeddings -------------------------------------------------------
    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._post(
            f"{self.base}/api/embed",
            json={"model": model, "input": texts},
            timeout=self.embed_timeout,
        )
        return resp.json().get("embeddings", [])

    def embed_one(self, model: str, text: str) -> list[float]:
        return self.embed(model, [text])[0]

    # ---- chat -------------------------------------------------------------
    def chat(self, model: str, system: str, user: str) -> str:
        resp = self._post(
            f"{self.base}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=self.chat_timeout,
        )
        return resp.json()["message"]["content"].strip()
