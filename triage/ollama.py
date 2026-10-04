"""Thin client for the Ollama HTTP API (embed + chat)."""

from __future__ import annotations

import requests

from .config import Config


class Ollama:
    def __init__(self, cfg: Config):
        self.base = cfg.ollama_url.rstrip("/")

    # ---- embeddings -------------------------------------------------------
    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self.base}/api/embed"
        resp = requests.post(url, json={"model": model, "input": texts}, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embeddings", [])

    def embed_one(self, model: str, text: str) -> list[float]:
        return self.embed(model, [text])[0]

    # ---- chat -------------------------------------------------------------
    def chat(self, model: str, system: str, user: str) -> str:
        url = f"{self.base}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
