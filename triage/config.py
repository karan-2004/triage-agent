"""Configuration for TriageAgent. Override with environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"


@dataclass
class Config:
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    embed_model: str = os.environ.get("TRIAGE_EMBED_MODEL", "nomic-embed-text")
    chat_model: str = os.environ.get("TRIAGE_CHAT_MODEL", "llama3.2:1b")
    # Tickets whose cosine similarity to a past thread is at/above this are
    # auto-answered from that thread's resolution rather than routed to a dev.
    similarity_threshold: float = float(os.environ.get("TRIAGE_SIM_THRESHOLD", "0.65"))
    top_k: int = int(os.environ.get("TRIAGE_TOP_K", "3"))
    index_dir: Path = Path(os.environ.get("TRIAGE_INDEX_DIR", str(INDEX_DIR)))


def load() -> Config:
    return Config()
