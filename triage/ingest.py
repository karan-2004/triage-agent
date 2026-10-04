"""Ingest a chat-log file into the vector index.

Two kinds of documents are indexed:

* ``ticket`` — one per thread, for "have we seen this before?"
* ``dev`` — one per developer, built from every message they wrote, so a new
  ticket can be routed to whoever's past responses it most resembles.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .config import Config
from .ollama import Ollama
from .store import VectorStore
from .ticket import NON_DEV_ROLES, Thread, parse_chat_log

EMBED_BATCH = 16


def build_dev_profiles(threads: list[Thread]) -> dict[str, str]:
    profiles: dict[str, list[str]] = defaultdict(list)
    for t in threads:
        for m in t.messages:
            sp = m.speaker.lower()
            if sp and sp not in NON_DEV_ROLES:
                profiles[sp].append(f"#{t.id}: {m.text}")
    return {dev: "\n".join(chunks) for dev, chunks in profiles.items()}


def ingest(cfg: Config, ticket_file: Path, index_path: Path | None = None) -> tuple[int, int]:
    text = Path(ticket_file).read_text()
    threads = parse_chat_log(text)
    if not threads:
        raise ValueError(f"No ticket threads found in {ticket_file}")

    ol = Ollama(cfg)
    store = VectorStore(index_path or cfg.index_dir)

    # 1. index individual threads (embed in small batches to stay responsive)
    thread_texts = [t.full_text() for t in threads]
    for i in range(0, len(threads), EMBED_BATCH):
        chunk = thread_texts[i:i + EMBED_BATCH]
        vecs = ol.embed(cfg.embed_model, chunk)
        for t, vec in zip(threads[i:i + EMBED_BATCH], vecs):
            store.add(
                doc_id=f"ticket:{t.id}",
                kind="ticket",
                text=t.full_text(),
                metadata={
                    "ticket_id": t.id,
                    "title": t.title,
                    "dev": t.primary_dev,
                    "devs": t.devs,
                    "priority": t.priority,
                    "resolved": t.resolved,
                    "resolution": t.resolution,
                },
                vector=vec,
            )

    # 2. index developer profiles for routing
    profiles = build_dev_profiles(threads)
    if profiles:
        dev_names = list(profiles.keys())
        dev_texts = [profiles[d] for d in dev_names]
        for i in range(0, len(dev_names), EMBED_BATCH):
            chunk = dev_texts[i:i + EMBED_BATCH]
            vecs = ol.embed(cfg.embed_model, chunk)
            for dev, vec in zip(dev_names[i:i + EMBED_BATCH], vecs):
                store.add(
                    doc_id=f"dev:{dev}",
                    kind="dev",
                    text=profiles[dev],
                    metadata={"dev": dev, "ticket_count": len(profiles[dev].splitlines())},
                    vector=vec,
                )

    store.save()
    return len(threads), len(profiles)
