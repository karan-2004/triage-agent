"""The triage brain: write an internal note from a past thread, or route to a dev."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .ollama import Ollama
from .store import VectorStore


@dataclass
class TriageResult:
    action: str  # "note" | "route"
    query: str = ""
    # note path
    note: str = ""
    sources: list[dict] = field(default_factory=list)
    # route path
    assigned_dev: str = ""
    route_score: float = 0.0
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "query": self.query,
            "note": self.note,
            "sources": self.sources,
            "assigned_dev": self.assigned_dev,
            "route_score": self.route_score,
            "reason": self.reason,
        }


SYSTEM = """You are a triage assistant for an internal dev team. You are given a
new support ticket and one or more similar PAST tickets that were already
resolved. Write a SHORT internal note addressed to the support team.

The note must:
- say whether this looks like a duplicate of a past ticket (name its ID)
- state the fix that was applied, and who handled it
- be direct and technical — no greeting, no sign-off, no invented details

Base the note ONLY on the provided context. Refer only to the PAST ticket IDs
in the context, never to the new ticket's own ID. If the context is thin or
unrelated, say the ticket needs triage instead of guessing."""


class TriageAgent:
    def __init__(self, cfg: Config, store: VectorStore | None = None):
        self.cfg = cfg
        self.store = store or self._load_store()
        self.ol = Ollama(cfg)

    def _load_store(self) -> VectorStore:
        store = VectorStore(self.cfg.index_dir)
        if not store.load():
            raise RuntimeError(
                f"No index found at {self.cfg.index_dir}. Run ingest first."
            )
        return store

    def triage(self, query: str) -> TriageResult:
        qvec = self.ol.embed_one(self.cfg.embed_model, query)

        similar = self.store.search(qvec, top_k=self.cfg.top_k, kind="ticket")
        if not similar:
            return TriageResult(action="route", query=query,
                                reason="No historical tickets indexed yet.")

        # Only auto-note from *resolved* threads — an open thread means we have
        # seen this before but don't yet have the fix.
        resolved = [s for s in similar if s["meta"].get("resolved")]
        if resolved and resolved[0]["score"] >= self.cfg.similarity_threshold:
            return self._note(query, resolved)

        return self._route(query, qvec, similar)

    def _note(self, query: str, similar: list[dict]) -> TriageResult:
        context = "\n\n".join(
            f"# Past ticket {s['meta'].get('ticket_id')} (similarity {s['score']:.2f})\n"
            f"TITLE: {s['meta'].get('title', '')}\n"
            f"RESOLUTION: {s['meta'].get('resolution', '')}\n"
            f"HANDLED BY: {s['meta'].get('dev', 'unknown')}"
            for s in similar
        )
        user = f"New ticket:\n{query}\n\nPast tickets:\n{context}"
        note = self.ol.chat(self.cfg.chat_model, SYSTEM, user)
        return TriageResult(
            action="note",
            query=query,
            note=note,
            sources=[{"id": s["id"], "score": s["score"],
                      "title": s["meta"].get("title", "")} for s in similar],
        )

    def _route(self, query: str, qvec: list[float], similar: list[dict] | None = None) -> TriageResult:
        best_ticket = self.store.search(qvec, top_k=1, kind="ticket")
        best_sim = best_ticket[0]["score"] if best_ticket else 0.0
        best_open = None
        if similar and similar[0]["score"] >= self.cfg.similarity_threshold and not similar[0]["meta"].get("resolved"):
            best_open = similar[0]

        devs = self.store.search(qvec, top_k=self.cfg.top_k, kind="dev")
        if not devs:
            return TriageResult(action="route", query=query,
                                reason="No developer profiles indexed yet.")
        best = devs[0]

        if best_open:
            reason = (
                f"A very similar ticket already exists (#{best_open['meta'].get('ticket_id')} "
                f"\"{best_open['meta'].get('title', '')}\", similarity {best_open['score']:.2f}) "
                f"but it is still open, so I routed to '{best['meta']['dev']}' — "
                f"their past work is closest."
            )
        else:
            reason = (
                f"No sufficiently similar resolved ticket found (best match {best_sim:.2f}). "
                f"Routed to '{best['meta']['dev']}' because their past work is closest "
                f"(similarity {best['score']:.2f})."
            )
        return TriageResult(
            action="route",
            query=query,
            assigned_dev=best["meta"]["dev"],
            route_score=best["score"],
            reason=reason,
        )
