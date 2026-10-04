"""Ticket data model + parser for free-form chat logs grouped by ticket ID.

The input is a plain chat transcript (no strict block format). Each message is
a line, optionally prefixed with a speaker (``name:`` or ``[name]``). Messages
are grouped into threads by the ticket ID they mention (``#123``, ``ticket-123``,
``TKT-123``, ``case 123``, …). Any message with no ticket ID attaches to the
most recent thread.

Roles: anyone who is *not* a known support-side role is treated as a developer.
A thread is ``resolved`` when a developer writes a closing message or when an
explicit ``resolution:`` / ``fixed:`` line appears.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TICKET_ID_RE = re.compile(r"(?i)(?:#|ticket[-_ ]?|tkt[-_ ]?|case[-_ ]?)\s*(\d+)")
BRACKET_SPEAKER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$")
COLON_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.-]*)\s*:\s*(.*)$")

NON_DEV_ROLES = {"support", "customer", "user", "client", "agent", "bot", "sysadmin"}

RESOLVE_KEYWORDS = (
    "resolved", "fixed", "shipped", "deployed", "closed", "patched",
    "fixed in", "done", "pushed through",
)

PRIORITY_MAP = {"p0": "critical", "p1": "high", "p2": "normal", "p3": "low"}
CRITICAL_HINTS = ("urgent", "down", "crash", "crashing", "critical", "outage", "production")


@dataclass
class Message:
    speaker: str  # '' if unknown (treated as the support/reporter side)
    text: str


@dataclass
class Thread:
    id: str
    title: str = ""
    priority: str = "normal"
    resolved: bool = False
    resolution: str = ""
    messages: list[Message] = field(default_factory=list)

    @property
    def devs(self) -> list[str]:
        seen: list[str] = []
        for m in self.messages:
            sp = m.speaker.lower()
            if sp and sp not in NON_DEV_ROLES and sp not in seen:
                seen.append(sp)
        return seen

    @property
    def primary_dev(self) -> str:
        return self.devs[0] if self.devs else "unassigned"

    def full_text(self) -> str:
        parts = [self.title]
        parts.extend(f"{m.speaker}: {m.text}" for m in self.messages if m.text)
        if self.resolution:
            parts.append(f"resolution: {self.resolution}")
        return "\n".join(p for p in parts if p)


def _is_support(speaker: str) -> bool:
    return (not speaker) or speaker.lower() in NON_DEV_ROLES


def parse_chat_log(text: str) -> list[Thread]:
    threads: dict[str, Thread] = {}
    order: list[str] = []
    current_id: str | None = None

    def get_thread(tid: str) -> Thread:
        if tid not in threads:
            threads[tid] = Thread(id=tid)
            order.append(tid)
        return threads[tid]

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        tid_match = TICKET_ID_RE.search(line)
        tid = tid_match.group(1) if tid_match else None

        speaker = ""
        rest = line
        mb = BRACKET_SPEAKER_RE.match(line)
        mc = COLON_SPEAKER_RE.match(line)
        if mb:
            speaker, rest = mb.group(1).strip(), mb.group(2).strip()
        elif mc:
            sp = mc.group(1)
            # avoid mistaking times like "14:00" for a speaker
            if not re.fullmatch(r"\d{1,2}", sp):
                speaker, rest = sp.strip(), mc.group(2).strip()

        if tid is not None:
            current_id = tid
            t = get_thread(tid)
        else:
            if current_id is None:
                continue  # orphan line before any ticket id
            t = get_thread(current_id)

        # strip the ticket-id token from the message body (keeps text clean)
        rest = TICKET_ID_RE.sub("", rest).strip()

        if not t.title and rest:
            t.title = rest
        t.messages.append(Message(speaker=speaker, text=rest))

        low = rest.lower()

        # explicit priority
        pm = re.search(r"priority\s*[:=]\s*(\w+)", low)
        if pm:
            t.priority = pm.group(1).lower()
        else:
            for tok, prio in PRIORITY_MAP.items():
                if re.search(rf"\b{tok}\b", low):
                    t.priority = prio
                    break

        # infer raised priority from criticality hints
        if t.priority == "normal" and any(re.search(rf"\b{h}\b", low) for h in CRITICAL_HINTS):
            t.priority = "high"

        # explicit resolution line
        rm = re.search(r"(?:resolution|fixed|fix)\s*[:=]\s*(.+)", line, re.I)
        if rm:
            t.resolution = rm.group(1).strip()
            t.resolved = True
        elif any(k in low for k in RESOLVE_KEYWORDS) and not _is_support(speaker):
            # a developer's closing message
            if not t.resolved:
                t.resolution = rest
                t.resolved = True

    return [threads[t] for t in order]


# Backwards-compatible alias
parse_tickets = parse_chat_log
