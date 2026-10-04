# TriageAgent

A local, open-source support-ticket triage agent for a dev team. Point it at a
chat log; it groups messages into ticket threads, builds an index, then for any
new ticket it either **writes a short internal note to the support team** (when
a similar resolved thread exists) or **routes it to the developer** whose past
responses it most resembles. It also prints case-based reports (open tickets
per dev, critical cases, and a full case list).

Built with local AI at its core (Ollama + open models). No data leaves your
machine.

## How it works

1. **Ingest** — parse a free-form chat log, group messages by ticket ID,
   embed each thread (`nomic-embed-text`), store in a local vector index. It
   also builds one "profile" embedding per developer from every message they
   wrote.
2. **Triage** — embed the new ticket and search:
   - if a past **resolved** thread is similar enough (≥ threshold) → an LLM
     writes a direct internal note for the support team (fix + who handled it);
   - otherwise → route to the dev whose past work is closest.
3. **Report** — case-by-case: open tickets per dev, critical cases, and a full
   case table.

## Setup

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b     # small + fast on CPU; use llama3.2 for better quality
   ```
3. Create a venv and install Python deps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

```bash
# 1. build the index
python cli.py ingest data/sample_tickets.txt

# 2. triage a new ticket
python cli.py triage "#999 the login page crashes on Safari"
python cli.py triage --file new_ticket.txt

# 3. report
python cli.py report data/sample_tickets.txt
```

Web UI:

```bash
streamlit run app.py
```

## Chat log format

No strict format. Each line is one message, optionally prefixed with a speaker
(`name:` or `[name]`). Messages are grouped by the ticket ID they mention.
A line with no ticket ID attaches to the most recent thread.

```text
support: #101 the login page crashes on Safari
alice: #101 which Safari version?
support: #101 17.4 on macOS 14
alice: #101 Safari-specific JS bug in the login handler. fixed in v2.3.1
```

- **Ticket ID**: `#123`, `ticket-123`, `TKT-123`, `case 123`, …
- **Dev vs support**: any speaker that isn't a known support role
  (`support`, `customer`, `user`, `client`, `agent`, `bot`) is a dev.
- **Resolved**: a dev message containing `resolved` / `fixed` / `shipped` /
  `deployed` / `closed` / `patched`, or an explicit `resolution:` / `fixed:` line.
- **Priority**: `P0`/`P1`/`P2`/`P3`, a `priority: critical` tag, or inferred
  from words like `crash`, `outage`, `production`, `down`.

## Configuration (env vars)

| Var | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base |
| `TRIAGE_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `TRIAGE_CHAT_MODEL` | `llama3.2:1b` | note-writing model |
| `TRIAGE_SIM_THRESHOLD` | `0.65` | similarity above which to auto-note |
| `TRIAGE_TOP_K` | `3` | neighbours to consider |

## Tests

```bash
source .venv/bin/activate
pip install pytest
pytest
```
