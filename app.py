"""Streamlit UI for TriageAgent.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from triage.config import load
from triage.agent import TriageAgent
from triage.ingest import ingest
from triage.reports import report_file
from triage.store import VectorStore

cfg = load()
SOURCE_FILE = cfg.index_dir / "source_tickets.txt"

st.set_page_config(page_title="TriageAgent", layout="wide")
st.title("TriageAgent")
st.caption("Local, open-source AI support-ticket triage — auto-reply or route to the right dev.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Setup")
    st.write(f"Embed: `{cfg.embed_model}`")
    st.write(f"Chat:  `{cfg.chat_model}`")
    st.divider()

    st.subheader("Train on ticket threads")
    ticket_text = st.text_area("Paste the chat log here", height=220,
                               help="A free-form chat log. Messages are grouped by ticket ID (#123, ticket-123, …).")
    uploaded = st.file_uploader("…or upload a .txt file", type=["txt"])
    if st.button("Build index", type="primary"):
        raw = ticket_text
        if uploaded is not None:
            raw = uploaded.getvalue().decode()
        if not raw.strip():
            st.error("Provide ticket threads first.")
        else:
            SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
            SOURCE_FILE.write_text(raw)
            with st.spinner("Indexing tickets… (first run loads the embedding model)"):
                try:
                    n_t, n_d = ingest(cfg, SOURCE_FILE, cfg.index_dir)
                    st.success(f"Indexed {n_t} tickets and {n_d} developer profiles.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    # index status
    store = VectorStore(cfg.index_dir)
    if store.load():
        n_tickets = sum(1 for d in store.docs if d["kind"] == "ticket")
        n_devs = sum(1 for d in store.docs if d["kind"] == "dev")
        st.caption(f"Index ready: {n_tickets} tickets, {n_devs} dev profiles.")

# ---------------------------------------------------------------- tabs
tab_triage, tab_report = st.tabs(["Triage", "Report"])

with tab_triage:
    st.subheader("New ticket")
    query = st.text_area("Paste the new ticket content", height=180,
                         placeholder="e.g. I can't reset my password — the reset link never arrives…")
    if st.button("Triage", type="primary"):
        if not query.strip():
            st.warning("Enter a ticket first.")
        elif not store.load():
            st.warning("No index yet — build it in the sidebar first.")
        else:
            with st.spinner("Triaging…"):
                try:
                    result = TriageAgent(cfg, store).triage(query)
                except Exception as exc:
                    st.error(f"Triage failed: {exc}")
                    st.stop()
            if result.action == "note":
                st.success("Matched a past ticket — wrote an internal note.")
                st.markdown("#### Note to support")
                st.write(result.note)
                with st.expander("Source threads"):
                    for s in result.sources:
                        st.write(f"- #{s['id']} ({s['score']:.2f}) — {s['title']}")
            else:
                st.info("No close match — routed to a developer.")
                st.markdown(f"#### Assigned to: `{result.assigned_dev}`")
                st.write(result.reason)

with tab_report:
    st.subheader("Ticket report")
    if SOURCE_FILE.exists():
        if st.button("Refresh report"):
            st.rerun()
        st.code(report_file(SOURCE_FILE), language="text")
    else:
        st.info("Build the index first to see a report.")
