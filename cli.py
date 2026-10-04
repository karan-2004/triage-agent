#!/usr/bin/env python3
"""TriageAgent CLI.

Usage:
    python cli.py ingest data/sample_tickets.txt
    python cli.py triage "Login page is blank on Safari 17"
    python cli.py triage --file new_ticket.txt
    python cli.py report data/sample_tickets.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from triage.config import load
from triage.ingest import ingest
from triage.agent import TriageAgent
from triage.reports import report_file
from triage.store import VectorStore


def cmd_ingest(args):
    cfg = load()
    if args.index_dir:
        cfg.index_dir = Path(args.index_dir)
    n_tickets, n_devs = ingest(cfg, Path(args.ticket_file))
    print(f"Ingested {n_tickets} tickets and {n_devs} developer profiles.")
    print(f"Index saved to {cfg.index_dir}")


def cmd_triage(args):
    cfg = load()
    if args.index_dir:
        cfg.index_dir = Path(args.index_dir)
    if args.file:
        query = Path(args.file).read_text()
    else:
        query = args.text

    agent = TriageAgent(cfg)
    result = agent.triage(query)

    print(json.dumps(result.as_dict(), indent=2))
    if result.action == "note":
        print("\n--- INTERNAL NOTE (to support) ---")
        print(result.note)
    else:
        print(f"\n--- ROUTED TO: {result.assigned_dev or 'nobody'} ---")
        print(result.reason)


def cmd_report(args):
    print(report_file(Path(args.ticket_file)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="triage", description="Local ticket triage agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Build the index from a ticket-thread file")
    p_ingest.add_argument("ticket_file")
    p_ingest.add_argument("--index-dir", default=None)
    p_ingest.set_defaults(func=cmd_ingest)

    p_triage = sub.add_parser("triage", help="Triage a new ticket")
    p_triage.add_argument("text", nargs="?", default="")
    p_triage.add_argument("--file", default=None)
    p_triage.add_argument("--index-dir", default=None)
    p_triage.set_defaults(func=cmd_triage)

    p_report = sub.add_parser("report", help="Print a ticket report")
    p_report.add_argument("ticket_file")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
