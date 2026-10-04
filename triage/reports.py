"""Case-based reports: ticket cases, open tickets per dev, critical cases."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .ticket import Thread, parse_chat_log

CRITICAL_PRIORITIES = {"critical", "high"}


def _case(t: Thread) -> dict:
    return {
        "id": t.id,
        "title": t.title,
        "priority": t.priority,
        "status": "resolved" if t.resolved else "open",
        "dev": t.primary_dev,
        "messages": len(t.messages),
    }


def summarize(threads: list[Thread]) -> dict:
    open_threads = [t for t in threads if not t.resolved]
    resolved_threads = [t for t in threads if t.resolved]

    open_per_dev: dict[str, int] = Counter(t.primary_dev for t in open_threads)
    critical_cases = [t for t in open_threads if t.priority in CRITICAL_PRIORITIES]

    return {
        "total": len(threads),
        "open": len(open_threads),
        "resolved": len(resolved_threads),
        "by_priority": dict(Counter(t.priority for t in threads)),
        "open_per_dev": dict(sorted(open_per_dev.items(), key=lambda x: -x[1])),
        "critical_cases": [_case(t) for t in critical_cases],
        "cases": [_case(t) for t in threads],
    }


def render(summary: dict) -> str:
    lines = [
        "TICKET REPORT",
        "=" * 40,
        f"Total tickets:  {summary['total']}",
        f"Open:           {summary['open']}",
        f"Resolved:       {summary['resolved']}",
        "",
        "By priority: " + ", ".join(f"{k}={v}" for k, v in sorted(summary["by_priority"].items())),
        "",
        "Open tickets per developer:",
    ]
    if summary["open_per_dev"]:
        for dev, n in summary["open_per_dev"].items():
            lines.append(f"  {dev:<15} {n}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"Critical cases ({len(summary['critical_cases'])} open):")
    if summary["critical_cases"]:
        for c in summary["critical_cases"]:
            lines.append(f"  [#{c['id']}] ({c['priority']}) {c['title']}  -> {c['dev']}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("All cases:")
    lines.append(f"  {'ID':>5}  {'PRIORITY':<10} {'STATUS':<9} {'DEV':<12} TITLE")
    lines.append("  " + "-" * 68)
    for c in summary["cases"]:
        title = (c["title"] or "")[:60]
        lines.append(
            f"  {c['id']:>5}  {c['priority']:<10} {c['status']:<9} {c['dev']:<12} {title}"
        )
    return "\n".join(lines)


def report_file(path: Path) -> str:
    threads = parse_chat_log(Path(path).read_text())
    return render(summarize(threads))
