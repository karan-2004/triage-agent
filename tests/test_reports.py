from triage.reports import render, summarize
from triage.ticket import Message, Thread


def _make_threads():
    t1 = Thread(id="101", title="login crash", priority="critical",
                messages=[Message("alice", "looking into it")])
    t2 = Thread(id="102", title="billing wrong", priority="high", resolved=True,
                messages=[Message("bob", "shipped the fix")])
    t3 = Thread(id="103", title="onboarding", priority="normal",
                messages=[Message("carol", "working on it")])
    return [t1, t2, t3]


def test_summarize_counts():
    s = summarize(_make_threads())
    assert s["total"] == 3
    assert s["open"] == 2
    assert s["resolved"] == 1


def test_open_per_dev_excludes_resolved():
    s = summarize(_make_threads())
    assert s["open_per_dev"] == {"alice": 1, "carol": 1}
    assert "bob" not in s["open_per_dev"]


def test_critical_cases_only_open_high_priority():
    s = summarize(_make_threads())
    ids = [c["id"] for c in s["critical_cases"]]
    assert ids == ["101"]


def test_cases_entry_shape():
    s = summarize(_make_threads())
    assert len(s["cases"]) == 3
    for c in s["cases"]:
        for k in ("id", "title", "priority", "status", "dev", "messages"):
            assert k in c


def test_render_contains_report_header_and_case_ids():
    s = summarize(_make_threads())
    r = render(s)
    assert "TICKET REPORT" in r
    assert "101" in r
    assert "102" in r
    assert "103" in r
