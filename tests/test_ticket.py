from triage.ticket import parse_chat_log


def test_groups_messages_by_ticket_id():
    text = (
        "support: #101 login broken\n"
        "ticket-102 billing error\n"
        "alice: TKT-103 api timeout\n"
        "case 104 signup failure\n"
    )
    threads = parse_chat_log(text)
    assert [t.id for t in threads] == ["101", "102", "103", "104"]


def test_no_ticket_id_attaches_to_recent_thread():
    text = (
        "ticket-200 first issue\n"
        "this is a followup without an id\n"
    )
    threads = parse_chat_log(text)
    assert len(threads) == 1
    assert threads[0].id == "200"
    assert len(threads[0].messages) == 2
    assert threads[0].messages[1].text == "this is a followup without an id"


def test_orphan_lines_before_any_ticket_are_skipped():
    text = (
        "this orphan line has no ticket\n"
        "ticket-201 real thread\n"
    )
    threads = parse_chat_log(text)
    assert len(threads) == 1
    assert threads[0].id == "201"


def test_speaker_detection_bracket_colon_and_empty():
    text = (
        "[support] #401 opening\n"
        "alice: hello there\n"
        "[bob] working on it\n"
        "no speaker line\n"
    )
    threads = parse_chat_log(text)
    msgs = threads[0].messages
    assert msgs[0].speaker == "support"
    assert msgs[1].speaker == "alice"
    assert msgs[2].speaker == "bob"
    assert msgs[3].speaker == ""


def test_time_is_not_treated_as_speaker():
    text = (
        "ticket-600 morning sync\n"
        "14:00 all hands\n"
    )
    threads = parse_chat_log(text)
    assert threads[0].messages[1].speaker == ""
    assert threads[0].messages[1].text == "14:00 all hands"


def test_priority_levels_and_tags_and_hints():
    text = (
        "ticket-700 P0 login broken\n"
        "ticket-701 P1 slow dashboard\n"
        "ticket-702 priority: critical\n"
        "ticket-703 the api is down\n"
    )
    threads = parse_chat_log(text)
    by_id = {t.id: t for t in threads}
    assert by_id["700"].priority == "critical"
    assert by_id["701"].priority == "high"
    assert by_id["702"].priority == "critical"
    assert by_id["703"].priority == "high"


def test_resolution_from_dev_closing_keyword_and_explicit_lines():
    text = (
        "alice: #800 fixed the login bug\n"
        "support: #801 resolution: rolled back the deploy\n"
        "customer: #802 fixed: restarted the server\n"
    )
    threads = parse_chat_log(text)
    by_id = {t.id: t for t in threads}
    assert by_id["800"].resolved
    assert by_id["800"].resolution == "fixed the login bug"
    assert by_id["801"].resolved
    assert by_id["801"].resolution == "rolled back the deploy"
    assert by_id["802"].resolved
    assert by_id["802"].resolution == "restarted the server"


def test_dev_vs_support_roles():
    text = (
        "ticket-900 support: page is broken\n"
        "customer: can't log in\n"
        "user: same here\n"
        "alice: I'll take a look\n"
    )
    t = parse_chat_log(text)[0]
    assert t.devs == ["alice"]
    assert t.primary_dev == "alice"


def test_unassigned_thread_primary_dev():
    t = parse_chat_log("ticket-902 support: something is wrong\n")[0]
    assert t.devs == []
    assert t.primary_dev == "unassigned"


def test_title_does_not_contain_ticket_id_token():
    t = parse_chat_log("support: #101 the login page crashes\n")[0]
    assert t.title == "the login page crashes"
    assert "#101" not in t.title
