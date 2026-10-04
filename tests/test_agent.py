import pytest

from triage.agent import TriageAgent, TriageResult
from triage.config import Config
from triage.store import VectorStore


class FakeOllama:
    def __init__(self, embeddings, chat_return="FAKE NOTE"):
        self.embeddings = embeddings
        self.chat_return = chat_return
        self.chat_calls = []

    def embed_one(self, model, text):
        return self.embeddings.get(text, [0.0, 0.0, 0.0])

    def chat(self, model, system, user):
        self.chat_calls.append((model, system, user))
        return self.chat_return


def _make_store(path, docs):
    store = VectorStore(path)
    for d in docs:
        store.add(d["id"], d["kind"], d["text"], d["meta"], d["vec"])
    return store


def test_triage_note_from_similar_resolved_ticket(tmp_path):
    store = _make_store(tmp_path / "s", [
        {"id": "ticket:42", "kind": "ticket", "text": "login crash",
         "meta": {"ticket_id": "42", "title": "login crash",
                  "resolution": "rebuilt cache", "dev": "alice", "resolved": True},
         "vec": [1.0, 0.0, 0.0]},
    ])
    agent = TriageAgent(Config(similarity_threshold=0.65), store)
    agent.ol = FakeOllama({"new ticket text": [1.0, 0.0, 0.0]},
                          chat_return="duplicate of #42")

    result = agent.triage("new ticket text")
    assert result.action == "note"
    assert result.note == "duplicate of #42"
    assert result.sources


def test_triage_routes_when_only_unresolved_similar(tmp_path):
    store = _make_store(tmp_path / "s", [
        {"id": "ticket:43", "kind": "ticket", "text": "still broken",
         "meta": {"ticket_id": "43", "title": "still broken", "resolved": False},
         "vec": [1.0, 0.0, 0.0]},
        {"id": "dev:alice", "kind": "dev", "text": "alice profile",
         "meta": {"dev": "alice"}, "vec": [0.0, 1.0, 0.0]},
    ])
    agent = TriageAgent(Config(similarity_threshold=0.65), store)
    agent.ol = FakeOllama({"q": [1.0, 0.0, 0.0]})

    result = agent.triage("q")
    assert result.action == "route"
    assert result.assigned_dev == "alice"
    assert "43" in result.reason


def test_triage_routes_to_top_dev_when_no_resolved_match(tmp_path):
    store = _make_store(tmp_path / "s", [
        {"id": "ticket:44", "kind": "ticket", "text": "billing",
         "meta": {"ticket_id": "44", "title": "billing", "resolved": True,
                  "resolution": "x", "dev": "old"},
         "vec": [0.0, 1.0, 0.0]},
        {"id": "dev:bob", "kind": "dev", "text": "bob profile",
         "meta": {"dev": "bob"}, "vec": [1.0, 0.0, 0.0]},
    ])
    agent = TriageAgent(Config(similarity_threshold=0.65), store)
    agent.ol = FakeOllama({"q": [1.0, 0.0, 0.0]})

    result = agent.triage("q")
    assert result.action == "route"
    assert result.assigned_dev == "bob"
    assert result.route_score == pytest.approx(1.0)


def test_triage_routes_when_no_ticket_docs(tmp_path):
    store = _make_store(tmp_path / "s", [
        {"id": "dev:alice", "kind": "dev", "text": "alice profile",
         "meta": {"dev": "alice"}, "vec": [1.0, 0.0, 0.0]},
    ])
    agent = TriageAgent(Config(), store)
    agent.ol = FakeOllama({"q": [1.0, 0.0, 0.0]})

    result = agent.triage("q")
    assert result.action == "route"
    assert result.reason == "No historical tickets indexed yet."


def test_triage_result_as_dict_keys():
    result = TriageResult(action="route", query="q")
    d = result.as_dict()
    for key in ("action", "query", "note", "sources",
                "assigned_dev", "route_score", "reason"):
        assert key in d
    assert d["action"] == "route"
    assert d["query"] == "q"
