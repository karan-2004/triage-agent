import numpy as np
import pytest

from triage.store import VectorStore


def test_add_then_search_orders_by_similarity(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("a", "ticket", "login", {}, [1, 0, 0])
    store.add("b", "ticket", "orthogonal", {}, [0, 1, 0])
    store.add("c", "ticket", "login again", {}, [1, 0, 0])

    results = store.search([1, 0, 0], top_k=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[-1]["score"] == pytest.approx(0.0)
    assert {r["id"] for r in results[:2]} == {"a", "c"}


def test_search_kind_filter(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("t1", "ticket", "login", {}, [1, 0, 0])
    store.add("d1", "dev", "alice profile", {}, [1, 0, 0])

    results = store.search([1, 0, 0], top_k=3, kind="dev")
    assert len(results) == 1
    assert results[0]["id"] == "d1"
    assert results[0]["kind"] == "dev"


def test_save_load_round_trip(tmp_path):
    store = VectorStore(tmp_path / "store")
    store.add("a", "ticket", "hello", {"x": 1}, [1, 0, 0])
    store.add("b", "dev", "world", {"y": 2}, [0, 1, 0])
    store.save()

    loaded = VectorStore(tmp_path / "store")
    assert loaded.load() is True
    assert loaded.docs == store.docs
    assert np.allclose(loaded.vectors, store.vectors)


def test_search_empty_store(tmp_path):
    store = VectorStore(tmp_path / "store")
    assert store.search([1, 0, 0]) == []
