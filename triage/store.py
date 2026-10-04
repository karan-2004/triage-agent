"""A tiny persistent vector store using numpy cosine similarity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.docs: list[dict] = []
        self.vectors: np.ndarray | None = None  # shape (n, dim)

    # ---- persistence ------------------------------------------------------
    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "docs.json").write_text(json.dumps(self.docs, indent=2))
        if self.vectors is not None:
            np.save(self.path / "vectors.npy", self.vectors)

    def load(self) -> bool:
        docs_path = self.path / "docs.json"
        vecs_path = self.path / "vectors.npy"
        if not docs_path.exists() or not vecs_path.exists():
            return False
        self.docs = json.loads(docs_path.read_text())
        self.vectors = np.load(vecs_path)
        return True

    # ---- mutation ---------------------------------------------------------
    def add(self, doc_id: str, kind: str, text: str, metadata: dict, vector: list[float]) -> None:
        self.docs.append({"id": doc_id, "kind": kind, "text": text, "meta": metadata})
        vec = np.asarray(vector, dtype=np.float32)[None, :]
        if self.vectors is None:
            self.vectors = vec
        else:
            self.vectors = np.vstack([self.vectors, vec])

    def __len__(self) -> int:
        return len(self.docs)

    # ---- query ------------------------------------------------------------
    def search(self, query_vec: list[float], top_k: int = 3, kind: str | None = None) -> list[dict]:
        if self.vectors is None or len(self.docs) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9
        sims = (self.vectors / norms) @ q

        ranked = []
        for i, score in enumerate(sims):
            doc = self.docs[i]
            if kind is not None and doc["kind"] != kind:
                continue
            ranked.append((float(score), doc))
        ranked.sort(key=lambda x: -x[0])
        return [{"score": s, **d} for s, d in ranked[:top_k]]
