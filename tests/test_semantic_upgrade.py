from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import indexer
import search


class FakeEmbedModel:
    def encode(self, text: str, normalize_embeddings: bool = True):
        del text, normalize_embeddings
        return np.linspace(0.0, 1.0, 384, dtype=np.float64)


class FakeQueryModel:
    def __init__(self, mapping: dict[str, list[float] | np.ndarray]):
        self.mapping = {
            key.lower().strip(): np.array(value, dtype=np.float64)
            for key, value in mapping.items()
        }

    def encode(self, text: str, normalize_embeddings: bool = True):
        del normalize_embeddings
        return self.mapping.get(text.lower().strip(), np.zeros(3, dtype=np.float64))


def _item(
    item_id: int,
    title: str,
    description: str,
    media_type: str = "article",
    domain: str = "example.com",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        url=f"https://example.com/{item_id}",
        title=title,
        description=description,
        media_type=media_type,
        domain=domain,
    )


def _hybrid_index(items: list[SimpleNamespace], semantic_vectors: list[list[float]]) -> dict:
    corpus = [f"{row.title or ''} {row.description or ''}".strip() for row in items]
    corpus_tokens = [search._tokenize(doc) for doc in corpus]

    bm25_model = None
    if search.BM25Okapi is not None and any(tokens for tokens in corpus_tokens):
        bm25_model = search.BM25Okapi(corpus_tokens)

    return {
        "items": items,
        "bm25": bm25_model,
        "corpus_tokens": corpus_tokens,
        "semantic_matrix": np.array(semantic_vectors, dtype=np.float64),
        "semantic_item_positions": list(range(len(items))),
    }


def test_embed_text_returns_384_float_vector(monkeypatch):
    monkeypatch.setattr(indexer, "_load_embed_model", lambda: FakeEmbedModel())

    vector = indexer.embed_text("Pagani Zonda", "Track-only V12 hypercar")

    assert vector is not None
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector[:10])


def test_embed_text_returns_none_for_empty_input(monkeypatch):
    monkeypatch.setattr(indexer, "_load_embed_model", lambda: FakeEmbedModel())

    assert indexer.embed_text("", "") is None
    assert indexer.embed_text(None, None) is None


def test_search_by_text_pagani_zonda_returns_pagani_or_zonda_top_result(monkeypatch):
    items = [
        _item(1, "Pagani Zonda R review", "Track monster and aero package"),
        _item(2, "Weekly newsletter", "General motorsport links"),
        _item(3, "Road trip guide", "Best scenic routes for spring"),
    ]
    index_data = _hybrid_index(items, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    query_model = FakeQueryModel({"pagani zonda": [1, 0, 0]})
    monkeypatch.setattr(search, "_load_query_embed_model", lambda: query_model)

    results = search.search_by_text("pagani zonda", top_n=3, search_index=index_data)

    assert results
    top = results[0]
    combined_text = f"{top.get('title') or ''} {top.get('description') or ''}".lower()
    assert "pagani" in combined_text or "zonda" in combined_text


def test_search_by_text_italian_supercar_has_semantic_recall_without_exact_terms(monkeypatch):
    items = [
        _item(1, "Modena Carbon Machine", "A twin-turbo flagship coupe from Emilia-Romagna"),
        _item(2, "Home gardening checklist", "How to maintain backyard flowers"),
        _item(3, "Laptop buying guide", "Choosing memory and CPU cores"),
    ]
    corpus = " ".join(f"{row.title} {row.description}".lower() for row in items)
    assert "italian" not in corpus
    assert "supercar" not in corpus

    index_data = _hybrid_index(items, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    query_model = FakeQueryModel({"italian supercar": [1, 0, 0]})
    monkeypatch.setattr(search, "_load_query_embed_model", lambda: query_model)

    results = search.search_by_text("italian supercar", top_n=3, search_index=index_data)

    assert results
    assert results[0]["id"] == 1
    assert results[0]["match_reason"] in {"semantic", "hybrid"}


def test_search_by_text_track_only_query_returns_track_concept_result(monkeypatch):
    items = [
        _item(1, "Weekend cooking notes", "Ideas for simple home dinners"),
        _item(2, "Circuit special prototype", "Non-homologated machine built only for closed-course use"),
        _item(3, "City commuting tips", "Choosing a daily electric scooter"),
    ]
    index_data = _hybrid_index(items, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    query_model = FakeQueryModel({"track only no road registration": [1, 0, 0]})
    monkeypatch.setattr(search, "_load_query_embed_model", lambda: query_model)

    results = search.search_by_text("track only no road registration", top_n=3, search_index=index_data)

    assert results
    assert results[0]["id"] == 2
    assert results[0]["match_reason"] in {"semantic", "hybrid"}


def test_search_by_text_v12_mid_engine_query_returns_engine_concept_result(monkeypatch):
    items = [
        _item(1, "Family camping planner", "Checklist for tents and sleeping bags"),
        _item(2, "Rear-cabin twelve-cylinder analysis", "Engine mounted behind cockpit with race aero"),
        _item(3, "Photography basics", "How shutter speed affects motion blur"),
    ]
    index_data = _hybrid_index(items, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    query_model = FakeQueryModel({"v12 mid-engine hypercar": [1, 0, 0]})
    monkeypatch.setattr(search, "_load_query_embed_model", lambda: query_model)

    results = search.search_by_text("V12 mid-engine hypercar", top_n=3, search_index=index_data)

    assert results
    assert results[0]["id"] == 2
    assert results[0]["match_reason"] in {"semantic", "hybrid"}
