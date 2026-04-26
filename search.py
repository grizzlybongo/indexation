"""Search and retrieval module for MédiaScrape.

This module provides:
- Text search using hybrid BM25 + semantic ranking
- Image similarity search using KNN over color histogram vectors
- Simple filters by media type and domain
- Dashboard-ready aggregate statistics
"""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Any
import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import func
try:
    from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional during bootstrap
    BM25Okapi = None  # type: ignore[assignment]
try:
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional during bootstrap
    SentenceTransformer = None  # type: ignore[assignment]

from indexer import EMBEDDING_MODEL_NAME, deserialize_vector, extract_features
from models import MediaItem, ScrapeSession, SessionLocal


# TODO (app.py): build_tfidf_index() should be cached using Flask app context, not rebuilt per request
# TODO (app.py): expose search_by_image_similarity() via a file upload endpoint
# TODO (visualizer.py): get_stats() top_domains can feed directly into bar chart
# TODO (indexer.py): consider increasing histogram bins from 8 to 16 for better KNN accuracy
# TODO (demo.ipynb): demonstrate all 6 search functions with real scraped data


def _to_result_dict(item: MediaItem, score: float = 0.0) -> dict[str, Any]:
    """Convert a MediaItem ORM object into a consistent API dictionary."""
    return {
        "id": item.id,
        "url": item.url,
        "title": item.title,
        "description": item.description,
        "media_type": item.media_type,
        "domain": item.domain,
        "score": float(score),
    }


TEXT_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize a string for BM25 with a lightweight, language-agnostic regex."""
    return [token for token in TEXT_TOKEN_PATTERN.findall((text or "").lower()) if token]


@lru_cache(maxsize=1)
def _load_query_embed_model() -> Any:
    """Load and cache the query embedding model once per process."""
    logger = logging.getLogger(__name__)

    if SentenceTransformer is None:
        logger.warning(
            "[search_by_text] sentence-transformers is unavailable. Falling back to BM25-only ranking."
        )
        return None

    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        logger.exception("[search_by_text] Failed loading query embedding model: %s", exc)
        return None


def build_hybrid_index() -> dict[str, Any]:
    """Build in-memory index structures for hybrid text retrieval."""
    db = SessionLocal()

    try:
        items = (
            db.query(MediaItem)
            .filter(MediaItem.is_indexed.is_(True))
            .order_by(MediaItem.id.asc())
            .all()
        )

    finally:
        db.close()

    if not items:
        return {
            "items": [],
            "bm25": None,
            "corpus_tokens": [],
            "semantic_matrix": None,
            "semantic_item_positions": [],
        }

    corpus = [f"{item.title or ''} {item.description or ''}".strip() for item in items]
    corpus_tokens = [_tokenize(doc) for doc in corpus]

    bm25_model = None
    if BM25Okapi is not None and any(tokens for tokens in corpus_tokens):
        bm25_model = BM25Okapi(corpus_tokens)

    semantic_vectors: list[list[float]] = []
    semantic_item_positions: list[int] = []
    for position, item in enumerate(items):
        vector = deserialize_vector(item.text_embedding)
        if vector is None:
            continue
        if len(vector) != 384:
            continue
        semantic_vectors.append(vector)
        semantic_item_positions.append(position)

    semantic_matrix = (
        np.array(semantic_vectors, dtype=np.float64)
        if semantic_vectors
        else None
    )

    return {
        "items": items,
        "bm25": bm25_model,
        "corpus_tokens": corpus_tokens,
        "semantic_matrix": semantic_matrix,
        "semantic_item_positions": semantic_item_positions,
    }


def reciprocal_rank_fusion(
    bm25_ranks: list[int], semantic_ranks: list[int], total_items: int, k: int = 60
) -> np.ndarray:
    """Fuse ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores = np.zeros(total_items, dtype=np.float64)

    for rank, idx in enumerate(bm25_ranks):
        scores[idx] += 1.0 / (k + rank + 1)

    for rank, idx in enumerate(semantic_ranks):
        scores[idx] += 1.0 / (k + rank + 1)

    return scores


def search_by_text(
    query: str,
    top_n: int = 10,
    search_index: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid retrieval with BM25 and semantic scoring fused by RRF."""
    if not query or not query.strip():
        return []

    index_data = search_index or build_hybrid_index()
    items: list[MediaItem] = index_data.get("items", [])
    if not items:
        return []

    query_tokens = _tokenize(query)
    bm25_scores = np.zeros(len(items), dtype=np.float64)

    bm25_model = index_data.get("bm25")
    if bm25_model is not None and query_tokens:
        bm25_scores = np.array(bm25_model.get_scores(query_tokens), dtype=np.float64)

    semantic_scores = np.zeros(len(items), dtype=np.float64)
    semantic_item_positions = np.array(index_data.get("semantic_item_positions", []), dtype=np.int64)
    semantic_matrix = index_data.get("semantic_matrix")

    if semantic_matrix is not None and semantic_item_positions.size > 0:
        model = _load_query_embed_model()
        if model is not None:
            try:
                query_vector = np.array(model.encode(query, normalize_embeddings=True), dtype=np.float64)
                query_norm = np.linalg.norm(query_vector)
                if query_norm > 0:
                    normalized_query = query_vector / query_norm
                    matrix_norms = np.linalg.norm(semantic_matrix, axis=1)
                    similarities = np.dot(semantic_matrix, normalized_query) / np.clip(matrix_norms, 1e-8, None)
                    for local_pos, global_pos in enumerate(semantic_item_positions.tolist()):
                        semantic_scores[global_pos] = float(similarities[local_pos])
            except Exception as exc:
                logging.getLogger(__name__).exception("[search_by_text] Semantic scoring failed: %s", exc)

    bm25_ranks = (
        np.argsort(bm25_scores)[::-1].tolist()
        if bm25_model is not None and query_tokens
        else []
    )

    semantic_ranks: list[int] = []
    if semantic_item_positions.size > 0:
        ranked_local = np.argsort(semantic_scores[semantic_item_positions])[::-1]
        semantic_ranks = semantic_item_positions[ranked_local].astype(int).tolist()

    if not bm25_ranks and not semantic_ranks:
        return []

    fused_scores = reciprocal_rank_fusion(
        bm25_ranks=bm25_ranks,
        semantic_ranks=semantic_ranks,
        total_items=len(items),
    )
    fused_ranks = np.argsort(fused_scores)[::-1]

    results: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for idx in fused_ranks:
        item = items[int(idx)]
        bm25_score = float(bm25_scores[int(idx)])
        semantic_score = float(semantic_scores[int(idx)])

        if bm25_score <= 0.0 and semantic_score <= 0.0:
            continue

        if bm25_score > 0.0 and semantic_score > 0.0:
            match_reason = "hybrid"
        elif semantic_score > 0.0:
            match_reason = "semantic"
        else:
            match_reason = "bm25"

        signature = f"{item.media_type}::{item.url}"
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        row = _to_result_dict(item, score=float(fused_scores[int(idx)]))
        row["match_reason"] = match_reason
        row["bm25_score"] = bm25_score
        row["semantic_score"] = semantic_score
        results.append(row)

        if len(results) >= top_n:
            break

    return results


def build_knn_index() -> tuple[NearestNeighbors | None, list[int]]:
    """Build a KNN index from image feature vectors stored in the DB."""
    # TODO (indexer.py): the richer the color histogram, the better KNN results
    db = SessionLocal()

    try:
        items = (
            db.query(MediaItem)
            .filter(
                MediaItem.is_indexed.is_(True),
                MediaItem.media_type == "image",
                MediaItem.feature_vector.is_not(None),
            )
            .order_by(MediaItem.id.asc())
            .all()
        )

        vectors: list[list[float]] = []
        item_ids: list[int] = []

        EXPECTED_DIMS = 112
        for item in items:
            vector = deserialize_vector(item.feature_vector)
            if vector is None:
                continue
            if len(vector) != EXPECTED_DIMS:
                continue   # skip stale vectors from old indexer versions
            vectors.append(vector)
            item_ids.append(item.id)

        if len(vectors) < 1:
            logging.getLogger(__name__).warning(
                "[build_knn_index] Warning: no indexed images with vectors. KNN disabled."
            )
            return None, []

        matrix = np.array(vectors, dtype=np.float64)
        knn = NearestNeighbors(metric="euclidean", algorithm="brute")
        knn.fit(matrix)

        return knn, item_ids

    finally:
        db.close()


def search_by_image_similarity(image_url: str, top_n: int = 5) -> list[dict[str, Any]]:
    """Find nearest indexed images by cosine distance over histogram vectors."""
    # TODO (app.py): accept uploaded image file instead of URL in Flask route
    if not image_url or not image_url.strip():
        return []

    query_vector = extract_features(image_url)
    if query_vector is None:
        return [{"error": "Could not extract features from query image URL."}]

    knn_model, item_ids = build_knn_index()
    if knn_model is None or not item_ids:
        return []

    query_array = np.array([query_vector], dtype=np.float64)
    k = min(max(int(top_n), 1), len(item_ids))

    distances, indices = knn_model.kneighbors(query_array, n_neighbors=k)

    db = SessionLocal()
    try:
        id_to_item = {
            item.id: item
            for item in db.query(MediaItem).filter(MediaItem.id.in_(item_ids)).all()
        }

        results: list[dict[str, Any]] = []
        for distance, vec_idx in zip(distances[0], indices[0]):
            item_id = item_ids[int(vec_idx)]
            item = id_to_item.get(item_id)
            if item is None:
                continue

            similarity = max(0.0, min(100.0, round((1.0 - float(distance)) * 100.0, 1)))
            if similarity > 85.0:
                similarity_label = "Very similar"
            elif similarity > 60.0:
                similarity_label = "Similar"
            else:
                similarity_label = "Loosely similar"

            results.append(
                {
                    "id": item.id,
                    "url": item.url,
                    "title": item.title,
                    "media_type": item.media_type,
                    "domain": item.domain,
                    "distance": float(distance),
                    "similarity": similarity,
                    "similarity_label": similarity_label,
                }
            )

        # Already nearest-first from kneighbors, but explicitly enforce ordering.
        results.sort(key=lambda row: row["distance"])
        return results

    finally:
        db.close()


def search_by_media_type(media_type: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return indexed items filtered by media type."""
    valid_types = {"image", "link", "article"}
    normalized = (media_type or "").strip().lower()

    if normalized not in valid_types:
        return []

    db = SessionLocal()
    try:
        rows = (
            db.query(MediaItem)
            .filter(MediaItem.media_type == normalized)
            .order_by(MediaItem.id.desc())
            .limit(max(int(limit), 0))
            .all()
        )
        return [_to_result_dict(item, score=0.0) for item in rows]
    finally:
        db.close()


def search_by_domain(domain: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return items filtered by case-insensitive partial domain match."""
    term = (domain or "").strip()
    if not term:
        return []

    db = SessionLocal()
    try:
        rows = (
            db.query(MediaItem)
            .filter(MediaItem.domain.ilike(f"%{term}%"))
            .order_by(MediaItem.id.desc())
            .limit(max(int(limit), 0))
            .all()
        )
        return [_to_result_dict(item, score=0.0) for item in rows]
    finally:
        db.close()


def get_stats() -> dict[str, Any]:
    """Return dashboard summary statistics for indexed media."""
    # TODO (visualizer.py): pass this dict directly to chart generation functions
    db = SessionLocal()

    try:
        total_items = db.query(func.count(MediaItem.id)).scalar() or 0
        total_images = db.query(func.count(MediaItem.id)).filter(MediaItem.media_type == "image").scalar() or 0
        total_links = db.query(func.count(MediaItem.id)).filter(MediaItem.media_type == "link").scalar() or 0
        total_articles = db.query(func.count(MediaItem.id)).filter(MediaItem.media_type == "article").scalar() or 0
        total_sessions = db.query(func.count(ScrapeSession.id)).scalar() or 0
        indexed_count = db.query(func.count(MediaItem.id)).filter(MediaItem.is_indexed.is_(True)).scalar() or 0

        top_domains_rows = (
            db.query(MediaItem.domain, func.count(MediaItem.id).label("count"))
            .group_by(MediaItem.domain)
            .order_by(func.count(MediaItem.id).desc())
            .limit(5)
            .all()
        )
        top_domains = [(domain, int(count)) for domain, count in top_domains_rows]

        indexed_ratio = float((indexed_count / total_items) * 100.0) if total_items > 0 else 0.0

        return {
            "total_items": int(total_items),
            "total_images": int(total_images),
            "total_links": int(total_links),
            "total_articles": int(total_articles),
            "total_sessions": int(total_sessions),
            "top_domains": top_domains,
            "indexed_ratio": indexed_ratio,
        }

    finally:
        db.close()
