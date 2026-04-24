"""Search and retrieval module for MédiaScrape.

This module provides:
- Text search using TF-IDF + cosine similarity
- Image similarity search using KNN over color histogram vectors
- Simple filters by media type and domain
- Dashboard-ready aggregate statistics
"""

from __future__ import annotations

from typing import Any
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sqlalchemy import func

from indexer import deserialize_vector, extract_features
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


def build_tfidf_index() -> tuple[TfidfVectorizer | None, np.ndarray | None, list[int]]:
    """Build a TF-IDF index from indexed media rows.

    Returns:
        (vectorizer, tfidf_matrix, item_ids)
    """
    # TODO (app.py): cache this index on startup, rebuild after each new scrape session
    db = SessionLocal()

    try:
        items = (
            db.query(MediaItem)
            .filter(MediaItem.is_indexed.is_(True))
            .order_by(MediaItem.id.asc())
            .all()
        )

        if not items:
            return None, None, []

        item_ids = [item.id for item in items]
        corpus = [f"{item.title or ''} {item.description or ''}".strip() for item in items]

        # Keep indexing robust even with very short/empty textual fields.
        if not any(text.strip() for text in corpus):
            return None, None, []

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_sparse = vectorizer.fit_transform(corpus)
        tfidf_matrix = tfidf_sparse.toarray()

        return vectorizer, tfidf_matrix, item_ids

    finally:
        db.close()


def search_by_text(query: str, top_n: int = 10) -> list[dict[str, Any]]:
    """Search indexed items using TF-IDF cosine similarity."""
    # TODO (app.py): pass top_n from Flask route query parameter
    if not query or not query.strip():
        return []

    vectorizer, tfidf_matrix, item_ids = build_tfidf_index()
    if vectorizer is None or tfidf_matrix is None or not item_ids:
        return []

    try:
        query_vec = vectorizer.transform([query]).toarray()
    except ValueError:
        # Common with query made only of stopwords after preprocessing.
        return []

    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    if similarities.size == 0:
        return []

    ranked_indices = np.argsort(similarities)[::-1]

    db = SessionLocal()
    try:
        id_to_item = {
            item.id: item
            for item in db.query(MediaItem).filter(MediaItem.id.in_(item_ids)).all()
        }

        results: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score <= 0.0:
                continue

            item_id = item_ids[int(idx)]
            item = id_to_item.get(item_id)
            if item is None:
                continue

            signature = f"{item.media_type}::{item.url}"
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            results.append(_to_result_dict(item, score=score))
            if len(results) >= top_n:
                break

        return results

    finally:
        db.close()


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

            results.append(
                {
                    "id": item.id,
                    "url": item.url,
                    "title": item.title,
                    "media_type": item.media_type,
                    "domain": item.domain,
                    "distance": float(distance),
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
