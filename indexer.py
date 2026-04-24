"""Data indexing module for MédiaScrape.

This module is the data processing layer between scraping and search:
- cleans raw scraped dictionaries with pandas
- extracts simple image features with NumPy
- persists indexed rows in SQLite via SQLAlchemy ORM

Used by:
- app.py: trigger indexing after scraping
- search.py: consume indexed text + vectors
- visualizer.py: consume cleaned tabular data
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
import logging

import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from models import MediaItem, ScrapeSession, SessionLocal
from scraper import REQUEST_HEADERS
from scraper import get_domain, scrape


# TODO (search.py): expose deserialize_vector() for KNN similarity queries
# TODO (visualizer.py): the cleaned DataFrame can be passed directly for chart generation
# TODO (app.py): run_indexer() will be called right after scrape() in the Flask route
# TODO (models.py): if more fields are needed by search.py, add them to MediaItem schema


def clean_data(items: list[dict]) -> pd.DataFrame:
    """Clean raw scraped items and return a normalized DataFrame.

    Steps performed:
    - convert list[dict] -> DataFrame
    - remove rows with missing/empty URL
    - strip whitespace in string columns
    - truncate description to 500 chars
    - normalize media_type to lowercase
    - drop duplicates by URL
    - fill missing title/description defaults
    """
    logger = logging.getLogger(__name__)
    logger.debug("[clean_data] Received %d raw items.", len(items))

    if not items:
        empty_df = pd.DataFrame(
            columns=[
                "url",
                "source_url",
                "media_type",
                "title",
                "description",
                "domain",
                "file_extension",
            ]
        )
        logger.debug("[clean_data] No items provided. Returning empty DataFrame.")
        return empty_df

    df = pd.DataFrame(items)

    # Ensure expected columns exist even if some dicts are incomplete.
    expected_columns = [
        "url",
        "source_url",
        "media_type",
        "title",
        "description",
        "domain",
        "file_extension",
    ]
    for column in expected_columns:
        if column not in df.columns:
            df[column] = None

    # Drop null/empty URL rows.
    df = df[df["url"].notna()].copy()
    df["url"] = df["url"].astype(str)
    df = df[df["url"].str.strip() != ""].copy()

    # Strip whitespace from all object/string columns.
    string_columns = df.select_dtypes(include=["object"]).columns
    for col in string_columns:
        df[col] = df[col].apply(lambda value: value.strip() if isinstance(value, str) else value)

    # Normalize media type.
    df["media_type"] = df["media_type"].fillna("").astype(str).str.lower().str.strip()

    # Truncate description to max 500 chars.
    df["description"] = df["description"].fillna("").astype(str).str.slice(0, 500)

    # Fill missing title/description with defaults.
    df["title"] = df["title"].fillna("").astype(str)
    df.loc[df["title"].str.strip() == "", "title"] = "Untitled"

    df.loc[df["description"].str.strip() == "", "description"] = "No description"

    # Fill missing source_url/domain using URL fallback logic.
    df["source_url"] = df["source_url"].fillna("").astype(str)
    df.loc[df["source_url"].str.strip() == "", "source_url"] = df["url"]

    df["domain"] = df["domain"].fillna("").astype(str)
    missing_domain = df["domain"].str.strip() == ""
    df.loc[missing_domain, "domain"] = df.loc[missing_domain, "source_url"].apply(get_domain)

    # Optional extension normalization.
    df["file_extension"] = df["file_extension"].fillna("").astype(str)
    df.loc[df["file_extension"].str.strip() == "", "file_extension"] = None

    before_drop = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    dropped_duplicates = before_drop - len(df)
    df.attrs["dropped_duplicates"] = dropped_duplicates

    logger.debug(
        "[clean_data] Cleaning complete. rows_in=%d, rows_out=%d, dropped_duplicates=%d",
        len(items),
        len(df),
        dropped_duplicates,
    )

    return df


def extract_features(image_url: str) -> list[float] | None:
    """Extract a weighted 112-dim visual feature vector from an image URL.

    Components:
    - HSV histogram (48 dims): 16 bins each for H, S, V channels
    - Edge-density histogram (16 dims): gradient-magnitude distribution on grayscale
    - Spatial HSV grid (48 dims): flattened 4x4 HSV pixel layout

    The three components are individually normalized, weighted, concatenated
    (48 + 16 + 48 = 112), and then L2-normalized.
    """
    # TODO (search.py): richer 112-dim mixed vectors improve KNN similarity accuracy
    logger = logging.getLogger(__name__)
    logger.debug("[extract_features] Processing image: %s", image_url)

    try:
        with requests.Session() as session:
            session.headers.update(REQUEST_HEADERS)
            response = session.get(image_url, timeout=5)
        if response.status_code != 200:
            logger.debug("[extract_features] Non-200 status (%s).", response.status_code)
            return None

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 1) HSV color histogram (48 dims).
        hsv_image = image.convert("HSV")
        hsv_array = np.array(hsv_image, dtype=np.float64)
        hue_hist, _ = np.histogram(hsv_array[:, :, 0], bins=16, range=(0, 256))
        sat_hist, _ = np.histogram(hsv_array[:, :, 1], bins=16, range=(0, 256))
        val_hist, _ = np.histogram(hsv_array[:, :, 2], bins=16, range=(0, 256))
        hsv_component = np.concatenate([hue_hist, sat_hist, val_hist]).astype(np.float64)
        hsv_sum = hsv_component.sum()
        if hsv_sum == 0:
            return None
        hsv_component = hsv_component / hsv_sum

        # 2) Edge-density histogram (16 dims).
        gray = image.convert("L")
        gray_arr = np.array(gray, dtype=np.float64)
        gx = np.abs(np.diff(gray_arr, axis=1, prepend=gray_arr[:, :1]))
        gy = np.abs(np.diff(gray_arr, axis=0, prepend=gray_arr[:1, :]))
        edges = gx + gy
        edge_hist, _ = np.histogram(edges, bins=16, range=(0, 512))
        edge_component = edge_hist.astype(np.float64)
        edge_sum = edge_component.sum()
        if edge_sum == 0:
            return None
        edge_component = edge_component / edge_sum

        # 3) Spatial HSV grid (48 dims from 4x4x3).
        grid_hsv = image.resize((4, 4), Image.LANCZOS).convert("HSV")
        grid_arr = np.array(grid_hsv, dtype=np.float64).reshape(16, 3)
        spatial_component = grid_arr.flatten()
        spatial_sum = spatial_component.sum()
        if spatial_sum == 0:
            return None
        spatial_component = spatial_component / spatial_sum

        # Weighted combination then final L2 normalization.
        weighted_hsv = hsv_component * 1.0
        weighted_edge = edge_component * 1.5
        weighted_spatial = spatial_component * 2.0

        vector = np.concatenate([weighted_hsv, weighted_edge, weighted_spatial]).astype(np.float64)
        vector = vector / (np.linalg.norm(vector) + 1e-8)

        return vector.tolist()

    except requests.exceptions.Timeout:
        logger.warning("[extract_features] Timeout while downloading image.")
    except requests.exceptions.RequestException as exc:
        logger.exception("[extract_features] Download failed: %s", exc)
    except UnidentifiedImageError:
        logger.warning("[extract_features] Content is not a valid image.")
    except OSError as exc:
        logger.exception("[extract_features] PIL failed to open image: %s", exc)
    except Exception as exc:
        logger.exception("[extract_features] Unexpected error: %s", exc)

    return None


def serialize_vector(vector: list[float] | None) -> str | None:
    """Serialize feature vector list (expected length: 112) to JSON for DB storage."""
    if vector is None:
        return None
    return json.dumps(vector)


def deserialize_vector(vector_str: str | None) -> list[float] | None:
    """Deserialize JSON vector string back to a list of floats (typically length 112)."""
    if vector_str is None:
        return None

    try:
        loaded = json.loads(vector_str)
        if not isinstance(loaded, list):
            return None
        return [float(value) for value in loaded]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def index_items(df: pd.DataFrame, session_id: int) -> dict[str, int]:
    """Insert cleaned items into the DB, extracting vectors for image rows.

    Rules:
    - skip URL if it already exists in DB
    - for image rows, extract + serialize feature vectors
    - keep processing even if one row fails
    """
    total_rows = int(len(df))
    inserted = 0
    skipped = 0
    failed = 0

    if total_rows == 0:
        return {"inserted": 0, "skipped": 0, "failed": 0, "total": 0}

    db = SessionLocal()

    try:
        seen_in_batch: set[str] = set()

        for _, row in df.iterrows():
            try:
                url = str(row.get("url", "")).strip()
                if not url:
                    failed += 1
                    continue

                # Defensive batch-level duplicate protection.
                if url in seen_in_batch:
                    skipped += 1
                    continue
                seen_in_batch.add(url)

                existing = db.query(MediaItem).filter(MediaItem.url == url).first()
                if existing is not None:
                    skipped += 1
                    continue

                media_type = str(row.get("media_type", "")).strip().lower()
                vector_json: str | None = None

                if media_type == "image":
                    vector = extract_features(url)
                    vector_json = serialize_vector(vector)

                item = MediaItem(
                    url=url,
                    source_url=str(row.get("source_url", url)),
                    media_type=media_type or "link",
                    title=str(row.get("title", "Untitled")),
                    description=str(row.get("description", "No description")),
                    domain=str(row.get("domain", get_domain(str(row.get("source_url", url))))),
                    file_extension=row.get("file_extension"),
                    feature_vector=vector_json,
                    is_indexed=True,
                    session_id=session_id,
                )

                db.add(item)
                db.commit()
                inserted += 1

            except SQLAlchemyError as exc:
                db.rollback()
                failed += 1
                logging.getLogger(__name__).exception("[index_items] DB error while inserting row: %s", exc)
            except Exception as exc:
                db.rollback()
                failed += 1
                logging.getLogger(__name__).exception("[index_items] Unexpected row failure: %s", exc)

        # Keep parent session total in sync with what was successfully indexed this run.
        session_row = db.query(ScrapeSession).filter(ScrapeSession.id == session_id).first()
        if session_row is not None:
            session_row.total_items = inserted + skipped
            db.commit()

    finally:
        db.close()

    return {"inserted": inserted, "skipped": skipped, "failed": failed, "total": total_rows}


def reindex_all_images() -> dict[str, int]:
    """Recompute feature vectors for every image media row in the database.

    Returns:
        {"updated": int, "failed": int}
    """
    db = SessionLocal()
    updated = 0
    failed = 0

    try:
        image_rows = db.query(MediaItem).filter(MediaItem.media_type == "image").all()

        for item in image_rows:
            try:
                vector = extract_features(item.url)
                if vector is None:
                    failed += 1
                    continue

                item.feature_vector = serialize_vector(vector)
                item.is_indexed = True
                db.commit()
                updated += 1

            except Exception as exc:
                db.rollback()
                failed += 1
                logging.getLogger(__name__).exception(
                    "[reindex_all_images] Failed for id=%s, url=%s: %s", item.id, item.url, exc
                )

    finally:
        db.close()

    return {"updated": updated, "failed": failed}


def run_indexer(scrape_results: dict) -> dict[str, Any]:
    """Run full indexing flow from scraper output dict.

    Input should be the result from `scraper.scrape()`.
    """
    items = scrape_results.get("items", []) if isinstance(scrape_results, dict) else []
    session_id = scrape_results.get("session_id") if isinstance(scrape_results, dict) else None

    base_summary = {
        "session_id": session_id,
        "scrape_total": int(scrape_results.get("total", 0)) if isinstance(scrape_results, dict) else 0,
        "scrape_images": int(scrape_results.get("images", 0)) if isinstance(scrape_results, dict) else 0,
        "scrape_links": int(scrape_results.get("links", 0)) if isinstance(scrape_results, dict) else 0,
        "scrape_articles": int(scrape_results.get("articles", 0)) if isinstance(scrape_results, dict) else 0,
    }

    if session_id is None:
        logging.getLogger(__name__).warning(
            "[run_indexer] Missing session_id. Cannot index without DB session context."
        )
        return {
            **base_summary,
            "inserted": 0,
            "skipped": 0,
            "failed": 0,
            "total": 0,
        }

    cleaned_df = clean_data(items)

    if cleaned_df.empty:
        logging.getLogger(__name__).debug("[run_indexer] No items to index after cleaning.")
        return {
            **base_summary,
            "inserted": 0,
            "skipped": int(cleaned_df.attrs.get("dropped_duplicates", 0)),
            "failed": 0,
            "total": 0,
        }

    index_summary = index_items(cleaned_df, session_id=session_id)

    # Include duplicates removed during cleaning in skipped total for clearer reporting.
    duplicates_removed = int(cleaned_df.attrs.get("dropped_duplicates", 0))
    combined_skipped = int(index_summary["skipped"]) + duplicates_removed

    combined = {
        **base_summary,
        "inserted": int(index_summary["inserted"]),
        "skipped": combined_skipped,
        "failed": int(index_summary["failed"]),
        "total": int(index_summary["total"]),
    }

    logging.getLogger(__name__).info(
        "[run_indexer] Indexing summary -> inserted=%d, skipped=%d, failed=%d, total=%d",
        combined["inserted"],
        combined["skipped"],
        combined["failed"],
        combined["total"],
    )

    return combined


def scrape_and_index(url: str) -> dict[str, Any]:
    """Convenience helper: scrape URL then index results in one call."""
    scraped = scrape(url)
    return run_indexer(scraped)
