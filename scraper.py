"""Web scraping module for MédiaScrape.

This module is the data acquisition layer. It:
- fetches HTML pages
- extracts images, links, and article-like text blocks
- records scrape session status in the database

Design goals:
- beginner-friendly and modular
- explicit print() tracing for each step
- safe fallbacks for common web-scraping edge cases
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

from models import MediaItem, ScrapeSession, SessionLocal, init_db

# Browser-like headers to reduce basic anti-bot blocking.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
    "Upgrade-Insecure-Requests": "1",
}

_MIN_CONTEXT_LENGTH = 40
_BOILERPLATE_FRAGMENTS = (
    "subscribe",
    "newsletter",
    "cookie",
    "privacy policy",
    "terms of use",
    "sign up",
    "log in",
    "advertisement",
)


# TODO (indexer.py): pass these raw dicts to indexer for NumPy feature extraction
# TODO (app.py): pagination — max_items param will be passed from Flask route
# TODO (search.py): tag high-value items (long descriptions) for TF-IDF priority


def normalize_url(url: str) -> str:
    """Ensure the URL has an HTTP/HTTPS scheme.

    If the user types `example.com`, we turn it into `https://example.com`.
    """
    cleaned = (url or "").strip()
    if not cleaned:
        return ""

    parsed = urlparse(cleaned)
    if parsed.scheme in {"http", "https"}:
        return cleaned

    return f"https://{cleaned}"


def fetch_page(url: str) -> BeautifulSoup | None:
    """Fetch a web page and return a parsed BeautifulSoup object.

    Returns None when the request fails or if the status code is not 200.
    """
    safe_url = normalize_url(url)
    if not safe_url:
        logger.debug("[fetch_page] Empty URL received.")
        return None

    logger.debug("[fetch_page] Fetching: %s", safe_url)

    try:
        with requests.Session() as session:
            session.headers.update(REQUEST_HEADERS)
            response = session.get(safe_url, timeout=10)
        logger.debug("[fetch_page] HTTP status: %s", response.status_code)

        if response.status_code == 403:
            logger.warning("[scraper] Access Denied (403). The site is blocking the request.")
            return None

        if response.status_code != 200:
            logger.debug("[fetch_page] Non-200 status code, returning None.")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        logger.debug("[fetch_page] Page parsed successfully.")
        return soup

    except requests.exceptions.Timeout:
        logger.warning("[fetch_page] Request timed out after 10 seconds.")
    except requests.exceptions.ConnectionError:
        logger.warning("[fetch_page] Connection error while reaching URL.")
    except requests.exceptions.HTTPError as exc:
        logger.exception("[fetch_page] HTTP error: %s", exc)
    except requests.exceptions.RequestException as exc:
        logger.exception("[fetch_page] Request failed: %s", exc)

    return None


def get_domain(url: str) -> str:
    """Extract a clean domain name from a URL.

    Example:
        https://www.bbc.com/news -> bbc.com
    """
    parsed = urlparse(url)
    hostname = (parsed.netloc or "").lower().strip()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    # Basic root-domain simplification for common hostnames:
    # - news.bbc.com -> bbc.com
    # - m.wikipedia.org -> wikipedia.org
    parts = [part for part in hostname.split(".") if part]
    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return hostname


def _extract_extension_from_url(url: str) -> str | None:
    """Extract a file extension from URL path (e.g., '.jpg')."""
    path = urlparse(url).path
    if not path or "." not in path:
        return None

    ext = "." + path.split(".")[-1].lower()

    # Keep it simple: only accept short extensions (e.g., jpg, png, html, webp).
    if 2 <= len(ext) <= 10:
        return ext
    return None


def _get_surrounding_paragraph_text(element: Any) -> str | None:
    """Capture nearby paragraph text while filtering boilerplate content."""

    def _passes_context_checks(p_tag: Any) -> bool:
        if p_tag is None or (getattr(p_tag, "name", "") or "").lower() != "p":
            return False

        text = p_tag.get_text(" ", strip=True)
        if len(text) < _MIN_CONTEXT_LENGTH:
            return False

        lower_text = text.lower()
        if any(fragment in lower_text for fragment in _BOILERPLATE_FRAGMENTS):
            return False

        return True

    # 1) Walk up parents looking for a valid <p>.
    for parent in element.parents:
        if _passes_context_checks(parent):
            return parent.get_text(" ", strip=True)

    # 2) Search within nearest <article>/<section>/<main> ancestor.
    container = None
    for parent in element.parents:
        parent_name = (getattr(parent, "name", "") or "").lower()
        if parent_name in {"article", "section", "main"}:
            container = parent
            break

    if container is not None:
        for paragraph in container.find_all("p"):
            if _passes_context_checks(paragraph):
                return paragraph.get_text(" ", strip=True)

    # 3) Fallback: scan forward paragraphs in document order.
    for paragraph in element.find_all_next("p"):
        if _passes_context_checks(paragraph):
            return paragraph.get_text(" ", strip=True)

    return None


def _is_in_noise_section(tag: Any) -> bool:
    """Return True when tag is nested inside common non-content sections."""
    noise_tags = {"nav", "footer", "aside", "header"}
    for parent in tag.parents:
        parent_name = (getattr(parent, "name", "") or "").lower()
        if parent_name in noise_tags:
            return True
    return False


def extract_images(soup: BeautifulSoup, source_url: str) -> list[dict[str, Any]]:
    """Extract image records from all <img> tags."""
    logger.debug("[extract_images] Looking for <img> tags...")
    domain = get_domain(source_url)
    images: list[dict[str, Any]] = []

    for img_tag in soup.find_all("img"):
        src = (img_tag.get("src") or "").strip()

        # Skip invalid/placeholder entries.
        if not src or len(src) < 10:
            continue

        absolute_url = urljoin(source_url, src)
        item = {
            "url": absolute_url,
            "source_url": source_url,
            "media_type": "image",
            "title": (img_tag.get("alt") or None),
            "description": _get_surrounding_paragraph_text(img_tag),
            "domain": domain,
            "file_extension": _extract_extension_from_url(absolute_url),
        }
        images.append(item)

    logger.debug("[extract_images] Extracted %d images.", len(images))
    return images


def extract_links(soup: BeautifulSoup, source_url: str) -> list[dict[str, Any]]:
    """Extract link records from all <a> tags with valid href values."""
    logger.debug("[extract_links] Looking for <a> tags...")
    domain = get_domain(source_url)
    links: list[dict[str, Any]] = []

    for link_tag in soup.find_all("a"):
        if _is_in_noise_section(link_tag):
            continue

        href = (link_tag.get("href") or "").strip()

        if not href:
            continue
        if href.startswith(("mailto:", "javascript:", "#")):
            continue

        absolute_url = urljoin(source_url, href)
        title = link_tag.get_text(" ", strip=True) or None

        if len((title or "").split()) < 3:
            continue

        item = {
            "url": absolute_url,
            "source_url": source_url,
            "media_type": "link",
            "title": title,
            "description": _get_surrounding_paragraph_text(link_tag),
            "domain": domain,
            "file_extension": _extract_extension_from_url(absolute_url),
        }
        links.append(item)

    logger.debug("[extract_links] Extracted %d links.", len(links))
    return links


def extract_articles(soup: BeautifulSoup, source_url: str) -> list[dict[str, Any]]:
    """Extract article-like content.

    Priority:
    - use <article> tags when available
    - otherwise fallback to <p> tags
    """
    logger.debug("[extract_articles] Looking for <article> tags...")
    domain = get_domain(source_url)
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = (meta_tag.get("content") or "").strip() if meta_tag else None
    if meta_desc == "":
        meta_desc = None
    articles: list[dict[str, Any]] = []

    article_blocks = soup.find_all("article")
    use_fallback_paragraphs = len(article_blocks) == 0

    if use_fallback_paragraphs:
        logger.debug("[extract_articles] No <article> found. Falling back to <p> tags.")
        article_blocks = soup.find_all("p")

    for block in article_blocks:
        title_tag = block.find("h1") or block.find("h2")
        if title_tag:
            title = title_tag.get_text(" ", strip=True) or None
        else:
            # For <p> fallback, use short snippet as title when no heading exists.
            fallback_text = block.get_text(" ", strip=True)
            title = (fallback_text[:80] + "...") if len(fallback_text) > 80 else (fallback_text or None)

        full_text = block.get_text(" ", strip=True)
        if not full_text:
            continue

        item = {
            "url": source_url,
            "source_url": source_url,
            "media_type": "article",
            "title": title,
            "description": meta_desc if meta_desc else full_text[:200],
            "domain": domain,
            "file_extension": ".html",
        }
        articles.append(item)

    logger.debug("[extract_articles] Extracted %d article blocks.", len(articles))
    return articles


def _deduplicate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate extracted records by URL while preserving order."""
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for item in items:
        item_url = item.get("url")
        if not item_url or item_url in seen:
            continue
        seen.add(item_url)
        deduped.append(item)

    return deduped


def scrape(url: str) -> dict[str, Any]:
    """Main entry point that runs the full scraping pipeline.

    Returns a summary dictionary with session metadata and extracted items.
    """
    init_db()  # Ensure tables exist.

    normalized_url = normalize_url(url)
    if not normalized_url:
        return {
            "error": "Empty URL provided.",
            "session_id": None,
            "total": 0,
            "images": 0,
            "links": 0,
            "articles": 0,
            "items": [],
        }

    db = SessionLocal()
    session_row = ScrapeSession(target_url=normalized_url, status="running")

    try:
        logger.info("[scrape] Starting new scrape session for: %s", normalized_url)
        db.add(session_row)
        db.commit()
        db.refresh(session_row)

        soup = fetch_page(normalized_url)
        if soup is None:
            session_row.status = "failed"
            session_row.finished_at = datetime.now(UTC)
            session_row.total_items = 0
            db.commit()

            return {
                "error": f"Failed to fetch page: {normalized_url}",
                "session_id": session_row.id,
                "total": 0,
                "images": 0,
                "links": 0,
                "articles": 0,
                "items": [],
            }

        images = extract_images(soup, normalized_url)
        links = extract_links(soup, normalized_url)
        articles = extract_articles(soup, normalized_url)

        combined = images + links + articles
        unique_items = _deduplicate_items(combined)

        session_row.status = "done"
        session_row.finished_at = datetime.now(UTC)
        session_row.total_items = len(unique_items)
        db.commit()

        logger.info(
            "[scrape] Completed successfully. images=%d, links=%d, articles=%d, total_unique=%d",
            len(images),
            len(links),
            len(articles),
            len(unique_items),
        )

        return {
            "session_id": session_row.id,
            "total": len(unique_items),
            "images": len(images),
            "links": len(links),
            "articles": len(articles),
            "items": unique_items,
        }

    except Exception as exc:  # Broad catch to safely update session status.
        logger.exception("[scrape] Unexpected failure: %s", exc)
        try:
            session_row.status = "failed"
            session_row.finished_at = datetime.now(UTC)
            db.commit()
        except Exception as db_exc:
            logger.exception("[scrape] Failed to mark session as failed: %s", db_exc)
            db.rollback()

        return {
            "error": str(exc),
            "session_id": getattr(session_row, "id", None),
            "total": 0,
            "images": 0,
            "links": 0,
            "articles": 0,
            "items": [],
        }

    finally:
        db.close()
