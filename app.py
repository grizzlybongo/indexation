"""Flask application entry point for MédiaScrape.

This app acts as the command center for:
- scraping new content
- indexing scraped items
- searching indexed content
- regenerating dashboard charts
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import requests
from flask import Flask, flash, g, jsonify, redirect, render_template, request, url_for

from indexer import reembed_all, reindex_all_images
from indexer import run_indexer as index_data
from models import MediaItem, ScrapeSession, SessionLocal as db
from scraper import scrape
from search import build_hybrid_index
from search import search_by_image_similarity as image_similarity
from search import search_by_text as text_search
from search import get_stats
from visualizer import generate_all_charts, plot_media_type_distribution, plot_top_domains


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("MEDIASCRAPE_SECRET_KEY", "mediascrape-dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///mediascrape.db"

# Ensure chart output directory exists at startup.
os.makedirs("static/charts", exist_ok=True)

# TODO: Step 8 - Create templates/index.html and templates/results.html to display this data.


def get_db_session():
    """Return a request-scoped SQLAlchemy session."""
    if "db_session" not in g:
        g.db_session = db()
    return g.db_session


def get_search_index() -> dict[str, Any]:
    """Return app-cached hybrid search index, building it on first access."""
    cache_key = "hybrid_search_index"
    if cache_key not in app.extensions:
        app.extensions[cache_key] = build_hybrid_index()
    return app.extensions[cache_key]


def invalidate_search_index() -> None:
    """Invalidate cached hybrid search index after data changes."""
    app.extensions.pop("hybrid_search_index", None)


@app.teardown_appcontext
def close_db_session(exception: Exception | None) -> None:
    """Close database session after each request."""
    session = g.pop("db_session", None)
    if session is not None:
        session.close()


@app.get("/")
def dashboard():
    """Dashboard route.

    Regenerates charts on each load and renders the home page.
    """
    media_types_chart = plot_media_type_distribution()
    top_domains_chart = plot_top_domains()

    charts = {
        "media_types": media_types_chart if media_types_chart and os.path.exists(media_types_chart) else None,
        "top_domains": top_domains_chart if top_domains_chart and os.path.exists(top_domains_chart) else None,
    }

    session = get_db_session()
    sessions = (
        session.query(ScrapeSession)
        .order_by(ScrapeSession.started_at.desc())
        .limit(10)
        .all()
    )

    stats = get_stats()

    return render_template(
        "index.html",
        charts=charts,
        sessions=sessions or [],
        stats=stats or {},
    )


@app.post("/scrape")
def run_scrape():
    """Scrape a user URL, index results, and refresh dashboard charts."""
    target_url = (request.form.get("url") or "").strip()

    if not target_url:
        flash("Please provide a URL to scrape.", "warning")
        return redirect(url_for("dashboard"))

    route_ts = datetime.now(UTC)

    try:
        scrape_results = scrape(target_url)

        if scrape_results.get("error"):
            error_message = str(scrape_results["error"])
            if "403" in error_message or "Access Denied" in error_message:
                flash("Access denied by target site (403). Try another URL.", "danger")
            elif "timeout" in error_message.lower():
                flash("The request timed out while scraping the page.", "danger")
            else:
                flash(f"Scrape failed: {error_message}", "danger")
            return redirect(url_for("dashboard"))

        index_summary = index_data(scrape_results)
        generate_all_charts()
        invalidate_search_index()

        flash(
            (
                f"Scrape completed at {route_ts.isoformat()} | "
                f"Indexed: {index_summary.get('inserted', 0)}, "
                f"Skipped: {index_summary.get('skipped', 0)}, "
                f"Failed: {index_summary.get('failed', 0)}"
            ),
            "success",
        )

    except requests.exceptions.Timeout:
        flash("The scrape request timed out. Please try again.", "danger")
    except requests.exceptions.RequestException:
        flash("Network error while scraping target URL.", "danger")
    except Exception as exc:
        message = str(exc)
        if "403" in message:
            flash("Access denied by target site (403).", "danger")
        else:
            flash(f"Unexpected scrape/index error: {message}", "danger")

    return redirect(url_for("dashboard"))


@app.get("/search")
def search_route():
    """Run text search and optional image similarity search."""
    query = (request.args.get("q") or "").strip()
    image_url = (request.args.get("image_url") or "").strip()

    text_results: list[dict[str, Any]] = []
    image_results: list[dict[str, Any]] = []

    if image_url:
        image_results = image_similarity(image_url, top_n=10)
    elif query:
        text_results = text_search(query, search_index=get_search_index())

        # If query looks like a URL, also try image-similarity mode.
        if query.startswith("http://") or query.startswith("https://"):
            image_results = image_similarity(query)

    return render_template(
        "results.html",
        query=query or image_url,
        text_results=text_results,
        image_results=image_results,
    )


@app.post("/reindex")
def reindex_route():
    """Recompute feature vectors for all image rows and return JSON summary."""
    summary = reindex_all_images()
    invalidate_search_index()
    return jsonify({"updated": int(summary.get("updated", 0)), "failed": int(summary.get("failed", 0))})


@app.post("/reembed")
def reembed_route():
    """Recompute text embeddings for all rows and return JSON summary."""
    summary = reembed_all()
    invalidate_search_index()
    return jsonify(
        {
            "updated": int(summary.get("updated", 0)),
            "failed": int(summary.get("failed", 0)),
            "skipped_null_text": int(summary.get("skipped_null_text", 0)),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
