"""Visualization module for MédiaScrape.

This module generates matplotlib PNG charts that can be served by Flask.
It is designed to be safe in web/server contexts:
- graceful placeholders when data is missing
- consistent style and sizes
- figures are always closed after saving
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import matplotlib

# Use a non-interactive backend so chart generation works in server/headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import func

from indexer import deserialize_vector
from models import MediaItem, ScrapeSession, SessionLocal
from search import get_stats


# Global styling rules requested by the project requirements.
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update(
    {
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


# TODO (app.py): chart paths returned by generate_all_charts() map directly to <img src=""> tags in templates
# TODO (search.py): get_stats() dict can replace direct DB queries in plot_media_type_distribution() and plot_top_domains()
# TODO (demo.ipynb): call generate_all_charts() and display inline with IPython.display
# TODO (models.py): if ScrapeSession gets a label/name field, use it as X axis label in scrape_timeline


def _ensure_output_path(save_path: str) -> Path:
    """Ensure output directory exists and return a Path object."""
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _save_and_close(fig: plt.Figure, save_path: str) -> str:
    """Apply layout, save PNG with required DPI, and close the figure."""
    output_path = _ensure_output_path(save_path)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path).replace("\\", "/")


def _save_placeholder(save_path: str, title: str, message: str) -> str:
    """Render a clean placeholder chart when there is not enough data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    return _save_and_close(fig, save_path)


def plot_media_type_distribution(save_path: str = "static/charts/media_types.png") -> str:
    """Generate a pie chart showing distribution of media types."""
    # TODO (app.py): call this after every scrape session to refresh the dashboard
    db = SessionLocal()

    try:
        rows = (
            db.query(MediaItem.media_type, func.count(MediaItem.id).label("count"))
            .group_by(MediaItem.media_type)
            .all()
        )

        if not rows:
            return _save_placeholder(save_path, "Media Type Distribution", "No data yet")

        valid_types = ["image", "link", "article"]
        count_map = {media_type: int(count) for media_type, count in rows if media_type in valid_types}
        labels = [media_type.capitalize() for media_type in valid_types if count_map.get(media_type, 0) > 0]
        values = [count_map[media_type] for media_type in valid_types if count_map.get(media_type, 0) > 0]

        if not values:
            return _save_placeholder(save_path, "Media Type Distribution", "No data yet")

        colors = {
            "Image": "#4C78A8",
            "Link": "#F58518",
            "Article": "#54A24B",
        }
        slice_colors = [colors.get(label, "#888888") for label in labels]

        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, _, _ = ax.pie(
            values,
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=slice_colors,
            pctdistance=0.8,
        )
        ax.set_title("Media Type Distribution")

        legend_labels = [f"{label}: {count}" for label, count in zip(labels, values)]
        ax.legend(wedges, legend_labels, title="Types", loc="center left", bbox_to_anchor=(1.0, 0.5))

        return _save_and_close(fig, save_path)

    finally:
        db.close()


def plot_top_domains(top_n: int = 10, save_path: str = "static/charts/top_domains.png") -> str:
    """Generate a horizontal bar chart of top scraped domains."""
    # TODO (search.py): get_stats() top_domains can feed this directly instead of re-querying
    db = SessionLocal()

    try:
        rows = (
            db.query(MediaItem.domain, func.count(MediaItem.id).label("count"))
            .group_by(MediaItem.domain)
            .order_by(func.count(MediaItem.id).desc())
            .limit(max(int(top_n), 1))
            .all()
        )

        if not rows:
            return _save_placeholder(save_path, "Top Scraped Domains", "No data yet")

        domains = [domain or "(unknown)" for domain, _ in rows]
        counts = [int(count) for _, count in rows]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(domains, counts, color="#4C78A8")
        ax.set_title("Top Scraped Domains")
        ax.set_xlabel("Item Count")
        ax.set_ylabel("Domain")

        # Most scraped at top.
        ax.invert_yaxis()

        # Count labels at end of bars.
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, str(count), va="center")

        return _save_and_close(fig, save_path)

    finally:
        db.close()


def plot_scrape_timeline(save_path: str = "static/charts/scrape_timeline.png") -> str:
    """Generate a timeline chart of total items per scrape session."""
    db = SessionLocal()

    try:
        sessions = db.query(ScrapeSession).order_by(ScrapeSession.started_at.asc()).all()

        if len(sessions) < 2:
            return _save_placeholder(save_path, "Items Scraped Per Session Over Time", "Not enough data yet")

        x_values = [session.started_at for session in sessions]
        y_values = [int(session.total_items or 0) for session in sessions]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_values, y_values, marker="o", linewidth=2, color="#F58518")
        ax.set_title("Items Scraped Per Session Over Time")
        ax.set_xlabel("Session Date/Time")
        ax.set_ylabel("Total Items")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        return _save_and_close(fig, save_path)

    finally:
        db.close()


def plot_indexed_vs_pending(save_path: str = "static/charts/indexed_ratio.png") -> str:
    """Generate a donut chart for indexing progress."""
    stats = get_stats()
    total_items = int(stats.get("total_items", 0))

    if total_items == 0:
        return _save_placeholder(save_path, "Indexing Progress", "No data yet")

    indexed_count = int(round((float(stats.get("indexed_ratio", 0.0)) / 100.0) * total_items))
    pending_count = max(total_items - indexed_count, 0)

    labels = ["Indexed", "Pending"]
    values = [indexed_count, pending_count]
    colors = ["#54A24B", "#E45756"]

    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, _, _ = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.8,
    )

    # Donut hole.
    center_circle = plt.Circle((0, 0), 0.55, fc="white")
    ax.add_artist(center_circle)

    ax.set_title("Indexing Progress")
    ax.text(0, 0, f"Total\n{total_items}", ha="center", va="center", fontsize=11)
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

    return _save_and_close(fig, save_path)


def plot_feature_vector_sample(
    n_samples: int = 5,
    save_path: str = "static/charts/feature_vectors.png",
) -> str:
    """Plot sample color-histogram vectors for indexed images."""
    # TODO (search.py): visually shows why similar histograms = similar images in KNN
    db = SessionLocal()

    try:
        rows = (
            db.query(MediaItem)
            .filter(MediaItem.media_type == "image", MediaItem.feature_vector.is_not(None))
            .order_by(MediaItem.id.asc())
            .limit(max(int(n_samples), 1))
            .all()
        )

        vectors: list[tuple[str, list[float]]] = []
        for item in rows:
            vector = deserialize_vector(item.feature_vector)
            if vector is None:
                continue
            if len(vector) != 24:
                continue
            label = f"ID {item.id}"
            vectors.append((label, vector))

        if not vectors:
            return _save_placeholder(save_path, "Color Histogram Feature Vectors (Sample)", "No image vectors yet")

        x = np.arange(24)
        bin_labels = [f"R{i}" for i in range(8)] + [f"G{i}" for i in range(8)] + [f"B{i}" for i in range(8)]

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, vector in vectors:
            ax.plot(x, vector, marker="o", linewidth=1.8, label=label)

        ax.set_title("Color Histogram Feature Vectors (Sample)")
        ax.set_xlabel("Histogram Bins")
        ax.set_ylabel("Normalized Frequency")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.legend(loc="upper right")

        return _save_and_close(fig, save_path)

    finally:
        db.close()


def generate_all_charts(db_session: Any = None, save_dir: str = "static/charts") -> dict[str, str | None]:
    """Generate all dashboard charts and return their output paths."""
    # TODO (app.py): call generate_all_charts() on startup and after each scrape session
    charts_dir = Path(save_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: dict[str, str | None] = {
        "media_types": str(charts_dir / "media_types.png").replace("\\", "/"),
        "top_domains": str(charts_dir / "top_domains.png").replace("\\", "/"),
        "scrape_timeline": str(charts_dir / "scrape_timeline.png").replace("\\", "/"),
        "indexed_ratio": str(charts_dir / "indexed_ratio.png").replace("\\", "/"),
        "feature_vectors": str(charts_dir / "feature_vectors.png").replace("\\", "/"),
    }

    generators = {
        "media_types": lambda: plot_media_type_distribution(chart_paths["media_types"]),
        "top_domains": lambda: plot_top_domains(10, chart_paths["top_domains"]),
        "scrape_timeline": lambda: plot_scrape_timeline(chart_paths["scrape_timeline"]),
        "indexed_ratio": lambda: plot_indexed_vs_pending(chart_paths["indexed_ratio"]),
        "feature_vectors": lambda: plot_feature_vector_sample(5, chart_paths["feature_vectors"]),
    }

    for key, generate in generators.items():
        try:
            chart_paths[key] = generate()
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "[generate_all_charts] Failed to generate '%s': %s", key, exc
            )
            chart_paths[key] = None

    return chart_paths
