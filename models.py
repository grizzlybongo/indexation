"""Database models for MédiaScrape.

This module defines the SQLite database schema used by the project.
It includes:
- ScrapeSession: one scraping run
- MediaItem: one scraped multimedia element

Other modules use these models as follows:
- indexer.py -> inserts/updates rows
- search.py -> reads/query rows
- app.py -> displays rows
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker


# -----------------------------------------------------------------------------
# Database connection setup
# -----------------------------------------------------------------------------
# Store the SQLite file in the project root (same folder as this file).
DB_PATH = Path(__file__).resolve().parent / "mediascrape.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# `future=True` enables SQLAlchemy 2.x style behavior.
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Reusable SQLAlchemy session factory for CRUD operations in other modules.
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# Base class that all ORM models inherit from.
Base = declarative_base()


class ScrapeSession(Base):
    """Represents one scraping run triggered by the user."""

    __tablename__ = "scrape_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    total_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False)

    # One-to-many relation: one scraping session can create many media rows.
    media_items: Mapped[list["MediaItem"]] = relationship(
        "MediaItem",
        back_populates="scrape_session",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            "ScrapeSession("
            f"id={self.id}, "
            f"target_url='{self.target_url}', "
            f"status='{self.status}', "
            f"total_items={self.total_items}"
            ")"
        )


class MediaItem(Base):
    """Represents one scraped media element (image/article/link/video)."""

    __tablename__ = "media_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), unique=True, nullable=False)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    media_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    file_extension: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Placeholder for future refinement:
    # indexer.py will compute a NumPy-based color histogram and serialize it to JSON.
    feature_vector: Mapped[str | None] = mapped_column(Text, nullable=True)

    scraped_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC), nullable=False)

    # Placeholder for future refinement:
    # indexer.py will toggle this to True after processing feature extraction/indexing.
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationship foreign key (many items -> one scrape session).
    session_id: Mapped[int] = mapped_column(ForeignKey("scrape_sessions.id"), nullable=False)
    scrape_session: Mapped[ScrapeSession] = relationship("ScrapeSession", back_populates="media_items")

    def __repr__(self) -> str:
        return (
            "MediaItem("
            f"id={self.id}, "
            f"media_type='{self.media_type}', "
            f"url='{self.url}', "
            f"domain='{self.domain}', "
            f"is_indexed={self.is_indexed}"
            ")"
        )


def init_db() -> None:
    """Create all tables in SQLite if they do not already exist."""
    Base.metadata.create_all(bind=engine)
