# MédiaScrape — Project Task Tracker

## Overall Progress
![Progress](https://progress-bar.dev/100/?title=Overall)
- **Modules done:** 10 / 10
- **Last updated:** 2026-03-30

---

## Task List

| # | File/Module | Status | Description | Depends On | Can Be Refined When |
|---|---|---|---|---|---|
| 1 | `requirements.txt` | ✅ Done | All dependencies listed and versioned | — | Any module is added |
| 2 | `models.py` | ✅ Done | SQLAlchemy ORM schema (MediaItem table) | requirements.txt | indexer.py is done |
| 3 | `scraper.py` | ✅ Done | requests + BeautifulSoup scraping logic | models.py | indexer.py, search.py done |
| 4 | `indexer.py` | ✅ Done | pandas cleaning + NumPy features + DB insert | scraper.py, models.py | search.py, visualizer.py done |
| 5 | `search.py` | ✅ Done | TF-IDF text search + KNN image similarity | indexer.py | app.py is done |
| 6 | `visualizer.py` | ✅ Done | matplotlib charts and stats generation | indexer.py | app.py is done |
| 7 | `app.py` | ✅ Done | Flask routes and app entry point | All modules | templates done |
| 8 | `templates/` | ✅ Done | index.html + results.html Flask templates | app.py | app.py is refined |
| 9 | `tests/test_scraper.py` | ⬜ Pending | pytest unit tests for core modules | scraper.py, indexer.py, search.py | Any module changes |
| 10 | `demo.ipynb` | ✅ Done | Jupyter notebook full pipeline demo | All modules | — |

---

## Status Legend
| Icon | Meaning |
|---|---|
| ⬜ Pending | Not started yet |
| 🔄 In Progress | Currently being built |
| ✅ Done | Completed and confirmed |
| ⚠️ Needs Refinement | Done but depends on a future module to be optimized |
| ❌ Ignored | Skipped intentionally |

---

## Refinement Notes
> This section tracks what should be revisited once a dependency is completed.

- [ ] `scraper.py` — pagination still pending until `app.py` defines max_items per request
- [ ] `scraper.py` — improve context extraction (currently nearest paragraph heuristic) once `search.py` defines relevance strategy
- [ ] `indexer.py` — batch insert optimization pending until `search.py` defines most queried fields
- [resolved] `indexer.py` — upgraded image vectors to 16-bin HSV (48 dims) and added `reindex_all_images()` for stale vector refresh
- [resolved] `models.py` — add extra columns if needed confirmed schema is sufficient for steps 1–4
- [resolved] `models.py` — replaced `datetime.utcnow()` with timezone-aware `datetime.now(UTC)` in runtime modules
- [ ] `requirements.txt` — re-check versions after all modules are finalized

---

## Notes & Decisions
- Database: SQLite (simple, no server needed)
- Search: TF-IDF for text, KNN for image similarity via color histograms
- Frontend: minimal Flask + Jinja2 templates, no JS framework
- Schema choice: each `MediaItem` is linked to one `ScrapeSession` via `session_id` (one-to-many)
- Validation: added dedicated `tests/test_step2_models.py` using in-memory SQLite to verify Step 2 schema behavior end-to-end
- Scraper design: modular extraction helpers return raw dict records, deduplicated by URL in `scrape()`
- Validation: added dedicated `tests/test_step3_scraper.py` with mocked HTTP calls and in-memory DB to verify scraper pipeline end-to-end
- Indexer design: clean data with pandas first, then persist unique rows while attaching optional image histogram vectors
- Validation: added dedicated `tests/test_integration_step1_to_4.py` to verify end-to-end pipeline from scrape to indexed DB persistence
- Visualization: added `visualizer.py` to generate PNG dashboard charts in `static/charts/` with placeholder fallbacks for empty datasets
- Refinement: scraper now uses `requests.Session` + browser-like headers and explicit 403 blocked-site messaging
- Refinement: added `pytest.ini` warning filters to keep test output clean and focused on actionable failures
- Application: added `app.py` Flask command-center routes (`/`, `/scrape`, `/search`) wired to scraper, indexer, search, and visualizer modules
- Templates: added `templates/index.html` and `templates/results.html` so Flask routes render without `TemplateNotFound`
- Refinement: indexer now uses HSV histograms with 16 bins per channel (48-length vectors) and supports full image-vector reindexing
 - Demo: notebook runs full pipeline on Wikipedia URLs, all charts render inline, explanations in French
