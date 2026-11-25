# \# INDXR

# 

# A web scraper and multimedia search engine built with Python and Flask.

# Give it a URL: it scrapes the page's images, links and articles, cleans and indexes them in SQLite, and lets you search the collection by \*\*meaning\*\* (hybrid BM25 + sentence-transformer search) or by \*\*visual similarity\*\* (KNN over image feature vectors). A dashboard shows charts of what has been collected.

# 

# \---

# 

# \## Features

# 

# \- \*\*Scraping\*\* — `requests` + BeautifulSoup extract images, links and article-like content from any page, with browser-like headers, URL normalization, deduplication and boilerplate filtering (nav/footer/sidebar text is skipped).

# \- \*\*Indexing\*\* — pandas cleans and normalizes raw records; each item is stored once (unique URL) and linked to the scrape session that found it.

# \- \*\*Hybrid text search\*\* — three-stage retrieval:

# &#x20; 1. \*\*BM25\*\* (`rank-bm25`) for exact keyword/model-name recall

# &#x20; 2. \*\*Semantic search\*\* with `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings, cosine similarity)

# &#x20; 3. \*\*Reciprocal Rank Fusion\*\* (k = 60) to merge both rankings without score normalization

# &#x20; Results include sub-scores and a match reason.

# \- \*\*Image similarity search\*\* — each image gets a 112-dim visual vector (48-dim HSV histogram + 16-dim edge-density histogram + 48-dim 4×4 spatial HSV grid, weighted and L2-normalized). A scikit-learn `NearestNeighbors` index (cosine distance) returns the closest images with a similarity percentage and label.

# \- \*\*Dashboard\*\* — matplotlib charts (media-type distribution, top domains, scrape timeline, indexed vs. pending, feature-vector samples) regenerated from the database, plus the 10 most recent scrape sessions.

# \- \*\*Maintenance endpoints\*\* — recompute all image vectors or all text embeddings in bulk.

# 

# \---

# 

# \## Tech stack

# 

# | Layer | Tools |

# |---|---|

# | Scraping | requests, BeautifulSoup4, lxml |

# | Data processing | pandas, NumPy, Pillow |

# | Search / ML | rank-bm25, sentence-transformers, scikit-learn |

# | Storage | SQLite via SQLAlchemy 2.0 ORM |

# | Web | Flask 3 + Jinja2 templates |

# | Charts | matplotlib |

# 

# \---

# 

# \## Project structure

# 

# ```

# indexation/

# ├── app.py                 # Flask app: routes, request-scoped DB session, cached search index

# ├── models.py              # SQLAlchemy schema (ScrapeSession, MediaItem) + init\_db()

# ├── scraper.py             # Fetch page, extract images/links/articles, deduplicate

# ├── indexer.py             # Clean data, extract image features, compute text embeddings, insert

# ├── search.py              # Hybrid text search (BM25 + semantic + RRF), image KNN, filters, stats

# ├── visualizer.py          # Dashboard chart generation (PNG → static/charts/)

# ├── templates/

# │   ├── index.html         # Dashboard: scrape form, search, charts, recent sessions

# │   └── results.html       # Text and image search results

# ├── static/charts/         # Generated chart images

# ├── tests/

# │   └── test\_semantic\_upgrade.py   # Embedding + hybrid search regression tests

# ├── demo.ipynb             # TF-IDF vs. hybrid comparison + PCA view of embeddings (French notes)

# ├── mediascrape.db         # SQLite database used by the app (sample data included)

# ├── requirements.txt

# ├── TASKS.md                       # Module-by-module progress tracker

# ├── MEDIASCRAPE\_UPGRADE\_PLAN.md    # Design plan for the semantic search upgrade

# └── SEMANTIC\_UPGRADE\_PROGRESS.md   # Step tracker and validation results for that upgrade

# ```

# 

# \### Data model

# 

# \- \*\*`ScrapeSession`\*\* — one scraping run: `target\_url`, `started\_at`, `finished\_at`, `total\_items`, `status`.

# \- \*\*`MediaItem`\*\* — one scraped element: `url` (unique), `source\_url`, `media\_type` (`image` / `link` / `article` / `video`), `title`, `description`, `domain`, `file\_extension`, `feature\_vector` (JSON, images only), `text\_embedding` (JSON, 384 floats or `NULL` when there is no meaningful text), `is\_indexed`, `scraped\_at`, `session\_id`.

# 

# \---

# 

# \## Getting started

# 

# \### Prerequisites

# 

# \- Python 3.11+ (the code uses `datetime.UTC`)

# \- About 80 MB of disk for the embedding model, downloaded automatically on first use and cached afterwards

# 

# \### Install

# 

# ```bash

# git clone https://github.com/grizzlybongo/indexation.git

# cd indexation

# 

# python -m venv .venv

# source .venv/bin/activate        # Windows: .venv\\Scripts\\activate

# 

# pip install -r requirements.txt

# ```

# 

# \### Run

# 

# ```bash

# python app.py

# ```

# 

# Then open <http://127.0.0.1:5000>. The app runs in Flask debug mode by default.

# 

# Optional: set a secret key for flash messages instead of the dev default.

# 

# ```bash

# export MEDIASCRAPE\_SECRET\_KEY="change-me"

# ```

# 

# Tables are created automatically on the first scrape (`init\_db()` also adds the `text\_embedding` column to older databases).

# 

# \---

# 

# \## Usage

# 

# 1\. \*\*Scrape\*\* — paste a URL into the dashboard form. The page is scraped, the results are indexed, and the charts and search cache are refreshed. A flash message reports how many items were indexed, skipped or failed. Sites that return 403 or time out get a clear error.

# 2\. \*\*Search by text\*\* — type a natural-language query such as `italian supercar` or `V12 mid-engine`. Hybrid search finds relevant items even when the exact words don't appear in them.

# 3\. \*\*Search by image\*\* — pass an image URL. The app extracts its feature vector and returns the most visually similar indexed images. A text query that starts with `http(s)://` also triggers image-similarity mode.

# 

# \### Routes

# 

# | Method | Path | Description |

# |---|---|---|

# | `GET` | `/` | Dashboard: charts, stats, recent scrape sessions |

# | `POST` | `/scrape` | Form field `url`. Scrape → index → regenerate charts |

# | `GET` | `/search?q=...` | Hybrid text search (and image mode if `q` is a URL) |

# | `GET` | `/search?image\_url=...` | Image similarity search (top 10) |

# | `POST` | `/reindex` | Recompute visual feature vectors for all images → JSON `{updated, failed}` |

# | `POST` | `/reembed` | Recompute text embeddings for all items → JSON `{updated, failed, skipped\_null\_text}` |

# 

# Example:

# 

# ```bash

# curl -X POST http://127.0.0.1:5000/reembed

# ```

# 

# \### Using the modules directly

# 

# ```python

# from indexer import scrape\_and\_index

# from search import search\_by\_text, search\_by\_image\_similarity

# 

# scrape\_and\_index("https://en.wikipedia.org/wiki/Supercar")

# search\_by\_text("fast italian hypercar", top\_n=5)

# search\_by\_image\_similarity("https://example.com/car.jpg", top\_n=5)

# ```

# 

# \---

# 

# \## Tests

# 

# ```bash

# pip install pytest

# pytest tests/ -q

# ```

# 

# `tests/test\_semantic\_upgrade.py` covers embedding generation (384-dim output, `None` for empty text) and semantic search regressions (`pagani zonda`, `italian supercar`, track-only and V12 queries).

# 

# \## Demo notebook

# 

# `demo.ipynb` runs the pipeline on the indexed data, compares a TF-IDF baseline with the hybrid search on several queries, and projects the stored text embeddings in 2D with PCA. Its explanations are in French.

# 

# \---

# 

# \## Known limitations and next steps

# 

# \- No pagination yet: the scraper processes one page per request.

# \- Context for images uses a nearest-paragraph heuristic.

# \- Older rows with empty titles rank weaker in semantic search; boosting descriptive fields is planned.

# \- Route-level tests for `/reembed` and cache invalidation are still to do.

# \- `pytest` is not listed in `requirements.txt`, and the `pytest.ini` mentioned in `TASKS.md` is not in the repository.

# \- `indxr.db` is empty and not used by the code; the app reads and writes `mediascrape.db`.

# 

# See `TASKS.md` and `SEMANTIC\_UPGRADE\_PROGRESS.md` for the full tracker.

# 

# \## Responsible scraping

# 

# Respect each site's `robots.txt` and terms of service, and avoid sending many requests in a short time.

