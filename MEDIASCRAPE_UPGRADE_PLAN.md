# MédiaScrape — Semantic Search Upgrade Plan

> **Goal:** Transform MédiaScrape into a cars and supercars focused semantic
> search engine. Replace TF-IDF keyword matching with sentence-transformer
> embeddings and hybrid BM25 ranking so that queries like "fast italian hypercar",
> "V12 mid-engine", or "carbon fiber supercar interior" return relevant results
> across makes, models, and articles — even when the exact words never appear
> in the scraped content.

---

## Reference Repositories

These existing GitHub projects should be used as reference and inspiration
during each upgrade step:

| Repo | What to borrow |
|---|---|
| [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) | Official semantic search examples, model loading, `util.semantic_search()` usage |
| [liamca/sqlite-hybrid-search](https://github.com/liamca/sqlite-hybrid-search) | Hybrid BM25 + vector search on SQLite — directly matches our stack |
| [nunenuh/hybrid-search](https://github.com/nunenuh/hybrid-search) | Reciprocal Rank Fusion (RRF) implementation combining BM25 and transformer scores |
| [kunci115/semantic-search](https://github.com/kunci115/semantic-search) | FAISS + SQLite backend pattern, plug-and-play with our existing models.py |
| [sentence-transformers semantic search README](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/README.md) | Asymmetric search setup (short query → long document), which is exactly our use case |

---

## Overview of Changes

```
Current stack                    Upgraded stack
─────────────────────────────    ─────────────────────────────────────────
TF-IDF (keyword matching)    →   Hybrid: BM25 + Sentence-Transformers
KNN over HSV histograms      →   KNN over 112-dim visual vectors (keep + fix)
SQLite (no vector column)    →   SQLite + new `text_embedding` JSON column
search.py (monolithic)       →   search.py split into semantic + hybrid layers
No score explanation         →   Results show semantic score + match reason
```

---

## Step 1 — Add `text_embedding` column to the database

**File:** `models.py`

Add a new column to the `MediaItem` table to store the sentence-transformer
embedding as a serialized JSON vector (same pattern as `feature_vector`):

```python
text_embedding = Column(Text, nullable=True)  # JSON-serialized list[float] of length 384
```

This stores the 384-dimensional embedding produced by `all-MiniLM-L6-v2`.
No migration needed — SQLite will add the column on next `init_db()` call
if you use `Base.metadata.create_all(engine, checkfirst=True)`.

---

## Step 2 — Install new dependencies

**File:** `requirements.txt`

Add these three packages:

```
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
```

`sentence-transformers` brings the embedding model.
`rank-bm25` is a lightweight pure-Python BM25 implementation — no Elasticsearch
or external server needed, works directly on our in-memory corpus.

---

## Step 3 — Add semantic embedding to the indexer

**File:** `indexer.py`

After cleaning each item and before inserting to DB, generate a 384-dim text
embedding for `title + description` using `all-MiniLM-L6-v2`:

```python
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(title: str, description: str) -> list[float] | None:
    text = f"{title} {description}".strip()
    if not text:
        return None
    vector = EMBED_MODEL.encode(text, normalize_embeddings=True)
    return vector.tolist()
```

Call `embed_text()` for every row in `index_items()` and store the result
as JSON in `item.text_embedding`. This runs once at index time — search
queries are then encoded on the fly.

**Why `all-MiniLM-L6-v2`:**
- 384 dimensions — small, fast, runs on CPU without GPU
- Trained for asymmetric semantic search (short query → longer document)
- Top performer on MTEB benchmark for its size class
- ~80MB download, cached locally after first use

---

## Step 4 — Replace `search_by_text()` with hybrid search

**File:** `search.py`

This is the core upgrade. Replace the existing TF-IDF function with a
two-stage hybrid retrieval pipeline:

### Stage 1 — BM25 (keyword recall)

Use `rank_bm25.BM25Okapi` to score all indexed items for exact and partial
keyword matches. BM25 is better than TF-IDF because it normalizes for
document length and term saturation. This stage catches exact car names and
model codes like "Pagani", "Zonda R", "LaFerrari", or "Huracán" even if the
semantic model scores them lower.

```python
from rank_bm25 import BM25Okapi

corpus_tokens = [doc.split() for doc in corpus]
bm25 = BM25Okapi(corpus_tokens)
bm25_scores = bm25.get_scores(query.split())
```

### Stage 2 — Semantic embedding (meaning recall)

Encode the query with the same `all-MiniLM-L6-v2` model and compute cosine
similarity against all stored `text_embedding` vectors:

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
query_embedding = model.encode(query, normalize_embeddings=True)
semantic_scores = util.cos_sim(query_embedding, corpus_embeddings).flatten()
```

### Stage 3 — Reciprocal Rank Fusion (RRF)

Merge BM25 ranks and semantic ranks using RRF. This avoids the score
normalization problem (BM25 and cosine similarity are on different scales).
Reference: [nunenuh/hybrid-search](https://github.com/nunenuh/hybrid-search)

```python
def reciprocal_rank_fusion(bm25_ranks: list[int], semantic_ranks: list[int],
                            k: int = 60) -> np.ndarray:
    scores = np.zeros(len(bm25_ranks))
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(semantic_ranks):
        scores[idx] += 1.0 / (k + rank + 1)
    return scores
```

The final ranked list is sorted by RRF score. Items that rank well in both
BM25 *and* semantic search float to the top — this is exactly what we want
for queries like "track hypercar" or "V12 supercar" returning actual car
articles and images rather than newsletter links or nav items.

---

## Step 5 — Cache the embedding index in Flask app context

**File:** `app.py`

Building the embedding index on every search request is slow. Cache it on
the Flask app context and rebuild only after a new scrape session:

```python
from flask import g

def get_search_index():
    if "search_index" not in g:
        g.search_index = build_hybrid_index()
    return g.search_index

@app.route("/scrape", methods=["POST"])
def scrape_route():
    # ... existing scrape + index logic ...
    g.pop("search_index", None)  # invalidate cache after new data
```

Reference: the official sentence-transformers docs recommend this pattern
for Flask/FastAPI — encode corpus once at startup, reuse for all queries.

---

## Step 6 — Add a `reembed_all()` function to the indexer

**File:** `indexer.py`

Same pattern as `reindex_all_images()` — iterate every `MediaItem`, compute
`embed_text(item.title, item.description)`, store in `item.text_embedding`,
commit. Expose as `POST /reembed` in `app.py`.

This must be run once after Step 3 is deployed to populate embeddings for
all items already in the DB.

---

## Step 7 — Improve image similarity display in the UI

**File:** `templates/results.html`

Currently image similarity shows a distance score that is hard to interpret.
Change it to show:

- A thumbnail preview of the result image URL (use an `<img>` tag with
  `onerror="this.style.display='none'"` for broken images)
- The distance converted to a 0–100 similarity percentage:
  `similarity = round((1 - distance) * 100, 1)`
- A label: "Very similar" (>85%), "Similar" (>60%), "Loosely similar" (<60%)

---

## Step 8 — Update `demo.ipynb`

Add a new section to the notebook demonstrating:

1. The difference between TF-IDF and semantic search on the same query
2. A side-by-side table: old TF-IDF results vs new hybrid results for
   queries like "italian hypercar", "track only supercar", "V12 engine",
   "carbon cockpit interior", and "most expensive production car"
3. A visualization of the embedding space using `matplotlib` + PCA to
   reduce 384-dim embeddings to 2D and plot clusters by media type

---

## Step 9 — Update `tests/test_scraper.py`

Add test cases for:

- `embed_text()` returns a list of 384 floats for normal input
- `embed_text()` returns `None` for empty input
- `search_by_text("pagani zonda")` returns results where the top result
  contains "pagani" or "zonda" in its title or description (regression test)
- `search_by_text("italian supercar")` returns at least one result even
  though neither "italian" nor "supercar" may appear verbatim in the corpus
  (this test would have failed with TF-IDF — it proves semantic search works)
- `search_by_text("track only no road registration")` returns results related
  to track-spec hypercars like Zonda R, FXX, or Senna GTR by concept
- `search_by_text("V12 mid-engine hypercar")` returns engine/performance
  related articles even when scraped text uses different terminology

---

## Upgrade Order (Dependency Graph)

```
Step 1 (models.py)
    └── Step 2 (requirements.txt)
            └── Step 3 (indexer.py — embed at index time)
                    └── Step 6 (reembed_all — backfill existing rows)
                            └── Step 4 (search.py — hybrid retrieval)
                                    └── Step 5 (app.py — cache index)
                                            └── Step 7 (results.html — UI)
                                                    └── Step 8 (demo.ipynb)
                                                            └── Step 9 (tests)
```

Do not skip Step 6 — without backfilling embeddings, search will return
empty results for all items scraped before this upgrade.

---

## Expected Results After Upgrade

| Query | Before (TF-IDF) | After (Hybrid Semantic) |
|---|---|---|
| `pagani zonda` | Finds by exact match only | Finds Zonda + visually/contextually related hypercars |
| `italian supercar` | 0 results | Returns Pagani, Ferrari, Lamborghini articles by concept |
| `V12 engine hypercar` | 0 results | Returns articles mentioning AMG V12, naturally aspirated engines |
| `carbon fiber interior cockpit` | 0 results | Returns matching image descriptions and race-car interior articles |
| `most expensive production car` | 0 results | Returns price-focused supercar articles by meaning |
| `track only no road registration` | 0 results | Returns articles about track-spec cars like Zonda R, FXX, Senna GTR |

---

## What Makes This Impressive

After this upgrade, MédiaScrape will:

1. **Understand meaning, not just words** — searching "track only V12 hypercar"
   finds relevant cars and articles across makes and models even if those exact
   words never appear in the scraped content
2. **Combine two retrieval strategies** — BM25 catches exact car names and model
   codes, semantic search catches concepts like "fast", "rare", "track-focused",
   RRF merges them without score normalization hacks
3. **Stay fully local** — no OpenAI API, no external vector database, no server.
   Everything runs on SQLite + CPU with cached model weights (~80MB)
4. **Cross-modal ready** — the same embedding column opens the door to text→image
   search in a future step (CLIP model swap), letting users search car images
   by typing a description like "red supercar low to the ground"
