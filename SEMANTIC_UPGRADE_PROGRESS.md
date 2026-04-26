# MediaScrape Semantic Upgrade Progress

## Goal
Migrate text retrieval from TF-IDF to hybrid semantic search (BM25 + sentence-transformer + RRF) while avoiding dummy semantic values.

## Current Status
- Phase: Semantic upgrade implementation complete
- Last updated: 2026-04-24
- Data policy: Keep real rows; store NULL text embeddings when text is weak/empty

## Working Now
- Legacy TF-IDF text search and KNN image similarity are operational
- Image feature reindex endpoint exists: POST /reindex
- text_embedding schema support is active in ORM + SQLite compatibility path
- indexer now computes text embeddings at insert time (no dummy embedding values)
- text embedding backfill endpoint exists: POST /reembed
- Hybrid text retrieval is active: BM25 + semantic + RRF
- App-level cached hybrid index is active and invalidates on scrape/reindex/reembed
- Image similarity results now expose human-readable similarity percentage and labels
- Semantic regression tests exist and pass
- demo.ipynb exists with TF-IDF vs hybrid comparison cells and PCA visualization scaffold
- demo.ipynb executed end-to-end successfully in the project kernel

## Not Working Yet
- No critical blockers identified in current upgrade scope

## Step Tracker

| Step | Scope | Status | Verification | Notes |
|---|---|---|---|---|
| 1 | Add text_embedding column in models.py | Completed | text_embedding_present=True | Includes compatibility for existing SQLite DB |
| 2 | Add sentence-transformers and rank-bm25 deps | Completed | both imports detected in venv | requirements and install done |
| 3 | Add embed_text + store embedding in indexer | Completed | embed_text returns 384-dim vector | empty input returns None |
| 4 | Replace search_by_text with hybrid BM25 + semantic + RRF | Completed | sanity queries returned ranked results | includes match_reason + sub-scores |
| 5 | Cache hybrid index in app context | Completed | app cache helper + invalidation wired | invalidates on scrape/reindex/reembed |
| 6 | Add reembed_all and POST /reembed | Completed | backfill run summary recorded | route + function wired |
| 7 | Improve image similarity UI output | Completed | payload includes similarity + label and template updated | thumbnail + percentage + label implemented |
| 8 | Create/update demo.ipynb semantic section | Completed | notebook executed successfully | TF-IDF vs hybrid table + PCA chart generated |
| 9 | Add semantic tests | Completed | 6 tests passed in 12.65s | embed_text + semantic query regressions covered |

## Phase Checklist

- [x] Phase 0: Tracking aligned
- [x] Phase 1: Schema + dependencies
- [x] Phase 2: Index-time embeddings
- [x] Phase 3: Backfill embeddings
- [x] Phase 4: Hybrid retrieval
- [x] Phase 5: Cache + performance
- [x] Phase 6: UI clarity
- [x] Phase 7: Notebook demo
- [x] Phase 8: Tests and hardening

## Validation Snapshot

- DB schema check: text_embedding_present=True
- Dependency check: rank_bm25_installed=True, sentence_transformers_installed=True
- embed_text smoke check: empty_is_none=True, vector_len=384
- reembed_all run summary: updated=2128, failed=0, skipped_null_text=10
- hybrid query sanity: `pagani zonda`, `italian supercar`, `track only no road registration` all returned results
- semantic tests: `pytest tests/test_semantic_upgrade.py -q` -> 6 passed
- notebook integrity: `demo.ipynb` parsed successfully (nbformat=4, cells=6)
- notebook run: all 5 code cells executed successfully in `.venv-2` kernel

## Next
1. Add optional route-level tests for `/reembed` and cache invalidation behavior.
2. Refine semantic ranking quality for empty-title legacy rows by boosting descriptive fields in scoring.
3. Optionally tune corpus curation toward cars/supercars-only sources for stronger domain precision.
