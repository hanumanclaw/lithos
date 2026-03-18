# feat/hybrid-search-rrf — Hybrid Search with RRF Demo

*2026-03-18T06:52:54Z by Showboat 0.6.1*
<!-- showboat-id: 1351a7d8-245c-4708-9b0a-ba11dee67894 -->

This document is the proof-of-work for PR #112 (`feat/hybrid-search-rrf`).

It demonstrates all three search modes exposed by `lithos_search` and the
`author`/`path_prefix` filter fix on semantic mode.

**Changes in this PR:**
- New `lithos_search` tool replaces the removed `lithos_semantic`
- Three search modes: `semantic`, `fulltext`, and the new default `hybrid`
- Hybrid mode fuses results from ChromaDB (vector) and Tantivy (BM25) using
  **Reciprocal Rank Fusion (RRF)** — parameter-free, robust to score-scale mismatches
- `author` and `path_prefix` filters now work in **all** modes (bug was: semantic
  mode silently ignored them before this PR)

## Test corpus

Five documents are seeded — three by `alice`, two by `bob`.

```bash
uv run python3 demo_hybrid.py 2>&1 | grep -v 'Loading\|Materializing\|LOAD REPORT\|UNEXPECTED\|Notes\|Warning\|weight\|Key\|Status\|^-\{20,\}$'
```

```output
   Building lithos-mcp @ file:///Users/hanuman/.openclaw/workspace/agents/lithos-dev/lithos
      Built lithos-mcp @ file:///Users/hanuman/.openclaw/workspace/agents/lithos-dev/lithos
Uninstalled 1 package in 0.83ms
Installed 1 package in 1ms
------------------------+------------+--+-

Seeded 5 documents:
  [author=alice] RRF Reranking Explained
  [author=alice] Vector Search with ChromaDB
  [author=bob  ] Full-Text Tantivy Index
  [author=alice] Hybrid Search Architecture
  [author=bob  ] Python Async Patterns

--- mode="semantic" (ChromaDB vector search) ---
Query: "how does rank fusion work"
  1. 'RRF Reranking Explained'  similarity=0.604
  2. 'Full-Text Tantivy Index'  similarity=0.181
  3. 'Hybrid Search Architecture'  similarity=0.126

--- mode="fulltext" (Tantivy BM25 search) ---
Query: "tantivy rust bm25"
  1. 'Full-Text Tantivy Index'  bm25_score=5.929

--- mode="hybrid" (RRF combining semantic + fulltext) ---
Query: "search ranking algorithm"
  1. 'Full-Text Tantivy Index'  rrf_score=0.0325
  2. 'Hybrid Search Architecture'  rrf_score=0.0323
  3. 'RRF Reranking Explained'  rrf_score=0.0318
  4. 'Vector Search with ChromaDB'  rrf_score=0.0315

--- RRF reranking effect: semantic-only vs hybrid ---
Query: "search ranking"
Semantic-only ranking:
  1. 'Full-Text Tantivy Index'  sim=0.490
  2. 'Hybrid Search Architecture'  sim=0.414
  3. 'RRF Reranking Explained'  sim=0.387
  4. 'Vector Search with ChromaDB'  sim=0.357
  5. 'Python Async Patterns'  sim=0.027
Hybrid/RRF ranking:
  1. 'Hybrid Search Architecture'  rrf=0.0325
  2. 'Full-Text Tantivy Index'  rrf=0.0325
  3. 'Vector Search with ChromaDB'  rrf=0.0315
  4. 'RRF Reranking Explained'  rrf=0.0315

--- mode="semantic" with author filter (fixed silent-filter bug) ---
  No filter:    5 results
  author=alice: 3 results
    - 'Hybrid Search Architecture'
    - 'Vector Search with ChromaDB'
    - 'RRF Reranking Explained'
  author=bob:   2 results
    - 'Full-Text Tantivy Index'
    - 'Python Async Patterns'

All author filter assertions passed.
```

**Key observations:**

1. **Semantic mode** surfaces 'RRF Reranking Explained' for *"how does rank fusion work"*
   even though those exact words never appear in the title — embedding recall at work.

2. **Fulltext mode** finds 'Full-Text Tantivy Index' for the literal query *"tantivy rust bm25"*.

3. **RRF reranking effect**: for *"search ranking"*, semantic-only ranks 'Full-Text Tantivy Index' #1
   (highest cosine similarity). After RRF fusion, 'Hybrid Search Architecture' shares the #1 spot
   because it also ranks highly in the full-text index — RRF rewards docs that appear consistently
   across both ranked lists, not just dominators in one.

4. **Author filter on semantic mode**: returns exactly alice's 3 docs and bob's 2 docs with
   zero cross-contamination — confirming the silent-filter bug is fixed.

## Unit tests

```bash
uv run pytest tests/test_search.py::TestChromaIndexFilters tests/test_search.py::TestHybridSearch -v --tb=short 2>&1 | tail -20
```

```output
platform darwin -- Python 3.12.12, pytest-9.0.2, pluggy-1.6.0 -- /Users/hanuman/.openclaw/workspace/agents/lithos-dev/lithos/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/hanuman/.openclaw/workspace/agents/lithos-dev/lithos
configfile: pyproject.toml
plugins: anyio-4.12.1, asyncio-1.3.0, cov-7.0.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 10 items

tests/test_search.py::TestChromaIndexFilters::test_author_filter_includes_matching PASSED [ 10%]
tests/test_search.py::TestChromaIndexFilters::test_author_filter_excludes_all_when_no_match PASSED [ 20%]
tests/test_search.py::TestChromaIndexFilters::test_path_prefix_filter_includes_matching PASSED [ 30%]
tests/test_search.py::TestChromaIndexFilters::test_author_and_path_prefix_combined PASSED [ 40%]
tests/test_search.py::TestChromaIndexFilters::test_semantic_search_engine_wires_author_filter PASSED [ 50%]
tests/test_search.py::TestChromaIndexFilters::test_semantic_search_engine_wires_path_prefix_filter PASSED [ 60%]
tests/test_search.py::TestHybridSearch::test_rrf_pure_function PASSED    [ 70%]
tests/test_search.py::TestHybridSearch::test_hybrid_mode_returns_results PASSED [ 80%]
tests/test_search.py::TestHybridSearch::test_fulltext_mode_via_engine PASSED [ 90%]
tests/test_search.py::TestHybridSearch::test_hybrid_deduplicates_by_doc_id PASSED [100%]

============================= 10 passed in 16.98s ==============================
```
