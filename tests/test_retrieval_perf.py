"""Performance-behaviour tests for the three retrieval optimisations:

1. get_collection() caches the PersistentClient / embedding model so it is only
   constructed once, not on every query.
2. _retrieve() issues at most one full metadata scan per call (currently issues two
   for list-style questions: one in _master_list_file_ids, one in _filename_match_chunks).
3. Keyword ($contains) lookups run concurrently rather than serially.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch, call

import pytest

import src.indexer as indexer_mod
from src.rag import _retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_collection(metadatas=None):
    """Minimal Chroma collection mock that satisfies all _retrieve call sites."""
    col = MagicMock()
    meta_rows = metadatas or []

    def _get(**kwargs):
        if kwargs.get("where_document") is not None:
            return {"ids": [], "documents": [], "metadatas": []}
        if kwargs.get("where") is not None:
            return {"ids": [], "documents": [], "metadatas": []}
        # full metadata scan
        return {"ids": [], "documents": [], "metadatas": meta_rows}

    col.get.side_effect = _get
    col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    return col


def _metadata_scan_count(collection):
    """Count collection.get calls that are full-table scans (no where/where_document)."""
    count = 0
    for c in collection.get.call_args_list:
        kw = c.kwargs or {}
        if kw.get("where") is None and kw.get("where_document") is None:
            count += 1
    return count


# ---------------------------------------------------------------------------
# 1. Collection caching
# ---------------------------------------------------------------------------

class TestCollectionCaching:
    def setup_method(self):
        indexer_mod._collection_cache = None

    def teardown_method(self):
        indexer_mod._collection_cache = None

    def test_persistent_client_constructed_once(self):
        """Calling get_collection() twice must not construct PersistentClient twice."""
        with patch("chromadb.PersistentClient") as mock_cls, \
             patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_or_create_collection.return_value = MagicMock()

            indexer_mod.get_collection()
            indexer_mod.get_collection()

            assert mock_cls.call_count == 1, (
                f"PersistentClient was constructed {mock_cls.call_count} times; expected 1"
            )

    def test_returns_same_object_on_repeated_calls(self):
        """get_collection() must return the identical object on the second call."""
        with patch("chromadb.PersistentClient") as mock_cls, \
             patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            singleton = MagicMock()
            mock_client.get_or_create_collection.return_value = singleton

            first = indexer_mod.get_collection()
            second = indexer_mod.get_collection()

            assert first is second

    def test_reset_forces_rebuild(self):
        """get_collection(reset=True) must rebuild even when a cached instance exists."""
        with patch("chromadb.PersistentClient") as mock_cls, \
             patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.get_or_create_collection.return_value = MagicMock()

            indexer_mod.get_collection()
            indexer_mod.get_collection(reset=True)

            assert mock_cls.call_count == 2


# ---------------------------------------------------------------------------
# 2. Single metadata scan
# ---------------------------------------------------------------------------

class TestSingleMetadataScan:
    def test_regular_question_at_most_one_scan(self):
        """A regular question must not cause more than one full metadata scan."""
        col = _make_collection()
        with patch("src.rag.get_collection", return_value=col):
            asyncio.run(_retrieve("what is the VPN policy?"))
        assert _metadata_scan_count(col) <= 1

    def test_list_question_at_most_one_scan(self):
        """A list-style question must not cause more than one full metadata scan."""
        col = _make_collection()
        with patch("src.rag.get_collection", return_value=col):
            asyncio.run(_retrieve("what software is approved?"))
        assert _metadata_scan_count(col) <= 1, (
            f"Expected ≤1 full metadata scan, got {_metadata_scan_count(col)}"
        )


# ---------------------------------------------------------------------------
# 3. Concurrent keyword lookups
# ---------------------------------------------------------------------------

class TestConcurrentKeywordLookups:
    def test_keyword_lookups_run_in_parallel(self):
        """Keyword lookups for multiple entities must run concurrently, not serially.

        Each $contains call is given a 50 ms artificial delay.  Sequential
        execution of 4 entities × 3 case variants = 12 calls × 50 ms = 600 ms.
        Concurrent execution should complete in roughly one delay period (~50 ms).
        We pass if elapsed < 50 % of the sequential estimate.
        """
        delay = 0.05  # 50 ms per lookup

        def slow_get(**kwargs):
            if kwargs.get("where_document") is not None:
                time.sleep(delay)
            return {"ids": [], "documents": [], "metadatas": []}

        col = MagicMock()
        col.get.side_effect = slow_get
        col.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}

        # Four distinct entity tokens → up to 12 keyword lookups
        question = "Microsoft Teams Slack Zoom Webex"

        with patch("src.rag.get_collection", return_value=col):
            start = time.monotonic()
            asyncio.run(_retrieve(question))
            elapsed = time.monotonic() - start

        entities = 4
        variants = 3
        sequential_estimate = entities * variants * delay
        assert elapsed < sequential_estimate * 0.5, (
            f"Keyword lookups appear sequential: took {elapsed:.3f}s "
            f"(sequential estimate {sequential_estimate:.3f}s, threshold "
            f"{sequential_estimate * 0.5:.3f}s)"
        )
