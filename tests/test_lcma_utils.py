"""Tests for lithos.lcma.utils — merge_and_normalize()."""

from __future__ import annotations

import pytest

from lithos.lcma.utils import Candidate, merge_and_normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate(
    node_id: str,
    score: float,
    scout: str = "vector",
    reasons: list[str] | None = None,
) -> Candidate:
    return Candidate(
        node_id=node_id,
        score=score,
        scouts=[scout],
        reasons=reasons or [f"matched via {scout}"],
    )


def _by_node(results: list[Candidate]) -> dict[str, Candidate]:
    return {c.node_id: c for c in results}


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list() -> None:
    assert merge_and_normalize([]) == []


# ---------------------------------------------------------------------------
# Single scout
# ---------------------------------------------------------------------------


class TestSingleScout:
    def test_single_candidate_scores_one(self) -> None:
        candidates = [_candidate("a", 0.8)]
        result = merge_and_normalize(candidates)
        assert len(result) == 1
        assert result[0].node_id == "a"
        assert result[0].score == pytest.approx(1.0)

    def test_two_candidates_normalised_correctly(self) -> None:
        # scores 0.2 and 0.6 → normalised 0.0 and 1.0
        candidates = [
            _candidate("a", 0.2),
            _candidate("b", 0.6),
        ]
        result = merge_and_normalize(candidates)
        by_node = _by_node(result)
        assert by_node["a"].score == pytest.approx(0.0)
        assert by_node["b"].score == pytest.approx(1.0)

    def test_equal_scores_all_map_to_one(self) -> None:
        candidates = [_candidate("a", 0.5), _candidate("b", 0.5)]
        result = merge_and_normalize(candidates)
        for c in result:
            assert c.score == pytest.approx(1.0)

    def test_three_candidates_min_max_span(self) -> None:
        # 0.0, 0.5, 1.0 → 0.0, 0.5, 1.0 (already spanning [0,1])
        candidates = [
            _candidate("a", 0.0),
            _candidate("b", 0.5),
            _candidate("c", 1.0),
        ]
        result = merge_and_normalize(candidates)
        by_node = _by_node(result)
        assert by_node["a"].score == pytest.approx(0.0)
        assert by_node["b"].score == pytest.approx(0.5)
        assert by_node["c"].score == pytest.approx(1.0)

    def test_scouts_and_reasons_preserved(self) -> None:
        c = _candidate("a", 0.7, scout="lexical", reasons=["bm25 hit"])
        result = merge_and_normalize([c])
        assert result[0].scouts == ["lexical"]
        assert result[0].reasons == ["bm25 hit"]


# ---------------------------------------------------------------------------
# Multiple scouts with different score ranges
# ---------------------------------------------------------------------------


class TestMultipleScouts:
    def test_each_scout_normalised_independently(self) -> None:
        """BM25 scores (0–500) and cosine scores (0–1) should each normalise
        to [0, 1] independently, not against the combined pool."""
        candidates = [
            _candidate("a", 400.0, scout="lexical"),  # BM25
            _candidate("b", 100.0, scout="lexical"),  # BM25
            _candidate("c", 0.9, scout="vector"),     # cosine
            _candidate("d", 0.3, scout="vector"),     # cosine
        ]
        result = merge_and_normalize(candidates)
        by_node = _by_node(result)

        # lexical group: lo=100, hi=400, span=300
        assert by_node["a"].score == pytest.approx(1.0)        # (400-100)/300 = 1.0
        assert by_node["b"].score == pytest.approx(0.0)        # (100-100)/300 = 0.0

        # vector group: lo=0.3, hi=0.9, span=0.6
        assert by_node["c"].score == pytest.approx(1.0)        # (0.9-0.3)/0.6 = 1.0
        assert by_node["d"].score == pytest.approx(0.0)        # (0.3-0.3)/0.6 = 0.0

    def test_scouts_with_single_member_each_score_one(self) -> None:
        candidates = [
            _candidate("a", 42.0, scout="lexical"),
            _candidate("b", 0.7, scout="vector"),
        ]
        result = merge_and_normalize(candidates)
        for c in result:
            assert c.score == pytest.approx(1.0)

    def test_no_cross_scout_score_contamination(self) -> None:
        """A low BM25 score must not drag down the normalised cosine score."""
        candidates = [
            _candidate("a", 0.001, scout="lexical"),  # very low BM25
            _candidate("b", 0.9, scout="vector"),
        ]
        result = merge_and_normalize(candidates)
        by_node = _by_node(result)
        # Each single-member group → both score 1.0
        assert by_node["a"].score == pytest.approx(1.0)
        assert by_node["b"].score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Duplicate node_id across scouts
# ---------------------------------------------------------------------------


class TestDuplicateNodeIdAcrossScouts:
    def test_max_score_wins(self) -> None:
        """Same node from two scouts — merged candidate keeps the higher score."""
        candidates = [
            _candidate("shared", 0.8, scout="vector"),   # normalises to 1.0 (solo)
            _candidate("shared", 0.4, scout="lexical"),  # normalises to 1.0 (solo)
        ]
        result = merge_and_normalize(candidates)
        assert len(result) == 1
        assert result[0].node_id == "shared"
        assert result[0].score == pytest.approx(1.0)

    def test_reasons_unioned(self) -> None:
        candidates = [
            _candidate("n1", 0.7, scout="vector", reasons=["semantic match"]),
            _candidate("n1", 200.0, scout="lexical", reasons=["keyword hit"]),
        ]
        result = merge_and_normalize(candidates)
        assert len(result) == 1
        merged = result[0]
        assert "semantic match" in merged.reasons
        assert "keyword hit" in merged.reasons

    def test_scouts_list_deduped(self) -> None:
        candidates = [
            _candidate("n1", 0.6, scout="vector"),
            _candidate("n1", 0.9, scout="vector"),   # same scout, different score
        ]
        result = merge_and_normalize(candidates)
        assert len(result) == 1
        assert result[0].scouts == ["vector"]

    def test_scouts_list_unioned_across_different_scouts(self) -> None:
        candidates = [
            _candidate("n1", 0.5, scout="vector"),
            _candidate("n1", 50.0, scout="lexical"),
        ]
        result = merge_and_normalize(candidates)
        assert len(result) == 1
        assert set(result[0].scouts) == {"vector", "lexical"}

    def test_three_scouts_same_node(self) -> None:
        candidates = [
            _candidate("alpha", 0.3, scout="vector"),
            _candidate("alpha", 80.0, scout="lexical"),
            _candidate("alpha", 0.5, scout="tags"),
        ]
        result = merge_and_normalize(candidates)
        assert len(result) == 1
        assert result[0].node_id == "alpha"
        # All three are solo in their group → each normalises to 1.0; max = 1.0
        assert result[0].score == pytest.approx(1.0)
        assert set(result[0].scouts) == {"vector", "lexical", "tags"}

    def test_input_not_mutated(self) -> None:
        """merge_and_normalize must not modify the original Candidate objects."""
        original_score = 0.3
        c = _candidate("x", original_score, scout="vector")
        merge_and_normalize([c])
        assert c.score == pytest.approx(original_score)

    def test_mixed_shared_and_unique_nodes(self) -> None:
        candidates = [
            _candidate("unique_v", 0.7, scout="vector"),
            _candidate("shared", 0.9, scout="vector"),
            _candidate("unique_l", 30.0, scout="lexical"),
            _candidate("shared", 10.0, scout="lexical"),
        ]
        result = merge_and_normalize(candidates)
        by_node = _by_node(result)

        assert set(by_node.keys()) == {"unique_v", "shared", "unique_l"}

        # vector group: lo=0.7, hi=0.9 → unique_v=0.0, shared=1.0
        # lexical group: lo=10, hi=30 → unique_l=1.0, shared=0.0
        # After merge, shared keeps max(1.0, 0.0) = 1.0
        assert by_node["shared"].score == pytest.approx(1.0)
        assert set(by_node["shared"].scouts) == {"vector", "lexical"}
