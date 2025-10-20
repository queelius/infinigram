#!/usr/bin/env python3
"""
Tests for transformation scoring system.
"""

import pytest
import math
from infinigram.scoring import (
    TransformationScorer,
    AdaptiveScorer,
    create_default_scorer,
    create_conservative_scorer,
    create_aggressive_scorer
)


class TestTransformationScorer:
    """Test TransformationScorer class."""

    def test_score_ranges(self):
        """Test that scores are in [0, 1] range."""
        scorer = TransformationScorer()

        context = b"the quick brown fox"
        transformed = b"the fast brown fox"
        transformations = ["synonym:quick->fast"]
        match_length = 15
        match_positions = [10, 50, 100]
        corpus_size = 1000

        score = scorer.score(
            context=context,
            transformed_context=transformed,
            transformations=transformations,
            match_length=match_length,
            match_positions=match_positions,
            corpus_size=corpus_size
        )

        assert 0.0 <= score <= 1.0

    def test_longer_match_higher_score(self):
        """Longer matches should score higher."""
        scorer = TransformationScorer()

        context = b"the quick brown fox"
        transformed = b"the quick brown fox"
        transformations = []
        corpus_size = 1000
        positions = [10, 20]

        # Short match
        score_short = scorer.score(
            context, transformed, transformations,
            match_length=5, match_positions=positions, corpus_size=corpus_size
        )

        # Long match
        score_long = scorer.score(
            context, transformed, transformations,
            match_length=15, match_positions=positions, corpus_size=corpus_size
        )

        assert score_long > score_short

    def test_more_frequent_higher_score(self):
        """More frequent patterns should score higher."""
        scorer = TransformationScorer()

        context = b"the quick brown fox"
        transformed = b"the quick brown fox"
        transformations = []
        match_length = 10
        corpus_size = 1000

        # Rare pattern
        score_rare = scorer.score(
            context, transformed, transformations,
            match_length, match_positions=[10], corpus_size=corpus_size
        )

        # Common pattern
        score_common = scorer.score(
            context, transformed, transformations,
            match_length, match_positions=[10, 20, 30, 40, 50], corpus_size=corpus_size
        )

        assert score_common > score_rare

    def test_fewer_transformations_higher_score(self):
        """Fewer transformations should score higher."""
        scorer = TransformationScorer()

        context = b"the quick brown fox"
        transformed = b"the fast brown fox"
        match_length = 10
        positions = [10, 20]
        corpus_size = 1000

        # No transformations (original)
        score_original = scorer.score(
            context, transformed, transformations=[],
            match_length=match_length, match_positions=positions, corpus_size=corpus_size
        )

        # One transformation
        score_one = scorer.score(
            context, transformed, transformations=["synonym:quick->fast"],
            match_length=match_length, match_positions=positions, corpus_size=corpus_size
        )

        # Multiple transformations
        score_multi = scorer.score(
            context, transformed,
            transformations=["synonym:quick->fast", "typo:fox->foks", "case:The->the"],
            match_length=match_length, match_positions=positions, corpus_size=corpus_size
        )

        assert score_original > score_one > score_multi

    def test_transformation_quality_matters(self):
        """Different transformers should have different reliability."""
        scorer = TransformationScorer()

        context = b"the quick brown fox"
        transformed = b"the fast brown fox"
        match_length = 10
        positions = [10, 20]
        corpus_size = 1000

        # Case normalization (very reliable: 0.99)
        score_case = scorer.score(
            context, transformed, transformations=["case:The->the"],
            match_length=match_length, match_positions=positions, corpus_size=corpus_size
        )

        # Synonym (less reliable: 0.85)
        score_synonym = scorer.score(
            context, transformed, transformations=["synonym:quick->fast"],
            match_length=match_length, match_positions=positions, corpus_size=corpus_size
        )

        assert score_case > score_synonym

    def test_empty_matches(self):
        """Empty matches should return low scores."""
        scorer = TransformationScorer()

        context = b"the quick brown fox"
        transformed = b"completely different"
        transformations = ["synonym:quick->fast"]

        score = scorer.score(
            context, transformed, transformations,
            match_length=0, match_positions=[], corpus_size=1000
        )

        # Should be lower than a good match, but transformation/depth still contribute
        # With 0 match and 0 frequency, score ≈ 0.3*0.85 + 0.1*0.717 ≈ 0.33
        assert score < 0.4

    def test_zero_context_length(self):
        """Zero context length should be handled gracefully."""
        scorer = TransformationScorer()

        # This shouldn't crash
        score = scorer._score_match_length(match_length=0, context_length=0)
        assert score == 0.0

    def test_score_batch(self):
        """Test batch scoring."""
        scorer = TransformationScorer()

        context = b"the quick brown fox"
        transformed_contexts = [
            (b"the fast brown fox", ["synonym:quick->fast"], 15, [10, 20]),
            (b"the quick brown fox", [], 19, [10, 20, 30]),
            (b"teh quick brown fox", ["typo:the->teh"], 18, [10]),
        ]

        scores = scorer.score_batch(
            context=context,
            transformed_contexts=transformed_contexts,
            corpus_size=1000
        )

        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_custom_weights(self):
        """Test custom weight configuration."""
        # Heavy match length weight
        scorer_length = TransformationScorer(
            match_length_weight=0.7,
            match_frequency_weight=0.1,
            transformation_weight=0.1,
            depth_weight=0.1
        )

        # Verify weights are normalized
        total = (scorer_length.match_length_weight +
                scorer_length.match_frequency_weight +
                scorer_length.transformation_weight +
                scorer_length.depth_weight)
        assert abs(total - 1.0) < 1e-6

    def test_custom_transformer_weights(self):
        """Test custom transformer reliability weights."""
        custom_weights = {
            'synonym': 0.5,  # Very unreliable
            'typo': 0.99,    # Very reliable
        }

        scorer = TransformationScorer(transformer_weights=custom_weights)

        context = b"test"
        transformed = b"test"

        # Synonym should score lower
        score_synonym = scorer.score(
            context, transformed, ["synonym:a->b"],
            match_length=10, match_positions=[1, 2], corpus_size=100
        )

        # Typo should score higher
        score_typo = scorer.score(
            context, transformed, ["typo:a->b"],
            match_length=10, match_positions=[1, 2], corpus_size=100
        )

        assert score_typo > score_synonym


class TestMatchLengthScoring:
    """Test match length scoring component."""

    def test_perfect_match(self):
        """Perfect match (full context) should score high."""
        scorer = TransformationScorer()
        score = scorer._score_match_length(match_length=100, context_length=100)
        assert score == 1.0

    def test_no_match(self):
        """No match should score 0."""
        scorer = TransformationScorer()
        score = scorer._score_match_length(match_length=0, context_length=100)
        assert score == 0.0

    def test_partial_match(self):
        """Partial match should be between 0 and 1."""
        scorer = TransformationScorer()
        score = scorer._score_match_length(match_length=50, context_length=100)
        assert 0.0 < score < 1.0

    def test_sqrt_scaling(self):
        """Should use sqrt for diminishing returns."""
        scorer = TransformationScorer()

        # 50% match should give sqrt(0.5) ≈ 0.707
        score = scorer._score_match_length(match_length=50, context_length=100)
        expected = math.sqrt(0.5)
        assert abs(score - expected) < 1e-6


class TestMatchFrequencyScoring:
    """Test match frequency scoring component."""

    def test_no_matches(self):
        """No matches should score 0."""
        scorer = TransformationScorer()
        score = scorer._score_match_frequency(match_positions=[], corpus_size=1000)
        assert score == 0.0

    def test_one_match(self):
        """One match should score low but not zero."""
        scorer = TransformationScorer()
        score = scorer._score_match_frequency(match_positions=[10], corpus_size=1000)
        # log(1+1)/log(101) = log(2)/log(101) ≈ 0.15
        assert 0.0 < score < 0.3

    def test_many_matches(self):
        """Many matches should score higher."""
        scorer = TransformationScorer()
        score = scorer._score_match_frequency(
            match_positions=list(range(100)),
            corpus_size=10000
        )
        # log(101)/log(101) = 1.0
        assert score == 1.0

    def test_logarithmic_scaling(self):
        """Should use logarithmic scaling."""
        scorer = TransformationScorer()

        # 10 matches
        score_10 = scorer._score_match_frequency([0]*10, corpus_size=1000)

        # 20 matches (double)
        score_20 = scorer._score_match_frequency([0]*20, corpus_size=1000)

        # Due to logarithmic scaling, doubling matches shouldn't double score
        assert score_20 < score_10 * 2


class TestTransformationQualityScoring:
    """Test transformation quality scoring component."""

    def test_no_transformations(self):
        """No transformations should score perfect (1.0)."""
        scorer = TransformationScorer()
        score = scorer._score_transformations(transformations=[])
        assert score == 1.0

    def test_single_case_transformation(self):
        """Case transformation (0.99 reliability) should score high."""
        scorer = TransformationScorer()
        score = scorer._score_transformations(["case:The->the"])
        assert abs(score - 0.99) < 1e-6

    def test_single_synonym_transformation(self):
        """Synonym transformation (0.85 reliability) should score lower."""
        scorer = TransformationScorer()
        score = scorer._score_transformations(["synonym:big->large"])
        assert abs(score - 0.85) < 1e-6

    def test_multiple_transformations_multiply(self):
        """Multiple transformations should multiply reliabilities."""
        scorer = TransformationScorer()

        # Two transformations: 0.99 * 0.95 = 0.9405
        score = scorer._score_transformations(["case:A->a", "typo:the->teh"])
        expected = 0.99 * 0.95
        assert abs(score - expected) < 1e-6

    def test_unknown_transformer(self):
        """Unknown transformer should use default (0.7)."""
        scorer = TransformationScorer()
        score = scorer._score_transformations(["unknown:foo->bar"])
        assert abs(score - 0.7) < 1e-6


class TestDepthScoring:
    """Test transformation depth scoring component."""

    def test_zero_depth(self):
        """No transformations should score perfect (1.0)."""
        scorer = TransformationScorer()
        score = scorer._score_depth(num_transformations=0)
        assert score == 1.0

    def test_exponential_decay(self):
        """Should use exponential decay."""
        scorer = TransformationScorer()

        # After 3 transformations: e^(-3/3) = e^(-1) ≈ 0.368
        score_3 = scorer._score_depth(3)
        expected_3 = math.exp(-1)
        assert abs(score_3 - expected_3) < 1e-3

        # After 6 transformations: e^(-6/3) = e^(-2) ≈ 0.135
        score_6 = scorer._score_depth(6)
        expected_6 = math.exp(-2)
        assert abs(score_6 - expected_6) < 1e-3

    def test_increasing_depth_decreases_score(self):
        """More transformations should decrease score."""
        scorer = TransformationScorer()

        scores = [scorer._score_depth(d) for d in range(5)]

        # Each subsequent score should be lower
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


class TestAdaptiveScorer:
    """Test AdaptiveScorer class."""

    def test_record_performance(self):
        """Test recording performance."""
        scorer = AdaptiveScorer()

        scorer.record_performance(score=0.8, correct=True)
        scorer.record_performance(score=0.5, correct=False)
        scorer.record_performance(score=0.9, correct=True)

        assert len(scorer.performance_history) == 3

    def test_analyze_performance_empty(self):
        """Empty history should return empty dict."""
        scorer = AdaptiveScorer()
        analysis = scorer.analyze_performance()
        assert analysis == {}

    def test_analyze_performance_bins(self):
        """Test performance analysis with binning."""
        scorer = AdaptiveScorer()

        # Add some data
        # High scores (0.8-1.0): 80% accuracy
        for _ in range(4):
            scorer.record_performance(0.9, correct=True)
        scorer.record_performance(0.9, correct=False)

        # Low scores (0.0-0.2): 50% accuracy
        scorer.record_performance(0.1, correct=True)
        scorer.record_performance(0.1, correct=False)

        analysis = scorer.analyze_performance()

        # Check high score bin
        if '0.8-1.0' in analysis:
            assert analysis['0.8-1.0']['accuracy'] == 0.8
            assert analysis['0.8-1.0']['count'] == 5

        # Check low score bin
        if '0.0-0.2' in analysis:
            assert analysis['0.0-0.2']['accuracy'] == 0.5
            assert analysis['0.0-0.2']['count'] == 2


class TestScorerFactories:
    """Test scorer factory functions."""

    def test_create_default_scorer(self):
        """Test default scorer creation."""
        scorer = create_default_scorer()
        assert isinstance(scorer, TransformationScorer)

        # Check default weights sum to 1.0
        total = (scorer.match_length_weight +
                scorer.match_frequency_weight +
                scorer.transformation_weight +
                scorer.depth_weight)
        assert abs(total - 1.0) < 1e-6

    def test_create_conservative_scorer(self):
        """Conservative scorer should penalize transformations heavily."""
        scorer = create_conservative_scorer()

        # Should have high depth penalty
        assert scorer.depth_weight > 0.15

    def test_create_aggressive_scorer(self):
        """Aggressive scorer should be more willing to try transformations."""
        scorer = create_aggressive_scorer()

        # Should have low depth penalty
        assert scorer.depth_weight < 0.15

    def test_conservative_vs_aggressive(self):
        """Test behavior difference between conservative and aggressive scorers.

        Conservative favors long matches (high match_length weight).
        Aggressive favors transformation quality (high transformation weight).
        """
        conservative = create_conservative_scorer()
        aggressive = create_aggressive_scorer()

        context = b"the quick brown fox"
        transformed = b"the fast brown fox"

        # Scenario: Poor match but good transformations (typo corrections)
        # Conservative should penalize poor match heavily
        # Aggressive should be more forgiving due to good transformation quality
        transformations = ["typo:quick->qwick", "typo:brown->brwn"]

        score_conservative = conservative.score(
            context, transformed, transformations,
            match_length=5,  # Poor match
            match_positions=[5],  # Rare pattern
            corpus_size=1000
        )

        score_aggressive = aggressive.score(
            context, transformed, transformations,
            match_length=5,  # Poor match
            match_positions=[5],  # Rare pattern
            corpus_size=1000
        )

        # With poor match, aggressive (which values transformation quality more)
        # should score higher than conservative (which values match length more)
        assert score_aggressive > score_conservative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
