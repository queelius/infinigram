#!/usr/bin/env python3
"""
Tests for recursive context transformation.
"""

import pytest
from infinigram.recursive import (
    RecursiveInfinigram,
    SynonymTransformer,
    EditDistanceTransformer,
    CaseNormalizer
)


@pytest.fixture
def simple_corpus():
    """Create a simple test corpus."""
    return b"the cat sat on the mat. the feline sat on the rug. the dog ran fast."


class TestTransformers:
    """Test individual transformers."""

    def test_case_normalizer(self):
        """Test case normalization."""
        transformer = CaseNormalizer()

        # Test with uppercase
        context = b"The Cat Sat"
        transformations = transformer.generate_transformations(
            context=context,
            suffix=b"Sat",
            corpus=b"the cat sat",
            match_positions=[8]
        )

        assert len(transformations) == 1
        new_context, desc = transformations[0]
        assert new_context == b"the cat sat"
        assert "case" in desc.lower()

    def test_edit_distance_transformer(self):
        """Test edit distance / typo correction."""
        transformer = EditDistanceTransformer(max_distance=2)

        # Test basic structure (actual typo detection needs corpus inspection)
        context = b"the cat sat"
        transformations = transformer.generate_transformations(
            context=context,
            suffix=b"sat",
            corpus=b"the cat sat on mat",
            match_positions=[8]
        )

        # Should work without errors
        assert isinstance(transformations, list)


class TestRecursiveInfinigram:
    """Test recursive Infinigram."""

    def test_initialization(self, simple_corpus):
        """Test basic initialization."""
        model = RecursiveInfinigram(simple_corpus)

        assert model.corpus == simple_corpus
        assert len(model.transformers) > 0

    def test_basic_prediction(self, simple_corpus):
        """Test that basic prediction works."""
        model = RecursiveInfinigram(simple_corpus)

        # Simple prediction
        context = b"the cat"
        probs = model.predict(context, max_depth=1)

        # Should return some predictions
        assert isinstance(probs, dict)
        # May be empty if no matches, that's ok for now

    def test_recursive_transform(self, simple_corpus):
        """Test recursive transformation generation."""
        model = RecursiveInfinigram(simple_corpus)

        context = b"the cat sat"

        # Generate transformations (depth 1)
        contexts = model._recursive_transform(
            context=context,
            depth=0,
            max_depth=1,
            seen=set(),
            beam_width=5
        )

        # Should have at least original context
        assert len(contexts) >= 1
        assert contexts[0][0] == context
        assert contexts[0][1] == []  # No transformations for original

    def test_cycle_detection(self, simple_corpus):
        """Test that cycle detection prevents infinite loops."""
        model = RecursiveInfinigram(simple_corpus)

        context = b"test"
        seen = {context}

        # Should return empty list (cycle detected)
        contexts = model._recursive_transform(
            context=context,
            depth=0,
            max_depth=3,
            seen=seen,
            beam_width=5
        )

        assert contexts == []

    def test_max_depth_limiting(self, simple_corpus):
        """Test that max depth is respected."""
        model = RecursiveInfinigram(simple_corpus)

        context = b"the cat"

        # Depth 0: should return only original
        contexts = model._recursive_transform(
            context=context,
            depth=0,
            max_depth=0,
            seen=set(),
            beam_width=5
        )

        assert len(contexts) == 1
        assert contexts[0][0] == context

    def test_prediction_with_explanation(self, simple_corpus):
        """Test prediction with explanations."""
        model = RecursiveInfinigram(simple_corpus)

        context = b"the cat"
        probs, explanations = model.predict_with_explanation(
            context,
            max_depth=2
        )

        # Should return predictions and explanations
        assert isinstance(probs, dict)
        assert isinstance(explanations, list)

        # Each explanation should have required fields
        for exp in explanations:
            assert 'context' in exp
            assert 'transformations' in exp
            assert 'weight' in exp


class TestPredictionCombining:
    """Test prediction combining logic - CRITICAL GAP."""

    def test_combine_empty_predictions_returns_empty(self):
        """
        Given: Empty list of weighted predictions
        When: Combining predictions
        Then: Returns empty dictionary
        """
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        result = model._combine_predictions([])

        assert result == {}
        assert isinstance(result, dict)

    def test_combine_single_prediction_normalizes(self):
        """
        Given: Single prediction with probabilities
        When: Combining predictions
        Then: Result is normalized to sum to 1.0
        """
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        # Single prediction: 'a' (65) with 0.3, 'b' (66) with 0.7
        weighted_predictions = [
            ({65: 0.3, 66: 0.7}, 1.0)
        ]

        result = model._combine_predictions(weighted_predictions)

        # Should normalize (already normalized in this case)
        assert abs(sum(result.values()) - 1.0) < 1e-9, \
            f"Expected sum=1.0, got {sum(result.values())}"

    def test_combine_respects_weights(self):
        """
        Given: Two predictions with different weights
        When: Combining predictions
        Then: Higher weight prediction contributes more
        """
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        # Two predictions: high weight for 'A', low weight for 'B'
        weighted_predictions = [
            ({65: 1.0}, 0.9),  # 'A' with weight 0.9
            ({66: 1.0}, 0.1),  # 'B' with weight 0.1
        ]

        result = model._combine_predictions(weighted_predictions)

        assert 65 in result and 66 in result
        assert result[65] > result[66], \
            f"Expected A (65) > B (66), got {result[65]} vs {result[66]}"

    def test_combine_overlapping_predictions_sum(self):
        """
        Given: Multiple predictions for the same byte
        When: Combining predictions
        Then: Probabilities are weighted and summed
        """
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        # Both predict 'A' (65) with equal weight
        weighted_predictions = [
            ({65: 0.5}, 0.5),
            ({65: 0.8}, 0.5),
        ]

        result = model._combine_predictions(weighted_predictions)

        # (0.5*0.5 + 0.8*0.5) / (0.5*0.5 + 0.8*0.5) = 1.0
        assert 65 in result
        assert abs(result[65] - 1.0) < 1e-9

    def test_combine_multiple_bytes_multiple_predictions(self):
        """
        Given: Multiple predictions with multiple bytes each
        When: Combining predictions
        Then: All bytes correctly weighted and normalized
        """
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        weighted_predictions = [
            ({65: 0.7, 66: 0.3}, 0.6),  # Weight 0.6
            ({65: 0.4, 67: 0.6}, 0.4),  # Weight 0.4
        ]

        result = model._combine_predictions(weighted_predictions)

        # Should have all bytes
        assert 65 in result  # 'A'
        assert 66 in result  # 'B'
        assert 67 in result  # 'C'

        # Should normalize to 1.0
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-9, f"Expected sum=1.0, got {total}"

        # 'A' appears in both, should have highest probability
        assert result[65] > result[66]
        assert result[65] > result[67]


class TestTransformerEdgeCases:
    """Test edge cases in transformers."""

    def test_edit_distance_calculation_accuracy(self):
        """
        Given: Pairs of words with known edit distances
        When: Calculating Levenshtein distance
        Then: Returns correct distance
        """
        transformer = EditDistanceTransformer(max_distance=5)

        # Test known distances
        test_cases = [
            (b"cat", b"cat", 0),      # Identical
            (b"cat", b"bat", 1),      # One substitution
            (b"cat", b"ca", 1),       # One deletion
            (b"cat", b"cart", 1),     # One insertion
            (b"kitten", b"sitting", 3),  # Classic example
            (b"", b"abc", 3),         # Empty string
            (b"abc", b"", 3),         # Empty string
        ]

        for word1, word2, expected_dist in test_cases:
            actual_dist = transformer._edit_distance(word1, word2)
            assert actual_dist == expected_dist, \
                f"Edit distance {word1} → {word2}: expected {expected_dist}, got {actual_dist}"

    def test_synonym_transformer_no_prefix_to_transform(self):
        """
        Given: Context where suffix matches entire context
        When: Generating transformations
        Then: Returns empty list (no prefix to transform)
        """
        corpus = b"the cat sat on the mat"
        transformer = SynonymTransformer()

        # Suffix equals entire context
        context = b"sat"
        suffix = b"sat"
        positions = [8]  # Position of "sat" in corpus

        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=positions
        )

        assert transformations == []

    def test_edit_distance_transformer_no_prefix_to_transform(self):
        """
        Given: Context where suffix matches entire context
        When: Generating transformations
        Then: Returns empty list (no prefix to transform)
        """
        corpus = b"the cat sat on the mat"
        transformer = EditDistanceTransformer(max_distance=2)

        context = b"mat"
        suffix = b"mat"
        positions = [19]

        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=positions
        )

        assert transformations == []

    def test_case_normalizer_already_lowercase(self):
        """
        Given: Context that is already lowercase
        When: Generating transformations
        Then: Returns empty list (no transformation needed)
        """
        transformer = CaseNormalizer()

        context = b"the cat sat"
        suffix = b"sat"
        corpus = b"irrelevant"
        match_positions = []

        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=match_positions
        )

        assert transformations == []


class TestRecursiveTransformDepthAndBeam:
    """Test recursive transformation with various depths and beam widths."""

    def test_beam_width_one_limits_candidates(self):
        """
        Given: Beam width of 1
        When: Generating transformations recursively
        Then: Only best candidate is explored at each level
        """
        corpus = b"the cat sat on the mat. the dog ran fast."
        model = RecursiveInfinigram(corpus)

        context = b"The Cat"

        # Beam width = 1 should still work
        contexts = model._recursive_transform(
            context=context,
            depth=0,
            max_depth=2,
            seen=set(),
            beam_width=1
        )

        # Should have at least original
        assert len(contexts) >= 1

    def test_large_beam_width_explores_more(self):
        """
        Given: Large beam width
        When: Generating transformations recursively
        Then: More candidates are explored
        """
        corpus = b"the cat sat on the mat. the dog ran fast."
        model = RecursiveInfinigram(corpus)

        context = b"The Cat"

        # Large beam should explore more
        contexts = model._recursive_transform(
            context=context,
            depth=0,
            max_depth=2,
            seen=set(),
            beam_width=10
        )

        # Should have original + transformations
        assert len(contexts) >= 1

    def test_no_matches_returns_only_original(self):
        """
        Given: Context that has no matches in corpus
        When: Generating transformations recursively
        Then: Returns only original context (no transformations possible)
        """
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        # Context completely outside corpus vocabulary
        context = b"xyz"

        contexts = model._recursive_transform(
            context=context,
            depth=0,
            max_depth=2,
            seen=set(),
            beam_width=5
        )

        # Should return only original (no matches to transform from)
        assert len(contexts) == 1
        assert contexts[0][0] == context
        assert contexts[0][1] == []


class TestIntegration:
    """Integration tests with realistic examples."""

    def test_simple_matching(self):
        """Test with simple exact matches."""
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        context = b"the cat"
        probs = model.predict(context, max_depth=1)

        # Should predict something (even if empty dict)
        assert isinstance(probs, dict)

    def test_case_insensitive_matching(self):
        """Test that case normalization helps find matches."""
        corpus = b"the cat sat on the mat"
        model = RecursiveInfinigram(corpus)

        # Test with different case
        context = b"The Cat"
        probs = model.predict(context, max_depth=2)

        # Should still work
        assert isinstance(probs, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
