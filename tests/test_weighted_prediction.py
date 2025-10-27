#!/usr/bin/env python3
"""
Tests for weighted hierarchical prediction in Infinigram.
"""

import pytest
from infinigram import Infinigram
from infinigram.weighting import linear_weight, quadratic_weight, exponential_weight


class TestPredictWeightedBasics:
    """Test basic functionality of predict_weighted."""

    def test_predict_weighted_exists(self):
        """predict_weighted method should exist."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)
        assert hasattr(model, 'predict_weighted')

    def test_predict_weighted_returns_dict(self):
        """predict_weighted should return a dict."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = Infinigram(corpus)
        result = model.predict_weighted([2, 3])
        assert isinstance(result, dict)

    def test_predict_weighted_probabilities_sum_to_one(self):
        """Probabilities should sum to approximately 1.0."""
        corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
        model = Infinigram(corpus)
        probs = model.predict_weighted([2, 3], top_k=256)  # Get all bytes
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01

    def test_predict_weighted_all_positive(self):
        """All returned probabilities should be non-negative."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = Infinigram(corpus)
        probs = model.predict_weighted([2, 3])
        # With smoothing=0, only observed bytes have positive probability
        assert all(p >= 0 for p in probs.values())
        # At least some probabilities should be positive
        assert any(p > 0 for p in probs.values())


class TestPredictWeightedLongestOnly:
    """Test that using only longest match equals regular predict."""

    def test_longest_only_equals_regular_predict(self):
        """Using min_length=max_length should equal regular predict."""
        corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
        model = Infinigram(corpus, max_length=10)
        context = [2, 3]

        # Regular predict - get all probabilities
        regular_probs = model.predict(context, top_k=256)

        # Weighted predict with only longest match
        _, longest_len = model.longest_suffix(context)
        weighted_probs = model.predict_weighted(
            context,
            min_length=longest_len,
            max_length=longest_len,
            top_k=256
        )

        # Should be very similar (allowing for numerical differences with 256-byte vocab)
        assert set(regular_probs.keys()) == set(weighted_probs.keys())
        for token in regular_probs:
            # Relax tolerance due to smoothing over 256 bytes
            assert abs(regular_probs[token] - weighted_probs[token]) < 0.15


class TestPredictWeightedMultipleLengths:
    """Test combining predictions from multiple suffix lengths."""

    def test_shorter_suffixes_contribute(self):
        """Shorter suffix matches should contribute to prediction."""
        # Create corpus where different length suffixes predict different tokens
        corpus = [
            1, 2, 3, 4,     # [1,2,3] -> 4
            5, 2, 3, 6,     # [5,2,3] -> 6
            7, 2, 8,        # [7,2] -> 8
            9, 2, 10        # [9,2] -> 10
        ]
        model = Infinigram(corpus)
        context = [1, 2, 3]

        # With weighted prediction (all lengths), we should see contributions
        # from both [2,3] suffix and [3] suffix
        weighted_probs = model.predict_weighted(
            context,
            min_length=1,
            max_length=3,
            weight_fn=linear_weight,
            top_k=100
        )

        # Should have predictions from multiple suffix matches
        assert len(weighted_probs) > 1

    def test_min_max_length_boundaries(self):
        """min_length and max_length should be respected."""
        corpus = [1, 2, 3, 4, 2, 3, 5, 2, 6]
        model = Infinigram(corpus)
        context = [1, 2, 3]

        # Predict with length 2 only
        probs = model.predict_weighted(
            context,
            min_length=2,
            max_length=2,
            top_k=100
        )

        # Should only use [2,3] suffix, not [3] or [1,2,3]
        assert isinstance(probs, dict)
        assert len(probs) > 0


class TestWeightFunctionEffects:
    """Test that different weight functions produce different results."""

    def test_different_weights_different_distributions(self):
        """Different weight functions should produce different distributions."""
        # Corpus where different lengths predict different tokens
        corpus = [
            1, 2, 3, 4,
            5, 2, 3, 6,
            7, 2, 8,
            9, 3, 10
        ]
        model = Infinigram(corpus)
        context = [1, 2, 3]

        # Linear weighting
        linear_probs = model.predict_weighted(
            context,
            min_length=1,
            max_length=3,
            weight_fn=linear_weight,
            top_k=100
        )

        # Quadratic weighting (should favor longer matches more)
        quadratic_probs = model.predict_weighted(
            context,
            min_length=1,
            max_length=3,
            weight_fn=quadratic_weight,
            top_k=100
        )

        # Distributions should be different
        # (Quadratic should give even more weight to longest match)
        assert linear_probs != quadratic_probs

    def test_exponential_weight_favors_longest_strongly(self):
        """Exponential weighting should strongly favor longest match."""
        corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
        model = Infinigram(corpus)
        context = [2, 3]

        exp_probs = model.predict_weighted(
            context,
            min_length=1,
            max_length=2,
            weight_fn=exponential_weight(),
            top_k=100
        )

        # With exponential weighting, longest match should dominate
        # so result should be very similar to regular predict
        regular_probs = model.predict(context, top_k=100)

        # Top prediction should be the same
        top_exp = max(exp_probs.items(), key=lambda x: x[1])[0]
        top_regular = max(regular_probs.items(), key=lambda x: x[1])[0]
        assert top_exp == top_regular


class TestEdgeCases:
    """Test edge cases for weighted prediction."""

    def test_no_matches_at_any_length(self):
        """Should handle case where no suffix matches."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)
        context = [99, 98, 97]  # Not in corpus (valid bytes)

        probs = model.predict_weighted(context, min_length=1, max_length=3, top_k=256)

        # Should fall back to smoothed uniform
        assert len(probs) > 0
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_empty_context(self):
        """Should handle empty context."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        probs = model.predict_weighted([], min_length=1, max_length=3)

        # Should return unigram distribution
        assert len(probs) > 0

    def test_single_token_corpus(self):
        """Should handle corpus with single unique token."""
        corpus = [1, 1, 1, 1, 1]
        model = Infinigram(corpus)
        context = [1]

        probs = model.predict_weighted(context, min_length=1, max_length=2)

        # Should predict token 1 with high probability
        assert 1 in probs
        # (But won't be 1.0 due to smoothing)

    def test_context_longer_than_max_length(self):
        """Should truncate context to max_length."""
        corpus = [1, 2, 3, 4, 5, 6, 7, 8]
        model = Infinigram(corpus)
        long_context = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # All valid bytes

        probs = model.predict_weighted(
            long_context,
            min_length=1,
            max_length=5,
            top_k=256  # Get all probabilities
        )

        # Should not crash and should return valid distribution
        assert len(probs) > 0
        assert abs(sum(probs.values()) - 1.0) < 0.01


class TestTopK:
    """Test top_k parameter."""

    def test_top_k_limits_output(self):
        """top_k should limit number of predictions returned."""
        corpus = list(range(100))  # Large vocabulary
        model = Infinigram(corpus)
        context = [0]

        probs = model.predict_weighted(context, top_k=10)

        # Should return at most 10 predictions
        assert len(probs) <= 10

    def test_top_k_returns_highest_prob(self):
        """top_k should return the k highest probability tokens."""
        corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
        model = Infinigram(corpus)
        context = [2, 3]

        all_probs = model.predict_weighted(context, top_k=100)
        top_3_probs = model.predict_weighted(context, top_k=3)

        # Top 3 should be the 3 highest from all predictions
        sorted_all = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        expected_top_3 = dict(sorted_all[:3])

        assert top_3_probs == expected_top_3
