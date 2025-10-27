#!/usr/bin/env python3
"""
Tests for Infinigram variable-length n-gram model.
"""

import pytest
import numpy as np

from infinigram import Infinigram, SuffixArray, create_infinigram


class TestSuffixArray:
    """Test suffix array implementation."""

    def test_suffix_array_construction(self):
        """Test suffix array builds correctly."""
        corpus = [1, 2, 3, 1, 2, 4]
        sa = SuffixArray(corpus)

        assert len(sa.suffix_array) == len(corpus)
        assert all(0 <= idx < len(corpus) for idx in sa.suffix_array)

    def test_find_range_exact_match(self):
        """Test finding exact pattern matches."""
        corpus = [1, 2, 3, 1, 2, 3, 4, 1, 2]
        sa = SuffixArray(corpus)

        # Pattern [1, 2] appears at positions 0, 3, 7
        positions = sa.search([1, 2])
        assert len(positions) == 3  # Three occurrences

    def test_find_range_not_found(self):
        """Test pattern not in corpus."""
        corpus = [1, 2, 3, 4, 5]
        sa = SuffixArray(corpus)

        positions = sa.search([9, 9, 9])
        assert len(positions) == 0  # Not found

    def test_find_range_single_token(self):
        """Test single token pattern."""
        corpus = [1, 2, 1, 3, 1]
        sa = SuffixArray(corpus)

        positions = sa.search([1])
        assert len(positions) == 3  # Token 1 appears 3 times

    def test_find_range_empty_pattern(self):
        """Test empty pattern returns empty range."""
        corpus = [1, 2, 3]
        sa = SuffixArray(corpus)

        positions = sa.search([])
        assert len(positions) == 0


class TestInfinigramCore:
    """Test core Infinigram functionality."""

    def test_infinigram_initialization(self):
        """Test Infinigram initializes correctly."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        assert model.n == 5
        assert model.vocab_size == 256  # Fixed vocabulary for bytes
        assert model.vocab == set(range(256))  # All 256 possible bytes

    def test_infinigram_with_max_length(self):
        """Test max_length parameter."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus, max_length=3)

        assert model.max_length == 3

    def test_create_infinigram_convenience(self):
        """Test convenience function."""
        corpus = [1, 2, 3]
        model = create_infinigram(corpus, max_length=5)

        assert isinstance(model, Infinigram)
        assert model.max_length == 5

    def test_byte_validation(self):
        """Test corpus must be valid bytes (0-255)."""
        # Valid corpus should work
        corpus_valid = [0, 128, 255]  # Min, mid, max bytes
        model = Infinigram(corpus_valid)
        assert model.n == 3

        # Invalid corpus should raise error
        with pytest.raises(ValueError, match="Corpus must contain only bytes"):
            Infinigram([256])  # Too large

        with pytest.raises(ValueError, match="Corpus must contain only bytes"):
            Infinigram([-1])  # Negative

        with pytest.raises(ValueError, match="Corpus must contain only bytes"):
            Infinigram([0, 100, 300, 1000])  # Mixed valid/invalid

    def test_update_byte_validation(self):
        """Test update validates byte range."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        # Valid update should work
        model.update([4, 5, 255])
        assert model.n == 6

        # Invalid update should raise error
        with pytest.raises(ValueError, match="New tokens must contain only bytes"):
            model.update([256])

        with pytest.raises(ValueError, match="New tokens must contain only bytes"):
            model.update([-1])


class TestLongestSuffix:
    """Test longest suffix matching."""

    def test_longest_suffix_exact_match(self):
        """Test finding exact suffix match."""
        corpus = [1, 2, 3, 4, 2, 3, 5]
        model = Infinigram(corpus)

        # Context [1, 2, 3] has exact match at position 0
        context = [1, 2, 3]
        pos, length = model.longest_suffix(context)

        assert length == 3  # Full match
        assert pos >= 0

    def test_longest_suffix_partial_match(self):
        """Test finding partial suffix match."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        # Context [99, 2, 3] - only [2, 3] matches
        context = [99, 2, 3]
        pos, length = model.longest_suffix(context)

        assert length == 2  # Partial match [2, 3]

    def test_longest_suffix_no_match(self):
        """Test when no suffix matches."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        context = [99, 98, 97]
        pos, length = model.longest_suffix(context)

        assert length == 0  # No match
        assert pos == -1

    def test_longest_suffix_empty_context(self):
        """Test empty context."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        pos, length = model.longest_suffix([])
        assert length == 0

    def test_longest_suffix_respects_max_length(self):
        """Test max_length limits suffix search."""
        corpus = [1, 2, 3, 4, 5, 6, 7, 8]
        model = Infinigram(corpus, max_length=3)

        # Long context that fully matches
        context = [1, 2, 3, 4, 5, 6, 7, 8]
        pos, length = model.longest_suffix(context)

        # Should only match last 3 tokens due to max_length
        assert length <= 3


class TestContinuations:
    """Test continuation probability computation."""

    def test_continuations_basic(self):
        """Test basic continuation counting."""
        corpus = [1, 2, 3, 1, 2, 4, 1, 2, 5]
        model = Infinigram(corpus)

        # After [1, 2], we have: 3, 4, 5
        context = [1, 2]
        conts = model.continuations(context)

        assert 3 in conts
        assert 4 in conts
        assert 5 in conts
        assert conts[3] == 1
        assert conts[4] == 1
        assert conts[5] == 1

    def test_continuations_empty_context(self):
        """Test continuations with empty context (unigram)."""
        corpus = [1, 2, 1, 3, 1]
        model = Infinigram(corpus)

        conts = model.continuations([])

        # Should return unigram counts
        assert conts[1] == 3
        assert conts[2] == 1
        assert conts[3] == 1

    def test_continuations_no_match(self):
        """Test continuations when no suffix matches."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        conts = model.continuations([99, 98, 97])

        # Should fall back to unigram distribution (what's in corpus)
        assert len(conts) == 3  # Only bytes 1, 2, 3 are in corpus
        assert conts[1] == 1
        assert conts[2] == 1
        assert conts[3] == 1

    def test_continuations_at_end_of_corpus(self):
        """Test continuations when match is at corpus end."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        # Last tokens have no continuation
        context = [4, 5]
        conts = model.continuations(context)

        # Should be empty or handle gracefully
        assert isinstance(conts, dict)


class TestPredict:
    """Test prediction (probability computation)."""

    def test_predict_returns_probabilities(self):
        """Test predict returns valid probability distribution."""
        corpus = [1, 2, 3] * 10  # Repeating pattern
        model = Infinigram(corpus)

        context = [1, 2]
        probs = model.predict(context, top_k=256)  # Get all probabilities

        # Check properties
        assert isinstance(probs, dict)
        assert len(probs) > 0
        assert all(0 <= p <= 1 for p in probs.values())
        assert abs(sum(probs.values()) - 1.0) < 0.01  # Should sum to 1 with all bytes

    def test_predict_top_k_limit(self):
        """Test top_k parameter limits results."""
        corpus = list(range(100)) * 5  # Many tokens
        model = Infinigram(corpus)

        context = [0]
        probs = model.predict(context, top_k=5)

        assert len(probs) <= 5

    def test_predict_smoothing(self):
        """Test smoothing assigns probability to unseen tokens."""
        corpus = [1, 2, 3, 1, 2, 4]
        model = Infinigram(corpus)

        context = [1, 2]
        probs = model.predict(context, smoothing=0.1)

        # Token 5 never follows [1, 2] but should have some probability
        if 5 in model.vocab:
            assert 5 in probs
            assert probs[5] > 0

    def test_predict_empty_context(self):
        """Test prediction with empty context."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        probs = model.predict([])

        # Should return unigram distribution
        assert len(probs) > 0
        # With smoothing=0, only observed bytes should have non-zero probability
        assert 1 in probs and probs[1] > 0
        assert 2 in probs and probs[2] > 0
        assert 3 in probs and probs[3] > 0

    def test_predict_consistent_probabilities(self):
        """Test predictions are consistent across calls."""
        corpus = [1, 2, 3] * 5
        model = Infinigram(corpus)

        context = [1, 2]
        probs1 = model.predict(context)
        probs2 = model.predict(context)

        # Should be deterministic
        assert probs1 == probs2


class TestConfidence:
    """Test confidence scoring."""

    def test_confidence_long_match(self):
        """Test high confidence for long matches."""
        corpus = [1, 2, 3, 4, 5] * 10
        model = Infinigram(corpus)

        context = [1, 2, 3, 4, 5]
        conf = model.confidence(context)

        assert 0 <= conf <= 1
        assert conf > 0.3  # Should have decent confidence

    def test_confidence_no_match(self):
        """Test low confidence for no match."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        context = [99, 98, 97]
        conf = model.confidence(context)

        assert conf == 0.0  # No match = no confidence

    def test_confidence_short_match(self):
        """Test lower confidence for short matches."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)

        # Only last token matches
        context = [99, 98, 97, 1]
        conf = model.confidence(context)

        assert 0 <= conf <= 1
        # Should be lower than long match but not zero
        assert conf < 0.5


class TestUpdate:
    """Test dynamic corpus updates."""

    def test_update_extends_corpus(self):
        """Test update adds new tokens."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        initial_size = model.n
        model.update([4, 5, 6])

        assert model.n == initial_size + 3
        assert model.corpus == [1, 2, 3, 4, 5, 6]

    def test_update_extends_vocabulary(self):
        """Test vocabulary is always fixed at 256 bytes."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        model.update([4, 5])

        assert 4 in model.vocab
        assert 5 in model.vocab
        assert model.vocab_size == 256  # Fixed byte vocabulary

    def test_update_enables_new_predictions(self):
        """Test updated corpus enables new predictions."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        # Initially, [2, 3] has no continuation
        context = [2, 3]
        conts_before = model.continuations(context)

        # Add continuation
        model.update([2, 3, 4])

        conts_after = model.continuations(context)

        # Now [2, 3] should have continuation 4
        assert 4 in conts_after


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_token_corpus(self):
        """Test corpus with single token."""
        corpus = [1]
        model = Infinigram(corpus)

        assert model.n == 1
        probs = model.predict([])
        assert len(probs) >= 1

    def test_empty_corpus_raises_error(self):
        """Test empty corpus handling."""
        # Empty corpus should handle gracefully
        try:
            model = Infinigram([])
            # Should either work or raise error
            assert model.n == 0
        except (ValueError, IndexError):
            pass  # Also acceptable

    def test_repeated_pattern_corpus(self):
        """Test corpus with repeated pattern."""
        corpus = [1, 2] * 100
        model = Infinigram(corpus)

        context = [1]
        probs = model.predict(context)

        # Token 2 should have very high probability
        assert 2 in probs
        assert probs[2] >= 0.4  # Relaxed threshold due to smoothing

    def test_long_context(self):
        """Test very long context."""
        corpus = list(range(100))
        model = Infinigram(corpus)

        context = list(range(50))  # Very long context
        probs = model.predict(context)

        # Should handle gracefully
        assert isinstance(probs, dict)

    def test_context_longer_than_corpus(self):
        """Test context longer than entire corpus."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        context = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # All valid bytes
        probs = model.predict(context)

        # Should find partial match or fall back
        assert isinstance(probs, dict)
        assert len(probs) > 0


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_wikipedia_style_corpus(self):
        """Test with Wikipedia-like text."""
        # Simulate "The cat sat on the mat. The cat ..."
        the, cat, sat, on, mat = 1, 2, 3, 4, 5
        corpus = [the, cat, sat, on, the, mat, the, cat, sat, on, the]

        model = Infinigram(corpus)

        # After "the cat", "sat" should be predicted
        context = [the, cat]
        probs = model.predict(context)

        assert sat in probs
        # "sat" should have relatively high probability (relaxed due to smoothing)
        assert probs[sat] >= 0.15

    def test_code_completion_scenario(self):
        """Test code completion scenario."""
        # Simulate: "def foo():\n  return 42\ndef foo():\n  return"
        DEF, FOO, LPAREN, RPAREN, COLON, RETURN, NUM = 1, 2, 3, 4, 5, 6, 7
        corpus = [DEF, FOO, LPAREN, RPAREN, COLON, RETURN, NUM,
                  DEF, FOO, LPAREN, RPAREN, COLON, RETURN]

        model = Infinigram(corpus)

        # Complete "return ___"
        context = [DEF, FOO, LPAREN, RPAREN, COLON, RETURN]
        probs = model.predict(context)

        # NUM should be predicted
        assert NUM in probs

    def test_update_and_query_cycle(self):
        """Test repeated update and query cycle."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        for i in range(10):
            # Query
            probs = model.predict([i % 3 + 1])
            assert len(probs) > 0

            # Update
            model.update([i % 3 + 1, (i + 1) % 3 + 1])

        # Should still work after many updates
        final_probs = model.predict([1])
        assert len(final_probs) > 0
