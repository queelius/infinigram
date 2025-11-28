#!/usr/bin/env python3
"""
Tests for exposing suffix array positions from Infinigram.
"""

import pytest
from infinigram import Infinigram


class TestSuffixArrayPositions:
    """Test that suffix array positions are properly exposed."""

    def test_find_all_suffix_matches_basic(self):
        """Test basic suffix match finding."""
        # Corpus: "the cat sat on the mat"
        corpus = list(b"the cat sat on the mat")
        model = Infinigram(corpus)

        # Context: "the cat"
        context = list(b"the cat")

        # Find all suffix matches
        matches = model.find_all_suffix_matches(context)

        # Should find matches for various suffix lengths
        assert len(matches) > 0

        # Matches should be sorted by decreasing length
        lengths = [length for length, _ in matches]
        assert lengths == sorted(lengths, reverse=True)

        # Each match should have positions
        for suffix_len, positions in matches:
            assert suffix_len > 0
            assert len(positions) > 0
            # Verify positions are valid
            for pos in positions:
                assert 0 <= pos < len(corpus)

    def test_find_all_suffix_matches_verifies_positions(self):
        """Test that returned positions actually match."""
        corpus = list(b"the cat sat. the cat ran.")
        model = Infinigram(corpus)

        context = list(b"the cat")
        matches = model.find_all_suffix_matches(context)

        # Verify each position actually contains the suffix
        for suffix_len, positions in matches:
            suffix = context[-suffix_len:]

            for pos in positions:
                # Extract suffix from corpus at this position
                corpus_suffix = corpus[pos:pos + suffix_len]
                assert corpus_suffix == suffix, \
                    f"Mismatch at pos {pos}: expected {suffix}, got {corpus_suffix}"

    def test_find_all_suffix_matches_multiple_lengths(self):
        """Test that we get multiple suffix lengths."""
        # Corpus with repeated patterns
        corpus = list(b"abc abcd abcde")
        model = Infinigram(corpus)

        # Context: "abcde"
        context = list(b"abcde")
        matches = model.find_all_suffix_matches(context)

        # Should find matches for "abcde", "abcd", "abc", etc.
        lengths = [length for length, _ in matches]

        # Should have multiple different lengths
        assert len(set(lengths)) > 1

    def test_find_all_suffix_matches_with_max_length(self):
        """Test that max_length is respected."""
        corpus = list(b"abcdefgh")
        model = Infinigram(corpus, max_length=3)

        context = list(b"abcdefgh")
        matches = model.find_all_suffix_matches(context)

        # Should not search beyond max_length
        max_match_len = max(length for length, _ in matches) if matches else 0
        assert max_match_len <= 3

    def test_find_all_suffix_matches_no_match(self):
        """Test with no matching suffixes."""
        corpus = list(b"abc")
        model = Infinigram(corpus)

        context = list(b"xyz")
        matches = model.find_all_suffix_matches(context)

        # Should return empty list
        assert matches == []

    def test_find_all_suffix_matches_empty_context(self):
        """Test with empty context."""
        corpus = list(b"abc")
        model = Infinigram(corpus)

        context = []
        matches = model.find_all_suffix_matches(context)

        # Should return empty list
        assert matches == []

    def test_find_all_suffix_matches_full_match(self):
        """Test when entire context matches."""
        corpus = list(b"hello world hello")
        model = Infinigram(corpus)

        context = list(b"hello")
        matches = model.find_all_suffix_matches(context)

        # Should find full match
        assert len(matches) > 0

        # First match should be full length
        longest_len, positions = matches[0]
        assert longest_len == len(context)

        # Should have multiple positions (appears twice)
        assert len(positions) >= 2

    def test_positions_enable_corpus_inspection(self):
        """Test that positions allow inspecting corpus context."""
        corpus = list(b"the big cat sat. the small dog ran.")
        model = Infinigram(corpus)

        context = list(b"cat")
        matches = model.find_all_suffix_matches(context)

        assert len(matches) > 0

        suffix_len, positions = matches[0]

        # We should be able to inspect what comes BEFORE the match
        for pos in positions:
            if pos >= 4:  # Enough space to look back
                # Look at 4 bytes before the match
                prefix = corpus[pos-4:pos]
                # Should be "big " (the word before "cat")
                assert prefix == list(b"big ")


class TestSuffixArrayUtilities:
    """Test SuffixArray utility methods (get_context, ngrams)."""

    def test_get_context_at_start(self):
        """Test get_context at corpus beginning."""
        corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        model = Infinigram(corpus)
        sa = model.sa  # Access suffix array via 'sa' attribute

        before, after = sa.get_context(0, window=3)
        assert before == [], "Should have no tokens before position 0"
        assert after == [1, 2, 3], "Should have first 3 tokens after"

    def test_get_context_at_end(self):
        """Test get_context at corpus end."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)
        sa = model.sa

        before, after = sa.get_context(4, window=3)
        assert before == [2, 3, 4], "Should have 3 tokens before position 4"
        assert after == [5], "Should only have 1 token at position 4"

    def test_get_context_middle(self):
        """Test get_context in middle of corpus."""
        corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        model = Infinigram(corpus)
        sa = model.sa

        before, after = sa.get_context(5, window=2)
        assert before == [4, 5], "Should have 2 tokens before"
        assert after == [6, 7], "Should have 2 tokens after"

    def test_get_context_default_window(self):
        """Test get_context with default window size."""
        corpus = list(range(100))
        model = Infinigram(corpus)
        sa = model.sa

        before, after = sa.get_context(50)  # Default window=10
        assert len(before) == 10, "Default window should be 10"
        assert len(after) == 10, "Default window should be 10"

    def test_get_context_zero_window(self):
        """Test get_context with window=0."""
        corpus = [1, 2, 3, 4, 5]
        model = Infinigram(corpus)
        sa = model.sa

        before, after = sa.get_context(2, window=0)
        assert before == [], "Window 0 should give empty before"
        assert after == [], "Window 0 should give empty after"

    def test_ngrams_basic(self):
        """Test ngrams iterator returns correct counts."""
        corpus = [1, 2, 3, 1, 2, 4]
        model = Infinigram(corpus)
        sa = model.sa

        bigrams = list(sa.ngrams(2))
        bigram_counts = dict(bigrams)

        # [1,2] appears twice, others once
        assert bigram_counts[(1, 2)] == 2, "Pattern [1,2] appears twice"
        assert bigram_counts[(2, 3)] == 1, "Pattern [2,3] appears once"
        assert bigram_counts[(3, 1)] == 1, "Pattern [3,1] appears once"
        assert bigram_counts[(2, 4)] == 1, "Pattern [2,4] appears once"

    def test_ngrams_sorted_by_frequency(self):
        """Test ngrams are sorted by descending frequency."""
        corpus = [1, 1, 1, 2, 2, 3]
        model = Infinigram(corpus)
        sa = model.sa

        unigrams = list(sa.ngrams(1))

        # Should be sorted by frequency (descending)
        counts = [count for _, count in unigrams]
        assert counts == sorted(counts, reverse=True), "Should be sorted by frequency descending"

        # First should be most frequent
        assert unigrams[0] == ((1,), 3), "Most frequent should be first"

    def test_ngrams_trigrams(self):
        """Test ngrams with n=3."""
        corpus = [1, 2, 3, 1, 2, 3, 4]
        model = Infinigram(corpus)
        sa = model.sa

        trigrams = list(sa.ngrams(3))
        trigram_counts = dict(trigrams)

        # [1,2,3] appears twice
        assert trigram_counts[(1, 2, 3)] == 2, "Pattern [1,2,3] appears twice"

    def test_ngrams_empty_for_n_greater_than_corpus(self):
        """Test ngrams returns empty when n > corpus length."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)
        sa = model.sa

        ngrams = list(sa.ngrams(5))  # n > len(corpus)
        assert ngrams == [], "Should be empty when n > corpus length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
