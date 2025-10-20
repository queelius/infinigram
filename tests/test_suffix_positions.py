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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
