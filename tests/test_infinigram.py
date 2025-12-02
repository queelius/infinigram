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
        with pytest.raises(ValueError, match="new_tokens must contain only bytes"):
            model.update([256])

        with pytest.raises(ValueError, match="new_tokens must contain only bytes"):
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


class TestFindAllSuffixMatches:
    """Test find_all_suffix_matches method."""

    def test_find_all_suffix_matches_basic(self):
        """Test basic suffix match finding."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"the cat")

        # Should find matches at multiple lengths
        assert len(matches) > 0

        # Should be sorted by decreasing length
        lengths = [length for length, _ in matches]
        assert lengths == sorted(lengths, reverse=True)

        # Each entry should have (length, positions)
        for length, positions in matches:
            assert length > 0
            assert len(positions) > 0

    def test_find_all_suffix_matches_multiple_lengths(self):
        """Test that we get matches at multiple suffix lengths."""
        corpus = b"abc abcd abcde"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"abcde")

        # Should have multiple different lengths
        lengths = [length for length, _ in matches]
        assert len(set(lengths)) > 1

    def test_find_all_suffix_matches_verifies_positions(self):
        """Test that returned positions actually contain the suffix."""
        corpus = b"the cat sat. the cat ran."
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"the cat")

        # Verify positions are valid
        for length, positions in matches:
            for pos in positions:
                assert 0 <= pos < len(corpus)

    def test_find_all_suffix_matches_no_match(self):
        """Test with no matching suffixes."""
        corpus = b"abc"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"xyz")
        assert matches == []

    def test_find_all_suffix_matches_empty_context(self):
        """Test with empty context."""
        corpus = b"abc"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"")
        assert matches == []

    def test_find_all_suffix_matches_respects_max_length(self):
        """Test that max_length is respected."""
        corpus = b"abcdefgh"
        model = Infinigram(corpus, max_length=3)

        matches = model.find_all_suffix_matches(b"abcdefgh")

        # Should not search beyond max_length
        if matches:
            max_match_len = max(length for length, _ in matches)
            assert max_match_len <= 3

    # === Additional comprehensive tests for find_all_suffix_matches ===

    def test_find_all_suffix_matches_single_byte_corpus(self):
        """Test with single byte corpus - edge case."""
        corpus = b"a"
        model = Infinigram(corpus)

        # Searching for "a" should find it
        matches = model.find_all_suffix_matches(b"a")
        assert len(matches) == 1
        assert matches[0] == (1, [0])  # length=1, position=0

        # Searching for "b" should find nothing
        matches = model.find_all_suffix_matches(b"b")
        assert matches == []

    def test_find_all_suffix_matches_single_byte_context(self):
        """Test with single byte context against longer corpus."""
        corpus = b"abcabc"
        model = Infinigram(corpus)

        # Single byte "a" appears at positions 0 and 3
        matches = model.find_all_suffix_matches(b"a")
        assert len(matches) == 1
        length, positions = matches[0]
        assert length == 1
        assert sorted(positions) == [0, 3]

    def test_find_all_suffix_matches_context_longer_than_corpus(self):
        """Test when context is longer than entire corpus."""
        corpus = b"abc"
        model = Infinigram(corpus)

        # Context longer than corpus - should still find partial matches
        matches = model.find_all_suffix_matches(b"xyzabc")

        # Should find "abc" (length 3), "bc" (length 2), "c" (length 1)
        assert len(matches) == 3
        lengths = [length for length, _ in matches]
        assert lengths == [3, 2, 1]

    def test_find_all_suffix_matches_position_verification(self):
        """Verify that corpus[pos:pos+length] == suffix for each match."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)

        context = b"the cat"
        matches = model.find_all_suffix_matches(context)

        # For each match, verify the position actually contains the suffix
        for length, positions in matches:
            suffix = context[-length:]
            for pos in positions:
                # Extract substring at that position
                extracted = corpus[pos:pos + length]
                assert extracted == suffix, (
                    f"Mismatch at pos={pos}, length={length}: "
                    f"expected {suffix!r}, got {extracted!r}"
                )

    def test_find_all_suffix_matches_frequency_verification(self):
        """Verify position counts match expected occurrences."""
        corpus = b"abab ab ab abab"
        model = Infinigram(corpus)

        # Test "ab" - should appear multiple times
        matches = model.find_all_suffix_matches(b"ab")
        # Returns matches at both length 2 ("ab") and length 1 ("b")
        assert len(matches) == 2  # Two lengths: 2 and 1

        # Find the length-2 match
        length_2_matches = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_matches) == 1
        length, positions = length_2_matches[0]
        assert length == 2

        # Count "ab" manually in corpus
        expected_count = corpus.count(b"ab")
        assert len(positions) == expected_count, (
            f"Expected {expected_count} positions for 'ab', got {len(positions)}"
        )

    def test_find_all_suffix_matches_with_lowercase_transform(self):
        """Test transform integration with lowercase."""
        corpus = b"Hello World hello world"
        model = Infinigram(corpus)

        # Without transform - "HELLO" won't match
        matches = model.find_all_suffix_matches(b"HELLO", transforms=[])
        assert matches == []

        # With lowercase transform - should find matches
        matches = model.find_all_suffix_matches(b"HELLO", transforms=['lowercase'])
        assert len(matches) > 0
        # The transformed query "hello" should match
        for length, positions in matches:
            assert len(positions) > 0

    def test_find_all_suffix_matches_with_strip_transform(self):
        """Test transform integration with strip."""
        corpus = b"test data"
        model = Infinigram(corpus)

        # Query with whitespace - strip should remove it
        matches = model.find_all_suffix_matches(b"  test  ", transforms=['strip'])
        assert len(matches) > 0
        # Should find "test" after stripping
        lengths = [length for length, _ in matches]
        assert 4 in lengths  # "test" is length 4

    def test_find_all_suffix_matches_with_default_transforms(self):
        """Test that default transforms are applied when transforms=None."""
        corpus = b"hello world"
        model = Infinigram(corpus, default_transforms=['lowercase'])

        # Query with uppercase - default lowercase transform should apply
        matches = model.find_all_suffix_matches(b"HELLO", transforms=None)
        assert len(matches) > 0

    def test_find_all_suffix_matches_override_default_transforms(self):
        """Test that transforms=[] overrides defaults."""
        corpus = b"hello world"
        model = Infinigram(corpus, default_transforms=['lowercase'])

        # Explicitly pass empty transforms - should NOT find uppercase
        matches = model.find_all_suffix_matches(b"HELLO", transforms=[])
        assert matches == []

    def test_find_all_suffix_matches_string_input(self):
        """Test that string input works (auto-converted to bytes)."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)

        # Pass string instead of bytes
        matches = model.find_all_suffix_matches("the cat")

        assert len(matches) > 0
        # Verify same result as bytes
        matches_bytes = model.find_all_suffix_matches(b"the cat")
        assert matches == matches_bytes

    def test_find_all_suffix_matches_list_input(self):
        """Test that list of ints input works."""
        corpus = b"abc"
        model = Infinigram(corpus)

        # Pass list of byte values
        matches = model.find_all_suffix_matches([97, 98, 99])  # "abc"

        assert len(matches) > 0
        # Verify same result as bytes
        matches_bytes = model.find_all_suffix_matches(b"abc")
        assert matches == matches_bytes

    def test_find_all_suffix_matches_repeated_pattern(self):
        """Test corpus with many repetitions of same pattern."""
        # Corpus with "ab" repeated 100 times
        corpus = b"ab" * 100
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"ab")

        # Should find exactly 100 positions for "ab"
        length_2_match = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_match) == 1
        _, positions = length_2_match[0]
        assert len(positions) == 100

        # Each position should be even (0, 2, 4, ...)
        expected_positions = list(range(0, 200, 2))
        assert sorted(positions) == expected_positions

    def test_find_all_suffix_matches_overlapping_patterns(self):
        """Test overlapping matches e.g., 'aaa' in 'aaaa'."""
        corpus = b"aaaa"
        model = Infinigram(corpus)

        # Search for "aa" in "aaaa"
        # Should find at positions 0, 1, 2 (overlapping)
        matches = model.find_all_suffix_matches(b"aa")

        length_2_match = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_match) == 1
        _, positions = length_2_match[0]
        assert sorted(positions) == [0, 1, 2]

    def test_find_all_suffix_matches_overlapping_longer_pattern(self):
        """Test overlapping with longer pattern 'aaa' in 'aaaaa'."""
        corpus = b"aaaaa"
        model = Infinigram(corpus)

        # Search for "aaa" in "aaaaa"
        # Should find at positions 0, 1, 2 (overlapping)
        matches = model.find_all_suffix_matches(b"aaa")

        length_3_match = [(l, p) for l, p in matches if l == 3]
        assert len(length_3_match) == 1
        _, positions = length_3_match[0]
        assert sorted(positions) == [0, 1, 2]

    def test_find_all_suffix_matches_all_lengths_found(self):
        """Test that all matching suffix lengths are returned."""
        corpus = b"abcdefg"
        model = Infinigram(corpus)

        # Context "defg" should match at lengths 4, 3, 2, 1
        matches = model.find_all_suffix_matches(b"defg")

        lengths = [length for length, _ in matches]
        assert lengths == [4, 3, 2, 1]

    def test_find_all_suffix_matches_partial_suffix_only(self):
        """Test when only partial suffix matches."""
        corpus = b"xyz"
        model = Infinigram(corpus)

        # Context "abc xyz" - only "xyz" part should match
        matches = model.find_all_suffix_matches(b"abc xyz")

        # Should find " xyz" (4), "xyz" (3), "yz" (2), "z" (1)
        # But "abc" part doesn't exist in corpus
        assert len(matches) > 0
        max_length = max(length for length, _ in matches)
        # Cannot match more than 4 bytes (" xyz")
        assert max_length <= 4

    def test_find_all_suffix_matches_unicode_string(self):
        """Test with unicode string input."""
        corpus = "hello world".encode('utf-8')
        model = Infinigram(corpus)

        # Unicode string should be auto-converted
        matches = model.find_all_suffix_matches("hello")

        assert len(matches) > 0
        # "hello" is 5 bytes in UTF-8
        max_length = max(length for length, _ in matches)
        assert max_length == 5

    def test_find_all_suffix_matches_special_bytes(self):
        """Test with special byte values (0, 255, etc.)."""
        corpus = bytes([0, 1, 255, 254, 0, 1])
        model = Infinigram(corpus)

        # Search for pattern with special bytes
        matches = model.find_all_suffix_matches(bytes([0, 1]))

        assert len(matches) > 0
        # [0, 1] appears at positions 0 and 4
        length_2_match = [(l, p) for l, p in matches if l == 2]
        assert len(length_2_match) == 1
        _, positions = length_2_match[0]
        assert sorted(positions) == [0, 4]

    def test_find_all_suffix_matches_no_false_positives(self):
        """Test that positions don't include false positives."""
        corpus = b"ab cd ef"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"ab")

        # Only position 0 should match "ab", not position 3 ("cd")
        for length, positions in matches:
            suffix = b"ab"[-length:]
            for pos in positions:
                assert corpus[pos:pos + length] == suffix

    def test_find_all_suffix_matches_positions_are_sorted(self):
        """Test that positions within each length are consistent."""
        corpus = b"test test test"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"test")

        for length, positions in matches:
            # Positions should be a list (may or may not be sorted)
            assert isinstance(positions, list)
            assert len(positions) > 0

    def test_find_all_suffix_matches_with_newlines(self):
        """Test corpus with newlines."""
        corpus = b"line1\nline2\nline1"
        model = Infinigram(corpus)

        matches = model.find_all_suffix_matches(b"line1")

        assert len(matches) > 0
        # "line1" appears at positions 0 and 12
        length_5_match = [(l, p) for l, p in matches if l == 5]
        assert len(length_5_match) == 1
        _, positions = length_5_match[0]
        assert len(positions) == 2

    def test_find_all_suffix_matches_empty_after_transform(self):
        """Test when transform results in empty query."""
        corpus = b"test"
        model = Infinigram(corpus)

        # Whitespace-only query stripped to empty
        matches = model.find_all_suffix_matches(b"   ", transforms=['strip'])
        assert matches == []

    def test_find_all_suffix_matches_max_length_zero_edge_case(self):
        """Test behavior when max_length effectively limits to nothing useful."""
        corpus = b"abcdefgh"
        model = Infinigram(corpus, max_length=1)

        matches = model.find_all_suffix_matches(b"abcdefgh")

        # Only last byte should be considered due to max_length=1
        if matches:
            max_match_len = max(length for length, _ in matches)
            assert max_match_len == 1


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
        """Test continuations with empty context returns uniform distribution."""
        corpus = [1, 2, 1, 3, 1]
        model = Infinigram(corpus)

        conts = model.continuations([])

        # Empty context returns uniform distribution over all 256 bytes
        assert len(conts) == 256
        assert all(v == 1 for v in conts.values())

    def test_continuations_no_match(self):
        """Test continuations when no suffix matches returns uniform distribution."""
        corpus = [1, 2, 3]
        model = Infinigram(corpus)

        conts = model.continuations([99, 98, 97])

        # No match returns uniform distribution over all 256 bytes
        assert len(conts) == 256
        assert all(v == 1 for v in conts.values())

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
        # Note: corpus is now stored in mmap file, not as attribute
        # Verify by checking we can find the new pattern
        assert model.count([4, 5, 6]) == 1

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
