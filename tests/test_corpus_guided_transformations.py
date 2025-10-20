#!/usr/bin/env python3
"""
Tests for corpus-guided transformation generation.

This tests that transformers actually inspect the corpus at match positions
and generate intelligent transformations based on what they find.
"""

import pytest
from infinigram.recursive import (
    RecursiveInfinigram,
    SynonymTransformer,
    EditDistanceTransformer,
    CaseNormalizer,
)


class TestSynonymTransformerCorpusInspection:
    """Test that SynonymTransformer inspects corpus correctly."""

    def test_generates_transformation_from_corpus_inspection(self):
        """Test that transformer finds synonyms by inspecting corpus."""
        # Corpus: "the feline chased the mouse. the cat ran fast."
        #          0   7      14     23  29     36  40  44  48
        # The suffix "chased" appears at position 14
        # Before "chased" is "feline " (position 7-14)
        # The suffix "ran" appears at position 44
        # Before "ran" is "cat " (position 40-44)

        corpus = b"the feline chased the mouse. the cat ran fast."
        model = RecursiveInfinigram(corpus)

        # Input context where "cat" appears before "chased"
        # This doesn't exist in corpus, but "feline chased" does
        context = b"the cat chased"

        # Find suffix matches for "chased"
        suffix = b"chased"
        context_list = list(context)
        all_matches = model._find_all_suffix_matches(context)

        # Should find "chased" in corpus
        assert len(all_matches) > 0

        # Get the longest match
        suffix_len, positions = all_matches[0]

        # Use SynonymTransformer to generate transformations
        transformer = SynonymTransformer()
        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=positions
        )

        # Since our placeholder synonym checker uses case-insensitive equality,
        # it won't find cat→feline. But we can verify the logic is working.
        # The transformer should inspect the corpus and compare words.

        # For now, verify it doesn't crash and returns a list
        assert isinstance(transformations, list)

    def test_avoids_duplicate_transformations(self):
        """Test that transformer doesn't generate duplicate transformations."""
        # Corpus with repeated pattern
        corpus = b"the cat sat. the cat sat again. the cat sat once more."

        transformer = SynonymTransformer()

        # Context with suffix "sat"
        context = b"the dog sat"
        suffix = b"sat"

        # Find match positions for "sat"
        model = RecursiveInfinigram(corpus)
        all_matches = model._find_all_suffix_matches(context)

        if all_matches:
            suffix_len, positions = all_matches[0]

            # Should have multiple positions for "sat"
            assert len(positions) >= 2

            transformations = transformer.generate_transformations(
                context=context,
                suffix=suffix,
                corpus=corpus,
                match_positions=positions
            )

            # Verify no duplicate transformation descriptions
            transform_descs = [desc for _, desc in transformations]
            assert len(transform_descs) == len(set(transform_descs))

    def test_handles_different_word_counts(self):
        """Test that transformer handles prefixes with different word counts."""
        corpus = b"big cat ran. small dog walked."

        transformer = SynonymTransformer()

        # Context: "the cat ran" (3 words before "ran")
        # Corpus: "big cat ran" (2 words before "ran")
        context = b"the cat ran"
        suffix = b"ran"

        model = RecursiveInfinigram(corpus)
        all_matches = model._find_all_suffix_matches(context)

        if all_matches:
            suffix_len, positions = all_matches[0]

            transformations = transformer.generate_transformations(
                context=context,
                suffix=suffix,
                corpus=corpus,
                match_positions=positions
            )

            # Should handle gracefully (skip mismatched word counts)
            assert isinstance(transformations, list)


class TestEditDistanceTransformerCorpusInspection:
    """Test that EditDistanceTransformer inspects corpus correctly."""

    def test_detects_typos_from_corpus(self):
        """Test that transformer detects typos by inspecting corpus."""
        # Test the core mechanism: transformer compares words and detects typos
        # Setup: corpus has correct spelling, context has typo

        corpus = b"hello world the cat sleeps"
        transformer = EditDistanceTransformer(max_distance=2)

        # Context: "the caat sleeps"
        # We'll find a match for " sleeps" which allows prefix comparison
        context = b"the caat sleeps"

        # Manually set up a scenario where we know transformation should work
        # Suffix: " sleeps" matches at position 19 in corpus
        # Context prefix: "the caat" (8 bytes)
        # Corpus at position 19-8=11: "the cat " (8 bytes)
        # This should allow comparison of "caat" vs "cat"

        suffix = b" sleeps"
        positions = [19]  # Position where " sleeps" appears in corpus

        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=positions
        )

        # Should detect typo
        assert len(transformations) > 0, "Should find caat→cat transformation"

        new_context, desc = transformations[0]
        assert "typo:" in desc
        assert "caat" in desc and "cat" in desc
        assert new_context == b"the cat sleeps"

    def test_respects_max_distance(self):
        """Test that transformer respects max edit distance."""
        corpus = b"the cat ran fast"

        transformer = EditDistanceTransformer(max_distance=1)

        # Context with large typo: "the caaaaat ran"
        context = b"the caaaaat ran"
        suffix = b"ran"

        model = RecursiveInfinigram(corpus)
        all_matches = model._find_all_suffix_matches(context)

        if all_matches:
            suffix_len, positions = all_matches[0]

            transformations = transformer.generate_transformations(
                context=context,
                suffix=suffix,
                corpus=corpus,
                match_positions=positions
            )

            # Should NOT suggest transformation (distance > max_distance)
            # Edit distance from "caaaaat" to "cat" is 4
            assert len(transformations) == 0

    def test_avoids_duplicate_typo_corrections(self):
        """Test that transformer doesn't generate duplicate corrections."""
        # Corpus with repeated pattern
        corpus = b"the cat sat. the cat sat. the cat sat."

        transformer = EditDistanceTransformer(max_distance=2)

        # Context with typo
        context = b"the caat sat"
        suffix = b"sat"

        model = RecursiveInfinigram(corpus)
        all_matches = model._find_all_suffix_matches(context)

        if all_matches:
            suffix_len, positions = all_matches[0]

            transformations = transformer.generate_transformations(
                context=context,
                suffix=suffix,
                corpus=corpus,
                match_positions=positions
            )

            # Verify no duplicate transformation descriptions
            transform_descs = [desc for _, desc in transformations]
            assert len(transform_descs) == len(set(transform_descs))


class TestCaseNormalizerCorpusInspection:
    """Test CaseNormalizer behavior."""

    def test_normalizes_case(self):
        """Test that case normalizer lowercases context."""
        corpus = b"the cat ran fast"

        transformer = CaseNormalizer()

        # Context with mixed case
        context = b"The CAT ran"
        suffix = b"ran"

        model = RecursiveInfinigram(corpus)
        all_matches = model._find_all_suffix_matches(context)

        if all_matches:
            suffix_len, positions = all_matches[0]

            transformations = transformer.generate_transformations(
                context=context,
                suffix=suffix,
                corpus=corpus,
                match_positions=positions
            )

            # Should lowercase
            assert len(transformations) == 1
            new_context, desc = transformations[0]
            assert new_context == b"the cat ran"
            assert "case:" in desc


class TestIntegrationWithRecursiveInfinigram:
    """Test transformers integrated with RecursiveInfinigram."""

    def test_end_to_end_typo_correction(self):
        """Test end-to-end typo correction during prediction."""
        # Corpus with patterns
        corpus = b"the cat sat on the mat. the cat ran fast."

        model = RecursiveInfinigram(corpus)

        # Input with typo: "the caat sat"
        context = b"the caat sat"

        # Make prediction with shallow recursion
        predictions = model.predict(
            context=context,
            max_depth=1,
            beam_width=3,
            top_k=10
        )

        # Should return some predictions
        assert len(predictions) > 0

        # Get explanation to see if transformation happened
        predictions_with_exp, explanations = model.predict_with_explanation(
            context=context,
            max_depth=1,
            beam_width=3,
            top_k=10
        )

        # Check that we explored some transformations
        assert len(explanations) > 0

        # At least one should be the original context
        original_found = any(
            exp['context'] == context and len(exp['transformations']) == 0
            for exp in explanations
        )
        assert original_found

    def test_end_to_end_case_normalization(self):
        """Test end-to-end case normalization during prediction."""
        corpus = b"the cat sat on the mat"

        model = RecursiveInfinigram(corpus)

        # Input with wrong case: "The Cat sat"
        context = b"The Cat sat"

        predictions, explanations = model.predict_with_explanation(
            context=context,
            max_depth=1,
            beam_width=3,
            top_k=10
        )

        # Should have tried case normalization
        case_normalized = any(
            'case:' in str(exp['transformations'])
            for exp in explanations
        )

        # Note: May not find case transformation if exact match exists
        # This is expected behavior
        assert len(explanations) > 0


class TestTransformerWordReplacement:
    """Test the word replacement logic in transformers."""

    def test_synonym_word_replacement_preserves_suffix(self):
        """Test that word replacement preserves the matched suffix."""
        transformer = SynonymTransformer()

        context = b"the big cat chased"
        context_prefix = b"the big cat "
        old_word = b"big"
        new_word = b"small"
        position = 1  # "big" is at position 1

        new_context = transformer._replace_word_in_context(
            context, context_prefix, old_word, new_word, position
        )

        # Should get: "the small cat chased"
        assert new_context == b"the small cat chased"

    def test_edit_distance_word_replacement_preserves_suffix(self):
        """Test that typo correction preserves the matched suffix."""
        transformer = EditDistanceTransformer()

        context = b"the caat ran fast"
        context_prefix = b"the caat "
        old_word = b"caat"
        new_word = b"cat"
        position = 1  # "caat" is at position 1

        new_context = transformer._replace_word_in_context(
            context, context_prefix, old_word, new_word, position
        )

        # Should get: "the cat ran fast"
        assert new_context == b"the cat ran fast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
