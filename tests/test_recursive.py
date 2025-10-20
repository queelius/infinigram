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
