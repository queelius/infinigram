#!/usr/bin/env python3
"""
Tests for evaluation framework.
"""

import pytest
from infinigram.infinigram import Infinigram
from infinigram.recursive import RecursiveInfinigram, CaseNormalizer
from infinigram.evaluation import (
    Evaluator,
    BenchmarkSuite,
    PredictionResult,
    EvaluationMetrics,
    create_synthetic_corpus,
    print_comparison_table
)


class TestEvaluator:
    """Test Evaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        corpus = b"the quick brown fox jumps over the lazy dog"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test Model")

        assert evaluator.model == model
        assert evaluator.model_name == "Test Model"

    def test_evaluate_simple(self):
        """Test basic evaluation."""
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        # Simple test data
        test_data = [
            (b"the cat", b" "),
            (b"cat sat", b" "),
        ]

        metrics, results = evaluator.evaluate(test_data, top_k=5)

        # Check we got results
        assert len(results) == 2
        assert isinstance(metrics, EvaluationMetrics)

        # Check result structure
        for result in results:
            assert isinstance(result, PredictionResult)
            assert result.context in [b"the cat", b"cat sat"]
            assert result.time_ms >= 0

    def test_evaluate_accuracy(self):
        """Test accuracy calculation."""
        corpus = b"aaabbbccc"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        # Test data where we know the answers
        test_data = [
            (b"aa", b"a"),  # Correct: next is 'a'
            (b"bb", b"b"),  # Correct: next is 'b'
            (b"cc", b"c"),  # Correct: next is 'c'
        ]

        metrics, results = evaluator.evaluate(test_data, top_k=5)

        # All should be correct
        assert metrics.num_samples == 3
        assert metrics.num_correct == 3
        assert metrics.accuracy == 1.0

    def test_evaluate_coverage(self):
        """Test coverage calculation."""
        corpus = b"abc"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        # Test data with some unmatchable contexts
        test_data = [
            (b"ab", b"c"),  # Covered
            (b"xyz", b"a"),  # Not covered (not in corpus)
        ]

        metrics, results = evaluator.evaluate(test_data, top_k=5)

        # Coverage should be 50% (1 out of 2 covered)
        assert metrics.num_samples == 2
        assert metrics.num_covered >= 1  # At least 'ab' should be covered
        assert 0.0 <= metrics.coverage <= 1.0

    def test_top_k_accuracy(self):
        """Test top-k accuracy calculation."""
        corpus = b"abcdefgh"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        test_data = [(b"abc", b"d")]

        metrics, results = evaluator.evaluate(test_data, top_k=10)

        # Check top-k accuracy dict exists
        assert 1 in metrics.top_k_accuracy
        assert 3 in metrics.top_k_accuracy
        assert 5 in metrics.top_k_accuracy


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""

    def test_initialization(self):
        """Test benchmark suite initialization."""
        corpus = b"test corpus"
        suite = BenchmarkSuite(corpus)
        assert suite.corpus == corpus

    def test_create_in_distribution_test(self):
        """Test in-distribution test creation."""
        corpus = b"the quick brown fox jumps over the lazy dog"
        suite = BenchmarkSuite(corpus)

        test_data = suite.create_in_distribution_test(
            num_samples=10,
            context_length=5
        )

        assert len(test_data) == 10

        for context, next_byte in test_data:
            assert len(context) == 5
            assert len(next_byte) == 1
            # Context + next_byte should exist in corpus
            assert context + next_byte in corpus

    def test_create_ood_test_case(self):
        """Test OOD test creation with case transformation."""
        corpus = b"The Quick Brown Fox"
        suite = BenchmarkSuite(corpus)

        test_data = suite.create_ood_test(
            transformations=['case'],
            num_samples=5,
            context_length=5
        )

        assert len(test_data) == 5

        for context, next_byte in test_data:
            assert len(context) == 5
            assert len(next_byte) == 1

    def test_create_ood_test_typo(self):
        """Test OOD test creation with typo transformation."""
        corpus = b"the quick brown fox"
        suite = BenchmarkSuite(corpus)

        test_data = suite.create_ood_test(
            transformations=['typo'],
            num_samples=5,
            context_length=5
        )

        assert len(test_data) == 5

    def test_create_ood_test_synonym(self):
        """Test OOD test creation with synonym transformation."""
        corpus = b"the big cat is fast"
        suite = BenchmarkSuite(corpus)

        test_data = suite.create_ood_test(
            transformations=['synonym'],
            num_samples=5,
            context_length=5
        )

        assert len(test_data) == 5

    def test_create_ood_test_multiple(self):
        """Test OOD test with multiple transformations."""
        corpus = b"The Quick Brown Fox"
        suite = BenchmarkSuite(corpus)

        test_data = suite.create_ood_test(
            transformations=['case', 'typo'],
            num_samples=5,
            context_length=5
        )

        assert len(test_data) == 5

    def test_compare_models(self):
        """Test model comparison."""
        corpus = b"the quick brown fox jumps over the lazy dog"

        # Create models
        vanilla = Infinigram(corpus)
        recursive = RecursiveInfinigram(
            corpus,
            transformers=[CaseNormalizer()]
        )

        models = {
            "Vanilla": vanilla,
            "Recursive": recursive,
        }

        # Create test data
        suite = BenchmarkSuite(corpus)
        in_dist = suite.create_in_distribution_test(num_samples=10, context_length=5)
        ood_case = suite.create_ood_test(['case'], num_samples=10, context_length=5)

        test_datasets = {
            "In-Dist": in_dist,
            "OOD-Case": ood_case,
        }

        # Compare
        results = suite.compare_models(
            models=models,
            test_datasets=test_datasets,
            top_k=5,
            verbose=False
        )

        # Check structure
        assert "Vanilla" in results
        assert "Recursive" in results
        assert "In-Dist" in results["Vanilla"]
        assert "OOD-Case" in results["Vanilla"]

        # Check metrics
        for model_name in models.keys():
            for dataset_name in test_datasets.keys():
                metrics = results[model_name][dataset_name]
                assert isinstance(metrics, EvaluationMetrics)
                assert 0.0 <= metrics.accuracy <= 1.0
                assert 0.0 <= metrics.coverage <= 1.0


class TestSyntheticCorpus:
    """Test synthetic corpus generation."""

    def test_create_synthetic_corpus(self):
        """Test synthetic corpus creation."""
        corpus = create_synthetic_corpus(size=1000)

        assert isinstance(corpus, bytes)
        assert len(corpus) >= 500  # At least close to target size
        assert len(corpus) <= 2000  # Not too much larger

    def test_synthetic_corpus_has_content(self):
        """Test that synthetic corpus has reasonable content."""
        corpus = create_synthetic_corpus(size=500)

        # Should contain some common words
        assert b"the" in corpus or b"fox" in corpus or b"cat" in corpus

        # Should have spaces and periods
        assert b" " in corpus
        assert b"." in corpus


class TestTransformations:
    """Test transformation functions."""

    def test_case_transform(self):
        """Test case transformation."""
        suite = BenchmarkSuite(b"test")

        text = b"Hello World"
        transformed = suite._apply_case_transform(text)

        # Should be same length
        assert len(transformed) == len(text)

        # Should have some changes (with high probability)
        # Can't guarantee changes due to randomness, just check it runs
        assert isinstance(transformed, bytes)

    def test_typo_transform(self):
        """Test typo transformation."""
        suite = BenchmarkSuite(b"test")

        text = b"hello world"
        transformed = suite._apply_typo_transform(text)

        # Should be same length
        assert len(transformed) == len(text)
        assert isinstance(transformed, bytes)

    def test_synonym_transform(self):
        """Test synonym transformation."""
        suite = BenchmarkSuite(b"test")

        text = b"the big cat is fast"
        transformed = suite._apply_synonym_transform(text)

        # Should replace 'big' with 'large' and 'fast' with 'quick'
        assert b"large" in transformed
        assert b"quick" in transformed
        assert b"big" not in transformed
        assert b"fast" not in transformed


class TestMetrics:
    """Test metrics computation."""

    def test_perplexity_calculation(self):
        """Test perplexity calculation."""
        corpus = b"abc"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        test_data = [(b"ab", b"c")]

        metrics, results = evaluator.evaluate(test_data, top_k=5)

        # Perplexity should be finite and positive
        assert metrics.perplexity > 0
        assert metrics.perplexity < float('inf')

    def test_rank_calculation(self):
        """Test rank calculation."""
        corpus = b"aaabbbccc"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        test_data = [(b"aa", b"a")]

        metrics, results = evaluator.evaluate(test_data, top_k=5)

        # Mean rank should be reasonable
        assert metrics.mean_rank >= 1
        assert metrics.median_rank >= 1


class TestPrintComparisonTable:
    """Test comparison table printing."""

    def test_print_comparison_table(self):
        """Test that printing doesn't crash."""
        corpus = b"test corpus"
        model = Infinigram(corpus)

        suite = BenchmarkSuite(corpus)
        test_data = suite.create_in_distribution_test(5, 3)

        results = suite.compare_models(
            models={"Model": model},
            test_datasets={"Test": test_data},
            top_k=5,
            verbose=False
        )

        # Should not crash
        print_comparison_table(results)


class TestEvaluatorEdgeCases:
    """Test evaluator edge cases for full coverage."""

    def test_evaluate_with_verbose_output(self):
        """
        Given: Test data with multiple samples
        When: Evaluating with verbose=True
        Then: Prints progress messages (covers logging lines)
        """
        corpus = b"the cat sat on the mat"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        # Create 100 samples to trigger progress printing
        test_data = [(b"the", b" ")] * 100

        # Should print progress without crashing
        metrics, results = evaluator.evaluate(test_data, top_k=5, verbose=True)

        assert len(results) == 100
        assert isinstance(metrics, EvaluationMetrics)

    def test_evaluate_with_no_predictions(self):
        """
        Given: Model that never returns predictions
        When: Evaluating
        Then: Handles gracefully with inf perplexity
        """
        # Mock model that always returns empty dict
        class NoOpModel:
            def predict(self, context, top_k=10):
                return {}

        model = NoOpModel()
        evaluator = Evaluator(model, "NoOp")

        test_data = [(b"test", b"x"), (b"data", b"y")]

        metrics, results = evaluator.evaluate(test_data, top_k=5)

        # Coverage should be 0%
        assert metrics.coverage == 0.0

        # Perplexity should be infinity (no predictions)
        assert metrics.perplexity == float('inf')

        # Mean probability should be 0
        assert metrics.mean_probability == 0.0

        # All predictions should be None
        assert all(r.predicted is None for r in results)


class TestBenchmarkSuiteVerbose:
    """Test benchmark suite with verbose output."""

    def test_compare_models_with_verbose(self):
        """
        Given: Multiple models and datasets
        When: Comparing with verbose=True
        Then: Prints comparison info (covers logging lines)
        """
        corpus = b"the cat sat on the mat"

        models = {
            "Vanilla": Infinigram(corpus),
            "Recursive": RecursiveInfinigram(corpus),
        }

        suite = BenchmarkSuite(corpus)
        test_data = suite.create_in_distribution_test(10, 5)

        test_datasets = {
            "Test": test_data,
        }

        # Should print verbose output
        results = suite.compare_models(
            models=models,
            test_datasets=test_datasets,
            top_k=5,
            verbose=True  # Enable verbose logging
        )

        assert "Vanilla" in results
        assert "Recursive" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
