#!/usr/bin/env python3
"""
Evaluation framework for Infinigram and RecursiveInfinigram.

Provides tools to benchmark and compare model performance on both
in-distribution and out-of-distribution (OOD) data.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import time
from infinigram.infinigram import Infinigram
from infinigram.recursive import RecursiveInfinigram


@dataclass
class PredictionResult:
    """Single prediction result."""
    context: bytes
    true_next: bytes
    predicted: Optional[bytes]
    probability: float
    rank: int  # Rank of true_next in predictions (1 = top prediction)
    correct: bool
    time_ms: float


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    # Accuracy metrics
    accuracy: float  # % of correct predictions
    top_k_accuracy: Dict[int, float]  # Top-k accuracy for k=1,3,5,10
    mean_rank: float  # Average rank of true next token
    median_rank: float  # Median rank of true next token

    # Coverage metrics
    coverage: float  # % of contexts that have predictions
    no_match_rate: float  # % of contexts with no suffix match

    # Quality metrics
    perplexity: float  # Perplexity of predictions
    mean_probability: float  # Average probability assigned to correct token

    # Performance metrics
    mean_time_ms: float  # Average prediction time
    median_time_ms: float  # Median prediction time
    total_time_s: float  # Total evaluation time

    # Additional stats
    num_samples: int
    num_correct: int
    num_covered: int


class Evaluator:
    """
    Evaluate Infinigram models on test data.

    Supports both vanilla Infinigram and RecursiveInfinigram.
    """

    def __init__(self, model, model_name: str = "Unknown"):
        """
        Initialize evaluator.

        Args:
            model: Infinigram or RecursiveInfinigram instance
            model_name: Name for logging/reporting
        """
        self.model = model
        self.model_name = model_name

    def evaluate(
        self,
        test_data: List[Tuple[bytes, bytes]],
        top_k: int = 10,
        verbose: bool = False
    ) -> Tuple[EvaluationMetrics, List[PredictionResult]]:
        """
        Evaluate model on test data.

        Args:
            test_data: List of (context, true_next_byte) pairs
            top_k: Number of top predictions to consider
            verbose: Print progress

        Returns:
            (metrics, detailed_results)
        """
        results = []
        start_time = time.time()

        for i, (context, true_next) in enumerate(test_data):
            if verbose and i % 100 == 0:
                print(f"Evaluating {i}/{len(test_data)}...")

            # Make prediction
            pred_start = time.time()
            predictions_dict = self.model.predict(context, top_k=top_k)
            pred_time = (time.time() - pred_start) * 1000  # ms

            # Convert dict to sorted list of (byte, probability) tuples
            if predictions_dict:
                predictions = sorted(
                    predictions_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_k]
            else:
                predictions = []

            # Analyze prediction
            if predictions:
                # Find rank of true next byte
                rank = None
                prob = 0.0

                # true_next is bytes, need to convert to int for comparison
                true_next_int = true_next[0] if len(true_next) > 0 else None

                for r, (pred_byte, pred_prob) in enumerate(predictions, 1):
                    if pred_byte == true_next_int:
                        rank = r
                        prob = pred_prob
                        break

                if rank is None:
                    # True byte not in top-k
                    rank = top_k + 1

                predicted = bytes([predictions[0][0]])
                top_prob = predictions[0][1]
                correct = (predicted == true_next)
            else:
                # No prediction available
                predicted = None
                top_prob = 0.0
                prob = 0.0
                rank = top_k + 1
                correct = False

            results.append(PredictionResult(
                context=context,
                true_next=true_next,
                predicted=predicted,
                probability=prob,
                rank=rank,
                correct=correct,
                time_ms=pred_time
            ))

        total_time = time.time() - start_time

        # Compute metrics
        metrics = self._compute_metrics(results, top_k, total_time)

        return metrics, results

    def _compute_metrics(
        self,
        results: List[PredictionResult],
        top_k: int,
        total_time: float
    ) -> EvaluationMetrics:
        """Compute aggregated metrics from results."""
        n = len(results)

        # Accuracy
        num_correct = sum(1 for r in results if r.correct)
        accuracy = num_correct / n if n > 0 else 0.0

        # Top-k accuracy
        top_k_acc = {}
        for k in [1, 3, 5, 10]:
            if k <= top_k:
                num_in_top_k = sum(1 for r in results if r.rank <= k)
                top_k_acc[k] = num_in_top_k / n if n > 0 else 0.0

        # Coverage
        num_covered = sum(1 for r in results if r.predicted is not None)
        coverage = num_covered / n if n > 0 else 0.0
        no_match_rate = 1.0 - coverage

        # Ranks
        ranks = [r.rank for r in results]
        mean_rank = sum(ranks) / len(ranks) if ranks else 0.0
        median_rank = sorted(ranks)[len(ranks) // 2] if ranks else 0.0

        # Perplexity (only for cases where we made a prediction)
        log_probs = []
        probs = []
        for r in results:
            if r.probability > 0:
                log_probs.append(math.log(r.probability))
                probs.append(r.probability)

        if log_probs:
            avg_log_prob = sum(log_probs) / len(log_probs)
            perplexity = math.exp(-avg_log_prob)
            mean_probability = sum(probs) / len(probs)
        else:
            perplexity = float('inf')
            mean_probability = 0.0

        # Timing
        times = [r.time_ms for r in results]
        mean_time = sum(times) / len(times) if times else 0.0
        median_time = sorted(times)[len(times) // 2] if times else 0.0

        return EvaluationMetrics(
            accuracy=accuracy,
            top_k_accuracy=top_k_acc,
            mean_rank=mean_rank,
            median_rank=median_rank,
            coverage=coverage,
            no_match_rate=no_match_rate,
            perplexity=perplexity,
            mean_probability=mean_probability,
            mean_time_ms=mean_time,
            median_time_ms=median_time,
            total_time_s=total_time,
            num_samples=n,
            num_correct=num_correct,
            num_covered=num_covered
        )


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for comparing models.
    """

    def __init__(self, corpus: bytes):
        """
        Initialize benchmark suite.

        Args:
            corpus: Training corpus
        """
        self.corpus = corpus

    def create_in_distribution_test(
        self,
        num_samples: int = 100,
        context_length: int = 20
    ) -> List[Tuple[bytes, bytes]]:
        """
        Create in-distribution test data from corpus.

        Samples random positions from the corpus.

        Args:
            num_samples: Number of test samples
            context_length: Length of context

        Returns:
            List of (context, next_byte) pairs
        """
        import random

        test_data = []
        max_pos = len(self.corpus) - context_length - 1

        for _ in range(num_samples):
            pos = random.randint(0, max_pos)
            context = self.corpus[pos:pos + context_length]
            next_byte = self.corpus[pos + context_length:pos + context_length + 1]
            test_data.append((context, next_byte))

        return test_data

    def create_ood_test(
        self,
        transformations: List[str],
        num_samples: int = 100,
        context_length: int = 20
    ) -> List[Tuple[bytes, bytes]]:
        """
        Create out-of-distribution test data.

        Applies transformations to corpus samples to create OOD data.

        Args:
            transformations: Types of transformations ('case', 'typo', 'synonym')
            num_samples: Number of test samples
            context_length: Length of context

        Returns:
            List of (transformed_context, next_byte) pairs
        """
        import random

        # First get in-distribution samples
        in_dist = self.create_in_distribution_test(num_samples, context_length)

        # Apply transformations
        test_data = []
        for context, next_byte in in_dist:
            transformed = context

            for transform_type in transformations:
                if transform_type == 'case':
                    transformed = self._apply_case_transform(transformed)
                elif transform_type == 'typo':
                    transformed = self._apply_typo_transform(transformed)
                elif transform_type == 'synonym':
                    transformed = self._apply_synonym_transform(transformed)

            test_data.append((transformed, next_byte))

        return test_data

    def _apply_case_transform(self, text: bytes) -> bytes:
        """Randomly flip case of letters."""
        import random
        result = []
        for byte in text:
            char = chr(byte)
            if random.random() < 0.3:  # 30% chance
                if char.isupper():
                    result.append(ord(char.lower()))
                elif char.islower():
                    result.append(ord(char.upper()))
                else:
                    result.append(byte)
            else:
                result.append(byte)
        return bytes(result)

    def _apply_typo_transform(self, text: bytes) -> bytes:
        """Randomly introduce typos."""
        import random
        result = list(text)

        # Introduce 1-2 typos
        num_typos = min(random.randint(1, 2), len(result) // 5)
        positions = random.sample(range(len(result)), num_typos)

        for pos in positions:
            # Simple character substitution
            if chr(result[pos]).isalpha():
                # Neighboring key on keyboard (simplified)
                result[pos] = ord(random.choice('abcdefghijklmnopqrstuvwxyz'))

        return bytes(result)

    def _apply_synonym_transform(self, text: bytes) -> bytes:
        """Replace words with synonyms (simple version)."""
        # Simple synonym mapping
        synonyms = {
            b'big': b'large',
            b'small': b'tiny',
            b'fast': b'quick',
            b'slow': b'sluggish',
            b'good': b'great',
            b'bad': b'poor',
            b'happy': b'glad',
            b'sad': b'unhappy',
        }

        result = text
        for original, replacement in synonyms.items():
            result = result.replace(original, replacement)

        return result

    def compare_models(
        self,
        models: Dict[str, Any],
        test_datasets: Dict[str, List[Tuple[bytes, bytes]]],
        top_k: int = 10,
        verbose: bool = True
    ) -> Dict[str, Dict[str, EvaluationMetrics]]:
        """
        Compare multiple models on multiple test datasets.

        Args:
            models: Dict of {model_name: model_instance}
            test_datasets: Dict of {dataset_name: test_data}
            top_k: Number of top predictions
            verbose: Print progress

        Returns:
            Dict of {model_name: {dataset_name: metrics}}
        """
        results = {}

        for model_name, model in models.items():
            if verbose:
                print(f"\nEvaluating {model_name}...")

            results[model_name] = {}
            evaluator = Evaluator(model, model_name)

            for dataset_name, test_data in test_datasets.items():
                if verbose:
                    print(f"  Dataset: {dataset_name}")

                metrics, _ = evaluator.evaluate(test_data, top_k=top_k, verbose=False)
                results[model_name][dataset_name] = metrics

                if verbose:
                    print(f"    Accuracy: {metrics.accuracy:.3f}")
                    print(f"    Coverage: {metrics.coverage:.3f}")
                    print(f"    Perplexity: {metrics.perplexity:.2f}")

        return results


def print_comparison_table(
    results: Dict[str, Dict[str, EvaluationMetrics]]
):
    """
    Print a formatted comparison table.

    Args:
        results: Output from BenchmarkSuite.compare_models()
    """
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)

    # Get all models and datasets
    models = list(results.keys())
    datasets = list(results[models[0]].keys())

    for dataset in datasets:
        print(f"\n{dataset}:")
        print("-" * 80)
        print(f"{'Model':<25} {'Acc':<8} {'Top-3':<8} {'Coverage':<10} {'PPL':<10} {'Time(ms)':<10}")
        print("-" * 80)

        for model in models:
            metrics = results[model][dataset]
            top3_acc = metrics.top_k_accuracy.get(3, 0.0)

            print(f"{model:<25} "
                  f"{metrics.accuracy:<8.3f} "
                  f"{top3_acc:<8.3f} "
                  f"{metrics.coverage:<10.3f} "
                  f"{metrics.perplexity:<10.2f} "
                  f"{metrics.mean_time_ms:<10.2f}")

    print("=" * 80)


def create_synthetic_corpus(size: int = 10000) -> bytes:
    """
    Create a synthetic corpus for testing.

    Args:
        size: Approximate size in bytes

    Returns:
        Synthetic corpus
    """
    import random

    # Common English words
    words = [
        b"the", b"quick", b"brown", b"fox", b"jumps", b"over", b"lazy", b"dog",
        b"cat", b"sits", b"on", b"mat", b"bird", b"flies", b"in", b"sky",
        b"fish", b"swims", b"water", b"tree", b"grows", b"forest", b"sun",
        b"shines", b"bright", b"moon", b"glows", b"night", b"star", b"twinkles",
    ]

    corpus_parts = []
    current_size = 0

    while current_size < size:
        # Random sentence
        sentence_length = random.randint(5, 15)
        sentence = b" ".join(random.choices(words, k=sentence_length)) + b". "
        corpus_parts.append(sentence)
        current_size += len(sentence)

    return b"".join(corpus_parts)
