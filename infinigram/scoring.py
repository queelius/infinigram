#!/usr/bin/env python3
"""
Transformation scoring for Infinigram.

Provides sophisticated scoring for weighted combination of predictions
from multiple transformed contexts. Used by predict_search() when performing
beam search over query transforms.
"""

from typing import List, Dict, Tuple
import math


class TransformationScorer:
    """
    Scores transformed contexts for weighted prediction combining.

    Considers multiple factors:
    - Match length (longer = better)
    - Match frequency (more occurrences = more confident)
    - Transformation depth (fewer transformations = better)
    - Transformation type (some transformers more reliable)
    """

    def __init__(
        self,
        match_length_weight: float = 0.4,
        match_frequency_weight: float = 0.2,
        transformation_weight: float = 0.3,
        depth_weight: float = 0.1,
        transformer_weights: Dict[str, float] = None
    ):
        """
        Initialize scorer with configurable weights.

        Args:
            match_length_weight: Weight for match length component (0-1)
            match_frequency_weight: Weight for match frequency component (0-1)
            transformation_weight: Weight for transformation quality component (0-1)
            depth_weight: Weight for transformation depth component (0-1)
            transformer_weights: Reliability weights for different transformers
                                Default: {'synonym': 0.9, 'typo': 0.95, 'case': 0.99}
        """
        # Normalize weights
        total = (match_length_weight + match_frequency_weight +
                transformation_weight + depth_weight)
        self.match_length_weight = match_length_weight / total
        self.match_frequency_weight = match_frequency_weight / total
        self.transformation_weight = transformation_weight / total
        self.depth_weight = depth_weight / total

        # Default transformer reliability weights
        if transformer_weights is None:
            transformer_weights = {
                'synonym': 0.85,     # Synonyms might not always preserve meaning
                'typo': 0.95,        # Typo corrections are usually reliable
                'case': 0.99,        # Case normalization is very safe
                'edit_distance': 0.95,  # Same as typo
            }
        self.transformer_weights = transformer_weights

    def score(
        self,
        context: bytes,
        transformed_context: bytes,
        transformations: List[str],
        match_length: int,
        match_positions: List[int],
        corpus_size: int
    ) -> float:
        """
        Score a transformed context.

        Args:
            context: Original context
            transformed_context: Context after transformations
            transformations: List of transformation descriptions applied
            match_length: Length of longest suffix match
            match_positions: Positions where suffix matches in corpus
            corpus_size: Total size of corpus

        Returns:
            Score in range [0, 1]
        """
        # Component 1: Match length score
        match_score = self._score_match_length(
            match_length,
            len(transformed_context)
        )

        # Component 2: Match frequency score
        frequency_score = self._score_match_frequency(
            match_positions,
            corpus_size
        )

        # Component 3: Transformation quality score
        transformation_score = self._score_transformations(transformations)

        # Component 4: Depth penalty
        depth_score = self._score_depth(len(transformations))

        # Weighted combination
        total_score = (
            self.match_length_weight * match_score +
            self.match_frequency_weight * frequency_score +
            self.transformation_weight * transformation_score +
            self.depth_weight * depth_score
        )

        return total_score

    def _score_match_length(self, match_length: int, context_length: int) -> float:
        """
        Score based on match length.

        Longer matches are better (more context = more confident).
        Returns value in [0, 1].
        """
        if context_length == 0:
            return 0.0

        # Normalized match length
        ratio = match_length / context_length

        # Apply sigmoid-like curve: f(x) = x^0.5
        # This gives diminishing returns for very long matches
        score = math.sqrt(ratio)

        return min(score, 1.0)

    def _score_match_frequency(self, match_positions: List[int], corpus_size: int) -> float:
        """
        Score based on how many times the pattern appears.

        More occurrences = more confident (pattern is common in corpus).
        Returns value in [0, 1].
        """
        if corpus_size == 0 or not match_positions:
            return 0.0

        num_matches = len(match_positions)

        # Logarithmic scaling: common patterns score higher,
        # but with diminishing returns
        # f(x) = log(x + 1) / log(101)  maps [0, 100] -> [0, 1]
        score = math.log(num_matches + 1) / math.log(101)

        return min(score, 1.0)

    def _score_transformations(self, transformations: List[str]) -> float:
        """
        Score based on transformation quality.

        Different transformers have different reliability.
        Returns value in [0, 1].
        """
        if not transformations:
            return 1.0  # No transformations = perfect (original context)

        # Multiply reliability scores of all transformations
        # This penalizes stacking multiple transformations
        cumulative_reliability = 1.0

        for transform in transformations:
            # Extract transformer type from description (e.g., "synonym:cat→feline")
            transformer_type = transform.split(':')[0] if ':' in transform else 'unknown'

            # Get reliability weight
            reliability = self.transformer_weights.get(transformer_type, 0.7)

            cumulative_reliability *= reliability

        return cumulative_reliability

    def _score_depth(self, num_transformations: int) -> float:
        """
        Score based on transformation depth.

        Fewer transformations = better (closer to original).
        Returns value in [0, 1].
        """
        if num_transformations == 0:
            return 1.0

        # Exponential decay: f(x) = e^(-x/3)
        # After 3 transformations, score ≈ 0.37
        # After 6 transformations, score ≈ 0.14
        score = math.exp(-num_transformations / 3.0)

        return score

    def score_batch(
        self,
        context: bytes,
        transformed_contexts: List[Tuple[bytes, List[str], int, List[int]]],
        corpus_size: int
    ) -> List[float]:
        """
        Score multiple transformed contexts.

        Args:
            context: Original context
            transformed_contexts: List of (transformed_context, transformations,
                                          match_length, match_positions)
            corpus_size: Total corpus size

        Returns:
            List of scores
        """
        scores = []
        for transformed_ctx, transforms, match_len, positions in transformed_contexts:
            score = self.score(
                context=context,
                transformed_context=transformed_ctx,
                transformations=transforms,
                match_length=match_len,
                match_positions=positions,
                corpus_size=corpus_size
            )
            scores.append(score)

        return scores


class AdaptiveScorer(TransformationScorer):
    """
    Adaptive scorer that adjusts weights based on observed performance.

    Can be used to learn optimal weights from data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.performance_history: List[Tuple[float, bool]] = []

    def record_performance(self, score: float, correct: bool):
        """
        Record whether a prediction with given score was correct.

        Args:
            score: The score assigned to the transformation
            correct: Whether the prediction was correct
        """
        self.performance_history.append((score, correct))

    def analyze_performance(self) -> Dict[str, float]:
        """
        Analyze performance across score ranges.

        Returns:
            Dictionary with accuracy statistics
        """
        if not self.performance_history:
            return {}

        # Bin scores into ranges
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_counts = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": {'correct': 0, 'total': 0}
                     for i in range(len(bins) - 1)}

        for score, correct in self.performance_history:
            # Find which bin
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i+1] or (i == len(bins) - 2 and score == 1.0):
                    bin_key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                    bin_counts[bin_key]['total'] += 1
                    if correct:
                        bin_counts[bin_key]['correct'] += 1
                    break

        # Calculate accuracy per bin
        results = {}
        for bin_key, counts in bin_counts.items():
            if counts['total'] > 0:
                accuracy = counts['correct'] / counts['total']
                results[bin_key] = {
                    'accuracy': accuracy,
                    'count': counts['total']
                }

        return results


def create_default_scorer() -> TransformationScorer:
    """Create scorer with default weights."""
    return TransformationScorer()


def create_conservative_scorer() -> TransformationScorer:
    """
    Create scorer that heavily prefers original context.

    Useful when corpus coverage is high.
    """
    return TransformationScorer(
        match_length_weight=0.5,
        match_frequency_weight=0.2,
        transformation_weight=0.1,
        depth_weight=0.2  # High depth penalty
    )


def create_aggressive_scorer() -> TransformationScorer:
    """
    Create scorer that is more willing to try transformations.

    Useful for OOD scenarios where corpus coverage is low.
    """
    return TransformationScorer(
        match_length_weight=0.3,
        match_frequency_weight=0.3,
        transformation_weight=0.3,
        depth_weight=0.1  # Low depth penalty
    )
