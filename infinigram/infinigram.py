#!/usr/bin/env python3
"""
Infinigram: Variable-length n-gram language model using suffix arrays.

This module provides the core Infinigram implementation that supports
variable-length context matching, enabling efficient queries for the
longest matching suffix and its continuation probabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from abc import ABC, abstractmethod


class LanguageModel(ABC):
    """Base interface for all language models in LangCalc."""

    @abstractmethod
    def predict(self, context: List[int], top_k: int = 50) -> Dict[int, float]:
        """Return probability distribution over next tokens."""
        pass


class SuffixArray:
    """
    Efficient suffix array for O(m log n) pattern matching.

    This is a simplified implementation optimized for Infinigram queries.
    For full implementation, see langcalc.data.suffix_array.
    """

    def __init__(self, corpus: List[int]):
        """
        Build suffix array from corpus.

        Args:
            corpus: Token sequence
        """
        self.corpus = corpus
        self.n = len(corpus)

        # Build suffix array: indices sorted by suffix lexicographic order
        self.sa = self._build_suffix_array()

    def _build_suffix_array(self) -> List[int]:
        """Build suffix array in O(n log n) time."""
        # Create list of (suffix, index) tuples
        suffixes = [(self.corpus[i:], i) for i in range(self.n)]

        # Sort by suffix
        suffixes.sort(key=lambda x: x[0])

        # Extract indices
        return [idx for _, idx in suffixes]

    def _compare_suffix(self, idx: int, pattern: List[int]) -> int:
        """
        Compare pattern with suffix starting at idx.

        Returns:
            -1 if pattern < suffix
             0 if pattern == suffix (up to pattern length)
             1 if pattern > suffix
        """
        m = len(pattern)
        for i in range(m):
            if idx + i >= self.n:
                return 1  # Pattern extends beyond corpus

            if pattern[i] < self.corpus[idx + i]:
                return -1
            elif pattern[i] > self.corpus[idx + i]:
                return 1

        return 0  # Match

    def find_range(self, pattern: List[int]) -> Tuple[int, int]:
        """
        Find range [left, right) of suffixes matching pattern.

        Args:
            pattern: Token sequence to search for

        Returns:
            (left, right) indices in suffix array, or (0, 0) if not found
        """
        if not pattern:
            return (0, 0)

        # Count all occurrences by linear scan (for now)
        # TODO: Optimize with binary search
        matches = []
        for i in range(self.n):
            idx = self.sa[i]
            # Check if pattern matches at this suffix
            match = True
            for j in range(len(pattern)):
                if idx + j >= self.n or self.corpus[idx + j] != pattern[j]:
                    match = False
                    break
            if match:
                matches.append(i)

        if not matches:
            return (0, 0)

        return (matches[0], matches[-1] + 1)


class Infinigram(LanguageModel):
    """
    Variable-length n-gram model using suffix arrays.

    Unlike traditional n-gram models with fixed order, Infinigram finds
    the longest matching suffix in the corpus and uses its continuations
    for prediction.

    Example:
        >>> corpus = [1, 2, 3, 4, 2, 3, 5]
        >>> model = Infinigram(corpus)
        >>> context = [1, 2, 3]
        >>> probs = model.predict(context)
        >>> # Finds longest match [2, 3] and returns P(next | [2, 3])
    """

    def __init__(self,
                 corpus: List[int],
                 max_length: Optional[int] = None,
                 min_count: int = 1,
                 smoothing: float = 0.01):
        """
        Initialize Infinigram from corpus.

        Args:
            corpus: Token sequence (list of integer token IDs)
            max_length: Maximum suffix length to consider (None = unlimited)
            min_count: Minimum frequency threshold for predictions
            smoothing: Smoothing parameter for unseen tokens
        """
        self.corpus = corpus
        self.n = len(corpus)
        self.max_length = max_length
        self.min_count = min_count
        self.smoothing = smoothing

        # Build suffix array
        self.sa = SuffixArray(corpus)

        # Compute vocabulary
        self.vocab = set(corpus)
        self.vocab_size = len(self.vocab)

    def longest_suffix(self, context: List[int]) -> Tuple[int, int]:
        """
        Find longest matching suffix in corpus.

        Args:
            context: Token sequence

        Returns:
            (position, length) of longest match, or (-1, 0) if no match
        """
        if not context:
            return (-1, 0)

        # Try increasingly shorter suffixes
        max_len = min(len(context), self.max_length if self.max_length else len(context))

        for length in range(max_len, 0, -1):
            suffix = context[-length:]
            start, end = self.sa.find_range(suffix)

            if end > start:  # Found matches
                # Return position of first match
                return (self.sa.sa[start], length)

        return (-1, 0)  # No match found

    def continuations(self, context: List[int]) -> Dict[int, int]:
        """
        Get continuation counts for longest matching suffix.

        Args:
            context: Token sequence

        Returns:
            Dict mapping next_token -> count
        """
        if not context:
            # Return unigram distribution
            return Counter(self.corpus)

        # Find longest matching suffix
        _, length = self.longest_suffix(context)

        if length == 0:
            # No match - return smoothed uniform
            return {token: 1 for token in self.vocab}

        # Get range of matching suffixes
        suffix = context[-length:]
        start, end = self.sa.find_range(suffix)

        # Count continuations
        continuations = defaultdict(int)
        for i in range(start, end):
            pos = self.sa.sa[i] + length
            if pos < self.n:  # Check bounds
                next_token = self.corpus[pos]
                continuations[next_token] += 1

        return dict(continuations)

    def predict(self, context: List[int], top_k: int = 50) -> Dict[int, float]:
        """
        Predict next token probabilities.

        Args:
            context: Token sequence
            top_k: Return only top k predictions

        Returns:
            Dict mapping token -> probability
        """
        # Get continuation counts
        counts = self.continuations(context)

        if not counts:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / self.vocab_size
            return {token: uniform_prob for token in list(self.vocab)[:top_k]}

        # Filter by min_count
        filtered = {token: count for token, count in counts.items()
                   if count >= self.min_count}

        if not filtered:
            filtered = counts  # Use all if filtering removes everything

        # Compute probabilities with smoothing
        total = sum(filtered.values())
        smoothed_total = total + self.smoothing * self.vocab_size

        probs = {}
        for token, count in filtered.items():
            probs[token] = (count + self.smoothing) / smoothed_total

        # Add smoothing for unseen tokens in vocab
        unseen_prob = self.smoothing / smoothed_total
        for token in self.vocab:
            if token not in probs:
                probs[token] = unseen_prob

        # Sort and return top k
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return dict(sorted_probs[:top_k])

    def confidence(self, context: List[int]) -> float:
        """
        Confidence score for prediction.

        Based on:
        - Length of matching suffix (longer = more confident)
        - Frequency of match (more occurrences = more confident)

        Args:
            context: Token sequence

        Returns:
            Confidence score in [0, 1]
        """
        pos, length = self.longest_suffix(context)

        if length == 0:
            return 0.0  # No match

        # Get number of matches
        suffix = context[-length:]
        start, end = self.sa.find_range(suffix)
        num_matches = end - start

        # Confidence based on match length and frequency
        length_score = min(length / 10.0, 1.0)  # Normalize by 10 tokens
        freq_score = min(num_matches / 100.0, 1.0)  # Normalize by 100 occurrences

        # Combine scores
        confidence = 0.7 * length_score + 0.3 * freq_score
        return confidence

    def update(self, new_tokens: List[int]):
        """
        Dynamically add new tokens to corpus.

        NOTE: This is a simple implementation that rebuilds the suffix array.
        For incremental updates, use IncrementalInfinigram.

        Args:
            new_tokens: Token sequence to add
        """
        # Extend corpus
        self.corpus.extend(new_tokens)
        self.n = len(self.corpus)

        # Update vocabulary
        self.vocab.update(new_tokens)
        self.vocab_size = len(self.vocab)

        # Rebuild suffix array (TODO: make incremental)
        self.sa = SuffixArray(self.corpus)

    def __repr__(self) -> str:
        return f"Infinigram(n={self.n}, vocab_size={self.vocab_size}, max_length={self.max_length})"


class IncrementalInfinigram(Infinigram):
    """
    Infinigram with efficient incremental updates.

    Uses an incremental suffix array that supports O(k log n) updates
    for k new tokens, avoiding full rebuild.

    TODO: Implement incremental suffix array in langcalc.data.incremental
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Use IncrementalSuffixArray instead of SuffixArray
        raise NotImplementedError("IncrementalInfinigram requires IncrementalSuffixArray")


# Convenience function
def create_infinigram(corpus: List[int], **kwargs) -> Infinigram:
    """
    Create an Infinigram model from corpus.

    Args:
        corpus: Token sequence
        **kwargs: Additional arguments passed to Infinigram

    Returns:
        Infinigram instance

    Example:
        >>> corpus = tokenize(load_wikipedia())
        >>> model = create_infinigram(corpus, max_length=50)
    """
    return Infinigram(corpus, **kwargs)
