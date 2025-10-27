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
from infinigram.suffix_array import SuffixArray


class Infinigram:
    """
    Variable-length n-gram model using suffix arrays.

    Unlike traditional n-gram models with fixed order, Infinigram finds
    the longest matching suffix in the corpus and uses its continuations
    for prediction.

    Infinigram operates on byte sequences (0-255), making it compatible with
    UTF-8 text, arbitrary binary data, and any byte-level representation.

    Example:
        >>> # Byte-level corpus
        >>> corpus = [72, 101, 108, 108, 111]  # "Hello" in ASCII
        >>> model = Infinigram(corpus)
        >>> context = [72, 101, 108]  # "Hel"
        >>> probs = model.predict(context)
        >>> # Finds longest match and returns P(next_byte | context)

        >>> # UTF-8 text example
        >>> text = "Hello world"
        >>> corpus = list(text.encode('utf-8'))
        >>> model = Infinigram(corpus)
        >>> context = list("Hello ".encode('utf-8'))
        >>> probs = model.predict(context)
    """

    def __init__(self,
                 corpus: List[int],
                 max_length: Optional[int] = None,
                 min_count: int = 1):
        """
        Initialize Infinigram from corpus.

        Args:
            corpus: Byte sequence (list of integers in range 0-255)
            max_length: Maximum suffix length to consider (None = unlimited)
            min_count: Minimum frequency threshold for predictions

        Raises:
            ValueError: If corpus contains values outside byte range (0-255)
        """
        # Validate byte range
        if corpus:
            invalid_bytes = [b for b in corpus if not (0 <= b <= 255)]
            if invalid_bytes:
                raise ValueError(
                    f"Corpus must contain only bytes (0-255). "
                    f"Found {len(invalid_bytes)} invalid values: "
                    f"{invalid_bytes[:10]}{'...' if len(invalid_bytes) > 10 else ''}"
                )

        self.corpus = corpus
        self.n = len(corpus)
        self.max_length = max_length
        self.min_count = min_count

        # Build suffix array
        self.sa = SuffixArray(corpus)

        # Fixed vocabulary: all 256 possible bytes
        self.vocab = set(range(256))
        self.vocab_size = 256

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

        # Limit context length if max_length is set
        if self.max_length:
            context = context[-self.max_length:]

        # Use suffix array's find_longest_suffix method
        return self.sa.find_longest_suffix(context)

    def find_all_suffix_matches(self, context: List[int]) -> List[Tuple[int, List[int]]]:
        """
        Find all suffix matches of varying lengths.

        Args:
            context: Token sequence

        Returns:
            List of (suffix_length, positions) tuples, sorted by decreasing length
        """
        if not context:
            return []

        matches = []

        # Limit context length if max_length is set
        max_len = min(len(context), self.max_length) if self.max_length else len(context)

        # Try all suffix lengths from longest to shortest
        for suffix_len in range(max_len, 0, -1):
            suffix = context[-suffix_len:]
            positions = self.sa.search(suffix)

            if positions:
                matches.append((suffix_len, positions))

        return matches

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
            # No match - fall back to unigram distribution
            return Counter(self.corpus)

        # Get all positions where the suffix occurs
        suffix = context[-length:]
        positions = self.sa.search(suffix)

        # Count continuations
        continuations = defaultdict(int)
        for pos in positions:
            next_pos = pos + length
            if next_pos < self.n:  # Check bounds
                next_token = self.corpus[next_pos]
                continuations[next_token] += 1

        return dict(continuations)

    def predict(self, context: List[int], top_k: int = 50, smoothing: float = 0.0) -> Dict[int, float]:
        """
        Predict next token probabilities.

        Args:
            context: Token sequence
            top_k: Return only top k predictions
            smoothing: Smoothing parameter for unseen tokens (default: 0.0)

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
        smoothed_total = total + smoothing * self.vocab_size

        probs = {}
        for token, count in filtered.items():
            probs[token] = (count + smoothing) / smoothed_total

        # Add smoothing for unseen tokens in vocab
        unseen_prob = smoothing / smoothed_total
        for token in self.vocab:
            if token not in probs:
                probs[token] = unseen_prob

        # Sort and return top k
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return dict(sorted_probs[:top_k])

    def predict_weighted(
        self,
        context: List[int],
        min_length: int = 1,
        max_length: Optional[int] = None,
        weight_fn: Optional[callable] = None,
        top_k: int = 50,
        smoothing: float = 0.0
    ) -> Dict[int, float]:
        """
        Predict using weighted combination of multiple suffix lengths.

        Instead of using only the longest matching suffix, this method combines
        predictions from all suffix lengths between min_length and max_length,
        weighted according to the weight function.

        Args:
            context: Token sequence
            min_length: Minimum suffix length to consider (default 1)
            max_length: Maximum suffix length (None = use self.max_length)
            weight_fn: Function mapping suffix_length -> weight
                      Default: lambda k: k (linear weighting)
            top_k: Return top k predictions (default 50)
            smoothing: Smoothing parameter for unseen tokens (default: 0.0)

        Returns:
            Dict mapping token -> probability

        Example:
            >>> from infinigram.weighting import quadratic_weight
            >>> model = Infinigram([1, 2, 3, 4, 2, 3, 5])
            >>> probs = model.predict_weighted(
            ...     [2, 3],
            ...     min_length=1,
            ...     max_length=3,
            ...     weight_fn=quadratic_weight
            ... )
        """
        # Default weight function
        if weight_fn is None:
            weight_fn = lambda k: float(k)  # Linear weighting

        # Determine max length
        if max_length is None:
            max_length = self.max_length if self.max_length else len(context)

        # Limit context length
        if self.max_length:
            context = context[-self.max_length:]

        # Collect weighted counts from all suffix lengths
        weighted_counts = defaultdict(float)
        total_weight = 0.0

        for length in range(min_length, min(max_length + 1, len(context) + 1)):
            # Get suffix of this length
            suffix = context[-length:] if length <= len(context) else context

            # Find all occurrences
            positions = self.sa.search(suffix)

            if not positions:
                continue  # No matches at this length

            # Get weight for this length
            weight = weight_fn(length)
            total_weight += weight

            # Count continuations
            for pos in positions:
                next_pos = pos + length
                if next_pos < self.n:
                    next_token = self.corpus[next_pos]
                    weighted_counts[next_token] += weight

        # If no matches at any length, fall back to smoothed uniform
        if total_weight == 0.0 or not weighted_counts:
            # Uniform over vocabulary with smoothing
            probs = {token: 1.0 / self.vocab_size for token in self.vocab}
        else:
            # Normalize weighted counts to probabilities
            total_count = sum(weighted_counts.values())

            # Apply smoothing
            smoothed_total = total_count + smoothing * self.vocab_size

            probs = {}
            for token, count in weighted_counts.items():
                probs[token] = (count + smoothing) / smoothed_total

            # Add smoothing for unseen tokens
            unseen_prob = smoothing / smoothed_total
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
        positions = self.sa.search(suffix)
        num_matches = len(positions)

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
            new_tokens: Byte sequence to add (0-255)

        Raises:
            ValueError: If new_tokens contains values outside byte range (0-255)
        """
        # Validate byte range
        if new_tokens:
            invalid_bytes = [b for b in new_tokens if not (0 <= b <= 255)]
            if invalid_bytes:
                raise ValueError(
                    f"New tokens must contain only bytes (0-255). "
                    f"Found {len(invalid_bytes)} invalid values: "
                    f"{invalid_bytes[:10]}{'...' if len(invalid_bytes) > 10 else ''}"
                )

        # Extend corpus
        self.corpus.extend(new_tokens)
        self.n = len(self.corpus)

        # Vocabulary is always fixed at 256 bytes (no need to update)

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
