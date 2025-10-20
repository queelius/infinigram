#!/usr/bin/env python3
"""
Recursive Context Transformation for Infinigram.

Enables OOD generalization through corpus-guided transformations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import numpy as np


class Transformer(ABC):
    """Base class for context transformers."""

    @abstractmethod
    def generate_transformations(
        self,
        context: bytes,
        suffix: bytes,
        corpus: bytes,
        match_positions: List[int]
    ) -> List[Tuple[bytes, str]]:
        """
        Generate possible transformations based on corpus matches.

        Args:
            context: Original input context
            suffix: Best matching suffix found
            corpus: Full corpus
            match_positions: Positions where suffix matches in corpus

        Returns:
            List of (transformed_context, transformation_description)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return transformer name."""
        pass


class SynonymTransformer(Transformer):
    """Replace words with synonyms found in corpus."""

    def __init__(self, use_wordnet: bool = True, min_similarity: float = 0.5):
        """
        Initialize synonym transformer.

        Args:
            use_wordnet: Whether to use WordNet for synonym detection
            min_similarity: Minimum similarity threshold for WordNet (0.0-1.0)
        """
        self.use_wordnet = use_wordnet
        self.min_similarity = min_similarity
        self.synonym_cache = {}

        # Try to import WordNet
        self.wordnet = None
        if use_wordnet:
            try:
                from nltk.corpus import wordnet as wn
                self.wordnet = wn
            except ImportError:
                # Fall back to placeholder if WordNet not available
                self.use_wordnet = False

    def generate_transformations(
        self,
        context: bytes,
        suffix: bytes,
        corpus: bytes,
        match_positions: List[int]
    ) -> List[Tuple[bytes, str]]:
        """
        Generate synonym-based transformations.

        Strategy:
        1. Look at what comes BEFORE suffix match in corpus
        2. Compare with what comes BEFORE suffix in context
        3. If words differ but are synonyms, generate transformation
        """
        transformations = []

        prefix_len = len(context) - len(suffix)
        if prefix_len == 0:
            return []  # No prefix to transform

        context_prefix = context[:prefix_len]

        # Track unique transformations to avoid duplicates
        seen_transformations = set()

        # Inspect corpus at each match position
        for pos in match_positions[:10]:  # Limit to avoid explosion
            if pos < prefix_len:
                continue

            # Extract corpus prefix (same length as context prefix)
            corpus_prefix = corpus[pos - prefix_len:pos]

            # Tokenize both prefixes
            context_words = self._tokenize(context_prefix)
            corpus_words = self._tokenize(corpus_prefix)

            # Need same number of words to compare
            if len(context_words) != len(corpus_words):
                continue

            # Find word differences
            for i, (ctx_word, corp_word) in enumerate(zip(context_words, corpus_words)):
                if ctx_word == corp_word:
                    continue  # Already matches

                # Check if they're synonyms
                if self._are_synonyms(ctx_word, corp_word):
                    # Generate transformed context
                    new_context = self._replace_word_in_context(
                        context, context_prefix, ctx_word, corp_word, position=i
                    )

                    # Create transformation description
                    ctx_word_str = ctx_word.decode('utf-8', errors='ignore')
                    corp_word_str = corp_word.decode('utf-8', errors='ignore')
                    transform_key = f"{ctx_word_str}→{corp_word_str}"

                    # Skip if we've already generated this transformation
                    if transform_key in seen_transformations:
                        continue

                    seen_transformations.add(transform_key)
                    transform_desc = f"synonym:{transform_key}"
                    transformations.append((new_context, transform_desc))

                    # Only generate one transformation per match position
                    break

        return transformations

    def _tokenize(self, text: bytes) -> List[bytes]:
        """Simple whitespace tokenization."""
        return text.split()

    def _are_synonyms(self, word1: bytes, word2: bytes) -> bool:
        """
        Check if two words are synonyms using WordNet.

        Uses multiple strategies:
        1. Check if words share synsets (exact synonyms)
        2. Check hypernym/hyponym relationships (cat/feline)
        3. Check path similarity in WordNet taxonomy
        """
        # Decode to strings
        try:
            str1 = word1.decode('utf-8').lower().strip()
            str2 = word2.decode('utf-8').lower().strip()
        except UnicodeDecodeError:
            return False

        # Empty words are not synonyms
        if not str1 or not str2:
            return False

        # Quick checks
        if str1 == str2:
            return True

        # Check cache
        cache_key = (str1, str2)
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]

        # Use WordNet if available
        if self.use_wordnet and self.wordnet:
            result = self._wordnet_similarity(str1, str2)
        else:
            # Fall back to exact match
            result = (str1 == str2)

        # Cache result
        self.synonym_cache[cache_key] = result
        return result

    def _wordnet_similarity(self, word1: str, word2: str) -> bool:
        """Check if two words are related using WordNet."""
        try:
            # Get synsets for both words (all parts of speech)
            synsets1 = self.wordnet.synsets(word1)
            synsets2 = self.wordnet.synsets(word2)

            # No synsets found for either word
            if not synsets1 or not synsets2:
                return False

            # Strategy 1: Check if they share any synsets (exact synonyms)
            set1 = set(synsets1)
            set2 = set(synsets2)
            if set1 & set2:
                return True

            # Strategy 2: Check hypernym/hyponym relationships (only for nouns)
            noun_synsets1 = [s for s in synsets1 if s.pos() == 'n']
            noun_synsets2 = [s for s in synsets2 if s.pos() == 'n']

            for s1 in noun_synsets1:
                for s2 in noun_synsets2:
                    # Check if s1 is a hyponym of s2 (s1 is a type of s2)
                    if s1 in s2.hyponyms():
                        return True
                    # Check if s2 is a hyponym of s1 (s2 is a type of s1)
                    if s2 in s1.hyponyms():
                        return True
                    # Check if they share a direct hypernym
                    if set(s1.hypernyms()) & set(s2.hypernyms()):
                        return True

            # Strategy 3: Check path similarity
            max_sim = 0.0
            for s1 in synsets1:
                for s2 in synsets2:
                    sim = s1.path_similarity(s2)
                    if sim:
                        max_sim = max(max_sim, sim)

            return max_sim >= self.min_similarity

        except Exception:
            # If anything goes wrong, fall back to False
            return False

    def _replace_word_in_context(self, context: bytes, context_prefix: bytes,
                                  old_word: bytes, new_word: bytes, position: int) -> bytes:
        """
        Replace word at given position in the context prefix.

        Args:
            context: Full context (prefix + suffix)
            context_prefix: The prefix part that needs word replacement
            old_word: Word to replace
            new_word: Replacement word
            position: Word index in the tokenized prefix

        Returns:
            New context with replaced word
        """
        # Get the suffix (part that matched)
        suffix = context[len(context_prefix):]

        # Check if prefix ends with whitespace
        prefix_ends_with_space = context_prefix.endswith(b' ')

        # Tokenize prefix and replace word
        words = context_prefix.split()
        if position < len(words):
            words[position] = new_word

        # Reconstruct prefix
        new_prefix = b' '.join(words)

        # Restore trailing space if original prefix had one
        if prefix_ends_with_space:
            new_prefix = new_prefix + b' '

        return new_prefix + suffix

    def get_name(self) -> str:
        return "synonym"


class EditDistanceTransformer(Transformer):
    """Fix typos using edit distance."""

    def __init__(self, max_distance: int = 2):
        """
        Initialize edit distance transformer.

        Args:
            max_distance: Maximum Levenshtein distance to consider
        """
        self.max_distance = max_distance

    def generate_transformations(
        self,
        context: bytes,
        suffix: bytes,
        corpus: bytes,
        match_positions: List[int]
    ) -> List[Tuple[bytes, str]]:
        """Generate typo-correction transformations."""
        transformations = []

        prefix_len = len(context) - len(suffix)
        if prefix_len == 0:
            return []

        context_prefix = context[:prefix_len]

        # Track unique transformations to avoid duplicates
        seen_transformations = set()

        for pos in match_positions[:10]:
            if pos < prefix_len:
                continue

            # Extract corpus prefix
            corpus_prefix = corpus[pos - prefix_len:pos]

            # Tokenize both prefixes
            context_words = self._tokenize(context_prefix)
            corpus_words = self._tokenize(corpus_prefix)

            # Need same number of words to compare
            if len(context_words) != len(corpus_words):
                continue

            # Find word differences with small edit distance
            for i, (ctx_word, corp_word) in enumerate(zip(context_words, corpus_words)):
                if ctx_word == corp_word:
                    continue

                dist = self._edit_distance(ctx_word, corp_word)
                if 0 < dist <= self.max_distance:
                    # Likely typo - fix it
                    new_context = self._replace_word_in_context(
                        context, context_prefix, ctx_word, corp_word, position=i
                    )

                    # Create transformation description
                    ctx_word_str = ctx_word.decode('utf-8', errors='ignore')
                    corp_word_str = corp_word.decode('utf-8', errors='ignore')
                    transform_key = f"{ctx_word_str}→{corp_word_str}"

                    # Skip if we've already generated this transformation
                    if transform_key in seen_transformations:
                        continue

                    seen_transformations.add(transform_key)
                    transform_desc = f"typo:{transform_key}"
                    transformations.append((new_context, transform_desc))

                    # Only generate one transformation per match position
                    break

        return transformations

    def _tokenize(self, text: bytes) -> List[bytes]:
        """Simple whitespace tokenization."""
        return text.split()

    def _edit_distance(self, word1: bytes, word2: bytes) -> int:
        """
        Compute Levenshtein distance.

        TODO: Use python-Levenshtein for speed
        """
        # Simple DP implementation
        s1, s2 = word1, word2
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def _replace_word_in_context(self, context: bytes, context_prefix: bytes,
                                  old_word: bytes, new_word: bytes, position: int) -> bytes:
        """
        Replace word at given position in the context prefix.

        Args:
            context: Full context (prefix + suffix)
            context_prefix: The prefix part that needs word replacement
            old_word: Word to replace
            new_word: Replacement word
            position: Word index in the tokenized prefix

        Returns:
            New context with replaced word
        """
        # Get the suffix (part that matched)
        suffix = context[len(context_prefix):]

        # Check if prefix ends with whitespace
        prefix_ends_with_space = context_prefix.endswith(b' ')

        # Tokenize prefix and replace word
        words = context_prefix.split()
        if position < len(words):
            words[position] = new_word

        # Reconstruct prefix
        new_prefix = b' '.join(words)

        # Restore trailing space if original prefix had one
        if prefix_ends_with_space:
            new_prefix = new_prefix + b' '

        return new_prefix + suffix

    def get_name(self) -> str:
        return "edit_distance"


class CaseNormalizer(Transformer):
    """Normalize case differences."""

    def generate_transformations(
        self,
        context: bytes,
        suffix: bytes,
        corpus: bytes,
        match_positions: List[int]
    ) -> List[Tuple[bytes, str]]:
        """Generate case normalization."""
        # Simple: just lowercase everything
        lowercased = context.lower()

        if lowercased != context:
            return [(lowercased, "case:lowercase")]

        return []

    def get_name(self) -> str:
        return "case"


class RecursiveInfinigram:
    """
    Infinigram with recursive context transformation.

    Enables OOD generalization by transforming input context based on
    patterns observed in the corpus, then re-querying suffix array.
    """

    def __init__(self, corpus: bytes, transformers: Optional[List[Transformer]] = None,
                 scorer: Optional['TransformationScorer'] = None):
        """
        Initialize recursive Infinigram.

        Args:
            corpus: Training corpus
            transformers: List of transformers to use (default: synonym, typo, case)
            scorer: Transformation scorer (default: uses default scorer)
        """
        self.corpus = corpus

        # Build suffix array
        from infinigram import Infinigram
        self.model = Infinigram(corpus)

        # Transformers
        if transformers is None:
            self.transformers = [
                SynonymTransformer(),
                EditDistanceTransformer(max_distance=2),
                CaseNormalizer(),
            ]
        else:
            self.transformers = transformers

        # Scorer
        if scorer is None:
            from infinigram.scoring import create_default_scorer
            self.scorer = create_default_scorer()
        else:
            self.scorer = scorer

        # Caches
        self._transformation_cache = {}
        self._match_cache = {}

    def predict(
        self,
        context: bytes,
        max_depth: int = 3,
        beam_width: int = 5,
        top_k: int = 50
    ) -> Dict[int, float]:
        """
        Predict next byte with recursive transformation.

        Args:
            context: Input context
            max_depth: Maximum recursion depth
            beam_width: Maximum candidates per level
            top_k: Return top K predictions

        Returns:
            Dictionary of {byte_value: probability}
        """
        # Generate all transformed contexts
        all_contexts = self._recursive_transform(
            context=context,
            depth=0,
            max_depth=max_depth,
            seen=set(),
            beam_width=beam_width
        )

        # For each context, get predictions
        weighted_predictions = []

        for transformed_context, transforms in all_contexts:
            # Query suffix array
            probs = self.model.predict(transformed_context, top_k=top_k)

            if probs:
                # Get match information for scoring
                suffix, positions = self._find_best_suffix_match(transformed_context)
                match_len = len(suffix)

                # Use scorer to compute weight
                weight = self.scorer.score(
                    context=context,
                    transformed_context=transformed_context,
                    transformations=transforms,
                    match_length=match_len,
                    match_positions=positions,
                    corpus_size=len(self.corpus)
                )

                weighted_predictions.append((probs, weight))

        # Combine predictions
        return self._combine_predictions(weighted_predictions)

    def _recursive_transform(
        self,
        context: bytes,
        depth: int,
        max_depth: int,
        seen: Set[bytes],
        beam_width: int
    ) -> List[Tuple[bytes, List[str]]]:
        """
        Recursively generate transformed contexts.

        Returns:
            List of (transformed_context, list_of_transformations)
        """
        # Base cases
        if depth >= max_depth:
            return [(context, [])]

        if context in seen:
            return []  # Cycle detected

        seen.add(context)

        # Start with original
        all_contexts = [(context, [])]

        # Find all suffix matches for current context
        all_matches = self._find_all_suffix_matches(context)

        if not all_matches:
            return all_contexts  # No matches, can't transform

        # Use the longest match for transformation generation
        suffix_len, positions = all_matches[0]
        suffix = context[-suffix_len:]

        # Generate transformations using each transformer
        candidates = []

        for transformer in self.transformers:
            transformations = transformer.generate_transformations(
                context=context,
                suffix=suffix,
                corpus=self.corpus,
                match_positions=positions
            )

            for new_context, transform_desc in transformations:
                if new_context != context:
                    # Score by match length
                    score = self._get_match_length(new_context)
                    candidates.append((score, new_context, transform_desc))

        # Keep top beam_width candidates
        candidates.sort(reverse=True, key=lambda x: x[0])
        top_candidates = candidates[:beam_width]

        # Recurse on top candidates
        for score, new_context, transform_desc in top_candidates:
            # Recursive call with copy of seen set (independent branches)
            deeper_contexts = self._recursive_transform(
                context=new_context,
                depth=depth + 1,
                max_depth=max_depth,
                seen=seen.copy(),
                beam_width=beam_width
            )

            for ctx, transforms in deeper_contexts:
                all_contexts.append((ctx, [transform_desc] + transforms))

        return all_contexts

    def _find_best_suffix_match(self, context: bytes) -> Tuple[bytes, List[int]]:
        """
        Find longest suffix match using suffix array.

        Returns:
            (suffix, positions) where suffix is the matched bytes
        """
        # Convert bytes to list of ints for Infinigram
        context_list = list(context)

        # Use Infinigram's new find_all_suffix_matches method
        matches = self.model.find_all_suffix_matches(context_list)

        if matches:
            # Return longest match (first in list)
            suffix_len, positions = matches[0]
            suffix = context[-suffix_len:]
            return suffix, positions

        return b'', []

    def _find_all_suffix_matches(self, context: bytes) -> List[Tuple[int, List[int]]]:
        """
        Find all suffix matches of varying lengths.

        Returns:
            List of (suffix_length, positions) tuples
        """
        context_list = list(context)
        return self.model.find_all_suffix_matches(context_list)

    def _get_match_length(self, context: bytes) -> int:
        """Get length of best suffix match."""
        suffix, positions = self._find_best_suffix_match(context)
        return len(suffix)

    def _combine_predictions(
        self,
        weighted_predictions: List[Tuple[Dict[int, float], float]]
    ) -> Dict[int, float]:
        """
        Combine predictions from multiple contexts.

        Args:
            weighted_predictions: List of (predictions, weight)

        Returns:
            Combined normalized predictions
        """
        combined = defaultdict(float)

        for probs, weight in weighted_predictions:
            for byte_val, prob in probs.items():
                combined[byte_val] += prob * weight

        # Normalize
        total = sum(combined.values())
        if total > 0:
            return {k: v / total for k, v in combined.items()}

        return {}

    def predict_with_explanation(
        self,
        context: bytes,
        max_depth: int = 3,
        beam_width: int = 5,
        top_k: int = 50
    ) -> Tuple[Dict[int, float], List[Dict]]:
        """
        Predict with explanations of transformations used.

        Returns:
            (predictions, explanations) where explanations is a list of dicts
        """
        # Generate all transformed contexts
        all_contexts = self._recursive_transform(
            context=context,
            depth=0,
            max_depth=max_depth,
            seen=set(),
            beam_width=beam_width
        )

        # Track explanations
        explanations = []
        weighted_predictions = []

        for transformed_context, transforms in all_contexts:
            # Query model
            probs = self.model.predict(transformed_context, top_k=top_k)

            if probs:
                # Get match information for scoring
                suffix, positions = self._find_best_suffix_match(transformed_context)
                match_len = len(suffix)

                # Use scorer to compute weight
                weight = self.scorer.score(
                    context=context,
                    transformed_context=transformed_context,
                    transformations=transforms,
                    match_length=match_len,
                    match_positions=positions,
                    corpus_size=len(self.corpus)
                )

                weighted_predictions.append((probs, weight))

                # Record explanation
                explanations.append({
                    'context': transformed_context,
                    'transformations': transforms,
                    'match_length': match_len,
                    'match_frequency': len(positions),
                    'weight': weight,
                    'predictions': probs
                })

        # Combine
        combined = self._combine_predictions(weighted_predictions)

        return combined, explanations
