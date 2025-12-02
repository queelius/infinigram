#!/usr/bin/env python3
"""
Infinigram: Variable-length n-gram language model using suffix arrays.

All models use mmap-backed storage for consistent behavior from 1KB to 100GB+.
Supports both:
- In-memory API: Infinigram(corpus) - creates temporary mmap model
- Persistent API: Infinigram.build(corpus, path) / Infinigram.load(path)
"""

import json
import math
import tempfile
import shutil
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

from infinigram.suffix_array import MmapSuffixArray, ChunkedMmapSuffixArray

# Default models directory
DEFAULT_MODELS_DIR = Path.home() / ".infinigram" / "models"


class Infinigram:
    """
    Variable-length n-gram model using mmap-backed suffix arrays.

    Unlike traditional n-gram models with fixed order, Infinigram finds
    the longest matching suffix in the corpus and uses its continuations
    for prediction.

    All models are disk-backed using memory-mapped files, enabling:
    - Consistent API for any corpus size (1KB to 100GB+)
    - Instant loading (mmap, not RAM)
    - Automatic persistence
    - Minimal memory footprint during queries

    Supports runtime query transforms:
        - transforms=['lowercase', 'strip'] - apply sequentially to query
        - search=['typo', 'synonym'] - beam search over alternatives

    Usage:
        # In-memory style (creates temporary mmap model)
        >>> model = Infinigram([1, 2, 3, 1, 2, 4])
        >>> probs = model.predict([1, 2])

        # With default transforms
        >>> model = Infinigram(corpus, default_transforms=['lowercase'])
        >>> probs = model.predict("THE CAT")  # matches "the cat"

        # Persistent model
        >>> model = Infinigram.build("The cat sat", "my_model")
        >>> model = Infinigram.load("my_model")
    """

    # Threshold for using chunked index (5GB)
    CHUNK_THRESHOLD = 5 * 1_000_000_000

    # Available runtime query transforms
    QUERY_TRANSFORMS = {
        'lowercase': lambda b: b.lower(),
        'uppercase': lambda b: b.upper(),
        'casefold': lambda b: b.decode('utf-8', errors='replace').casefold().encode('utf-8'),
        'strip': lambda b: b.strip(),
        'normalize_whitespace': lambda b: b' '.join(b.split()),
    }

    def __init__(
        self,
        corpus_or_path: Union[List[int], bytes, str, Path],
        max_length: Optional[int] = None,
        min_count: int = 1,
        default_transforms: Optional[List[str]] = None,
        default_search: Optional[List[str]] = None,
        default_max_depth: int = 2,
        default_beam_width: int = 3,
        _from_path: bool = False
    ):
        """
        Initialize Infinigram model.

        Args:
            corpus_or_path: Either:
                - List[int]: byte values (0-255) - creates temp model
                - bytes: byte sequence - creates temp model
                - str/Path: if _from_path=True, path to existing model dir
                          if _from_path=False, string corpus to build temp model
            max_length: Maximum suffix length to consider (None = unlimited)
            min_count: Minimum frequency threshold for predictions
            default_transforms: Transforms to apply by default (e.g., ['lowercase'])
            default_search: Search transforms for fuzzy matching (e.g., ['typo'])
            default_max_depth: Default depth for search mode
            default_beam_width: Default beam width for search mode
            _from_path: Internal flag - True if loading from existing model
        """
        self.max_length = max_length
        self.min_count = min_count
        self._temp_dir = None

        # Transform defaults
        self.default_transforms = default_transforms or []
        self.default_search = default_search or []
        self.default_max_depth = default_max_depth
        self.default_beam_width = default_beam_width

        # Fixed vocabulary: all 256 possible bytes
        self.vocab = set(range(256))
        self.vocab_size = 256

        if _from_path:
            # Loading from existing model directory
            self._load_from_path(corpus_or_path)
        elif isinstance(corpus_or_path, (list, bytes)):
            # In-memory corpus - create temp mmap model
            self._build_from_corpus(corpus_or_path)
        elif isinstance(corpus_or_path, str):
            # Check if it's a path to existing model or a string corpus
            path = Path(corpus_or_path)
            if path.exists() and (path / "meta.json").exists():
                # It's an existing model directory
                self._load_from_path(corpus_or_path)
            else:
                # It's a string corpus - encode and build
                self._build_from_corpus(corpus_or_path.encode('utf-8'))
        elif isinstance(corpus_or_path, Path):
            self._load_from_path(corpus_or_path)
        else:
            raise ValueError(
                f"Invalid argument type: {type(corpus_or_path)}. "
                f"Expected List[int], bytes, str, or Path."
            )

    def _validate_corpus(self, corpus: Union[List[int], bytes]) -> bytes:
        """Validate and convert corpus to bytes."""
        if isinstance(corpus, bytes):
            return corpus

        # Validate byte values
        for i, val in enumerate(corpus):
            if not isinstance(val, int) or val < 0 or val > 255:
                raise ValueError(
                    f"Corpus must contain only bytes (0-255), "
                    f"but found {val} at position {i}"
                )
        return bytes(corpus)

    def _build_from_corpus(self, corpus: Union[List[int], bytes]):
        """Build a temporary mmap model from in-memory corpus."""
        import pydivsufsort

        corpus_bytes = self._validate_corpus(corpus)

        # Create temporary directory (will be cleaned up on close)
        self._temp_dir = tempfile.mkdtemp(prefix="infinigram_")
        self.model_path = Path(self._temp_dir)

        # Write corpus
        corpus_file = self.model_path / "corpus.bin"
        with open(corpus_file, 'wb') as f:
            f.write(corpus_bytes)

        # Build suffix array (use int64 to support large corpora >2GB)
        sa_array = pydivsufsort.divsufsort(corpus_bytes).astype(np.int64)
        np.save(self.model_path / "suffix_array.npy", sa_array)

        # Write metadata
        self.meta = {
            'n': len(corpus_bytes),
            'corpus_size': len(corpus_bytes),
            'max_length': self.max_length,
            'min_count': self.min_count,
        }
        with open(self.model_path / "meta.json", 'w') as f:
            json.dump(self.meta, f, indent=2)

        # Load the mmap suffix array
        self._sa = MmapSuffixArray(str(self.model_path))
        self._is_chunked = False
        self.n = self._sa.n

    def _load_from_path(self, model_path: Union[str, Path]):
        """Load model from existing directory."""
        self.model_path = Path(model_path)

        # Load metadata
        meta_file = self.model_path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Use Infinigram.build() to create a new model."
            )

        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        # Determine if chunked or regular
        self._is_chunked = 'num_chunks' in self.meta

        # Load the appropriate suffix array
        if self._is_chunked:
            self._sa = ChunkedMmapSuffixArray(str(self.model_path))
        else:
            self._sa = MmapSuffixArray(str(self.model_path))

        self.n = self._sa.n

    @classmethod
    def build(
        cls,
        corpus: Union[str, bytes, Path, List[int]],
        model_path: Union[str, Path],
        projections: Optional[List[str]] = None,
        chunk_size_gb: float = 5.0,
        max_length: Optional[int] = None,
        min_count: int = 1,
        verbose: bool = True
    ) -> 'Infinigram':
        """
        Build a new persistent Infinigram model from corpus.

        Args:
            corpus: Text string, bytes, list of bytes, or path to corpus file
            model_path: Directory to save model files
            projections: List of projection names to apply (e.g., ['lowercase'])
            chunk_size_gb: Chunk size for large corpora (default 5GB)
            max_length: Maximum suffix length for predictions
            min_count: Minimum frequency threshold
            verbose: Print progress messages

        Returns:
            Infinigram model ready for queries

        Example:
            >>> model = Infinigram.build("Hello world", "hello_model")
            >>> model = Infinigram.build(Path("corpus.txt"), "corpus_model")
        """
        import pydivsufsort

        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine corpus source and convert to bytes
        if isinstance(corpus, list):
            # List of byte values - validate and convert
            for i, val in enumerate(corpus):
                if not isinstance(val, int) or val < 0 or val > 255:
                    raise ValueError(
                        f"Corpus must contain only bytes (0-255), "
                        f"but found {val} at position {i}"
                    )
            corpus_bytes = bytes(corpus)
            corpus_size = len(corpus_bytes)

            # Apply projections if specified
            if projections:
                corpus_bytes = cls._apply_projections(corpus_bytes, projections)
                corpus_size = len(corpus_bytes)

            # Write corpus to file
            corpus_file = model_dir / "corpus.bin"
            with open(corpus_file, 'wb') as f:
                f.write(corpus_bytes)
            corpus_path = corpus_file

        elif isinstance(corpus, (str, bytes)) and not Path(corpus).exists():
            # In-memory corpus (string or bytes, not a file path)
            if isinstance(corpus, str):
                corpus_bytes = corpus.encode('utf-8')
            else:
                corpus_bytes = corpus

            # Apply projections if specified
            if projections:
                corpus_bytes = cls._apply_projections(corpus_bytes, projections)

            corpus_size = len(corpus_bytes)

            # Write corpus to file
            corpus_file = model_dir / "corpus.bin"
            with open(corpus_file, 'wb') as f:
                f.write(corpus_bytes)
            corpus_path = corpus_file

        elif isinstance(corpus, (str, Path)) and Path(corpus).exists():
            # File path
            corpus_path = Path(corpus)
            corpus_size = corpus_path.stat().st_size

            # Copy corpus to model directory
            dest_corpus = model_dir / "corpus.bin"
            if corpus_path != dest_corpus:
                if verbose:
                    print(f"Copying corpus to {dest_corpus}...")
                shutil.copy(corpus_path, dest_corpus)
            corpus_path = dest_corpus

        else:
            raise ValueError(f"Invalid corpus: {corpus}")

        if verbose:
            print(f"Building model at {model_dir}")
            print(f"Corpus size: {corpus_size:,} bytes ({corpus_size/1e9:.2f} GB)")

        # Build suffix array
        start = time.time()

        if corpus_size > cls.CHUNK_THRESHOLD:
            # Use chunked index for large corpora
            if verbose:
                print(f"Using chunked index (threshold: {cls.CHUNK_THRESHOLD/1e9:.1f} GB)")
            sa = ChunkedMmapSuffixArray.build(
                str(corpus_path),
                str(model_dir),
                chunk_size_gb=chunk_size_gb
            )
            is_chunked = True
        else:
            # Use regular mmap index
            if verbose:
                print("Building suffix array...")

            # Load corpus
            with open(corpus_path, 'rb') as f:
                corpus_bytes = f.read()

            # Build suffix array using pydivsufsort
            # Use int64 to support corpora >2GB
            sa_array = pydivsufsort.divsufsort(corpus_bytes).astype(np.int64)

            # Save suffix array
            np.save(model_dir / "suffix_array.npy", sa_array)

            # Save metadata
            meta = {
                'n': len(corpus_bytes),
                'corpus_size': corpus_size,
                'projections': projections or [],
                'max_length': max_length,
                'min_count': min_count,
            }
            with open(model_dir / "meta.json", 'w') as f:
                json.dump(meta, f, indent=2)

            is_chunked = False

        elapsed = time.time() - start
        if verbose:
            print(f"Built in {elapsed:.1f}s ({corpus_size/elapsed/1e6:.1f} MB/sec)")

        # If chunked, update meta with additional fields
        if is_chunked:
            meta_file = model_dir / "meta.json"
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            meta['projections'] = projections or []
            meta['max_length'] = max_length
            meta['min_count'] = min_count
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

        return cls(str(model_dir), max_length=max_length, min_count=min_count, _from_path=True)

    @classmethod
    def load(cls, model_path: Union[str, Path], **kwargs) -> 'Infinigram':
        """
        Load an existing Infinigram model.

        Args:
            model_path: Path to model directory, or model name (looks in ~/.infinigram/models/)
            **kwargs: Override model settings (max_length, min_count)

        Returns:
            Infinigram model

        Example:
            >>> model = Infinigram.load("wikipedia")  # loads ~/.infinigram/models/wikipedia
            >>> model = Infinigram.load("/data/my_model")  # absolute path
        """
        path = Path(model_path)

        # If not absolute, check default models directory
        if not path.is_absolute() and not path.exists():
            default_path = DEFAULT_MODELS_DIR / model_path
            if default_path.exists():
                path = default_path

        return cls(str(path), _from_path=True, **kwargs)

    @staticmethod
    def _apply_projections(corpus: bytes, projections: List[str]) -> bytes:
        """Apply text projections/augmentations to corpus."""
        text = corpus.decode('utf-8', errors='replace')

        for proj in projections:
            if proj == 'lowercase':
                text = text.lower()
            elif proj == 'uppercase':
                text = text.upper()
            elif proj == 'strip':
                text = text.strip()
            # Add more projections as needed

        return text.encode('utf-8')

    # === Query Transform Methods ===

    @classmethod
    def list_transforms(cls) -> List[str]:
        """List available query transforms."""
        return list(cls.QUERY_TRANSFORMS.keys())

    def _apply_transforms(
        self,
        query: bytes,
        transforms: Optional[List[str]] = None
    ) -> bytes:
        """
        Apply transforms sequentially to a query.

        Args:
            query: Query bytes
            transforms: List of transform names, or None to use defaults

        Returns:
            Transformed query bytes
        """
        # Use defaults if not specified
        if transforms is None:
            transforms = self.default_transforms

        # Apply each transform in sequence
        for transform in transforms:
            if transform not in self.QUERY_TRANSFORMS:
                raise ValueError(
                    f"Unknown transform '{transform}'. "
                    f"Available: {list(self.QUERY_TRANSFORMS.keys())}"
                )
            query = self.QUERY_TRANSFORMS[transform](query)

        return query

    def _normalize_query(
        self,
        query: Union[bytes, List[int], str],
        transforms: Optional[List[str]] = None
    ) -> bytes:
        """
        Normalize query to bytes and apply transforms.

        Args:
            query: Query in any supported format
            transforms: Transforms to apply (None = use defaults)

        Returns:
            Normalized and transformed query bytes
        """
        # Convert to bytes
        if isinstance(query, str):
            query = query.encode('utf-8')
        elif isinstance(query, list):
            query = bytes(query)

        # Apply transforms
        return self._apply_transforms(query, transforms)

    # === Query Methods ===

    def search(
        self,
        pattern: Union[bytes, List[int], str],
        transforms: Optional[List[str]] = None
    ) -> List[int]:
        """
        Search for all occurrences of pattern.

        Args:
            pattern: Bytes, list of byte values, or string
            transforms: Transforms to apply to pattern (None = use defaults,
                       [] = no transforms)

        Returns:
            List of positions where pattern occurs
        """
        pattern = self._normalize_query(pattern, transforms)

        if self._is_chunked:
            # ChunkedMmapSuffixArray returns (chunk_idx, pos) tuples
            results = self._sa.search(pattern)
            return results
        else:
            return self._sa.search(pattern)

    def count(
        self,
        pattern: Union[bytes, List[int], str],
        transforms: Optional[List[str]] = None
    ) -> int:
        """
        Count occurrences of pattern.

        Args:
            pattern: Bytes, list of byte values, or string
            transforms: Transforms to apply to pattern (None = use defaults,
                       [] = no transforms)

        Returns:
            Number of occurrences
        """
        pattern = self._normalize_query(pattern, transforms)
        return self._sa.count(pattern)

    def get_byte(self, position: int, chunk_idx: Optional[int] = None) -> int:
        """
        Get byte at a specific position.

        Args:
            position: Position in corpus (or chunk if chunked)
            chunk_idx: Chunk index (required for chunked models)

        Returns:
            Byte value at position
        """
        if self._is_chunked:
            if chunk_idx is None:
                raise ValueError("chunk_idx required for chunked models")
            return self._sa.chunks[chunk_idx]._corpus_mmap[position]
        else:
            return self._sa._corpus_mmap[position]

    def get_chunk_size(self, chunk_idx: int) -> int:
        """Get size of a specific chunk (for chunked models)."""
        if not self._is_chunked:
            return self.n
        return self._sa.chunks[chunk_idx].n

    def longest_suffix(
        self,
        context: Union[bytes, List[int], str],
        transforms: Optional[List[str]] = None
    ) -> Tuple[int, int]:
        """
        Find longest matching suffix in corpus.

        Args:
            context: Byte sequence or string
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            (position, length) of longest match, or (-1, 0) if no match
        """
        context = self._normalize_query(context, transforms)

        if not context:
            return (-1, 0)

        # Limit context length if max_length is set
        if self.max_length:
            context = context[-self.max_length:]

        # Binary search for longest matching suffix
        for length in range(len(context), 0, -1):
            suffix = context[-length:]
            # Use transforms=[] since context is already transformed
            if self.count(suffix, transforms=[]) > 0:
                # Found a match - get first position
                results = self.search(suffix, transforms=[])
                if results:
                    if self._is_chunked:
                        # Return first position (chunk_idx, pos)
                        return (results[0], length)
                    else:
                        return (results[0], length)
        return (-1, 0)

    def find_all_suffix_matches(
        self,
        context: Union[bytes, List[int], str],
        transforms: Optional[List[str]] = None
    ) -> List[Tuple[int, List]]:
        """
        Find all matching suffixes at different lengths.

        For context "abc", searches for "abc", "bc", "c" and returns
        all matches with their corpus positions.

        Args:
            context: Byte sequence or string
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            List of (length, positions) tuples, sorted by decreasing length.
            For chunked models, positions are (chunk_idx, pos) tuples.

        Example:
            >>> model = Infinigram(b"the cat sat on the mat")
            >>> model.find_all_suffix_matches(b"the cat")
            [(7, [0]), (4, [0, 15]), (1, [2, 8, 17])]  # "the cat", " cat", "t"
        """
        context = self._normalize_query(context, transforms)

        if not context:
            return []

        # Limit context length if max_length is set
        if self.max_length:
            context = context[-self.max_length:]

        matches = []

        # Search for each suffix length (longest to shortest)
        for length in range(len(context), 0, -1):
            suffix = context[-length:]
            # transforms=[] since context already transformed
            results = self.search(suffix, transforms=[])

            if results:
                matches.append((length, list(results)))

        return matches

    def continuations(
        self,
        context: Union[bytes, List[int], str],
        transforms: Optional[List[str]] = None
    ) -> Dict[int, int]:
        """
        Get continuation counts for longest matching suffix.

        Args:
            context: Byte sequence or string
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            Dict mapping next_byte -> count
        """
        context = self._normalize_query(context, transforms)

        if not context:
            # Return uniform for empty context
            return {b: 1 for b in range(256)}

        # Limit context length if max_length is set
        if self.max_length:
            context = context[-self.max_length:]

        # Find longest matching suffix
        for length in range(len(context), 0, -1):
            suffix = context[-length:]
            # Use transforms=[] since context is already transformed
            results = self.search(suffix, transforms=[])

            if not results:
                continue

            # Count continuations
            continuations = defaultdict(int)

            if self._is_chunked:
                for chunk_idx, pos in results:
                    next_pos = pos + length
                    chunk_size = self.get_chunk_size(chunk_idx)
                    if next_pos < chunk_size:
                        next_byte = self.get_byte(next_pos, chunk_idx)
                        continuations[next_byte] += 1
            else:
                for pos in results:
                    next_pos = pos + length
                    if next_pos < self.n:
                        next_byte = self.get_byte(next_pos)
                        continuations[next_byte] += 1

            if continuations:
                return dict(continuations)

        # No match found - return uniform
        return {b: 1 for b in range(256)}

    def predict(
        self,
        context: Union[bytes, List[int], str],
        top_k: int = 50,
        smoothing: float = 0.0,
        transforms: Optional[List[str]] = None
    ) -> Dict[int, float]:
        """
        Predict next byte probabilities.

        Args:
            context: Byte sequence or string
            top_k: Return only top k predictions
            smoothing: Laplace smoothing parameter (default: 0.0)
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            Dict mapping byte -> probability
        """
        # Get continuation counts (transforms applied inside continuations)
        counts = self.continuations(context, transforms=transforms)

        if not counts:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / self.vocab_size
            return {token: uniform_prob for token in list(self.vocab)[:top_k]}

        # Filter by min_count
        filtered = {token: count for token, count in counts.items()
                   if count >= self.min_count}

        if not filtered:
            filtered = counts

        # Compute probabilities with smoothing
        total = sum(filtered.values())
        smoothed_total = total + smoothing * self.vocab_size

        probs = {}
        for token, count in filtered.items():
            probs[token] = (count + smoothing) / smoothed_total

        # Add smoothing for unseen tokens
        if smoothing > 0:
            unseen_prob = smoothing / smoothed_total
            for token in self.vocab:
                if token not in probs:
                    probs[token] = unseen_prob

        # Sort and return top k
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return dict(sorted_probs[:top_k])

    def predict_logprobs(
        self,
        context: Union[bytes, List[int], str],
        top_k: int = 50,
        smoothing: float = 1e-10,
        transforms: Optional[List[str]] = None
    ) -> Dict[int, float]:
        """
        Predict next byte log-probabilities.

        Args:
            context: Byte sequence or string
            top_k: Return only top k predictions
            smoothing: Smoothing to avoid log(0) (default: 1e-10)
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            Dict mapping byte -> log probability
        """
        probs = self.predict(context, top_k=top_k, smoothing=smoothing, transforms=transforms)
        return {token: math.log(p) for token, p in probs.items()}

    def predict_weighted(
        self,
        context: Union[bytes, List[int], str],
        min_length: int = 1,
        max_length: Optional[int] = None,
        weight_fn: Optional[callable] = None,
        top_k: int = 50,
        smoothing: float = 0.0,
        transforms: Optional[List[str]] = None
    ) -> Dict[int, float]:
        """
        Predict using weighted combination of multiple suffix lengths.

        Args:
            context: Byte sequence or string
            min_length: Minimum suffix length to consider
            max_length: Maximum suffix length (None = use model's max_length)
            weight_fn: Function mapping suffix_length -> weight
            top_k: Return top k predictions
            smoothing: Smoothing parameter
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            Dict mapping byte -> probability
        """
        # Apply transforms once at entry
        context = self._normalize_query(context, transforms)

        if weight_fn is None:
            weight_fn = lambda k: float(k)  # Linear weighting

        if max_length is None:
            max_length = self.max_length if self.max_length else len(context)

        if self.max_length:
            context = context[-self.max_length:]

        # Collect weighted counts from all suffix lengths
        weighted_counts = defaultdict(float)
        total_weight = 0.0

        for length in range(min_length, min(max_length + 1, len(context) + 1)):
            suffix = context[-length:] if length <= len(context) else context
            # transforms=[] since context already transformed
            results = self.search(suffix, transforms=[])

            if not results:
                continue

            weight = weight_fn(length)
            total_weight += weight

            if self._is_chunked:
                for chunk_idx, pos in results:
                    next_pos = pos + length
                    chunk_size = self.get_chunk_size(chunk_idx)
                    if next_pos < chunk_size:
                        next_byte = self.get_byte(next_pos, chunk_idx)
                        weighted_counts[next_byte] += weight
            else:
                for pos in results:
                    next_pos = pos + length
                    if next_pos < self.n:
                        next_byte = self.get_byte(next_pos)
                        weighted_counts[next_byte] += weight

        if total_weight == 0.0 or not weighted_counts:
            probs = {token: 1.0 / self.vocab_size for token in self.vocab}
        else:
            total_count = sum(weighted_counts.values())
            smoothed_total = total_count + smoothing * self.vocab_size

            probs = {}
            for token, count in weighted_counts.items():
                probs[token] = (count + smoothing) / smoothed_total

            if smoothing > 0:
                unseen_prob = smoothing / smoothed_total
                for token in self.vocab:
                    if token not in probs:
                        probs[token] = unseen_prob

        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return dict(sorted_probs[:top_k])

    def predict_backoff(
        self,
        context: Union[bytes, List[int], str],
        backoff_factor: float = 0.4,
        min_count_threshold: int = 1,
        top_k: int = 50,
        smoothing: float = 0.0,
        transforms: Optional[List[str]] = None
    ) -> Dict[int, float]:
        """
        Predict using Stupid Backoff smoothing.

        Uses longest matching suffix if it has enough counts, otherwise
        backs off to shorter suffixes with a penalty factor. This is the
        algorithm from Brants et al. (2007) "Large Language Models in
        Machine Translation".

        Unlike predict_weighted which combines all lengths, backoff gives
        strong preference to longer matches and only uses shorter when
        the longer match is insufficient.

        Args:
            context: Byte sequence or string
            backoff_factor: Penalty when backing off (default 0.4, as in paper)
            min_count_threshold: Minimum count to use a match without backoff
            top_k: Return top k predictions
            smoothing: Laplace smoothing for final probabilities
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            Dict mapping byte -> probability

        Example:
            >>> model = Infinigram(b"the cat sat on the mat")
            >>> model.predict_backoff(b"the cat")
            # Uses longest match "the cat" if count >= threshold,
            # otherwise backs off to "cat", " cat", etc.
        """
        context = self._normalize_query(context, transforms)

        if not context:
            # Uniform for empty context
            return {b: 1.0 / self.vocab_size for b in list(self.vocab)[:top_k]}

        # Limit context length if max_length is set
        if self.max_length:
            context = context[-self.max_length:]

        # Get all suffix matches using find_all_suffix_matches
        # Returns [(length, positions), ...] sorted by decreasing length
        all_matches = self.find_all_suffix_matches(context, transforms=[])

        if not all_matches:
            # No matches - uniform distribution
            return {b: 1.0 / self.vocab_size for b in list(self.vocab)[:top_k]}

        # Collect continuations with backoff weights
        weighted_counts = defaultdict(float)
        cumulative_backoff = 1.0
        used_any = False

        for length, positions in all_matches:
            if not positions:
                continue

            # Count continuations at this length
            continuations = defaultdict(int)
            if self._is_chunked:
                for chunk_idx, pos in positions:
                    next_pos = pos + length
                    chunk_size = self.get_chunk_size(chunk_idx)
                    if next_pos < chunk_size:
                        next_byte = self.get_byte(next_pos, chunk_idx)
                        continuations[next_byte] += 1
            else:
                for pos in positions:
                    next_pos = pos + length
                    if next_pos < self.n:
                        next_byte = self.get_byte(next_pos)
                        continuations[next_byte] += 1

            if not continuations:
                continue

            total_count = sum(continuations.values())

            # Check if this level has enough counts
            if total_count >= min_count_threshold:
                # Use this level with current backoff weight
                for token, count in continuations.items():
                    # Weight by backoff factor and relative frequency
                    weighted_counts[token] += cumulative_backoff * count
                used_any = True

                # If we have a strong match (many counts), stop here
                if total_count >= min_count_threshold * 5:
                    break

            # Apply backoff penalty for next (shorter) level
            cumulative_backoff *= backoff_factor

        if not used_any or not weighted_counts:
            # Fall back to uniform
            return {b: 1.0 / self.vocab_size for b in list(self.vocab)[:top_k]}

        # Convert to probabilities with optional smoothing
        total = sum(weighted_counts.values())
        smoothed_total = total + smoothing * self.vocab_size

        probs = {}
        for token, count in weighted_counts.items():
            probs[token] = (count + smoothing) / smoothed_total

        if smoothing > 0:
            unseen_prob = smoothing / smoothed_total
            for token in self.vocab:
                if token not in probs:
                    probs[token] = unseen_prob

        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        return dict(sorted_probs[:top_k])

    def predict_search(
        self,
        context: Union[bytes, List[int], str],
        search: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        beam_width: Optional[int] = None,
        top_k: int = 50,
        smoothing: float = 0.0
    ) -> Dict[int, float]:
        """
        Predict using beam search over transform space.

        Explores combinations of transforms to find best matches when
        exact matching fails or is weak.

        Args:
            context: Byte sequence or string
            search: Transform names to explore (None = use default_search)
            max_depth: Maximum transform chain depth (None = use default_max_depth)
            beam_width: Candidates to keep per level (None = use default_beam_width)
            top_k: Return top k predictions
            smoothing: Smoothing parameter

        Returns:
            Dict mapping byte -> probability

        Example:
            >>> model.predict_search("teh CAT", search=['lowercase', 'typo'])
            # Explores: "teh CAT", "teh cat", "the CAT", "the cat", ...
        """
        # Normalize context
        if isinstance(context, str):
            context = context.encode('utf-8')
        elif isinstance(context, list):
            context = bytes(context)

        # Use defaults if not specified
        search_transforms = search if search is not None else self.default_search
        max_depth = max_depth if max_depth is not None else self.default_max_depth
        beam_width = beam_width if beam_width is not None else self.default_beam_width

        # If no search transforms, fall back to regular predict
        if not search_transforms:
            return self.predict(context, top_k=top_k, smoothing=smoothing, transforms=[])

        # Beam search over transform space
        # Each candidate is (transformed_context, transforms_applied, score)
        candidates = [(context, [], self._score_match(context))]
        all_predictions = []

        for depth in range(max_depth + 1):
            # Collect predictions from current candidates
            for ctx, transforms_applied, score in candidates:
                if score > 0:  # Only predict if there's a match
                    probs = self.predict(ctx, top_k=top_k, smoothing=smoothing, transforms=[])
                    # Weight by match quality (score) and transform depth penalty
                    depth_penalty = 0.9 ** len(transforms_applied)
                    weight = score * depth_penalty
                    all_predictions.append((probs, weight))

            if depth >= max_depth:
                break

            # Generate new candidates by applying transforms
            new_candidates = []
            for ctx, transforms_applied, _ in candidates:
                for transform in search_transforms:
                    if transform in self.QUERY_TRANSFORMS:
                        try:
                            new_ctx = self.QUERY_TRANSFORMS[transform](ctx)
                            if new_ctx != ctx:  # Only if transform changed something
                                new_score = self._score_match(new_ctx)
                                new_transforms = transforms_applied + [transform]
                                new_candidates.append((new_ctx, new_transforms, new_score))
                        except Exception:
                            pass  # Skip failed transforms

            # Keep top beam_width candidates by score
            new_candidates.sort(key=lambda x: -x[2])
            candidates = new_candidates[:beam_width]

            if not candidates:
                break  # No more candidates to explore

        # Combine all predictions
        return self._combine_weighted_predictions(all_predictions, top_k)

    def _score_match(self, context: bytes) -> float:
        """Score a context by match quality (0-1)."""
        _, length = self.longest_suffix(context, transforms=[])
        if length == 0:
            return 0.0
        # Score based on match length relative to context
        return min(length / max(len(context), 1), 1.0)

    def _combine_weighted_predictions(
        self,
        predictions: List[Tuple[Dict[int, float], float]],
        top_k: int
    ) -> Dict[int, float]:
        """Combine multiple weighted prediction distributions."""
        if not predictions:
            return {b: 1.0 / self.vocab_size for b in list(self.vocab)[:top_k]}

        # Weighted average of predictions
        combined = defaultdict(float)
        total_weight = sum(w for _, w in predictions)

        if total_weight == 0:
            return {b: 1.0 / self.vocab_size for b in list(self.vocab)[:top_k]}

        for probs, weight in predictions:
            for token, prob in probs.items():
                combined[token] += prob * weight / total_weight

        # Normalize and return top k
        sorted_probs = sorted(combined.items(), key=lambda x: -x[1])
        return dict(sorted_probs[:top_k])

    def confidence(
        self,
        context: Union[bytes, List[int], str],
        transforms: Optional[List[str]] = None
    ) -> float:
        """
        Confidence score for prediction.

        Based on match length and frequency.

        Args:
            context: Byte sequence
            transforms: Transforms to apply (None = use defaults, [] = none)

        Returns:
            Confidence score in [0, 1]
        """
        context = self._normalize_query(context, transforms)

        # transforms=[] since context already transformed
        _, length = self.longest_suffix(context, transforms=[])

        if length == 0:
            return 0.0

        suffix = context[-length:]
        num_matches = self.count(suffix, transforms=[])

        length_score = min(length / 10.0, 1.0)
        freq_score = min(num_matches / 100.0, 1.0)

        return 0.7 * length_score + 0.3 * freq_score

    def get_context(
        self,
        position: int,
        window: int = 50,
        chunk_idx: Optional[int] = None
    ) -> bytes:
        """
        Get context around a position.

        Args:
            position: Position in corpus
            window: Number of bytes before and after
            chunk_idx: Chunk index (for chunked models)

        Returns:
            Bytes around position
        """
        if self._is_chunked:
            if chunk_idx is None:
                raise ValueError("chunk_idx required for chunked models")
            return self._sa.get_context(chunk_idx, position, window)
        else:
            return self._sa.get_context(position, window)

    def update(self, new_tokens: Union[bytes, List[int]]):
        """
        Update corpus with new tokens and rebuild index.

        Note: This rebuilds the entire suffix array, which is O(n log n).
        For frequent updates, consider batching.

        Args:
            new_tokens: New byte values to append
        """
        if isinstance(new_tokens, list):
            # Validate new tokens
            for i, val in enumerate(new_tokens):
                if not isinstance(val, int) or val < 0 or val > 255:
                    raise ValueError(
                        f"new_tokens must contain only bytes (0-255), "
                        f"but found {val} at position {i}"
                    )
            new_tokens = bytes(new_tokens)

        # Read existing corpus
        corpus_file = self.model_path / "corpus.bin"
        with open(corpus_file, 'rb') as f:
            existing_corpus = f.read()

        # Append new tokens
        updated_corpus = existing_corpus + new_tokens

        # Rebuild
        import pydivsufsort

        # Write updated corpus
        with open(corpus_file, 'wb') as f:
            f.write(updated_corpus)

        # Rebuild suffix array (use int64 for large corpora support)
        sa_array = pydivsufsort.divsufsort(updated_corpus).astype(np.int64)
        np.save(self.model_path / "suffix_array.npy", sa_array)

        # Update metadata
        self.meta['n'] = len(updated_corpus)
        self.meta['corpus_size'] = len(updated_corpus)
        with open(self.model_path / "meta.json", 'w') as f:
            json.dump(self.meta, f, indent=2)

        # Reload suffix array
        self._sa = MmapSuffixArray(str(self.model_path))
        self.n = self._sa.n

    @property
    def sa(self):
        """Access to underlying suffix array (for backwards compatibility)."""
        return self._sa

    def close(self):
        """Close memory-mapped files and clean up temporary directory."""
        if hasattr(self._sa, 'close'):
            self._sa.close()

        # Clean up temporary directory if we created one
        if self._temp_dir is not None:
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass  # Best effort cleanup

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self) -> str:
        chunked_str = f", chunks={self._sa.num_chunks}" if self._is_chunked else ""
        return f"Infinigram(n={self.n:,}{chunked_str}, path={self.model_path})"

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def list_models(cls, models_dir: Optional[Path] = None) -> List[str]:
        """
        List available models in the models directory.

        Args:
            models_dir: Directory to search (default: ~/.infinigram/models)

        Returns:
            List of model names
        """
        if models_dir is None:
            models_dir = DEFAULT_MODELS_DIR

        if not models_dir.exists():
            return []

        models = []
        for item in models_dir.iterdir():
            if item.is_dir() and (item / "meta.json").exists():
                models.append(item.name)

        return sorted(models)


# Backwards compatibility aliases
class IdentityAdapter:
    """Identity adapter for backwards compatibility."""
    def encode(self, x):
        return x
    def decode(self, x):
        return x


def create_infinigram(corpus: Union[str, bytes, List[int]], **kwargs) -> Infinigram:
    """
    Create an Infinigram model from corpus (backwards compatibility).

    Args:
        corpus: Byte sequence or text
        **kwargs: Additional arguments passed to Infinigram

    Returns:
        Infinigram model
    """
    return Infinigram(corpus, **kwargs)
