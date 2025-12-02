"""
Suffix array implementation for efficient n-gram operations.

Uses pydivsufsort (C library) for O(n) construction at ~15-30 MB/sec.
"""

from typing import List, Tuple, Iterator, Union
import numpy as np
import pydivsufsort


class SuffixArray:
    """
    Efficient suffix array for n-gram operations.

    Supports:
    - Fast longest suffix matching
    - Pattern search in O(m log n) time
    - Range queries for all occurrences
    - Memory-efficient storage

    Performance: ~15-30 MB/sec construction via pydivsufsort
    """

    def __init__(self, tokens: Union[List[int], bytes]):
        """
        Build a suffix array from token sequence.

        Args:
            tokens: List of token ids (0-255) or bytes
        """
        # Normalize to bytes for efficient storage
        if isinstance(tokens, bytes):
            self._bytes = tokens
        else:
            self._bytes = bytes(tokens)

        self.tokens = self._bytes  # Compatibility alias
        self.n = len(self._bytes)

        # Build suffix array
        # Always use int64 to support large corpora (>2GB chunks)
        if self.n == 0:
            self.suffix_array = np.array([], dtype=np.int64)
        elif self.n == 1:
            self.suffix_array = np.array([0], dtype=np.int64)
        else:
            self.suffix_array = pydivsufsort.divsufsort(self._bytes).astype(np.int64)

        # LCP array built lazily on first access
        self._lcp_array = None

    @property
    def lcp_array(self) -> np.ndarray:
        """Lazily build LCP array on first access."""
        if self._lcp_array is None:
            self._lcp_array = self._build_lcp_array()
        return self._lcp_array

    def _build_lcp_array(self) -> np.ndarray:
        """
        Build Longest Common Prefix array using Kasai's algorithm.
        Time complexity: O(n)
        """
        if self.n == 0:
            return np.array([], dtype=np.int32)

        lcp = np.zeros(self.n, dtype=np.int32)
        rank = np.zeros(self.n, dtype=np.int32)

        for i in range(self.n):
            rank[self.suffix_array[i]] = i

        k = 0
        for i in range(self.n):
            if rank[i] == self.n - 1:
                k = 0
                continue

            j = self.suffix_array[rank[i] + 1]

            while i + k < self.n and j + k < self.n and \
                  self._bytes[i + k] == self._bytes[j + k]:
                k += 1

            lcp[rank[i]] = k

            if k > 0:
                k -= 1

        return lcp

    def find_longest_suffix(self, query: Union[List[int], bytes]) -> Tuple[int, int]:
        """
        Find the longest suffix of query that appears in the text.

        Returns:
            (position, length) of the longest matching suffix
        """
        if isinstance(query, bytes):
            query = list(query)

        best_pos = -1
        best_len = 0

        for suffix_len in range(1, len(query) + 1):
            suffix = query[-suffix_len:]
            positions = self.search(suffix)

            if positions:
                best_pos = positions[0]
                best_len = suffix_len
            else:
                break

        return best_pos, best_len

    def search(self, pattern: Union[List[int], bytes]) -> List[int]:
        """
        Search for all occurrences of pattern.

        Returns:
            List of starting positions where pattern occurs
        """
        if not pattern:
            return []

        if isinstance(pattern, bytes):
            pattern = list(pattern)

        left = self._binary_search_left(pattern)
        right = self._binary_search_right(pattern)

        if left >= right:
            return []

        return [int(self.suffix_array[i]) for i in range(left, right)]

    def _binary_search_left(self, pattern: List[int]) -> int:
        """Find leftmost position where pattern could be inserted."""
        left, right = 0, self.n

        while left < right:
            mid = (left + right) // 2
            suffix_start = self.suffix_array[mid]

            if self._compare_pattern_at(pattern, suffix_start) > 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _binary_search_right(self, pattern: List[int]) -> int:
        """Find rightmost position where pattern could be inserted."""
        left, right = 0, self.n

        while left < right:
            mid = (left + right) // 2
            suffix_start = self.suffix_array[mid]

            if self._compare_pattern_at(pattern, suffix_start, prefix_only=True) >= 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _compare_pattern_at(self, pattern: List[int], pos: int,
                           prefix_only: bool = False) -> int:
        """Compare pattern with suffix starting at pos."""
        pattern_len = len(pattern)
        suffix_len = self.n - pos

        for i in range(min(pattern_len, suffix_len)):
            if pattern[i] < self._bytes[pos + i]:
                return -1
            elif pattern[i] > self._bytes[pos + i]:
                return 1

        if prefix_only:
            return 0 if pattern_len <= suffix_len else 1

        if pattern_len < suffix_len:
            return -1
        elif pattern_len > suffix_len:
            return 1
        else:
            return 0

    def get_context(self, position: int, window: int = 10) -> Tuple[List[int], List[int]]:
        """Get context around a position."""
        start = max(0, position - window)
        end = min(self.n, position + window)

        before = list(self._bytes[start:position])
        after = list(self._bytes[position:end])

        return before, after

    def ngrams(self, n: int) -> Iterator[Tuple[Tuple[int, ...], int]]:
        """Iterate over all n-grams with their frequencies."""
        ngram_counts = {}

        for i in range(self.n - n + 1):
            ngram = tuple(self._bytes[i:i + n])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        for ngram, count in sorted(ngram_counts.items(), key=lambda x: -x[1]):
            yield ngram, count

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return f"SuffixArray(n={self.n})"


class PersistedSuffixArray(SuffixArray):
    """
    Suffix array with disk persistence for instant loading.

    First load: builds and saves to disk (~15-30 MB/sec)
    Subsequent loads: instant from disk (~300+ MB/sec)

    Example:
        >>> sa = PersistedSuffixArray.load_or_build("corpus.txt", "/tmp/index")
    """

    @classmethod
    def load_or_build(
        cls,
        corpus_path: str,
        index_path: str,
        rebuild: bool = False
    ) -> 'PersistedSuffixArray':
        """
        Load existing index or build and save a new one.

        Args:
            corpus_path: Path to corpus file
            index_path: Directory to store/load index files
            rebuild: Force rebuild even if index exists
        """
        import json
        from pathlib import Path

        index_dir = Path(index_path)
        corpus_file = Path(corpus_path)

        sa_file = index_dir / "suffix_array.npy"
        corpus_cache = index_dir / "corpus.bin"
        meta_file = index_dir / "meta.json"

        # Check if we can load existing index
        if not rebuild and sa_file.exists() and corpus_cache.exists() and meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            corpus_mtime = corpus_file.stat().st_mtime if corpus_file.exists() else 0
            corpus_size = corpus_file.stat().st_size if corpus_file.exists() else 0

            if meta.get('corpus_mtime') == corpus_mtime and meta.get('corpus_size') == corpus_size:
                print(f"Loading pre-built index from {index_dir}...")
                return cls._load_from_cache(index_dir)
            else:
                print(f"Corpus changed, rebuilding index...")

        # Build new index
        print(f"Building index for {corpus_path}...")

        with open(corpus_path, 'rb') as f:
            corpus_bytes = f.read()

        import time
        start = time.time()
        instance = cls(corpus_bytes)
        elapsed = time.time() - start
        print(f"Built in {elapsed:.2f}s ({len(corpus_bytes)/elapsed:,.0f} bytes/sec)")

        instance._save_to_cache(index_dir, corpus_path)

        return instance

    def _save_to_cache(self, index_dir, corpus_path: str):
        """Save suffix array and metadata to disk."""
        import json
        from pathlib import Path

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        corpus_file = Path(corpus_path)

        np.save(index_dir / "suffix_array.npy", self.suffix_array)

        with open(index_dir / "corpus.bin", 'wb') as f:
            f.write(self._bytes)

        if self._lcp_array is not None:
            np.save(index_dir / "lcp_array.npy", self._lcp_array)

        meta = {
            'corpus_mtime': corpus_file.stat().st_mtime if corpus_file.exists() else 0,
            'corpus_size': corpus_file.stat().st_size if corpus_file.exists() else 0,
            'n': self.n,
        }
        with open(index_dir / "meta.json", 'w') as f:
            json.dump(meta, f)

        print(f"Index saved to {index_dir}")

    @classmethod
    def _load_from_cache(cls, index_dir) -> 'PersistedSuffixArray':
        """Load suffix array from disk cache."""
        from pathlib import Path
        import time

        index_dir = Path(index_dir)
        start = time.time()

        with open(index_dir / "corpus.bin", 'rb') as f:
            corpus_bytes = f.read()

        instance = object.__new__(cls)
        instance._bytes = corpus_bytes
        instance.tokens = corpus_bytes
        instance.n = len(corpus_bytes)

        instance.suffix_array = np.load(index_dir / "suffix_array.npy")

        lcp_file = index_dir / "lcp_array.npy"
        instance._lcp_array = np.load(lcp_file) if lcp_file.exists() else None

        elapsed = time.time() - start
        print(f"Loaded in {elapsed:.3f}s ({len(corpus_bytes)/elapsed/1_000_000:.0f} MB/sec)")

        return instance

    def save(self, index_dir: str, corpus_path: str = ""):
        """Save suffix array to disk for later fast loading."""
        self._save_to_cache(index_dir, corpus_path)


class MmapSuffixArray:
    """
    Memory-mapped suffix array for datasets larger than RAM.

    Uses mmap for both corpus and suffix array, allowing the OS to
    page in only the needed portions. Can handle multi-GB datasets
    with minimal memory footprint.

    Build once, query with <100MB RAM regardless of corpus size.

    Example:
        >>> # First time: builds index (requires RAM for build)
        >>> MmapSuffixArray.build("wiki.txt", "/data/wiki_index")

        >>> # Query time: uses ~100MB regardless of corpus size
        >>> sa = MmapSuffixArray("/data/wiki_index")
        >>> positions = sa.search(b"Einstein")
    """

    def __init__(self, index_path: str):
        """
        Open a pre-built memory-mapped index.

        Args:
            index_path: Directory containing the index files
        """
        import mmap
        from pathlib import Path

        index_dir = Path(index_path)

        if not (index_dir / "corpus.bin").exists():
            raise FileNotFoundError(
                f"Index not found at {index_dir}. "
                f"Use MmapSuffixArray.build() to create it first."
            )

        # Memory-map the corpus
        self._corpus_file = open(index_dir / "corpus.bin", 'rb')
        self._corpus_mmap = mmap.mmap(
            self._corpus_file.fileno(), 0, access=mmap.ACCESS_READ
        )
        self.n = len(self._corpus_mmap)

        # Memory-map the suffix array using np.load which handles header correctly
        self.suffix_array = np.load(
            index_dir / "suffix_array.npy",
            mmap_mode='r'
        )

        # LCP array is optional
        lcp_file = index_dir / "lcp_array.npy"
        if lcp_file.exists():
            self._lcp_array = np.load(lcp_file, mmap_mode='r')
        else:
            self._lcp_array = None

    @classmethod
    def build(
        cls,
        corpus_path: str,
        index_path: str,
        chunk_size: int = 100_000_000  # 100MB chunks for progress
    ) -> 'MmapSuffixArray':
        """
        Build a memory-mapped index from a corpus file.

        Note: Building requires loading the corpus into RAM once.
        After building, queries use minimal memory via mmap.

        Args:
            corpus_path: Path to corpus file
            index_path: Directory to save index files
            chunk_size: Chunk size for progress reporting
        """
        import json
        import shutil
        from pathlib import Path
        import time

        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        corpus_file = Path(corpus_path)
        corpus_size = corpus_file.stat().st_size

        print(f"Building mmap index for {corpus_path}")
        print(f"Corpus size: {corpus_size:,} bytes ({corpus_size/1_000_000_000:.2f} GB)")

        # Copy corpus to index directory
        print("Copying corpus...")
        shutil.copy(corpus_path, index_dir / "corpus.bin")

        # Load corpus and build suffix array
        print("Loading corpus into memory for suffix array construction...")
        with open(corpus_path, 'rb') as f:
            corpus_bytes = f.read()

        print("Building suffix array...")
        start = time.time()
        # Use int64 to support corpora >2GB
        sa = pydivsufsort.divsufsort(corpus_bytes).astype(np.int64)
        elapsed = time.time() - start
        print(f"Built in {elapsed:.1f}s ({len(corpus_bytes)/elapsed/1_000_000:.1f} MB/sec)")

        # Save suffix array in numpy format
        print("Saving suffix array...")
        np.save(index_dir / "suffix_array.npy", sa)

        # Save metadata
        meta = {
            'corpus_size': corpus_size,
            'corpus_mtime': corpus_file.stat().st_mtime,
            'n': len(corpus_bytes),
        }
        with open(index_dir / "meta.json", 'w') as f:
            json.dump(meta, f)

        # Free memory
        del corpus_bytes
        del sa

        print(f"Index saved to {index_dir}")
        print(f"Total index size: {sum(f.stat().st_size for f in index_dir.iterdir()):,} bytes")

        # Return mmap instance
        return cls(index_path)

    def search(self, pattern: Union[List[int], bytes]) -> List[int]:
        """Search for all occurrences of pattern."""
        if not pattern:
            return []

        if isinstance(pattern, bytes):
            pattern = list(pattern)

        left = self._binary_search_left(pattern)
        right = self._binary_search_right(pattern)

        if left >= right:
            return []

        return [int(self.suffix_array[i]) for i in range(left, right)]

    def _binary_search_left(self, pattern: List[int]) -> int:
        left, right = 0, self.n

        while left < right:
            mid = (left + right) // 2
            if self._compare_at(pattern, self.suffix_array[mid]) > 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _binary_search_right(self, pattern: List[int]) -> int:
        left, right = 0, self.n

        while left < right:
            mid = (left + right) // 2
            if self._compare_at(pattern, self.suffix_array[mid], prefix_only=True) >= 0:
                left = mid + 1
            else:
                right = mid

        return left

    def _compare_at(self, pattern: List[int], pos: int, prefix_only: bool = False) -> int:
        """Compare pattern with suffix at position pos."""
        pattern_len = len(pattern)
        suffix_len = self.n - pos

        for i in range(min(pattern_len, suffix_len)):
            corpus_byte = self._corpus_mmap[pos + i]
            if pattern[i] < corpus_byte:
                return -1
            elif pattern[i] > corpus_byte:
                return 1

        if prefix_only:
            return 0 if pattern_len <= suffix_len else 1

        if pattern_len < suffix_len:
            return -1
        elif pattern_len > suffix_len:
            return 1
        return 0

    def get_context(self, position: int, window: int = 50) -> bytes:
        """Get context around a position."""
        start = max(0, position - window)
        end = min(self.n, position + window)
        return self._corpus_mmap[start:end]

    def count(self, pattern: Union[List[int], bytes]) -> int:
        """Count occurrences of pattern (faster than len(search()))."""
        if not pattern:
            return 0

        if isinstance(pattern, bytes):
            pattern = list(pattern)

        left = self._binary_search_left(pattern)
        right = self._binary_search_right(pattern)

        return max(0, right - left)

    def close(self):
        """Close memory-mapped files."""
        self._corpus_mmap.close()
        self._corpus_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"MmapSuffixArray(n={self.n:,})"


class ChunkedMmapSuffixArray:
    """
    Chunked suffix array for corpora larger than available RAM.

    Splits corpus into chunks, builds separate indexes, and merges
    query results. Handles corpora up to hundreds of GB with limited RAM.

    Memory usage: ~9x chunk_size during build, ~100MB during queries.

    Example:
        >>> # Build index for 20GB corpus with 64GB RAM (4 chunks of 5GB)
        >>> ChunkedMmapSuffixArray.build("wikipedia.txt", "wiki_index", chunk_size_gb=5)

        >>> # Query uses <100MB RAM regardless of corpus size
        >>> sa = ChunkedMmapSuffixArray("wiki_index")
        >>> count = sa.count(b"Einstein")
    """

    def __init__(self, index_path: str):
        """
        Open a pre-built chunked index.

        Args:
            index_path: Directory containing chunk index directories
        """
        import json
        from pathlib import Path

        index_dir = Path(index_path)

        meta_file = index_dir / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"Index not found at {index_dir}. "
                f"Use ChunkedMmapSuffixArray.build() to create it first."
            )

        with open(meta_file, 'r') as f:
            self.meta = json.load(f)

        self.n = self.meta['total_size']
        self.num_chunks = self.meta['num_chunks']
        self.chunk_offsets = self.meta['chunk_offsets']

        # Load all chunk indexes (memory-mapped, so minimal RAM)
        self.chunks = []
        for i in range(self.num_chunks):
            chunk_dir = index_dir / f"chunk_{i}"
            self.chunks.append(MmapSuffixArray(str(chunk_dir)))

    @classmethod
    def build(
        cls,
        corpus_path: str,
        index_path: str,
        chunk_size_gb: float = 5.0
    ) -> 'ChunkedMmapSuffixArray':
        """
        Build a chunked index from a large corpus file.

        Args:
            corpus_path: Path to corpus file
            index_path: Directory to save index
            chunk_size_gb: Size of each chunk in GB (default 5GB)

        Returns:
            ChunkedMmapSuffixArray instance ready for queries
        """
        import json
        from pathlib import Path
        import time

        index_dir = Path(index_path)
        index_dir.mkdir(parents=True, exist_ok=True)

        corpus_file = Path(corpus_path)
        corpus_size = corpus_file.stat().st_size
        chunk_size = int(chunk_size_gb * 1_000_000_000)

        num_chunks = (corpus_size + chunk_size - 1) // chunk_size
        print(f"Building chunked index for {corpus_path}")
        print(f"Corpus size: {corpus_size:,} bytes ({corpus_size/1e9:.2f} GB)")
        print(f"Chunk size: {chunk_size_gb} GB")
        print(f"Number of chunks: {num_chunks}")
        print()

        chunk_offsets = [0]
        total_start = time.time()

        with open(corpus_path, 'rb') as f:
            for chunk_idx in range(num_chunks):
                chunk_dir = index_dir / f"chunk_{chunk_idx}"
                chunk_dir.mkdir(exist_ok=True)

                # Read chunk
                start_offset = chunk_idx * chunk_size
                f.seek(start_offset)
                chunk_data = f.read(chunk_size)

                print(f"Chunk {chunk_idx + 1}/{num_chunks}: {len(chunk_data):,} bytes")

                # Save chunk corpus
                with open(chunk_dir / "corpus.bin", 'wb') as cf:
                    cf.write(chunk_data)

                # Build suffix array
                print("  Building suffix array...")
                start = time.time()
                # Use int64 to support chunks >2GB
                sa = pydivsufsort.divsufsort(chunk_data).astype(np.int64)
                elapsed = time.time() - start
                print(f"  Built in {elapsed:.1f}s ({len(chunk_data)/elapsed/1e6:.1f} MB/sec)")

                # Save suffix array
                np.save(chunk_dir / "suffix_array.npy", sa)

                # Save chunk metadata
                chunk_meta = {
                    'size': len(chunk_data),
                    'offset': start_offset,
                }
                with open(chunk_dir / "meta.json", 'w') as mf:
                    json.dump(chunk_meta, mf)

                chunk_offsets.append(start_offset + len(chunk_data))

                # Free memory
                del chunk_data
                del sa

                print()

        # Save overall metadata
        meta = {
            'total_size': corpus_size,
            'num_chunks': num_chunks,
            'chunk_size': chunk_size,
            'chunk_offsets': chunk_offsets,
        }
        with open(index_dir / "meta.json", 'w') as f:
            json.dump(meta, f)

        total_elapsed = time.time() - total_start
        print(f"Total build time: {total_elapsed:.1f}s ({corpus_size/total_elapsed/1e6:.1f} MB/sec)")
        print(f"Index saved to {index_dir}")

        return cls(str(index_dir))

    def search(self, pattern: Union[List[int], bytes]) -> List[Tuple[int, int]]:
        """
        Search for all occurrences across all chunks.

        Returns:
            List of (chunk_idx, position_in_chunk) tuples
        """
        results = []
        for chunk_idx, chunk in enumerate(self.chunks):
            positions = chunk.search(pattern)
            for pos in positions:
                results.append((chunk_idx, pos))
        return results

    def count(self, pattern: Union[List[int], bytes]) -> int:
        """Count total occurrences across all chunks."""
        total = 0
        for chunk in self.chunks:
            total += chunk.count(pattern)
        return total

    def get_context(self, chunk_idx: int, position: int, window: int = 50) -> bytes:
        """Get context from a specific chunk."""
        if chunk_idx >= len(self.chunks):
            raise IndexError(f"Chunk {chunk_idx} out of range")
        return self.chunks[chunk_idx].get_context(position, window)

    def close(self):
        """Close all chunk indexes."""
        for chunk in self.chunks:
            chunk.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        return self.n

    def __repr__(self):
        return f"ChunkedMmapSuffixArray(n={self.n:,}, chunks={self.num_chunks})"
