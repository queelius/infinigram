#!/usr/bin/env python3
"""
Disk-backed storage layer for Infinigram datasets.

Implements hybrid JSONL + SQLite storage:
- documents.jsonl: Source of truth (human-readable)
- index.db: SQLite index for fast random access and tag queries
- suffix_array.npz: Cached suffix array for predictions
"""

import json
import sqlite3
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Tuple, Any
import time


class Dataset:
    """
    Represents a dataset on disk.

    Storage layout:
        dataset_dir/
            documents.jsonl    # Source of truth
            index.db          # SQLite index
            metadata.json     # Dataset metadata
            suffix_array.npz  # Cached suffix array (lazy)
    """

    def __init__(self, path: Path):
        """Initialize dataset from directory path."""
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)

        # File paths
        self.jsonl_path = self.path / "documents.jsonl"
        self.index_path = self.path / "index.db"
        self.metadata_path = self.path / "metadata.json"
        self.sa_cache_path = self.path / "suffix_array.npz"

        # Ensure files exist
        self.jsonl_path.touch(exist_ok=True)

        # Open SQLite index
        self.db = self._init_index()

        # Load or create metadata
        self.metadata = self._load_metadata()

        # Build index if needed
        if self._index_needs_rebuild():
            self._rebuild_index()

    def _init_index(self) -> sqlite3.Connection:
        """Initialize SQLite index database."""
        conn = sqlite3.connect(str(self.index_path))

        # Document index: maps doc_id to byte position in JSONL
        conn.execute("""
            CREATE TABLE IF NOT EXISTS doc_index (
                id INTEGER PRIMARY KEY,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)

        # Tag index: maps tags to document IDs
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                doc_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                created_at REAL DEFAULT (julianday('now')),
                PRIMARY KEY (doc_id, tag),
                FOREIGN KEY (doc_id) REFERENCES doc_index(id) ON DELETE CASCADE
            )
        """)

        # Index for fast tag lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)
        """)

        # Index metadata: track when index was last built
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()
        return conn

    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata from JSON."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        else:
            # Create default metadata
            metadata = {
                "name": self.path.name,
                "projections": [],
                "config": {
                    "max_length": None,
                    "min_count": 1
                },
                "created_at": time.time()
            }
            self._save_metadata(metadata)
            return metadata

    def _save_metadata(self, metadata: Optional[Dict[str, Any]] = None):
        """Save metadata to JSON."""
        if metadata is not None:
            self.metadata = metadata

        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _index_needs_rebuild(self) -> bool:
        """Check if index needs to be rebuilt."""
        # Get last index build time
        cursor = self.db.execute(
            "SELECT value FROM index_meta WHERE key = 'last_build_mtime'"
        )
        row = cursor.fetchone()

        if not row:
            return True  # Never built

        last_build_mtime = float(row[0])
        current_mtime = self.jsonl_path.stat().st_mtime

        return current_mtime > last_build_mtime

    def _rebuild_index(self):
        """Rebuild index from JSONL file."""
        print(f"Rebuilding index for {self.path.name}...")

        # Clear existing index
        self.db.execute("DELETE FROM doc_index")
        self.db.execute("DELETE FROM tags")

        # Scan JSONL and build index
        with open(self.jsonl_path, 'rb') as f:
            offset = 0
            doc_id = 0

            while True:
                line_start = offset
                line = f.readline()

                if not line:
                    break

                line_length = len(line)

                # Insert into index
                self.db.execute(
                    "INSERT INTO doc_index (id, byte_offset, byte_length) VALUES (?, ?, ?)",
                    (doc_id, line_start, line_length)
                )

                offset += line_length
                doc_id += 1

        # Update metadata
        self.db.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('last_build_mtime', ?)",
            (str(self.jsonl_path.stat().st_mtime),)
        )

        self.db.commit()
        print(f"Index rebuilt: {doc_id} documents")

    # ========================================================================
    # Document Access
    # ========================================================================

    def count_documents(self) -> int:
        """Count total documents."""
        cursor = self.db.execute("SELECT COUNT(*) FROM doc_index")
        return cursor.fetchone()[0]

    def read_document(self, index: int) -> str:
        """
        Read document by index (0-based).

        Fast O(1) access using SQLite index.
        """
        cursor = self.db.execute(
            "SELECT byte_offset, byte_length FROM doc_index WHERE id = ?",
            (index,)
        )
        row = cursor.fetchone()

        if not row:
            raise IndexError(f"Document {index} not found")

        offset, length = row

        with open(self.jsonl_path, 'rb') as f:
            f.seek(offset)
            line = f.read(length)
            data = json.loads(line)
            return data['text']

    def iter_documents(self, start: int = 0, limit: Optional[int] = None) -> Iterator[str]:
        """
        Iterate documents with optional range.

        Args:
            start: Starting document index
            limit: Maximum number of documents to return
        """
        query = "SELECT byte_offset, byte_length FROM doc_index WHERE id >= ?"
        params = [start]

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.db.execute(query, params)

        with open(self.jsonl_path, 'rb') as f:
            for offset, length in cursor:
                f.seek(offset)
                line = f.read(length)
                data = json.loads(line)
                yield data['text']

    def add_document(self, text: str) -> int:
        """
        Add document to dataset.

        Returns:
            Document ID (index) of newly added document
        """
        # Get current end position
        offset = self.jsonl_path.stat().st_size

        # Create JSON line
        line = json.dumps({"text": text}) + '\n'
        line_bytes = line.encode('utf-8')

        # Append to JSONL
        with open(self.jsonl_path, 'ab') as f:
            f.write(line_bytes)

        # Get new document ID
        doc_id = self.count_documents()

        # Update index
        self.db.execute(
            "INSERT INTO doc_index (id, byte_offset, byte_length) VALUES (?, ?, ?)",
            (doc_id, offset, len(line_bytes))
        )
        self.db.commit()

        # Invalidate suffix array cache
        self._invalidate_suffix_array_cache()

        return doc_id

    def remove_document(self, index: int):
        """
        Remove document by index.

        Uses tombstone approach: removes from index but leaves in JSONL.
        Call compact() to rewrite JSONL without deleted documents.
        """
        # Verify document exists
        cursor = self.db.execute("SELECT id FROM doc_index WHERE id = ?", (index,))
        if not cursor.fetchone():
            raise IndexError(f"Document {index} not found")

        # Remove from index (tombstone)
        self.db.execute("DELETE FROM doc_index WHERE id = ?", (index,))
        self.db.commit()

        # Invalidate suffix array cache
        self._invalidate_suffix_array_cache()

    def compact(self):
        """
        Rewrite JSONL file without deleted documents.

        This reclaims space from tombstoned documents.
        """
        # Get all indexed documents in order
        cursor = self.db.execute(
            "SELECT id FROM doc_index ORDER BY id"
        )
        doc_ids = [row[0] for row in cursor]

        # Create temporary JSONL
        temp_path = self.jsonl_path.with_suffix('.jsonl.tmp')

        with open(temp_path, 'w') as out:
            for doc_id in doc_ids:
                text = self.read_document(doc_id)
                out.write(json.dumps({"text": text}) + '\n')

        # Atomic replace
        temp_path.replace(self.jsonl_path)

        # Rebuild index
        self._rebuild_index()

    # ========================================================================
    # Tagging Support
    # ========================================================================

    def add_tag(self, doc_id: int, tag: str):
        """Add tag to document."""
        # Validate tag (no numeric tags to avoid collision with indices)
        if tag.isdigit():
            raise ValueError(f"Tags cannot be purely numeric: '{tag}'")

        try:
            self.db.execute(
                "INSERT INTO tags (doc_id, tag) VALUES (?, ?)",
                (doc_id, tag)
            )
            self.db.commit()
        except sqlite3.IntegrityError:
            # Tag already exists for this document
            pass

    def remove_tag(self, doc_id: int, tag: str):
        """Remove tag from document."""
        self.db.execute(
            "DELETE FROM tags WHERE doc_id = ? AND tag = ?",
            (doc_id, tag)
        )
        self.db.commit()

    def get_tags(self, doc_id: int) -> List[str]:
        """Get all tags for a document."""
        cursor = self.db.execute(
            "SELECT tag FROM tags WHERE doc_id = ? ORDER BY tag",
            (doc_id,)
        )
        return [row[0] for row in cursor]

    def find_by_tag(self, tag: str) -> List[int]:
        """Find all document IDs with given tag."""
        cursor = self.db.execute(
            "SELECT doc_id FROM tags WHERE tag = ? ORDER BY doc_id",
            (tag,)
        )
        return [row[0] for row in cursor]

    def resolve_tag(self, tag: str) -> Optional[int]:
        """
        Resolve tag to document ID.

        Returns first document with this tag, or None if not found.
        """
        docs = self.find_by_tag(tag)
        return docs[0] if docs else None

    # ========================================================================
    # Suffix Array Cache Management
    # ========================================================================

    def _invalidate_suffix_array_cache(self):
        """Invalidate suffix array cache (called on document changes)."""
        if self.sa_cache_path.exists():
            self.sa_cache_path.unlink()

    def get_corpus_size(self) -> int:
        """Get total corpus size in bytes."""
        return self.jsonl_path.stat().st_size

    def build_corpus(self) -> bytes:
        """
        Build corpus by reading all documents.

        Returns concatenated documents separated by NULL bytes.
        This is used for building suffix arrays.
        """
        docs = []
        for text in self.iter_documents():
            docs.append(text.encode('utf-8'))

        # Join with NULL byte separator
        return b'\x00'.join(docs)

    # ========================================================================
    # Metadata Management
    # ========================================================================

    def get_projections(self) -> List[str]:
        """Get list of active projections."""
        return self.metadata.get('projections', [])

    def set_projections(self, projections: List[str]):
        """Set active projections."""
        self.metadata['projections'] = projections
        self._save_metadata()
        self._invalidate_suffix_array_cache()

    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.metadata.get('config', {})

    def update_config(self, **kwargs):
        """Update dataset configuration."""
        config = self.metadata.get('config', {})
        config.update(kwargs)
        self.metadata['config'] = config
        self._save_metadata()

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        num_docs = self.count_documents()
        corpus_size = self.get_corpus_size()

        # Count tags
        cursor = self.db.execute("SELECT COUNT(DISTINCT tag) FROM tags")
        num_tags = cursor.fetchone()[0]

        return {
            'name': self.metadata['name'],
            'num_documents': num_docs,
            'corpus_size': corpus_size,
            'num_tags': num_tags,
            'projections': self.get_projections(),
            'has_suffix_array_cache': self.sa_cache_path.exists(),
        }

    def close(self):
        """Close database connection."""
        self.db.close()

    def __repr__(self):
        return f"Dataset({self.path.name}, docs={self.count_documents()})"
