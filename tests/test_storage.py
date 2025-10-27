#!/usr/bin/env python3
"""
Tests for the storage layer (Dataset class).
"""

import pytest
import tempfile
from pathlib import Path
from infinigram.storage import Dataset


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "test_dataset"
        dataset = Dataset(dataset_path)
        yield dataset
        dataset.close()


class TestDatasetBasics:
    """Test basic dataset operations."""

    def test_create_empty_dataset(self, temp_dataset):
        """Test creating an empty dataset."""
        assert temp_dataset.count_documents() == 0
        assert temp_dataset.get_corpus_size() >= 0

    def test_add_document(self, temp_dataset):
        """Test adding a document."""
        doc_id = temp_dataset.add_document("Hello, world!")
        assert doc_id == 0
        assert temp_dataset.count_documents() == 1

    def test_read_document(self, temp_dataset):
        """Test reading a document by index."""
        temp_dataset.add_document("First document")
        temp_dataset.add_document("Second document")

        assert temp_dataset.read_document(0) == "First document"
        assert temp_dataset.read_document(1) == "Second document"

    def test_read_nonexistent_document(self, temp_dataset):
        """Test reading a document that doesn't exist."""
        with pytest.raises(IndexError):
            temp_dataset.read_document(999)

    def test_iter_documents(self, temp_dataset):
        """Test iterating over documents."""
        temp_dataset.add_document("Doc 1")
        temp_dataset.add_document("Doc 2")
        temp_dataset.add_document("Doc 3")

        docs = list(temp_dataset.iter_documents())
        assert docs == ["Doc 1", "Doc 2", "Doc 3"]

    def test_iter_documents_with_range(self, temp_dataset):
        """Test iterating with start and limit."""
        for i in range(10):
            temp_dataset.add_document(f"Doc {i}")

        # Start from index 3
        docs = list(temp_dataset.iter_documents(start=3))
        assert len(docs) == 7
        assert docs[0] == "Doc 3"

        # Limit to 5 documents
        docs = list(temp_dataset.iter_documents(start=0, limit=5))
        assert len(docs) == 5
        assert docs[-1] == "Doc 4"


class TestDocumentRemoval:
    """Test document removal and compaction."""

    def test_remove_document(self, temp_dataset):
        """Test removing a document."""
        temp_dataset.add_document("Doc 1")
        temp_dataset.add_document("Doc 2")
        temp_dataset.add_document("Doc 3")

        temp_dataset.remove_document(1)
        assert temp_dataset.count_documents() == 2

        # Remaining documents
        cursor = temp_dataset.db.execute("SELECT id FROM doc_index ORDER BY id")
        ids = [row[0] for row in cursor]
        assert ids == [0, 2]

    def test_compact(self, temp_dataset):
        """Test compacting dataset after deletions."""
        temp_dataset.add_document("Doc 1")
        temp_dataset.add_document("Doc 2")
        temp_dataset.add_document("Doc 3")

        temp_dataset.remove_document(1)
        temp_dataset.compact()

        # After compaction, IDs should be sequential
        assert temp_dataset.count_documents() == 2
        assert temp_dataset.read_document(0) == "Doc 1"
        assert temp_dataset.read_document(1) == "Doc 3"


class TestTagging:
    """Test document tagging."""

    def test_add_tag(self, temp_dataset):
        """Test adding a tag to a document."""
        doc_id = temp_dataset.add_document("Tagged document")
        temp_dataset.add_tag(doc_id, "important")

        tags = temp_dataset.get_tags(doc_id)
        assert "important" in tags

    def test_add_multiple_tags(self, temp_dataset):
        """Test adding multiple tags to a document."""
        doc_id = temp_dataset.add_document("Multi-tagged document")
        temp_dataset.add_tag(doc_id, "math")
        temp_dataset.add_tag(doc_id, "algebra")
        temp_dataset.add_tag(doc_id, "tutorial")

        tags = temp_dataset.get_tags(doc_id)
        assert len(tags) == 3
        assert set(tags) == {"math", "algebra", "tutorial"}

    def test_numeric_tag_forbidden(self, temp_dataset):
        """Test that purely numeric tags are forbidden."""
        doc_id = temp_dataset.add_document("Document")

        with pytest.raises(ValueError, match="purely numeric"):
            temp_dataset.add_tag(doc_id, "123")

    def test_remove_tag(self, temp_dataset):
        """Test removing a tag."""
        doc_id = temp_dataset.add_document("Document")
        temp_dataset.add_tag(doc_id, "tag1")
        temp_dataset.add_tag(doc_id, "tag2")

        temp_dataset.remove_tag(doc_id, "tag1")

        tags = temp_dataset.get_tags(doc_id)
        assert "tag1" not in tags
        assert "tag2" in tags

    def test_find_by_tag(self, temp_dataset):
        """Test finding documents by tag."""
        doc1 = temp_dataset.add_document("Doc 1")
        doc2 = temp_dataset.add_document("Doc 2")
        doc3 = temp_dataset.add_document("Doc 3")

        temp_dataset.add_tag(doc1, "important")
        temp_dataset.add_tag(doc3, "important")

        results = temp_dataset.find_by_tag("important")
        assert results == [doc1, doc3]

    def test_resolve_tag(self, temp_dataset):
        """Test resolving a tag to document ID."""
        doc_id = temp_dataset.add_document("Tagged doc")
        temp_dataset.add_tag(doc_id, "mylink")

        resolved = temp_dataset.resolve_tag("mylink")
        assert resolved == doc_id

        # Non-existent tag
        assert temp_dataset.resolve_tag("nonexistent") is None


class TestMetadata:
    """Test metadata management."""

    def test_default_metadata(self, temp_dataset):
        """Test default metadata is created."""
        assert temp_dataset.metadata['name'] == "test_dataset"
        assert 'projections' in temp_dataset.metadata
        assert 'config' in temp_dataset.metadata

    def test_set_projections(self, temp_dataset):
        """Test setting projections."""
        temp_dataset.set_projections(['lowercase', 'uppercase'])

        projections = temp_dataset.get_projections()
        assert projections == ['lowercase', 'uppercase']

    def test_update_config(self, temp_dataset):
        """Test updating configuration."""
        temp_dataset.update_config(max_length=100, min_count=2)

        config = temp_dataset.get_config()
        assert config['max_length'] == 100
        assert config['min_count'] == 2


class TestCorpusBuilding:
    """Test corpus building for suffix arrays."""

    def test_build_corpus(self, temp_dataset):
        """Test building corpus from documents."""
        temp_dataset.add_document("Hello")
        temp_dataset.add_document("World")

        corpus = temp_dataset.build_corpus()
        assert corpus == b"Hello\x00World"

    def test_build_corpus_empty(self, temp_dataset):
        """Test building corpus from empty dataset."""
        corpus = temp_dataset.build_corpus()
        assert corpus == b""


class TestIndexRebuilding:
    """Test index rebuilding."""

    def test_index_rebuild_on_manual_jsonl_edit(self, temp_dataset):
        """Test that index is rebuilt if JSONL is modified externally."""
        temp_dataset.add_document("Doc 1")

        # Simulate external modification by touching the JSONL file
        import time
        time.sleep(0.1)
        temp_dataset.jsonl_path.touch()

        # Create new dataset instance (simulates restart)
        temp_dataset.close()
        dataset2 = Dataset(temp_dataset.path)

        # Index should be rebuilt
        assert dataset2.count_documents() == 1

        dataset2.close()


class TestStats:
    """Test statistics gathering."""

    def test_get_stats(self, temp_dataset):
        """Test getting dataset statistics."""
        temp_dataset.add_document("Doc 1")
        temp_dataset.add_document("Doc 2")
        temp_dataset.add_tag(0, "tag1")

        stats = temp_dataset.get_stats()

        assert stats['name'] == "test_dataset"
        assert stats['num_documents'] == 2
        assert stats['num_tags'] == 1
        assert stats['corpus_size'] > 0
        assert stats['has_suffix_array_cache'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
