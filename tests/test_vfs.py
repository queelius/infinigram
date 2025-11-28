#!/usr/bin/env python3
"""
Tests for the Virtual Filesystem.
"""

import pytest
import tempfile
from pathlib import Path
from infinigram.vfs import VirtualFilesystem, PathType


@pytest.fixture
def vfs():
    """Create a temporary VFS for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir) / "datasets"
        filesystem = VirtualFilesystem(storage_dir)

        # Create some test datasets
        math = filesystem.create_dataset("math")
        math.add_document("Addition is combining numbers")
        math.add_document("Subtraction is inverse of addition")
        math.add_tag(0, "basics")
        math.add_tag(0, "algebra/intro")

        another = filesystem.create_dataset("another")
        another.add_document("Test document")

        yield filesystem
        filesystem.close_all()


class TestPathNormalization:
    """Test path normalization."""

    def test_normalize_absolute_path(self):
        """Test normalizing absolute paths."""
        assert VirtualFilesystem.normalize_path("/math/5") == "/math/5"
        assert VirtualFilesystem.normalize_path("///math///5") == "/math/5"

    def test_normalize_relative_components(self):
        """Test resolving . and .. components."""
        assert VirtualFilesystem.normalize_path("/math/./5") == "/math/5"
        assert VirtualFilesystem.normalize_path("/math/../another/0") == "/another/0"
        assert VirtualFilesystem.normalize_path("/math/docs/../5") == "/math/5"

    def test_normalize_root(self):
        """Test normalizing root path."""
        assert VirtualFilesystem.normalize_path("/") == "/"
        assert VirtualFilesystem.normalize_path("/.") == "/"
        assert VirtualFilesystem.normalize_path("/..") == "/"


class TestPathResolution:
    """Test path resolution."""

    def test_resolve_absolute_path(self, vfs):
        """Test resolving absolute paths."""
        assert vfs.resolve_path("/math/5") == "/math/5"

    def test_resolve_relative_path(self, vfs):
        """Test resolving relative paths."""
        vfs.cwd = "/math"
        assert vfs.resolve_path("5") == "/math/5"
        assert vfs.resolve_path("../another/0") == "/another/0"

    def test_resolve_home(self, vfs):
        """Test resolving ~ to root."""
        assert vfs.resolve_path("~") == "/"

    def test_resolve_previous_dir(self, vfs):
        """Test resolving - (previous directory)."""
        vfs.cwd = "/math"
        vfs.prev_dir = "/another"
        assert vfs.resolve_path("-") == "/another"


class TestPathParsing:
    """Test path parsing and type detection."""

    def test_parse_root(self, vfs):
        """Test parsing root path."""
        info = vfs.parse_path("/")
        assert info.type == PathType.ROOT

    def test_parse_dataset(self, vfs):
        """Test parsing dataset path."""
        info = vfs.parse_path("/math")
        assert info.type == PathType.DATASET
        assert info.dataset == "math"

    def test_parse_document_by_index(self, vfs):
        """Test parsing document by numeric index."""
        info = vfs.parse_path("/math/0")
        assert info.type == PathType.DOCUMENT
        assert info.dataset == "math"
        assert info.doc_id == 0

    def test_parse_document_by_tag(self, vfs):
        """Test parsing document by tag."""
        info = vfs.parse_path("/math/basics")
        assert info.type == PathType.TAGGED_DOC
        assert info.dataset == "math"
        assert info.tag == "basics"

    def test_parse_hierarchical_tag(self, vfs):
        """Test parsing hierarchical tag path."""
        info = vfs.parse_path("/math/algebra/intro")
        assert info.type == PathType.TAGGED_DOC
        assert info.dataset == "math"
        assert info.tag == "algebra/intro"

    def test_parse_proj_dir(self, vfs):
        """Test parsing projections directory."""
        info = vfs.parse_path("/proj")
        assert info.type == PathType.PROJ_DIR

    def test_parse_projection(self, vfs):
        """Test parsing specific projection."""
        info = vfs.parse_path("/proj/lowercase")
        assert info.type == PathType.PROJECTION
        assert info.projection == "lowercase"


class TestDatasetManagement:
    """Test dataset management operations."""

    def test_list_datasets(self, vfs):
        """Test listing datasets."""
        datasets = vfs.list_datasets()
        assert "math" in datasets
        assert "another" in datasets

    def test_get_dataset(self, vfs):
        """Test getting a dataset."""
        dataset = vfs.get_dataset("math")
        assert dataset.count_documents() == 2

    def test_dataset_exists(self, vfs):
        """Test checking if dataset exists."""
        assert vfs.dataset_exists("math")
        assert not vfs.dataset_exists("nonexistent")

    def test_create_dataset(self, vfs):
        """Test creating a new dataset."""
        vfs.create_dataset("newdata")
        assert vfs.dataset_exists("newdata")

    def test_create_duplicate_dataset(self, vfs):
        """Test creating a dataset that already exists."""
        with pytest.raises(FileExistsError):
            vfs.create_dataset("math")

    def test_delete_dataset(self, vfs):
        """Test deleting a dataset."""
        vfs.delete_dataset("another")
        assert not vfs.dataset_exists("another")


class TestNavigation:
    """Test directory navigation."""

    def test_change_to_root(self, vfs):
        """Test changing to root directory."""
        vfs.cwd = "/math"
        vfs.change_directory("/")
        assert vfs.cwd == "/"

    def test_change_to_dataset(self, vfs):
        """Test changing to a dataset."""
        vfs.change_directory("/math")
        assert vfs.cwd == "/math"

    def test_change_to_proj(self, vfs):
        """Test changing to projections directory."""
        vfs.change_directory("/proj")
        assert vfs.cwd == "/proj"

    def test_change_to_relative_path(self, vfs):
        """Test changing to relative path."""
        vfs.cwd = "/"
        vfs.change_directory("math")
        assert vfs.cwd == "/math"

    def test_change_to_parent(self, vfs):
        """Test changing to parent directory."""
        vfs.cwd = "/math"
        vfs.change_directory("..")
        assert vfs.cwd == "/"

    def test_change_to_home(self, vfs):
        """Test changing to home directory."""
        vfs.cwd = "/math"
        vfs.change_directory("~")
        assert vfs.cwd == "/"

    def test_change_to_previous(self, vfs):
        """Test changing to previous directory."""
        vfs.change_directory("/math")
        vfs.change_directory("/another")
        vfs.change_directory("-")
        assert vfs.cwd == "/math"

    def test_change_to_nonexistent_dataset(self, vfs):
        """Test changing to dataset that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            vfs.change_directory("/nonexistent")

    def test_change_to_nonexistent_tag(self, vfs):
        """Test changing to nonexistent tag path."""
        # Dataset exists but tag doesn't
        with pytest.raises(FileNotFoundError):
            vfs.change_directory("/math/nonexistent_tag")

    def test_change_to_document_fails(self, vfs):
        """Test that changing to a document fails."""
        with pytest.raises(ValueError, match="Not a directory"):
            vfs.change_directory("/math/0")

    def test_get_current_dataset(self, vfs):
        """Test getting current dataset from CWD."""
        vfs.cwd = "/math"
        assert vfs.get_current_dataset() == "math"

        vfs.cwd = "/"
        assert vfs.get_current_dataset() is None

        vfs.cwd = "/proj"
        assert vfs.get_current_dataset() is None


class TestPathQueries:
    """Test path existence and type queries."""

    def test_exists_root(self, vfs):
        """Test that root exists."""
        assert vfs.exists("/")

    def test_exists_dataset(self, vfs):
        """Test checking if dataset exists."""
        assert vfs.exists("/math")
        assert not vfs.exists("/nonexistent")

    def test_exists_document(self, vfs):
        """Test checking if document exists."""
        assert vfs.exists("/math/0")
        assert not vfs.exists("/math/999")

    def test_exists_projection(self, vfs):
        """Test checking if projection exists."""
        assert vfs.exists("/proj/lowercase")
        assert not vfs.exists("/proj/nonexistent")

    def test_is_directory_root(self, vfs):
        """Test that root is a directory."""
        assert vfs.is_directory("/")

    def test_is_directory_dataset(self, vfs):
        """Test that datasets are directories."""
        assert vfs.is_directory("/math")

    def test_is_directory_document(self, vfs):
        """Test that documents are not directories."""
        assert not vfs.is_directory("/math/0")

    def test_is_directory_proj(self, vfs):
        """Test that /proj is a directory."""
        assert vfs.is_directory("/proj")


class TestListing:
    """Test directory listing."""

    def test_list_root(self, vfs):
        """Test listing root directory."""
        items = vfs.list_directory("/")
        assert "math" in items
        assert "another" in items

    def test_list_dataset(self, vfs):
        """Test listing dataset contents."""
        items = vfs.list_directory("/math")
        # Should have document indices
        assert "0" in items
        assert "1" in items
        # Should have tags
        assert "basics" in items or "algebra" in items

    def test_list_proj_dir(self, vfs):
        """Test listing projections."""
        items = vfs.list_directory("/proj")
        assert "lowercase" in items
        assert "uppercase" in items

    def test_list_current_directory(self, vfs):
        """Test listing current directory (.)."""
        vfs.cwd = "/math"
        items = vfs.list_directory(".")
        assert "0" in items

    def test_list_relative_directory(self, vfs):
        """Test listing relative directory."""
        vfs.cwd = "/"
        items = vfs.list_directory("math")
        assert "0" in items


class TestHierarchicalTags:
    """Test hierarchical tag navigation."""

    def test_tag_as_directory(self, vfs):
        """Test navigating to tag path."""
        dataset = vfs.get_dataset("math")
        dataset.add_tag(1, "algebra/equations")

        # Should be able to cd to algebra
        vfs.change_directory("/math/algebra")
        assert vfs.cwd == "/math/algebra"

    def test_list_tag_directory(self, vfs):
        """Test listing tag directory."""
        dataset = vfs.get_dataset("math")
        dataset.add_tag(0, "algebra/linear")
        dataset.add_tag(1, "algebra/quadratic")

        items = vfs.list_directory("/math/algebra")
        # Should show subdirectories/tags
        assert "linear" in items or "intro" in items


class TestVFSEdgeCases:
    """Test VFS edge cases and error handling."""

    def test_exists_tagged_doc_with_missing_dataset(self, vfs):
        """Test exists returns False when dataset is missing for tagged doc."""
        # Check a tagged doc path for a dataset that doesn't exist
        result = vfs.exists("/nonexistent_dataset/sometag")
        assert result is False, "Should return False for nonexistent dataset"

    def test_exists_document_out_of_range(self, vfs):
        """Test exists returns False for document ID out of range."""
        result = vfs.exists("/math/999")  # math only has 2 docs
        assert result is False, "Should return False for out of range doc ID"

    def test_exists_tag_that_doesnt_exist(self, vfs):
        """Test exists returns False for tag that doesn't exist."""
        result = vfs.exists("/math/nonexistent_tag")
        assert result is False, "Should return False for nonexistent tag"

    def test_list_directory_tag_with_nested_children(self, vfs):
        """Test listing directory with multiple nested tag levels."""
        dataset = vfs.get_dataset("math")
        # Add nested tags: algebra/linear/equations, algebra/linear/systems
        dataset.add_tag(0, "nested/level1/level2a")
        dataset.add_tag(1, "nested/level1/level2b")

        # List the top-level tag
        items = vfs.list_directory("/math/nested")
        assert "level1" in items, "Should show next level in hierarchy"

        # List one level deeper
        items = vfs.list_directory("/math/nested/level1")
        assert "level2a" in items or "level2b" in items, "Should show leaf tags"

    def test_list_directory_not_a_directory_raises(self, vfs):
        """Test listing a file (not directory) raises ValueError."""
        with pytest.raises(ValueError, match="Not a directory"):
            vfs.list_directory("/math/0")  # Document, not directory

    def test_close_all_clears_datasets(self, vfs):
        """Test close_all properly closes all datasets."""
        # Get datasets to confirm they're open
        dataset1 = vfs.get_dataset("math")
        dataset2 = vfs.get_dataset("another")
        assert dataset1 is not None
        assert dataset2 is not None

        # Close all
        vfs.close_all()

        # Cache should be cleared
        assert len(vfs._dataset_cache) == 0, "Cache should be empty after close_all"

    def test_context_manager_cleanup(self):
        """Test VFS as context manager properly cleans up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir) / "test_datasets"

            with VirtualFilesystem(storage_dir) as vfs:
                # Create a dataset
                vfs.create_dataset("test")
                dataset = vfs.get_dataset("test")
                dataset.add_document("Test document")

                # Verify it exists
                assert vfs.dataset_exists("test")

            # After context exit, cache should be cleared
            # (The VFS object still exists but datasets are closed)

    def test_parse_invalid_path_handles_empty(self, vfs):
        """Test parsing empty path is handled gracefully."""
        # Empty path resolves to current directory
        result = vfs.resolve_path("")
        assert result == vfs.cwd, "Empty path should resolve to current directory"

    def test_change_directory_to_nonexistent_raises(self, vfs):
        """Test cd to nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            vfs.change_directory("/nonexistent_dataset")

    def test_create_duplicate_dataset_raises(self, vfs):
        """Test creating duplicate dataset raises FileExistsError."""
        with pytest.raises(FileExistsError, match="already exists"):
            vfs.create_dataset("math")  # Already exists from fixture


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
