#!/usr/bin/env python3
"""
Virtual Filesystem for Infinigram.

Provides Unix-like path-based navigation and access to datasets.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from infinigram.storage import Dataset


class PathType(Enum):
    """Types of paths in the VFS."""
    ROOT = "/"
    DATASET = "/dataset"
    DOCUMENT = "/dataset/index"
    TAGGED_DOC = "/dataset/tag/path"
    PROJ_DIR = "/proj"
    PROJECTION = "/proj/name"


class PathInfo:
    """Information about a resolved path."""

    def __init__(self, path_type: PathType, **kwargs):
        self.type = path_type
        self.dataset = kwargs.get('dataset')
        self.doc_id = kwargs.get('doc_id')
        self.tag = kwargs.get('tag')
        self.projection = kwargs.get('projection')

    def __repr__(self):
        parts = [f"PathInfo({self.type.name}"]
        if self.dataset:
            parts.append(f"dataset={self.dataset}")
        if self.doc_id is not None:
            parts.append(f"doc_id={self.doc_id}")
        if self.tag:
            parts.append(f"tag={self.tag}")
        if self.projection:
            parts.append(f"projection={self.projection}")
        return ", ".join(parts) + ")"


class VirtualFilesystem:
    """
    Virtual filesystem for Infinigram datasets.

    Hierarchy:
        /                      (root - lists datasets)
        /<dataset>/            (dataset directory)
        /<dataset>/<index>     (document by index)
        /<dataset>/<tag>       (document by tag)
        /proj/                 (projections directory)
        /proj/<name>           (projection definition)
    """

    def __init__(self, storage_dir: Path):
        """
        Initialize VFS.

        Args:
            storage_dir: Directory containing datasets (e.g., ~/.infinigram/datasets/)
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Current working directory
        self.cwd = '/'

        # Previous directory (for cd -)
        self.prev_dir = None

        # Cache of open datasets
        self._dataset_cache: Dict[str, Dataset] = {}

        # Projection registry (built-in projections)
        self.projections = {
            'lowercase': lambda text: text.lower(),
            'uppercase': lambda text: text.upper(),
            'title': lambda text: text.title(),
            'strip': lambda text: text.strip(),
        }

    # ========================================================================
    # Path Utilities
    # ========================================================================

    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize path by resolving . and .. components.

        Examples:
            /math/../another → /another
            /math/./5 → /math/5
            math//5 → /math/5
            /.. → / (can't go above root)
        """
        if not path.startswith('/'):
            path = '/' + path

        parts = []
        for part in path.split('/'):
            if part == '..':
                # Can't go above root
                if parts:
                    parts.pop()
            elif part and part != '.':
                parts.append(part)

        return '/' + '/'.join(parts) if parts else '/'

    def resolve_path(self, path: str) -> str:
        """
        Resolve path relative to current working directory.

        Args:
            path: Absolute or relative path

        Returns:
            Absolute normalized path

        Examples:
            Current: /math
            Input: 5 → /math/5
            Input: ../another/0 → /another/0
            Input: /proj/lowercase → /proj/lowercase
        """
        if path == '~':
            return '/'
        elif path == '-':
            return self.prev_dir if self.prev_dir else self.cwd
        elif path.startswith('/'):
            # Absolute path
            return self.normalize_path(path)
        else:
            # Relative path
            combined = f"{self.cwd}/{path}"
            return self.normalize_path(combined)

    def parse_path(self, path: str) -> PathInfo:
        """
        Parse path and determine its type.

        Args:
            path: Absolute path (should be normalized)

        Returns:
            PathInfo describing the path

        Raises:
            ValueError: If path is invalid
        """
        path = self.normalize_path(path)
        parts = [p for p in path.split('/') if p]

        # Root
        if not parts:
            return PathInfo(PathType.ROOT)

        # Special directories
        if parts[0] == 'proj':
            if len(parts) == 1:
                return PathInfo(PathType.PROJ_DIR)
            elif len(parts) == 2:
                return PathInfo(PathType.PROJECTION, projection=parts[1])
            else:
                raise ValueError(f"Invalid projection path: {path}")

        # Dataset paths
        dataset_name = parts[0]

        if len(parts) == 1:
            # /dataset
            return PathInfo(PathType.DATASET, dataset=dataset_name)

        # /dataset/something
        # Could be: index (0, 1, 2...) or tag (algebra/equations)
        remainder = '/'.join(parts[1:])

        # Try to parse as index
        if remainder.isdigit():
            doc_id = int(remainder)
            return PathInfo(PathType.DOCUMENT, dataset=dataset_name, doc_id=doc_id)

        # Otherwise, it's a tag
        return PathInfo(PathType.TAGGED_DOC, dataset=dataset_name, tag=remainder)

    # ========================================================================
    # Dataset Management
    # ========================================================================

    def list_datasets(self) -> List[str]:
        """List all dataset names."""
        datasets = []
        for item in self.storage_dir.iterdir():
            if item.is_dir():
                datasets.append(item.name)
        return sorted(datasets)

    def get_dataset(self, name: str) -> Dataset:
        """
        Get dataset by name.

        Returns cached instance if available, otherwise opens it.
        """
        if name not in self._dataset_cache:
            dataset_path = self.storage_dir / name
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset '{name}' not found")
            self._dataset_cache[name] = Dataset(dataset_path)

        return self._dataset_cache[name]

    def create_dataset(self, name: str) -> Dataset:
        """Create a new dataset."""
        dataset_path = self.storage_dir / name
        if dataset_path.exists():
            raise FileExistsError(f"Dataset '{name}' already exists")

        dataset = Dataset(dataset_path)
        self._dataset_cache[name] = dataset
        return dataset

    def dataset_exists(self, name: str) -> bool:
        """Check if dataset exists."""
        return (self.storage_dir / name).exists()

    def delete_dataset(self, name: str):
        """
        Delete a dataset.

        Removes from cache and deletes directory.
        """
        # Close and remove from cache
        if name in self._dataset_cache:
            self._dataset_cache[name].close()
            del self._dataset_cache[name]

        # Delete directory
        dataset_path = self.storage_dir / name
        if dataset_path.exists():
            import shutil
            shutil.rmtree(dataset_path)

    # ========================================================================
    # Navigation
    # ========================================================================

    def change_directory(self, path: str) -> str:
        """
        Change current working directory.

        Args:
            path: Path to change to

        Returns:
            New current directory

        Raises:
            ValueError: If path doesn't exist or isn't a directory
        """
        # Handle special cases
        if not path or path == '~':
            new_path = '/'
        elif path == '-':
            new_path = self.prev_dir if self.prev_dir else self.cwd
        else:
            new_path = self.resolve_path(path)

        # Validate path exists and is a directory
        path_info = self.parse_path(new_path)

        if path_info.type == PathType.ROOT:
            # Root is always valid
            pass
        elif path_info.type == PathType.PROJ_DIR:
            # /proj is always valid
            pass
        elif path_info.type == PathType.DATASET:
            # Check dataset exists
            if not self.dataset_exists(path_info.dataset):
                raise FileNotFoundError(f"Dataset '{path_info.dataset}' not found")
        elif path_info.type == PathType.TAGGED_DOC:
            # Check dataset exists
            if not self.dataset_exists(path_info.dataset):
                raise FileNotFoundError(f"Dataset '{path_info.dataset}' not found")
            # Check that tag path exists (has documents or subtags)
            dataset = self.get_dataset(path_info.dataset)
            # Check if this exact tag exists OR if there are tags with this prefix
            exact_match = dataset.resolve_tag(path_info.tag) is not None
            prefix = path_info.tag + '/'
            cursor = dataset.db.execute(
                "SELECT COUNT(*) FROM tags WHERE tag LIKE ? OR tag = ?",
                (prefix + '%', path_info.tag)
            )
            has_subtags = cursor.fetchone()[0] > 0
            if not exact_match and not has_subtags:
                raise FileNotFoundError(f"Tag path '{path_info.tag}' not found in dataset '{path_info.dataset}'")
        elif path_info.type in (PathType.DOCUMENT, PathType.PROJECTION):
            raise ValueError(f"Not a directory: {new_path}")

        # Save previous directory
        self.prev_dir = self.cwd

        # Update current directory
        self.cwd = new_path

        return self.cwd

    def get_current_dataset(self) -> Optional[str]:
        """
        Get current dataset name from CWD.

        Returns:
            Dataset name if CWD is in a dataset, None otherwise
        """
        path_info = self.parse_path(self.cwd)

        if path_info.type in (PathType.DATASET, PathType.DOCUMENT, PathType.TAGGED_DOC):
            return path_info.dataset

        return None

    # ========================================================================
    # Path Queries
    # ========================================================================

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        try:
            resolved = self.resolve_path(path)
            path_info = self.parse_path(resolved)

            if path_info.type == PathType.ROOT:
                return True
            elif path_info.type == PathType.PROJ_DIR:
                return True
            elif path_info.type == PathType.PROJECTION:
                return path_info.projection in self.projections
            elif path_info.type == PathType.DATASET:
                return self.dataset_exists(path_info.dataset)
            elif path_info.type == PathType.DOCUMENT:
                if not self.dataset_exists(path_info.dataset):
                    return False
                dataset = self.get_dataset(path_info.dataset)
                try:
                    dataset.read_document(path_info.doc_id)
                    return True
                except IndexError:
                    return False
            elif path_info.type == PathType.TAGGED_DOC:
                if not self.dataset_exists(path_info.dataset):
                    return False
                dataset = self.get_dataset(path_info.dataset)
                return dataset.resolve_tag(path_info.tag) is not None

        except (ValueError, FileNotFoundError):
            return False

        return False

    def is_directory(self, path: str) -> bool:
        """Check if path is a directory."""
        try:
            resolved = self.resolve_path(path)
            path_info = self.parse_path(resolved)

            return path_info.type in (
                PathType.ROOT,
                PathType.DATASET,
                PathType.PROJ_DIR,
                PathType.TAGGED_DOC  # Tags can act like directories
            )
        except (ValueError, FileNotFoundError):
            return False

    # ========================================================================
    # Listing
    # ========================================================================

    def list_directory(self, path: str = '.') -> List[str]:
        """
        List contents of directory.

        Args:
            path: Directory to list (default: current directory)

        Returns:
            List of names in directory

        Raises:
            ValueError: If path is not a directory
        """
        resolved = self.resolve_path(path)
        path_info = self.parse_path(resolved)

        if path_info.type == PathType.ROOT:
            # List datasets
            return self.list_datasets()

        elif path_info.type == PathType.PROJ_DIR:
            # List projections
            return sorted(self.projections.keys())

        elif path_info.type == PathType.DATASET:
            # List documents in dataset
            dataset = self.get_dataset(path_info.dataset)
            count = dataset.count_documents()

            # Return both indices and tags
            items = [str(i) for i in range(count)]

            # Add unique tag prefixes
            cursor = dataset.db.execute("SELECT DISTINCT tag FROM tags ORDER BY tag")
            tags = [row[0] for row in cursor]

            # For hierarchical tags, only show top-level
            top_level_tags = set()
            for tag in tags:
                if '/' in tag:
                    top_level = tag.split('/')[0]
                    top_level_tags.add(top_level)
                else:
                    top_level_tags.add(tag)

            items.extend(sorted(top_level_tags))
            return items

        elif path_info.type == PathType.TAGGED_DOC:
            # List documents under this tag path
            dataset = self.get_dataset(path_info.dataset)

            # Find all tags that start with this path
            prefix = path_info.tag + '/'
            cursor = dataset.db.execute(
                "SELECT DISTINCT tag FROM tags WHERE tag LIKE ? OR tag = ? ORDER BY tag",
                (prefix + '%', path_info.tag)
            )

            items = []
            for (tag,) in cursor:
                if tag == path_info.tag:
                    # This exact tag exists
                    items.append(str(dataset.resolve_tag(tag)))
                elif tag.startswith(prefix):
                    # Child tag
                    remainder = tag[len(prefix):]
                    if '/' in remainder:
                        # Nested deeper - show next level
                        next_level = remainder.split('/')[0]
                        if next_level not in items:
                            items.append(next_level)
                    else:
                        # Leaf tag
                        items.append(remainder)

            return sorted(set(items))

        else:
            raise ValueError(f"Not a directory: {path}")

    # ========================================================================
    # Cleanup
    # ========================================================================

    def close_all(self):
        """Close all open datasets."""
        for dataset in self._dataset_cache.values():
            dataset.close()
        self._dataset_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all()
