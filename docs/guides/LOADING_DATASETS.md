# Loading Datasets in Infinigram

## Quick Answer

**No, you don't need an absolute path!** You can use either:
1. **Relative paths** - relative to your storage directory
2. **Absolute paths** - if you want full control

## Basic Usage

### Option 1: Using VirtualFilesystem (Recommended)

The `VirtualFilesystem` manages datasets for you with Unix-like paths:

```python
from pathlib import Path
from infinigram.vfs import VirtualFilesystem

# Create VFS with a storage directory
# This can be relative or absolute
vfs = VirtualFilesystem(storage_dir=Path("./datasets"))
# OR
vfs = VirtualFilesystem(storage_dir=Path.home() / ".infinigram" / "datasets")

# Load existing dataset by name (no path needed!)
dataset = vfs.get_dataset("math")

# The dataset is located at: storage_dir/math/
```

### Option 2: Using Dataset Directly

If you want direct control, use the `Dataset` class:

```python
from pathlib import Path
from infinigram.storage import Dataset

# Relative path (relative to current working directory)
dataset = Dataset(Path("./my_datasets/math"))

# Absolute path
dataset = Dataset(Path("/home/user/datasets/math"))

# Using home directory
dataset = Dataset(Path.home() / "datasets" / "math")
```

## Complete Examples

### Example 1: Create and Load with VFS

```python
from pathlib import Path
from infinigram.vfs import VirtualFilesystem

# Setup VFS with relative path
vfs = VirtualFilesystem(storage_dir=Path("./infinigram_data"))

# Create a new dataset
math_dataset = vfs.create_dataset("math")

# Add some documents
math_dataset.add_document("Addition is combining numbers")
math_dataset.add_document("Subtraction is the inverse of addition")
math_dataset.add_tag(0, "basics")

# Later, load the dataset (same session or different session)
vfs2 = VirtualFilesystem(storage_dir=Path("./infinigram_data"))
loaded_dataset = vfs2.get_dataset("math")

print(f"Documents: {loaded_dataset.count_documents()}")  # Output: 2
print(loaded_dataset.get_document(0))  # Output: "Addition is combining numbers"
```

### Example 2: Direct Dataset Loading

```python
from pathlib import Path
from infinigram.storage import Dataset

# Load dataset from relative path
dataset = Dataset(Path("./data/my_corpus"))

# Add documents
dataset.add_document("The cat sat on the mat")
dataset.add_tag(0, "animals/cat")

# Later, load from the same path
dataset2 = Dataset(Path("./data/my_corpus"))
print(dataset2.count_documents())  # Output: 1
```

### Example 3: Using Environment Variables

```python
import os
from pathlib import Path
from infinigram.vfs import VirtualFilesystem

# Get storage directory from environment variable
storage_dir = Path(os.getenv("INFINIGRAM_DATA", "./infinigram_data"))

vfs = VirtualFilesystem(storage_dir=storage_dir)
dataset = vfs.get_dataset("my_dataset")
```

## Path Resolution Details

### VirtualFilesystem

When you use `VirtualFilesystem(storage_dir=Path("./datasets"))`:

1. The `storage_dir` is where ALL datasets are stored
2. Each dataset is a subdirectory: `storage_dir/dataset_name/`
3. You access datasets by name, not by full path

```
./datasets/              # storage_dir
├── math/               # dataset "math"
│   ├── documents.jsonl
│   ├── index.db
│   └── metadata.json
├── science/            # dataset "science"
│   ├── documents.jsonl
│   ├── index.db
│   └── metadata.json
└── history/            # dataset "history"
    ├── documents.jsonl
    ├── index.db
    └── metadata.json
```

To load:
```python
vfs = VirtualFilesystem(Path("./datasets"))
math_ds = vfs.get_dataset("math")         # loads ./datasets/math/
science_ds = vfs.get_dataset("science")   # loads ./datasets/science/
```

### Dataset Direct

When you use `Dataset(path)`:

1. The `path` is the **full path to the dataset directory**
2. Can be relative or absolute
3. The directory contains: `documents.jsonl`, `index.db`, `metadata.json`

```python
# These all work:
Dataset(Path("./my_data"))                    # relative
Dataset(Path("/home/user/data"))              # absolute
Dataset(Path.home() / "infinigram" / "data")  # home directory
Dataset(Path.cwd() / "datasets" / "math")     # current working dir
```

## Common Patterns

### Pattern 1: Default Location

```python
from pathlib import Path
from infinigram.vfs import VirtualFilesystem

# Use a consistent default location
DEFAULT_STORAGE = Path.home() / ".infinigram" / "datasets"

vfs = VirtualFilesystem(storage_dir=DEFAULT_STORAGE)
```

### Pattern 2: Project-Local Data

```python
from pathlib import Path
from infinigram.vfs import VirtualFilesystem

# Store datasets in project directory
vfs = VirtualFilesystem(storage_dir=Path("./data"))
```

### Pattern 3: Shared System Data

```python
from pathlib import Path
from infinigram.vfs import VirtualFilesystem

# System-wide location
vfs = VirtualFilesystem(storage_dir=Path("/var/lib/infinigram/datasets"))
```

## Checking if Dataset Exists

### With VFS

```python
vfs = VirtualFilesystem(Path("./datasets"))

if vfs.dataset_exists("math"):
    dataset = vfs.get_dataset("math")
else:
    dataset = vfs.create_dataset("math")
```

### With Dataset

```python
from pathlib import Path

dataset_path = Path("./datasets/math")

if dataset_path.exists():
    dataset = Dataset(dataset_path)
else:
    # Create new (Dataset creates directory automatically)
    dataset = Dataset(dataset_path)
```

## Listing Available Datasets

```python
vfs = VirtualFilesystem(Path("./datasets"))

# List all datasets
datasets = vfs.list_datasets()
print(f"Available datasets: {datasets}")

# Load each one
for name in datasets:
    ds = vfs.get_dataset(name)
    print(f"{name}: {ds.count_documents()} documents")
```

## Error Handling

```python
from infinigram.vfs import VirtualFilesystem
from pathlib import Path

vfs = VirtualFilesystem(Path("./datasets"))

try:
    dataset = vfs.get_dataset("nonexistent")
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
    # Create it instead
    dataset = vfs.create_dataset("nonexistent")
```

## Best Practices

### ✅ Recommended

1. **Use VirtualFilesystem** for most cases - it manages paths for you
2. **Use relative paths** for project-specific data
3. **Use `Path.home()`** for user-specific data
4. **Check existence** before loading if not sure dataset exists

### ❌ Avoid

1. Hard-coding absolute paths in code
2. Using string paths instead of `Path` objects
3. Creating datasets in system directories without permissions

## Summary

**To answer your question directly:**

```python
# NO absolute path needed!

from pathlib import Path
from infinigram.vfs import VirtualFilesystem

# Use relative path for storage directory
vfs = VirtualFilesystem(Path("./my_datasets"))

# Load dataset by name (just the name, not a path!)
dataset = vfs.get_dataset("math")

# The dataset is automatically loaded from: ./my_datasets/math/
```

The path you provide to `VirtualFilesystem()` can be relative or absolute - your choice! Then you access datasets by simple names, no paths needed.
