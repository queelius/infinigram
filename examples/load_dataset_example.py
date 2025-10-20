#!/usr/bin/env python3
"""
Example: Loading datasets in Infinigram

Shows different ways to create and load datasets.
"""

from pathlib import Path
from infinigram.vfs import VirtualFilesystem
from infinigram.storage import Dataset


def example_1_vfs_relative():
    """Example 1: Using VFS with relative path."""
    print("=" * 60)
    print("Example 1: VFS with Relative Path")
    print("=" * 60)

    # Use relative path for storage
    vfs = VirtualFilesystem(storage_dir=Path("./example_datasets"))

    # Create a dataset
    print("\nCreating dataset 'math'...")
    math = vfs.create_dataset("math")
    math.add_document("Addition is combining numbers")
    math.add_document("Subtraction is inverse of addition")
    math.add_tag(0, "basics/addition")
    math.add_tag(1, "basics/subtraction")
    print(f"Created dataset with {math.count_documents()} documents")

    # Load the dataset (simulating a new session)
    print("\nLoading dataset 'math'...")
    vfs2 = VirtualFilesystem(storage_dir=Path("./example_datasets"))
    loaded = vfs2.get_dataset("math")
    print(f"Loaded dataset with {loaded.count_documents()} documents")
    print(f"Document 0: {loaded.read_document(0)}")
    print(f"Document 1: {loaded.read_document(1)}")

    # List all datasets
    print(f"\nAvailable datasets: {vfs.list_datasets()}")

    # Clean up
    vfs.delete_dataset("math")
    print("\nCleaned up dataset")


def example_2_direct_loading():
    """Example 2: Direct dataset loading."""
    print("\n" + "=" * 60)
    print("Example 2: Direct Dataset Loading")
    print("=" * 60)

    # Create dataset with relative path
    dataset_path = Path("./example_data/science")
    print(f"\nCreating dataset at: {dataset_path.absolute()}")

    dataset = Dataset(dataset_path)
    dataset.add_document("Physics studies matter and energy")
    dataset.add_document("Chemistry studies substances")
    dataset.add_tag(0, "physics")
    dataset.add_tag(1, "chemistry")
    print(f"Created dataset with {dataset.count_documents()} documents")

    # Close and reload
    dataset.close()

    print("\nReloading dataset...")
    dataset2 = Dataset(dataset_path)
    print(f"Loaded dataset with {dataset2.count_documents()} documents")
    print(f"Document 0: {dataset2.read_document(0)}")

    # Clean up
    dataset2.close()
    import shutil
    shutil.rmtree(dataset_path.parent)
    print("\nCleaned up dataset")


def example_3_check_existence():
    """Example 3: Checking if dataset exists."""
    print("\n" + "=" * 60)
    print("Example 3: Checking Dataset Existence")
    print("=" * 60)

    vfs = VirtualFilesystem(storage_dir=Path("./example_datasets"))

    dataset_name = "history"

    # Check and create if needed
    if vfs.dataset_exists(dataset_name):
        print(f"\nDataset '{dataset_name}' exists, loading...")
        dataset = vfs.get_dataset(dataset_name)
    else:
        print(f"\nDataset '{dataset_name}' doesn't exist, creating...")
        dataset = vfs.create_dataset(dataset_name)
        dataset.add_document("Ancient Rome was founded in 753 BC")
        dataset.add_tag(0, "ancient/rome")

    print(f"Dataset has {dataset.count_documents()} documents")

    # Clean up
    vfs.delete_dataset(dataset_name)
    print("\nCleaned up dataset")


def example_4_home_directory():
    """Example 4: Using home directory."""
    print("\n" + "=" * 60)
    print("Example 4: Home Directory Storage")
    print("=" * 60)

    # Store in home directory
    storage_dir = Path.home() / ".infinigram_example" / "datasets"
    print(f"\nStorage directory: {storage_dir}")

    vfs = VirtualFilesystem(storage_dir=storage_dir)

    # Create dataset
    dataset = vfs.create_dataset("temp")
    dataset.add_document("Test document")
    print(f"Created dataset at: {storage_dir / 'temp'}")

    # List datasets
    print(f"Datasets in home: {vfs.list_datasets()}")

    # Clean up
    vfs.delete_dataset("temp")
    import shutil
    if storage_dir.parent.exists():
        shutil.rmtree(storage_dir.parent)
    print("\nCleaned up")


def example_5_navigation():
    """Example 5: Unix-like navigation."""
    print("\n" + "=" * 60)
    print("Example 5: Unix-like Navigation")
    print("=" * 60)

    vfs = VirtualFilesystem(storage_dir=Path("./example_datasets"))

    # Create some datasets
    for name in ["math", "science", "history"]:
        ds = vfs.create_dataset(name)
        ds.add_document(f"This is {name}")

    # Navigate like Unix
    print("\nCurrent directory:", vfs.cwd)

    vfs.change_directory("/math")
    print("After 'cd /math':", vfs.cwd)

    vfs.change_directory("..")
    print("After 'cd ..':", vfs.cwd)

    vfs.change_directory("science")
    print("After 'cd science':", vfs.cwd)

    vfs.change_directory("-")
    print("After 'cd -' (previous):", vfs.cwd)

    # List directory
    print("\nContents of '/':", vfs.list_directory("/"))

    # Clean up
    for name in ["math", "science", "history"]:
        vfs.delete_dataset(name)
    print("\nCleaned up datasets")


if __name__ == "__main__":
    print("Infinigram Dataset Loading Examples")
    print("====================================\n")

    example_1_vfs_relative()
    example_2_direct_loading()
    example_3_check_existence()
    example_4_home_directory()
    example_5_navigation()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
