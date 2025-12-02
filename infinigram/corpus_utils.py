#!/usr/bin/env python3
"""
Corpus utilities for Infinigram.

Provides helpers for building corpora from documents, handling separators,
and applying augmentations.
"""

from typing import List, Union, Callable, Optional, Iterator, Any
from pathlib import Path
import json


def build_corpus_from_documents(
    documents: Union[List[str], List[List[int]]],
    separator: bytes = b"\n\n",
    encoding: str = 'utf-8'
) -> List[int]:
    """
    Build a byte-level corpus from multiple documents with separators.

    Documents are separated by a byte sequence (default: double newline)
    to prevent spurious cross-document patterns.

    Args:
        documents: List of text strings or byte sequences
        separator: Byte sequence to insert between documents (default: b"\\n\\n")
        encoding: Text encoding if documents are strings (default: 'utf-8')

    Returns:
        Byte sequence with separators between documents

    Example:
        >>> docs = ["the cat sat", "the cat ran"]
        >>> corpus = build_corpus_from_documents(docs)
        >>> # Result: b"the cat sat\\n\\nthe cat ran"
        >>> # In bytes: [116, 104, 101, ..., 10, 10, ...]

        >>> # Byte-level documents
        >>> byte_docs = [[1, 2, 3], [4, 5, 6]]
        >>> corpus = build_corpus_from_documents(byte_docs, separator=b"\\x00")
        >>> # Result: [1, 2, 3, 0, 4, 5, 6]
    """
    separator_bytes = list(separator)
    corpus_bytes = []

    for i, doc in enumerate(documents):
        # Convert to bytes if needed
        if isinstance(doc, str):
            doc_bytes = list(doc.encode(encoding))
        elif isinstance(doc, (list, bytes)):
            doc_bytes = list(doc)
        else:
            raise TypeError(f"Document must be str, list, or bytes. Got {type(doc)}")

        # Validate byte range
        invalid = [b for b in doc_bytes if not (0 <= b <= 255)]
        if invalid:
            raise ValueError(
                f"Document {i} contains invalid byte values: {invalid[:10]}"
            )

        corpus_bytes.extend(doc_bytes)

        # Add separator between documents (not after last)
        if i < len(documents) - 1:
            corpus_bytes.extend(separator_bytes)

    return corpus_bytes


def build_corpus_with_augmentation(
    documents: Union[List[str], List[List[int]]],
    augmentations: List[Callable[[Union[str, List[int]]], Union[str, List[int]]]],
    separator: bytes = b"\n\n",
    encoding: str = 'utf-8'
) -> List[int]:
    """
    Build corpus with augmented variants of each document.

    Each document is augmented using the provided functions, and all variants
    (original + augmentations) are treated as separate documents in the corpus.

    Args:
        documents: List of text strings or byte sequences
        augmentations: List of augmentation functions
        separator: Byte sequence to separate documents (default: b"\\n\\n")
        encoding: Text encoding if documents are strings (default: 'utf-8')

    Returns:
        Byte sequence with original and augmented documents

    Example:
        >>> def lowercase(text):
        ...     return text.lower()
        >>> def uppercase(text):
        ...     return text.upper()
        >>>
        >>> docs = ["Hello World"]
        >>> corpus = build_corpus_with_augmentation(
        ...     docs,
        ...     augmentations=[lowercase, uppercase]
        ... )
        >>> # Corpus contains:
        >>> # "Hello World\\n\\nhello world\\n\\nHELLO WORLD"
    """
    augmented_docs = []

    for doc in documents:
        # Add original
        augmented_docs.append(doc)

        # Add augmented variants
        for aug_fn in augmentations:
            augmented_doc = aug_fn(doc)
            augmented_docs.append(augmented_doc)

    # Build corpus with separators
    return build_corpus_from_documents(
        augmented_docs,
        separator=separator,
        encoding=encoding
    )


def text_to_bytes(text: str, encoding: str = 'utf-8') -> List[int]:
    """
    Convert text to byte sequence.

    Args:
        text: Input text
        encoding: Text encoding (default: 'utf-8')

    Returns:
        Byte sequence (0-255)

    Example:
        >>> text_to_bytes("Hello")
        [72, 101, 108, 108, 111]
        >>> text_to_bytes("café")
        [99, 97, 102, 195, 169]  # UTF-8 encoding of é
    """
    return list(text.encode(encoding))


def bytes_to_text(byte_sequence: List[int], encoding: str = 'utf-8') -> str:
    """
    Convert byte sequence to text.

    Args:
        byte_sequence: Byte sequence (0-255)
        encoding: Text encoding (default: 'utf-8')

    Returns:
        Decoded text (invalid sequences replaced with �)

    Example:
        >>> bytes_to_text([72, 101, 108, 108, 111])
        'Hello'
        >>> bytes_to_text([99, 97, 102, 195, 169])
        'café'
    """
    return bytes(byte_sequence).decode(encoding, errors='replace')


def validate_byte_sequence(byte_sequence: List[int]) -> None:
    """
    Validate that a sequence contains only valid bytes (0-255).

    Args:
        byte_sequence: Sequence to validate

    Raises:
        ValueError: If sequence contains invalid values

    Example:
        >>> validate_byte_sequence([0, 128, 255])  # OK
        >>> validate_byte_sequence([256])  # Raises ValueError
    """
    invalid = [b for b in byte_sequence if not (0 <= b <= 255)]
    if invalid:
        raise ValueError(
            f"Sequence contains {len(invalid)} invalid byte values: "
            f"{invalid[:10]}{'...' if len(invalid) > 10 else ''}"
        )


# =============================================================================
# Large Dataset Loading Utilities
# =============================================================================

def load_text_file(path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Load text from a file.

    Args:
        path: Path to text file
        encoding: Text encoding (default: 'utf-8')

    Returns:
        File contents as string
    """
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def iter_jsonl(
    path: Union[str, Path],
    text_field: str = 'text',
    encoding: str = 'utf-8'
) -> Iterator[str]:
    """
    Iterate over text documents in a JSONL file.

    Args:
        path: Path to JSONL file
        text_field: Field name containing text (default: 'text')
        encoding: Text encoding (default: 'utf-8')

    Yields:
        Text content from each line

    Example:
        >>> for doc in iter_jsonl("corpus.jsonl"):
        ...     print(len(doc))
    """
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if text_field in obj:
                    yield obj[text_field]


def build_corpus_from_jsonl(
    path: Union[str, Path],
    text_field: str = 'text',
    separator: bytes = b"\n\n",
    max_documents: Optional[int] = None,
    encoding: str = 'utf-8'
) -> List[int]:
    """
    Build corpus from a JSONL file.

    Args:
        path: Path to JSONL file
        text_field: Field name containing text (default: 'text')
        separator: Byte sequence between documents (default: b"\\n\\n")
        max_documents: Maximum documents to load (None = all)
        encoding: Text encoding (default: 'utf-8')

    Returns:
        Byte corpus as List[int]

    Example:
        >>> corpus = build_corpus_from_jsonl("wiki.jsonl", max_documents=1000)
        >>> model = Infinigram(corpus)
    """
    documents = []
    for i, text in enumerate(iter_jsonl(path, text_field, encoding)):
        if max_documents is not None and i >= max_documents:
            break
        documents.append(text)

    return build_corpus_from_documents(documents, separator=separator, encoding=encoding)


def load_huggingface_dataset(
    dataset_name: str,
    config: Optional[str] = None,
    split: str = 'train',
    text_field: str = 'text',
    max_documents: Optional[int] = None,
    separator: bytes = b"\n\n",
    streaming: bool = True,
    trust_remote_code: bool = False
) -> List[int]:
    """
    Load corpus from a HuggingFace dataset.

    Requires the `datasets` library: pip install datasets

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'wikitext', 'bookcorpus')
        config: Dataset configuration name (e.g., 'wikitext-2-raw-v1')
        split: Dataset split (default: 'train')
        text_field: Field containing text (default: 'text')
        max_documents: Maximum documents to load (None = all, use with caution!)
        separator: Byte sequence between documents (default: b"\\n\\n")
        streaming: Use streaming mode to avoid downloading entire dataset
        trust_remote_code: Allow running custom code from the dataset

    Returns:
        Byte corpus as List[int]

    Example:
        >>> # Load WikiText-103 (first 10k documents)
        >>> corpus = load_huggingface_dataset(
        ...     "wikitext",
        ...     config="wikitext-103-raw-v1",
        ...     split="train",
        ...     text_field="text",
        ...     max_documents=10000
        ... )
        >>> model = Infinigram(corpus)

        >>> # Load WikiText-2 (smaller, for testing)
        >>> corpus = load_huggingface_dataset(
        ...     "wikitext",
        ...     config="wikitext-2-raw-v1",
        ...     split="train",
        ...     max_documents=1000
        ... )
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with: pip install datasets"
        )

    config_str = f"/{config}" if config else ""
    print(f"Loading {dataset_name}{config_str} ({split})...")

    # Load dataset (streaming to avoid memory issues)
    kwargs = {
        'split': split,
        'streaming': streaming,
        'trust_remote_code': trust_remote_code,
    }

    if streaming:
        dataset = load_dataset(dataset_name, config, **kwargs)
    else:
        dataset = load_dataset(dataset_name, config, **kwargs)

    documents = []
    for i, example in enumerate(dataset):
        if max_documents is not None and i >= max_documents:
            break

        text = example.get(text_field, '')
        if text and text.strip():
            documents.append(text)

        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1} documents...")

    print(f"Building corpus from {len(documents)} documents...")
    corpus = build_corpus_from_documents(documents, separator=separator)
    print(f"Corpus size: {len(corpus):,} bytes ({len(corpus) / 1_000_000:.1f} MB)")

    return corpus


def save_corpus_to_jsonl(
    documents: List[str],
    path: Union[str, Path],
    text_field: str = 'text',
    encoding: str = 'utf-8'
) -> None:
    """
    Save documents to a JSONL file.

    Args:
        documents: List of text documents
        path: Output path
        text_field: Field name for text (default: 'text')
        encoding: Text encoding (default: 'utf-8')

    Example:
        >>> docs = ["Document 1", "Document 2"]
        >>> save_corpus_to_jsonl(docs, "corpus.jsonl")
    """
    with open(path, 'w', encoding=encoding) as f:
        for doc in documents:
            obj = {text_field: doc}
            f.write(json.dumps(obj) + '\n')
