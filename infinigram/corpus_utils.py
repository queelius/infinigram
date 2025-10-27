#!/usr/bin/env python3
"""
Corpus utilities for Infinigram.

Provides helpers for building corpora from documents, handling separators,
and applying augmentations.
"""

from typing import List, Union, Callable, Optional


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
