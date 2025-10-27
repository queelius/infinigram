#!/usr/bin/env python3
"""
Tests for UTF-8 handling, document separators, and token adapters.
"""

import pytest
from infinigram import Infinigram, IdentityAdapter
from infinigram.corpus_utils import (
    build_corpus_from_documents,
    build_corpus_with_augmentation,
    text_to_bytes,
    bytes_to_text,
    validate_byte_sequence
)


class TestUTF8Encoding:
    """Test UTF-8 encoding and decoding."""

    def test_ascii_text_to_bytes(self):
        """Test ASCII text conversion."""
        text = "Hello World"
        byte_seq = text_to_bytes(text)

        # ASCII is single-byte UTF-8
        assert len(byte_seq) == len(text)
        assert all(0 <= b <= 127 for b in byte_seq)  # ASCII range

        # Round-trip
        decoded = bytes_to_text(byte_seq)
        assert decoded == text

    def test_utf8_multibyte_characters(self):
        """Test multi-byte UTF-8 characters."""
        text = "café"  # é is multi-byte in UTF-8
        byte_seq = text_to_bytes(text)

        # "café" = [c, a, f, é]
        # é = 0xC3 0xA9 in UTF-8 (2 bytes)
        assert len(byte_seq) == 5  # 4 characters, but é uses 2 bytes
        assert byte_seq == [99, 97, 102, 195, 169]

        # Round-trip
        decoded = bytes_to_text(byte_seq)
        assert decoded == text

    def test_chinese_characters(self):
        """Test Chinese UTF-8 characters (3 bytes each)."""
        text = "你好"  # "Hello" in Chinese
        byte_seq = text_to_bytes(text)

        # Each Chinese character uses 3 bytes in UTF-8
        assert len(byte_seq) == 6  # 2 chars × 3 bytes
        assert all(0 <= b <= 255 for b in byte_seq)

        # Round-trip
        decoded = bytes_to_text(byte_seq)
        assert decoded == text

    def test_emoji(self):
        """Test emoji (4-byte UTF-8 characters)."""
        text = "Hello 🔥 World"
        byte_seq = text_to_bytes(text)

        # 🔥 uses 4 bytes in UTF-8
        assert len(byte_seq) == 16  # 12 ASCII + 4 emoji bytes
        assert all(0 <= b <= 255 for b in byte_seq)

        # Round-trip
        decoded = bytes_to_text(byte_seq)
        assert decoded == text

    def test_mixed_unicode(self):
        """Test mixed ASCII, Latin, and emoji."""
        text = "Hello café 世界 🌍"
        byte_seq = text_to_bytes(text)

        assert all(0 <= b <= 255 for b in byte_seq)

        # Round-trip
        decoded = bytes_to_text(byte_seq)
        assert decoded == text

    def test_invalid_utf8_sequence(self):
        """Test handling of invalid UTF-8 sequences."""
        # Invalid UTF-8: single continuation byte
        invalid_seq = [0xFF, 0xFE]

        # Should replace invalid sequences with replacement character
        text = bytes_to_text(invalid_seq)
        assert "�" in text  # U+FFFD replacement character


class TestDocumentSeparators:
    """Test document separator handling."""

    def test_build_corpus_from_text_documents(self):
        """Test building corpus from text documents."""
        docs = ["Hello world", "Goodbye world"]
        corpus = build_corpus_from_documents(docs)

        # Should contain both documents separated by \n\n
        assert isinstance(corpus, list)
        assert all(0 <= b <= 255 for b in corpus)

        # Verify separator is present
        sep_bytes = list(b"\n\n")
        # Find separator in corpus
        has_sep = any(
            corpus[i:i+len(sep_bytes)] == sep_bytes
            for i in range(len(corpus) - len(sep_bytes))
        )
        assert has_sep

    def test_build_corpus_from_byte_documents(self):
        """Test building corpus from byte sequences."""
        docs = [[1, 2, 3], [4, 5, 6]]
        corpus = build_corpus_from_documents(docs, separator=b"\x00")

        # Should be: [1, 2, 3, 0, 4, 5, 6]
        assert corpus == [1, 2, 3, 0, 4, 5, 6]

    def test_custom_separator(self):
        """Test custom separator."""
        docs = ["doc1", "doc2"]
        corpus = build_corpus_from_documents(docs, separator=b"<SEP>")

        sep_bytes = list(b"<SEP>")
        # Verify custom separator is used
        has_sep = any(
            corpus[i:i+len(sep_bytes)] == sep_bytes
            for i in range(len(corpus) - len(sep_bytes))
        )
        assert has_sep

    def test_no_separator_after_last_document(self):
        """Test that separator is not added after last document."""
        docs = [[1, 2], [3, 4]]
        corpus = build_corpus_from_documents(docs, separator=b"\x00")

        # Should be [1, 2, 0, 3, 4] (no trailing separator)
        assert corpus == [1, 2, 0, 3, 4]
        assert corpus[-1] != 0

    def test_single_document_no_separator(self):
        """Test single document has no separator."""
        docs = ["Hello world"]
        corpus = build_corpus_from_documents(docs)

        # Should not contain separator
        text_bytes = list("Hello world".encode('utf-8'))
        assert corpus == text_bytes


class TestAugmentation:
    """Test corpus augmentation."""

    def test_build_corpus_with_augmentation(self):
        """Test building corpus with augmentations."""
        docs = ["Hello"]

        def lowercase(text):
            return text.lower()

        def uppercase(text):
            return text.upper()

        corpus = build_corpus_with_augmentation(
            docs,
            augmentations=[lowercase, uppercase]
        )

        # Should contain: "Hello\n\nhello\n\nHELLO"
        decoded = bytes_to_text(corpus)
        assert "Hello" in decoded
        assert "hello" in decoded
        assert "HELLO" in decoded

    def test_augmentation_preserves_order(self):
        """Test augmentations are applied in order."""
        docs = [[1, 2, 3]]

        def add_ten(seq):
            return [x + 10 for x in seq]

        def double(seq):
            return [x * 2 for x in seq]

        corpus = build_corpus_with_augmentation(
            docs,
            augmentations=[add_ten, double],
            separator=b"\x00"
        )

        # Original: [1, 2, 3]
        # Add ten: [11, 12, 13]
        # Double: [2, 4, 6]
        # Result: [1, 2, 3, 0, 11, 12, 13, 0, 2, 4, 6]
        assert corpus == [1, 2, 3, 0, 11, 12, 13, 0, 2, 4, 6]


class TestIdentityAdapter:
    """Test IdentityAdapter."""

    def test_identity_adapter_bytes_to_tokens(self):
        """Test bytes to tokens is identity."""
        adapter = IdentityAdapter()
        bytes_seq = [1, 2, 3, 255]
        tokens = adapter.bytes_to_tokens(bytes_seq)

        assert tokens == bytes_seq
        assert tokens is not bytes_seq  # Should be a copy

    def test_identity_adapter_tokens_to_bytes(self):
        """Test tokens to bytes is identity."""
        adapter = IdentityAdapter()
        tokens = [0, 128, 255]
        bytes_seq = adapter.tokens_to_bytes(tokens)

        assert bytes_seq == tokens
        assert bytes_seq is not tokens  # Should be a copy

    def test_identity_adapter_text_to_bytes(self):
        """Test text to bytes."""
        adapter = IdentityAdapter()
        text = "Hello"
        bytes_seq = adapter.text_to_bytes(text)

        assert bytes_seq == [72, 101, 108, 108, 111]

    def test_identity_adapter_bytes_to_text(self):
        """Test bytes to text."""
        adapter = IdentityAdapter()
        bytes_seq = [72, 101, 108, 108, 111]
        text = adapter.bytes_to_text(bytes_seq)

        assert text == "Hello"

    def test_identity_adapter_validates_byte_range(self):
        """Test adapter validates byte range."""
        adapter = IdentityAdapter()

        # Valid bytes should work
        adapter.bytes_to_tokens([0, 128, 255])

        # Invalid bytes should raise error
        with pytest.raises(ValueError, match="must contain only values 0-255"):
            adapter.bytes_to_tokens([256])

        with pytest.raises(ValueError, match="must contain only values 0-255"):
            adapter.tokens_to_bytes([-1])


class TestInfinigramWithUTF8:
    """Test Infinigram with UTF-8 text."""

    def test_infinigram_with_english_text(self):
        """Test Infinigram with English text."""
        text = "the cat sat on the mat the cat"
        corpus = list(text.encode('utf-8'))

        model = Infinigram(corpus)

        # Query: "the cat"
        context = list("the cat".encode('utf-8'))
        probs = model.predict(context)

        # Should predict ' ' (space) with high probability
        space_byte = ord(' ')
        assert space_byte in probs
        assert probs[space_byte] > 0.1

    def test_infinigram_with_unicode_text(self):
        """Test Infinigram with Unicode text."""
        text = "café café"
        corpus = list(text.encode('utf-8'))

        model = Infinigram(corpus)

        # Query: "caf"
        context = list("caf".encode('utf-8'))
        probs = model.predict(context, top_k=10)

        # Should predict 0xC3 (first byte of é)
        assert 195 in probs  # 0xC3
        assert probs[195] > 0.1

    def test_infinigram_learns_multibyte_patterns(self):
        """Test that Infinigram learns multi-byte UTF-8 patterns."""
        # Repeated pattern with Chinese characters
        text = "你好 你好 你好"
        corpus = list(text.encode('utf-8'))

        model = Infinigram(corpus, max_length=20)

        # Find pattern "你"
        ni_bytes = list("你".encode('utf-8'))  # 3 bytes
        pos, length = model.longest_suffix(ni_bytes)

        # Should find the pattern
        assert length == 3


class TestByteValidation:
    """Test byte sequence validation."""

    def test_validate_byte_sequence_valid(self):
        """Test validation of valid byte sequences."""
        validate_byte_sequence([0, 128, 255])  # Should not raise
        validate_byte_sequence([])  # Empty is valid
        validate_byte_sequence([1] * 1000)  # Large is valid

    def test_validate_byte_sequence_invalid(self):
        """Test validation rejects invalid sequences."""
        with pytest.raises(ValueError, match="invalid byte values"):
            validate_byte_sequence([256])

        with pytest.raises(ValueError, match="invalid byte values"):
            validate_byte_sequence([-1])

        with pytest.raises(ValueError, match="invalid byte values"):
            validate_byte_sequence([0, 100, 300, 500])


class TestIntegrationWithDocuments:
    """Integration tests with realistic documents."""

    def test_multi_document_corpus(self):
        """Test corpus built from multiple documents."""
        docs = [
            "The quick brown fox",
            "The lazy dog",
            "The quick cat"
        ]

        corpus = build_corpus_from_documents(docs)
        model = Infinigram(corpus, max_length=50)

        # Query: "The quick"
        context = list("The quick".encode('utf-8'))
        probs = model.predict(context, top_k=20)

        # Should see predictions from multiple documents
        # ' b' (from "brown") and ' c' (from "cat")
        assert len(probs) > 0

    def test_augmented_corpus_improves_coverage(self):
        """Test that augmentation increases pattern coverage."""
        docs = ["Hello World"]

        # Build without augmentation
        corpus_plain = build_corpus_from_documents(docs)
        model_plain = Infinigram(corpus_plain)

        # Build with augmentation
        corpus_aug = build_corpus_with_augmentation(
            docs,
            augmentations=[str.lower, str.upper]
        )
        model_aug = Infinigram(corpus_aug)

        # Augmented corpus should be larger
        assert len(corpus_aug) > len(corpus_plain)

        # Query lowercase: "hello"
        context_lower = list("hello".encode('utf-8'))

        # Augmented model should have this pattern
        pos_aug, len_aug = model_aug.longest_suffix(context_lower)
        assert len_aug > 0  # Should find "hello" from augmentation
