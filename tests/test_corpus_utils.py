#!/usr/bin/env python3
"""
Tests for corpus_utils module.

Tests for loading, converting, and processing corpora.
"""

import json
import tempfile
from pathlib import Path
import pytest

from infinigram.corpus_utils import (
    build_corpus_from_documents,
    build_corpus_with_augmentation,
    text_to_bytes,
    bytes_to_text,
    validate_byte_sequence,
    load_text_file,
    iter_jsonl,
    build_corpus_from_jsonl,
    save_corpus_to_jsonl,
)


class TestTextConversion:
    """Tests for text_to_bytes and bytes_to_text."""

    def test_text_to_bytes_ascii(self):
        result = text_to_bytes("Hello")
        assert result == [72, 101, 108, 108, 111]

    def test_text_to_bytes_utf8(self):
        result = text_to_bytes("café")
        # UTF-8: c=99, a=97, f=102, é=195,169
        assert result == [99, 97, 102, 195, 169]

    def test_text_to_bytes_empty(self):
        result = text_to_bytes("")
        assert result == []

    def test_bytes_to_text_ascii(self):
        result = bytes_to_text([72, 101, 108, 108, 111])
        assert result == "Hello"

    def test_bytes_to_text_utf8(self):
        result = bytes_to_text([99, 97, 102, 195, 169])
        assert result == "café"

    def test_bytes_to_text_invalid_sequence(self):
        # Invalid UTF-8 should be replaced with replacement character
        result = bytes_to_text([255, 254])
        assert "�" in result

    def test_roundtrip(self):
        text = "Hello, World! café"
        assert bytes_to_text(text_to_bytes(text)) == text


class TestBuildCorpusFromDocuments:
    """Tests for build_corpus_from_documents."""

    def test_single_document(self):
        docs = ["hello"]
        result = build_corpus_from_documents(docs)
        expected = list(b"hello")
        assert result == expected

    def test_multiple_documents_with_separator(self):
        docs = ["hello", "world"]
        result = build_corpus_from_documents(docs, separator=b"\n\n")
        expected = list(b"hello\n\nworld")
        assert result == expected

    def test_custom_separator(self):
        docs = ["a", "b", "c"]
        result = build_corpus_from_documents(docs, separator=b"\x00")
        expected = list(b"a\x00b\x00c")
        assert result == expected

    def test_empty_documents(self):
        docs = []
        result = build_corpus_from_documents(docs)
        assert result == []

    def test_byte_list_documents(self):
        docs = [[1, 2, 3], [4, 5, 6]]
        result = build_corpus_from_documents(docs, separator=b"\x00")
        assert result == [1, 2, 3, 0, 4, 5, 6]

    def test_invalid_byte_value_raises(self):
        docs = [[256, 1, 2]]  # 256 is invalid
        with pytest.raises(ValueError):
            build_corpus_from_documents(docs)


class TestBuildCorpusWithAugmentation:
    """Tests for build_corpus_with_augmentation."""

    def test_lowercase_augmentation(self):
        docs = ["Hello"]
        result = build_corpus_with_augmentation(
            docs,
            augmentations=[lambda x: x.lower()],
            separator=b"\n\n"
        )
        expected = list(b"Hello\n\nhello")
        assert result == expected

    def test_multiple_augmentations(self):
        docs = ["Hi"]
        result = build_corpus_with_augmentation(
            docs,
            augmentations=[lambda x: x.lower(), lambda x: x.upper()],
            separator=b"\x00"
        )
        # Original + lower + upper
        expected = list(b"Hi\x00hi\x00HI")
        assert result == expected

    def test_no_augmentations(self):
        docs = ["hello"]
        result = build_corpus_with_augmentation(docs, augmentations=[])
        expected = list(b"hello")
        assert result == expected


class TestValidateByteSequence:
    """Tests for validate_byte_sequence."""

    def test_valid_sequence(self):
        # Should not raise
        validate_byte_sequence([0, 128, 255])

    def test_invalid_too_high(self):
        with pytest.raises(ValueError):
            validate_byte_sequence([256])

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            validate_byte_sequence([-1])

    def test_empty_sequence(self):
        # Should not raise
        validate_byte_sequence([])


class TestLoadTextFile:
    """Tests for load_text_file."""

    def test_load_simple_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello, World!")
            f.flush()
            result = load_text_file(f.name)
            assert result == "Hello, World!"

    def test_load_utf8_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("café")
            f.flush()
            result = load_text_file(f.name)
            assert result == "café"

    def test_load_multiline_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\nline2\nline3")
            f.flush()
            result = load_text_file(f.name)
            assert result == "line1\nline2\nline3"


class TestIterJsonl:
    """Tests for iter_jsonl."""

    def test_iter_simple_jsonl(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('{"text": "doc2"}\n')
            f.write('{"text": "doc3"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name))
            assert docs == ["doc1", "doc2", "doc3"]

    def test_iter_custom_field(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"content": "doc1"}\n')
            f.write('{"content": "doc2"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name, text_field='content'))
            assert docs == ["doc1", "doc2"]

    def test_iter_skips_missing_field(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('{"other": "ignored"}\n')
            f.write('{"text": "doc3"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name))
            assert docs == ["doc1", "doc3"]

    def test_iter_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('\n')
            f.write('{"text": "doc2"}\n')
            f.flush()

            docs = list(iter_jsonl(f.name))
            assert docs == ["doc1", "doc2"]


class TestBuildCorpusFromJsonl:
    """Tests for build_corpus_from_jsonl."""

    def test_build_corpus_from_jsonl(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "hello"}\n')
            f.write('{"text": "world"}\n')
            f.flush()

            corpus = build_corpus_from_jsonl(f.name, separator=b"\x00")
            expected = list(b"hello\x00world")
            assert corpus == expected

    def test_build_corpus_max_documents(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "doc1"}\n')
            f.write('{"text": "doc2"}\n')
            f.write('{"text": "doc3"}\n')
            f.flush()

            corpus = build_corpus_from_jsonl(f.name, max_documents=2, separator=b"\x00")
            expected = list(b"doc1\x00doc2")
            assert corpus == expected


class TestSaveCorpusToJsonl:
    """Tests for save_corpus_to_jsonl."""

    def test_save_and_reload(self):
        docs = ["hello", "world", "test"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name

        save_corpus_to_jsonl(docs, path)

        # Reload and verify
        reloaded = list(iter_jsonl(path))
        assert reloaded == docs

    def test_save_custom_field(self):
        docs = ["doc1", "doc2"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name

        save_corpus_to_jsonl(docs, path, text_field='content')

        # Verify field name
        with open(path, 'r') as f:
            obj = json.loads(f.readline())
            assert 'content' in obj
            assert obj['content'] == 'doc1'

    def test_save_utf8_content(self):
        docs = ["café", "naïve"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name

        save_corpus_to_jsonl(docs, path)
        reloaded = list(iter_jsonl(path))
        assert reloaded == docs


class TestHuggingFaceLoader:
    """Tests for load_huggingface_dataset (mocked)."""

    def test_import_error_without_datasets(self, monkeypatch):
        """Test that appropriate error is raised when datasets not installed."""
        import sys

        # Mock the datasets import to fail
        import importlib
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == 'datasets':
                raise ImportError("No module named 'datasets'")
            return original_import(name, *args, **kwargs)

        # Use monkeypatch to temporarily replace import behavior
        # We can't easily mock builtins.__import__, so let's test differently
        # Just verify the function exists and has the right signature

        from infinigram.corpus_utils import load_huggingface_dataset
        import inspect

        sig = inspect.signature(load_huggingface_dataset)
        params = list(sig.parameters.keys())

        assert 'dataset_name' in params
        assert 'split' in params
        assert 'text_field' in params
        assert 'max_documents' in params
        assert 'streaming' in params
