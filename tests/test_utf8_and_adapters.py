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

    def test_repr(self):
        """Test IdentityAdapter repr includes class name and encoding."""
        adapter = IdentityAdapter()
        repr_str = repr(adapter)
        assert "IdentityAdapter" in repr_str, "repr should include class name"
        assert "utf-8" in repr_str, "repr should include default encoding"

    def test_repr_with_custom_encoding(self):
        """Test IdentityAdapter repr with custom encoding."""
        adapter = IdentityAdapter(encoding='latin-1')
        repr_str = repr(adapter)
        assert "IdentityAdapter" in repr_str
        assert "latin-1" in repr_str, "repr should show custom encoding"

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


class MockTokenizer:
    """Mock tokenizer for testing TokenProbabilityAdapter."""

    def __init__(self, vocab: dict = None):
        """
        Initialize mock tokenizer.

        Args:
            vocab: Optional dict mapping token_id -> text
        """
        # Simple vocabulary: ASCII printable characters as single-char tokens
        self._vocab = vocab or {i: chr(i) for i in range(32, 127)}
        self._reverse = {v: k for k, v in self._vocab.items()}

    def encode(self, text: str) -> list:
        """Encode text to token IDs (character-level)."""
        return [self._reverse.get(c, ord(c)) for c in text]

    def decode(self, tokens: list) -> str:
        """Decode token IDs to text."""
        return ''.join(self._vocab.get(t, chr(t)) for t in tokens)

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._vocab)


class TestTokenProbabilityAdapter:
    """Tests for TokenProbabilityAdapter."""

    def test_import(self):
        """Test TokenProbabilityAdapter can be imported."""
        from infinigram.adapters import TokenProbabilityAdapter
        assert TokenProbabilityAdapter is not None

    def test_init(self):
        """Test adapter initialization."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"the cat sat on the mat")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        assert adapter.model is model
        assert adapter.tokenizer is tokenizer

    def test_token_probability_single_byte(self):
        """Test probability for single-byte token."""
        from infinigram.adapters import TokenProbabilityAdapter

        # Corpus where 'a' always follows 'c'
        corpus = list(b"ca ca ca ca")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        # Token 'a' (ASCII 97) should have high probability after 'c'
        prob = adapter.token_probability("c", ord('a'))
        assert prob > 0.5

    def test_token_probability_multi_byte(self):
        """Test probability for multi-byte token."""
        from infinigram.adapters import TokenProbabilityAdapter

        # Create tokenizer with multi-char tokens
        vocab = {
            1: "the",
            2: " cat",
            3: " sat",
        }
        tokenizer = MockTokenizer(vocab)

        corpus = list(b"the cat sat the cat sat the cat")
        model = Infinigram(corpus)

        adapter = TokenProbabilityAdapter(model, tokenizer)

        # After "the", " cat" (token 2) should have high probability
        prob_cat = adapter.token_probability("the", 2)
        assert prob_cat > 0.1

    def test_token_log_probability(self):
        """Test log probability computation."""
        from infinigram.adapters import TokenProbabilityAdapter
        import math

        corpus = list(b"ab ab ab ab")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        # Get both log and linear probability
        log_p = adapter.token_log_probability("a", ord('b'))
        p = adapter.token_probability("a", ord('b'))

        # They should be consistent
        assert abs(math.exp(log_p) - p) < 1e-6

    def test_token_probabilities_batch(self):
        """Test batch probability computation."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"ab ac ad ae")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        # Get probabilities for multiple tokens
        token_ids = [ord('b'), ord('c'), ord('d'), ord('e')]
        probs = adapter.token_probabilities("a", token_ids)

        assert len(probs) == 4
        assert all(p >= 0 for p in probs.values())

    def test_token_probabilities_normalized(self):
        """Test normalized probability computation."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"ab ac ab ac")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        token_ids = [ord('b'), ord('c')]
        probs = adapter.token_probabilities("a", token_ids, normalize=True)

        # Normalized probabilities should sum to 1
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6

    def test_mix_probabilities(self):
        """Test probability mixing with LLM."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"ab ab ab ab")  # Corpus strongly prefers 'b' after 'a'
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        # LLM prefers 'c' after 'a'
        llm_probs = {ord('b'): 0.2, ord('c'): 0.8}

        # Mix with alpha=0.5
        mixed = adapter.mix_probabilities("a", llm_probs, alpha=0.5)

        # Both tokens should be present
        assert ord('b') in mixed
        assert ord('c') in mixed

        # Probabilities should be normalized
        total = sum(mixed.values())
        assert abs(total - 1.0) < 1e-6

    def test_mix_probabilities_alpha_extremes(self):
        """Test mixing at alpha extremes."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"ab ab ab")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        llm_probs = {ord('b'): 0.3, ord('x'): 0.7}

        # alpha=1.0 should give pure LLM
        mixed_llm = adapter.mix_probabilities("a", llm_probs, alpha=1.0, normalize=False)
        assert abs(mixed_llm[ord('b')] - 0.3) < 1e-6
        assert abs(mixed_llm[ord('x')] - 0.7) < 1e-6

        # alpha=0.0 should give pure corpus
        mixed_corpus = adapter.mix_probabilities("a", llm_probs, alpha=0.0, normalize=False)
        # 'b' should have corpus probability, 'x' should be very low
        assert mixed_corpus[ord('b')] > mixed_corpus[ord('x')]

    def test_mix_probabilities_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"test")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        with pytest.raises(ValueError, match="alpha must be in"):
            adapter.mix_probabilities("t", {ord('e'): 1.0}, alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            adapter.mix_probabilities("t", {ord('e'): 1.0}, alpha=-0.1)

    def test_mix_log_probabilities(self):
        """Test log-domain probability mixing."""
        from infinigram.adapters import TokenProbabilityAdapter
        import math

        corpus = list(b"ab ab ab")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)

        # LLM log probabilities
        llm_log_probs = {ord('b'): math.log(0.3), ord('c'): math.log(0.7)}

        # Mix in log domain
        mixed_log = adapter.mix_log_probabilities("a", llm_log_probs, alpha=0.5)

        # Should produce valid log probabilities
        assert ord('b') in mixed_log
        assert ord('c') in mixed_log
        assert all(v <= 0 for v in mixed_log.values())  # Log probs are <= 0

    def test_repr(self):
        """Test string representation."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"test")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)
        repr_str = repr(adapter)

        assert "TokenProbabilityAdapter" in repr_str
        assert "MockTokenizer" in repr_str

    def test_context_types(self):
        """Test that various context types work."""
        from infinigram.adapters import TokenProbabilityAdapter

        corpus = list(b"ab ab ab")
        model = Infinigram(corpus)
        tokenizer = MockTokenizer()

        adapter = TokenProbabilityAdapter(model, tokenizer)
        token_id = ord('b')

        # String context
        p1 = adapter.token_probability("a", token_id)

        # Bytes context
        p2 = adapter.token_probability(b"a", token_id)

        # List context
        p3 = adapter.token_probability([ord('a')], token_id)

        # All should give same result
        assert abs(p1 - p2) < 1e-6
        assert abs(p2 - p3) < 1e-6

    def test_chain_rule_correctness(self):
        """Test that chain rule is applied correctly for multi-byte tokens."""
        from infinigram.adapters import TokenProbabilityAdapter

        # Use a corpus with clear patterns where bytes appear in top_k
        corpus = list(b"ab ab ab ab ab ab ab ab")
        model = Infinigram(corpus)

        # Token 1 = "ab"
        vocab = {1: "ab", 2: "cd"}
        tokenizer = MockTokenizer(vocab)

        adapter = TokenProbabilityAdapter(model, tokenizer)

        # Test 1: Single byte token matches model.predict() directly
        # After "a", get P("b") via adapter
        p_b_via_adapter = adapter.token_probability("a", ord('b'))

        # Get P("b"|"a") directly from model
        p_b_direct = model.predict([ord('a')]).get(ord('b'), 0)

        # Single byte token: adapter should match direct prediction
        assert abs(p_b_via_adapter - p_b_direct) < 1e-6

        # Test 2: Verify multi-byte tokens use chain rule
        # P("ab" | "ab ") should be P('a'|"ab ") * P('b'|"ab a")
        context = "ab "

        p_ab_token = adapter.token_probability(context, 1)

        # Manually compute via chain rule
        ctx_bytes = list(context.encode('utf-8'))
        p_a = model.predict(ctx_bytes).get(ord('a'), adapter.smoothing)

        ctx_with_a = ctx_bytes + [ord('a')]
        p_b_given_a = model.predict(ctx_with_a).get(ord('b'), adapter.smoothing)

        expected = p_a * p_b_given_a

        # Should match
        assert abs(p_ab_token - expected) < 1e-6
