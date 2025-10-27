#!/usr/bin/env python3
"""
Token adapters for Infinigram.

Adapters convert between byte sequences (Infinigram's native representation)
and token sequences (used by LLM tokenizers like TikToken, SentencePiece, etc.).
"""

from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class TokenAdapter(ABC):
    """
    Abstract base class for token adapters.

    Token adapters provide bidirectional conversion between:
    - Byte sequences (List[int] with values 0-255)
    - Token sequences (List[int] with tokenizer-specific IDs)

    This allows Infinigram to work with any tokenization scheme while
    maintaining its byte-level core.
    """

    @abstractmethod
    def bytes_to_tokens(self, byte_sequence: List[int]) -> List[int]:
        """
        Convert byte sequence to token sequence.

        Args:
            byte_sequence: List of bytes (0-255)

        Returns:
            List of token IDs

        Example:
            >>> adapter = IdentityAdapter()
            >>> adapter.bytes_to_tokens([72, 101, 108, 108, 111])
            [72, 101, 108, 108, 111]
        """
        pass

    @abstractmethod
    def tokens_to_bytes(self, token_sequence: List[int]) -> List[int]:
        """
        Convert token sequence to byte sequence.

        Args:
            token_sequence: List of token IDs

        Returns:
            List of bytes (0-255)

        Example:
            >>> adapter = IdentityAdapter()
            >>> adapter.tokens_to_bytes([72, 101, 108, 108, 111])
            [72, 101, 108, 108, 111]
        """
        pass

    @abstractmethod
    def text_to_bytes(self, text: str) -> List[int]:
        """
        Convert text to byte sequence.

        Args:
            text: Input text string

        Returns:
            List of bytes (0-255)

        Example:
            >>> adapter = IdentityAdapter()
            >>> adapter.text_to_bytes("Hello")
            [72, 101, 108, 108, 111]
        """
        pass

    @abstractmethod
    def bytes_to_text(self, byte_sequence: List[int]) -> str:
        """
        Convert byte sequence to text.

        Args:
            byte_sequence: List of bytes (0-255)

        Returns:
            Decoded text string

        Example:
            >>> adapter = IdentityAdapter()
            >>> adapter.bytes_to_text([72, 101, 108, 108, 111])
            'Hello'
        """
        pass


class IdentityAdapter(TokenAdapter):
    """
    Identity token adapter.

    This adapter treats bytes as tokens (no transformation).
    Token IDs are simply byte values (0-255).

    This is the default adapter and demonstrates the simplest
    possible tokenization scheme.

    Example:
        >>> adapter = IdentityAdapter()
        >>> text = "Hello"
        >>> bytes_seq = adapter.text_to_bytes(text)  # [72, 101, 108, 108, 111]
        >>> tokens = adapter.bytes_to_tokens(bytes_seq)  # [72, 101, 108, 108, 111]
        >>> assert bytes_seq == tokens  # Identity!
    """

    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize identity adapter.

        Args:
            encoding: Text encoding to use (default: 'utf-8')
        """
        self.encoding = encoding

    def bytes_to_tokens(self, byte_sequence: List[int]) -> List[int]:
        """
        Identity: bytes == tokens.

        Args:
            byte_sequence: List of bytes (0-255)

        Returns:
            Same list (identity transformation)
        """
        # Validate byte range
        if byte_sequence:
            invalid = [b for b in byte_sequence if not (0 <= b <= 255)]
            if invalid:
                raise ValueError(
                    f"Byte sequence must contain only values 0-255. "
                    f"Found {len(invalid)} invalid values: {invalid[:10]}"
                )
        return list(byte_sequence)  # Return copy

    def tokens_to_bytes(self, token_sequence: List[int]) -> List[int]:
        """
        Identity: tokens == bytes.

        Args:
            token_sequence: List of token IDs (0-255)

        Returns:
            Same list (identity transformation)
        """
        # In identity adapter, tokens must also be valid bytes
        if token_sequence:
            invalid = [t for t in token_sequence if not (0 <= t <= 255)]
            if invalid:
                raise ValueError(
                    f"Token sequence must contain only values 0-255. "
                    f"Found {len(invalid)} invalid values: {invalid[:10]}"
                )
        return list(token_sequence)  # Return copy

    def text_to_bytes(self, text: str) -> List[int]:
        """
        Convert text to UTF-8 bytes.

        Args:
            text: Input text

        Returns:
            UTF-8 byte sequence
        """
        return list(text.encode(self.encoding))

    def bytes_to_text(self, byte_sequence: List[int]) -> str:
        """
        Convert bytes to text using UTF-8 decoding.

        Args:
            byte_sequence: UTF-8 byte sequence

        Returns:
            Decoded text (invalid sequences replaced with �)
        """
        return bytes(byte_sequence).decode(self.encoding, errors='replace')

    def __repr__(self) -> str:
        return f"IdentityAdapter(encoding='{self.encoding}')"
