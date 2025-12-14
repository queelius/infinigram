#!/usr/bin/env python3
"""
Token adapters for Infinigram.

Adapters convert between byte sequences (Infinigram's native representation)
and token sequences (used by LLM tokenizers like TikToken, SentencePiece, etc.).

Includes TokenProbabilityAdapter for rigorous byte-to-token probability
marginalization, enabling probability mixing with LLMs.
"""

from typing import List, Dict, Tuple, Optional, Union, Callable, Protocol
from abc import ABC, abstractmethod
import math


class Tokenizer(Protocol):
    """Protocol for LLM tokenizers (tiktoken, transformers, etc.)."""

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        ...

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary."""
        ...


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


class TokenProbabilityAdapter:
    """
    Adapter for computing token-level probabilities from byte-level Infinigram.

    This enables rigorous probability mixing between Infinigram and LLMs by
    marginalizing byte probabilities into token probabilities via the chain rule.

    For a token t that encodes to bytes [b₁, b₂, ..., bₖ]:

        P_infini(t | context) = ∏ᵢ P(bᵢ | context, b₁, ..., bᵢ₋₁)

    This is exact, not an approximation. A token is just a named byte sequence.

    Example:
        >>> from infinigram import Infinigram
        >>> from infinigram.adapters import TokenProbabilityAdapter
        >>> import tiktoken
        >>>
        >>> # Create Infinigram model
        >>> corpus = b"The quick brown fox jumps over the lazy dog."
        >>> model = Infinigram(corpus)
        >>>
        >>> # Create adapter with GPT-4 tokenizer
        >>> tokenizer = tiktoken.encoding_for_model("gpt-4")
        >>> adapter = TokenProbabilityAdapter(model, tokenizer)
        >>>
        >>> # Compute token probability
        >>> context = "The quick brown"
        >>> token_id = tokenizer.encode(" fox")[0]
        >>> prob = adapter.token_probability(context, token_id)
        >>>
        >>> # Mix with LLM probabilities
        >>> llm_probs = {token_id: 0.3, ...}  # From LLM
        >>> mixed = adapter.mix_probabilities(context, llm_probs, alpha=0.7)
    """

    def __init__(
        self,
        model: "Infinigram",
        tokenizer: Tokenizer,
        log_domain: bool = True,
        smoothing: float = 1e-10,
    ):
        """
        Initialize the token probability adapter.

        Args:
            model: Infinigram model instance
            tokenizer: LLM tokenizer (must implement encode/decode)
            log_domain: If True, compute in log domain for numerical stability
            smoothing: Minimum probability for unseen bytes (avoids log(0))
        """
        self.model = model
        self.tokenizer = tokenizer
        self.log_domain = log_domain
        self.smoothing = smoothing

        # Cache token -> bytes mapping for efficiency
        self._token_bytes_cache: Dict[int, bytes] = {}

    def _get_token_bytes(self, token_id: int) -> bytes:
        """Get the byte representation of a token (cached)."""
        if token_id not in self._token_bytes_cache:
            try:
                text = self.tokenizer.decode([token_id])
                self._token_bytes_cache[token_id] = text.encode('utf-8')
            except Exception:
                # Some tokens may not decode cleanly
                self._token_bytes_cache[token_id] = b''
        return self._token_bytes_cache[token_id]

    def token_log_probability(
        self,
        context: Union[str, bytes, List[int]],
        token_id: int,
    ) -> float:
        """
        Compute log P(token | context) via byte-level chain rule.

        Args:
            context: The conditioning context (string, bytes, or byte list)
            token_id: The token ID to compute probability for

        Returns:
            Log probability of the token given context

        The probability is computed as:
            log P(token) = Σᵢ log P(bᵢ | context, b₁, ..., bᵢ₋₁)

        where [b₁, ..., bₖ] are the bytes encoding the token.
        """
        # Normalize context to byte list
        if isinstance(context, str):
            context_bytes = list(context.encode('utf-8'))
        elif isinstance(context, bytes):
            context_bytes = list(context)
        else:
            context_bytes = list(context)

        # Get token's byte representation
        token_bytes = self._get_token_bytes(token_id)
        if not token_bytes:
            return float('-inf')

        # Compute log probability via chain rule
        log_prob = 0.0
        current_context = context_bytes.copy()

        for byte_val in token_bytes:
            # Get byte distribution from Infinigram
            byte_probs = self.model.predict(current_context)

            # Get probability for this byte (with smoothing)
            p = byte_probs.get(byte_val, self.smoothing)
            p = max(p, self.smoothing)  # Ensure non-zero

            log_prob += math.log(p)
            current_context.append(byte_val)

        return log_prob

    def token_probability(
        self,
        context: Union[str, bytes, List[int]],
        token_id: int,
    ) -> float:
        """
        Compute P(token | context) via byte-level chain rule.

        Args:
            context: The conditioning context
            token_id: The token ID to compute probability for

        Returns:
            Probability of the token given context (0 to 1)
        """
        log_p = self.token_log_probability(context, token_id)
        if log_p == float('-inf'):
            return 0.0
        return math.exp(log_p)

    def token_probabilities(
        self,
        context: Union[str, bytes, List[int]],
        token_ids: List[int],
        normalize: bool = False,
    ) -> Dict[int, float]:
        """
        Compute probabilities for multiple tokens.

        Args:
            context: The conditioning context
            token_ids: List of token IDs to compute probabilities for
            normalize: If True, normalize probabilities to sum to 1

        Returns:
            Dictionary mapping token IDs to probabilities
        """
        probs = {}
        for token_id in token_ids:
            probs[token_id] = self.token_probability(context, token_id)

        if normalize and probs:
            total = sum(probs.values())
            if total > 0:
                probs = {t: p / total for t, p in probs.items()}

        return probs

    def mix_probabilities(
        self,
        context: Union[str, bytes, List[int]],
        llm_probs: Dict[int, float],
        alpha: float = 0.5,
        normalize: bool = True,
    ) -> Dict[int, float]:
        """
        Mix LLM token probabilities with corpus-based probabilities.

        Computes:
            P_final(t) = α × P_LLM(t) + (1-α) × P_infini(t)

        Args:
            context: The conditioning context
            llm_probs: Dictionary of token_id -> probability from LLM
            alpha: Mixing weight for LLM (0 = pure corpus, 1 = pure LLM)
            normalize: If True, normalize final distribution

        Returns:
            Dictionary mapping token IDs to mixed probabilities

        Example:
            >>> # Get top-k tokens from LLM
            >>> llm_output = llm.predict(context, top_k=100)
            >>> # Mix with corpus probabilities
            >>> mixed = adapter.mix_probabilities(context, llm_output, alpha=0.7)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        # Compute corpus probabilities for LLM's tokens
        token_ids = list(llm_probs.keys())
        corpus_probs = self.token_probabilities(context, token_ids)

        # Mix probabilities
        mixed = {}
        for token_id in token_ids:
            p_llm = llm_probs[token_id]
            p_corpus = corpus_probs.get(token_id, 0.0)
            mixed[token_id] = alpha * p_llm + (1 - alpha) * p_corpus

        # Normalize if requested
        if normalize and mixed:
            total = sum(mixed.values())
            if total > 0:
                mixed = {t: p / total for t, p in mixed.items()}

        return mixed

    def mix_log_probabilities(
        self,
        context: Union[str, bytes, List[int]],
        llm_log_probs: Dict[int, float],
        alpha: float = 0.5,
    ) -> Dict[int, float]:
        """
        Mix probabilities in log domain (more numerically stable).

        Uses log-sum-exp for mixing:
            log P_final(t) = log(α × exp(log P_LLM) + (1-α) × exp(log P_infini))

        Args:
            context: The conditioning context
            llm_log_probs: Dictionary of token_id -> log probability from LLM
            alpha: Mixing weight for LLM

        Returns:
            Dictionary mapping token IDs to mixed log probabilities
        """
        mixed = {}
        for token_id, llm_log_p in llm_log_probs.items():
            corpus_log_p = self.token_log_probability(context, token_id)

            # Log-sum-exp: log(α*exp(a) + (1-α)*exp(b))
            if alpha == 1.0:
                mixed[token_id] = llm_log_p
            elif alpha == 0.0:
                mixed[token_id] = corpus_log_p
            else:
                log_alpha = math.log(alpha)
                log_1_alpha = math.log(1 - alpha)

                a = log_alpha + llm_log_p
                b = log_1_alpha + corpus_log_p

                # Numerically stable log-sum-exp
                max_val = max(a, b)
                mixed[token_id] = max_val + math.log(
                    math.exp(a - max_val) + math.exp(b - max_val)
                )

        return mixed

    def __repr__(self) -> str:
        return (
            f"TokenProbabilityAdapter(model={self.model!r}, "
            f"tokenizer={type(self.tokenizer).__name__}, "
            f"log_domain={self.log_domain})"
        )
