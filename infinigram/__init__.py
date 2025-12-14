"""
Infinigram: Variable-Length N-gram Language Models

An efficient implementation of variable-length n-gram language models using
suffix arrays for O(m log n) pattern matching.
"""

from infinigram.infinigram import Infinigram, create_infinigram
from infinigram.suffix_array import SuffixArray
from infinigram.adapters import TokenAdapter, IdentityAdapter, TokenProbabilityAdapter
from infinigram import weighting
from infinigram import corpus_utils

__version__ = "0.4.2"  # TokenProbabilityAdapter for LLM probability mixing
__author__ = "Alex Towell"
__email__ = "lex@metafunctor.com"

__all__ = [
    "Infinigram",
    "create_infinigram",
    "SuffixArray",
    "TokenAdapter",
    "IdentityAdapter",
    "TokenProbabilityAdapter",
    "weighting",
    "corpus_utils",
]
