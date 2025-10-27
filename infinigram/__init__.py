"""
Infinigram: Variable-Length N-gram Language Models

An efficient implementation of variable-length n-gram language models using
suffix arrays for O(m log n) pattern matching.
"""

from infinigram.infinigram import Infinigram, create_infinigram
from infinigram.suffix_array import SuffixArray
from infinigram.adapters import TokenAdapter, IdentityAdapter
from infinigram import weighting
from infinigram import corpus_utils

__version__ = "0.2.0"  # Phase 1: Hierarchical weighting + Byte-level core
__author__ = "Alex Towell"
__email__ = "lex@metafunctor.com"

__all__ = [
    "Infinigram",
    "create_infinigram",
    "SuffixArray",
    "TokenAdapter",
    "IdentityAdapter",
    "weighting",
    "corpus_utils",
]
