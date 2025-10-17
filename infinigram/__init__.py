"""
Infinigram: Variable-Length N-gram Language Models

An efficient implementation of variable-length n-gram language models using
suffix arrays for O(m log n) pattern matching.
"""

from infinigram.infinigram import Infinigram, create_infinigram
from infinigram.suffix_array import SuffixArray

__version__ = "0.1.0"
__author__ = "Alex Towell"
__email__ = "lex@metafunctor.com"

__all__ = [
    "Infinigram",
    "create_infinigram",
    "SuffixArray",
]
