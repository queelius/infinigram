# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Infinigram is a **high-speed, corpus-based language model** that uses suffix arrays for variable-length n-gram pattern matching. Unlike traditional neural LMs or fixed-order n-grams, Infinigram:

- **Trains instantly**: Models are corpora (no gradient descent needed)
- **Finds variable-length patterns**: Automatically uses longest matching context
- **Provides exact matching**: Every prediction traces back to actual corpus occurrences
- **Runs extremely fast**: Orders of magnitude faster than neural inference
- **Enables LLM grounding**: Can be mixed with neural LM probabilities for domain adaptation

**Key use case**: Mix Infinigram probabilities with LLM probabilities to ground outputs in specific corpora (technical docs, legal text, domain-specific knowledge) without expensive fine-tuning.

## Commands

### Testing
```bash
# Run all tests (299 tests)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=infinigram --cov-report=html

# Run specific test file
pytest tests/test_infinigram.py
pytest tests/test_recursive.py
pytest tests/test_evaluation.py

# Run specific test class
pytest tests/test_infinigram.py::TestPredict

# Run specific test
pytest tests/test_infinigram.py::TestPredict::test_predict_returns_probabilities

# Run tests with keyword filter
pytest tests/ -k "recursive"
pytest tests/ -k "scoring"
```

### Development
```bash
# Install in development mode (editable)
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Run example demo
python examples/demo.py

# Launch interactive REPL
python -m infinigram.repl
# or
infinigram-repl

# Start REST API server
python -m infinigram.server.api
```

### Code Quality (when dev dependencies are installed)
```bash
# Format code
black infinigram/ tests/

# Check style
flake8 infinigram/ tests/

# Type checking
mypy infinigram/
```

## Architecture

### Core Components

```
infinigram/
├── __init__.py           # Public API exports (v0.2.0)
├── suffix_array.py       # Suffix array with binary search (O(m log n) queries)
├── infinigram.py         # Core language model class
├── recursive.py          # RecursiveInfinigram with context transformations
├── scoring.py            # Transformation scoring for weighted predictions
├── evaluation.py         # Evaluation framework for OOD generalization
├── weighting.py          # Hierarchical suffix weighting functions
├── adapters.py           # Token adapters (byte-level, identity)
├── corpus_utils.py       # Corpus building and augmentation utilities
├── storage.py            # Hybrid JSONL+SQLite storage for datasets
├── vfs.py                # Virtual filesystem with Unix-like path navigation
├── repl.py               # Interactive REPL shell
└── server/
    ├── __init__.py
    ├── api.py            # FastAPI REST server
    └── models.py         # Pydantic models for API
```

### Key Classes

1. **`SuffixArray`** (`suffix_array.py`)
   - Efficient suffix array with binary search
   - **Construction**: O(n log n) time using lexicographic sorting
   - **LCP array**: Longest Common Prefix array for optimizations
   - **Pattern search**: `search(pattern)` returns all occurrence positions
   - **Longest suffix**: `find_longest_suffix(query)` finds longest matching suffix

2. **`Infinigram`** (`infinigram.py`)
   - Core language model implementation
   - Key methods:
     - `longest_suffix(context)`: Returns (position, length) of longest match
     - `continuations(context)`: Returns token counts following the pattern
     - `predict(context, top_k)`: Returns probability distribution with Laplace smoothing
     - `predict_weighted(context)`: Hierarchical multi-length matching
     - `confidence(context)`: Returns 0-1 score based on match quality
     - `update(new_tokens)`: Appends tokens and rebuilds suffix array

3. **`RecursiveInfinigram`** (`recursive.py`)
   - Extends Infinigram with context transformation for OOD generalization
   - Transformers: `SynonymTransformer`, `TypoTransformer`, `CaseTransformer`
   - Corpus-guided transformation generation
   - WordNet integration for synonym expansion

4. **`TransformationScorer`** (`scoring.py`)
   - Scores transformed contexts for weighted prediction combining
   - Considers: match length, match frequency, transformation depth, transformer type
   - Configurable component weights

5. **`VirtualFilesystem`** (`vfs.py`)
   - Unix-like path navigation for datasets
   - Hierarchy: `/` → `/<dataset>/` → `/<dataset>/<doc_id>`
   - Commands: `cd`, `ls`, `pwd`, `cat`

6. **`InfinigramREPL`** (`repl.py`)
   - Interactive shell with prompt_toolkit
   - Dataset management: `/dataset`, `/use`, `/datasets`
   - Predictions: `/predict`, `/complete`
   - Augmentations: `/augment lowercase uppercase`
   - Bash passthrough: `!ls`, `!head file.txt`

### Algorithm Flow

When `predict(context)` is called:
1. Find longest matching suffix of `context` in the corpus (via suffix array binary search)
2. If match found: collect all tokens that follow this pattern in the corpus
3. Apply Laplace smoothing: `P(token) = (count + smoothing) / (total + smoothing * vocab_size)`
4. Return top_k predictions as probability distribution

When `RecursiveInfinigram.predict(context)` is called:
1. Try direct prediction with base Infinigram
2. If match is short, apply transformers to generate alternative contexts
3. Score each transformation using `TransformationScorer`
4. Combine predictions using weighted averaging
5. Return blended probability distribution

### Design Decisions

- **Byte-level processing**: All input is bytes (UTF-8 encoded), enabling universal text handling
- **Token representation**: Integer token IDs for suffix array operations
- **Smoothing**: Default 0.01 Laplace smoothing ensures non-zero probabilities
- **Updates**: `update()` rebuilds entire suffix array (incremental updates planned)
- **Memory efficiency**: O(n) space vs O(V^n) for hash-based n-grams
- **Speed**: Suffix array binary search provides O(m log n) pattern matching

## Test Structure

The test suite contains **299 tests** across 14 test files:

```
tests/
├── test_infinigram.py              # Core Infinigram tests (36 tests)
├── test_suffix_positions.py        # Suffix array position tests
├── test_recursive.py               # RecursiveInfinigram tests (36 tests)
├── test_recursive_integration.py   # Integration tests for recursive
├── test_scoring.py                 # TransformationScorer tests
├── test_evaluation.py              # Evaluation framework tests (48 tests)
├── test_corpus_guided_transformations.py  # Transformer tests
├── test_wordnet_integration.py     # WordNet synonym tests
├── test_weighting.py               # Weight function tests
├── test_weighted_prediction.py     # Hierarchical prediction tests
├── test_storage.py                 # Storage layer tests
├── test_vfs.py                     # Virtual filesystem tests
├── test_repl_navigation.py         # REPL navigation tests
└── test_utf8_and_adapters.py       # UTF-8 and adapter tests
```

Test categories:
- **Unit tests**: Individual component testing
- **Integration tests**: Component interaction testing
- **Robustness tests**: Edge cases and error handling
- **Evaluation tests**: OOD generalization scenarios

## Key Algorithms

1. **Suffix Array Binary Search** (`suffix_array.py:130-160`):
   - `_binary_search_left()`: Find leftmost occurrence
   - `_binary_search_right()`: Find rightmost occurrence
   - `_compare_pattern_at()`: Pattern comparison with exact/prefix modes

2. **Longest Suffix Matching** (`suffix_array.py:81-105`):
   - Try increasingly long suffixes until no match found
   - Return position and length of best match

3. **Probability Calculation** (`infinigram.py:114-145`):
   - Get continuation counts for longest match
   - Apply Laplace smoothing
   - Normalize to probability distribution

4. **Transformation Scoring** (`scoring.py`):
   - Match length component (longer = better)
   - Match frequency component (more occurrences = confident)
   - Transformation depth penalty (fewer transforms = better)
   - Transformer reliability weights

## Usage Examples

### Basic Usage
```python
from infinigram import Infinigram

# Create model from integer token sequence
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
model = Infinigram(corpus, max_length=10)

# Predict next token
context = [2, 3]
probs = model.predict(context)  # Returns {4: 0.657, 5: 0.330, ...}

# Get match info
position, length = model.longest_suffix(context)
confidence = model.confidence(context)

# Update with new data
model.update([2, 3, 7, 8])
```

### Byte-Level Text Processing
```python
from infinigram import Infinigram
from infinigram.corpus_utils import text_to_bytes, bytes_to_text

# Build from text
text = "the cat sat on the mat"
corpus = list(text.encode('utf-8'))
model = Infinigram(corpus, max_length=20)

# Predict from text context
context = list("the cat".encode('utf-8'))
probs = model.predict(context)
```

### RecursiveInfinigram with Transformations
```python
from infinigram.recursive import RecursiveInfinigram, SynonymTransformer

# Create with synonym transformer
model = RecursiveInfinigram(
    corpus=corpus,
    transformers=[SynonymTransformer(use_wordnet=True)],
    max_depth=2
)

# Predict with OOD context (uses transformations)
probs = model.predict(context)
```

### REPL Session
```bash
$ python -m infinigram.repl
infinigram> /dataset demo
infinigram [demo]> /load the cat sat on the mat
infinigram [demo]> /predict the cat
infinigram [demo]> /complete the --max 20
infinigram [demo]> /augment lowercase
infinigram [demo]> /datasets
infinigram [demo]> /help
```

## Important Implementation Notes

### Current State
- **Version**: 0.2.0 (Post-LangCalc Independence)
- **Clean separation**: SuffixArray in own module, Infinigram uses composition
- **Binary search**: Fully implemented with left/right boundary search
- **All tests passing**: 299/299 tests pass

### When Modifying Code
- `predict()` is the primary public API - changes affect all downstream users
- Suffix array construction in `SuffixArray.__init__()` and `Infinigram.update()` - performance-critical
- `_compare_pattern_at()` is subtle - handles exact and prefix-only matching modes
- `find_longest_suffix()` tries progressively longer suffixes

### Performance Targets
- **Construction**: 1M tokens/second
- **Query latency**: <10ms for 100-token context
- **Throughput**: 1000+ queries/second on single CPU
- **Memory**: <10 bytes per corpus token

## Documentation

Documentation is organized under `docs/` for MkDocs:
- `docs/index.md` - Getting started
- `docs/features/` - Feature documentation (scoring, wordnet, evaluation)
- `docs/guides/` - User guides (REPL, examples)
- `docs/development/` - Development docs (benchmarks, future work)

Build docs locally:
```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

## Future Roadmap

See [ARCHITECTURE.md](../ARCHITECTURE.md) for detailed roadmap:
- Phase 1: Core API Enhancements (current)
- Phase 2: REST API Server (implemented)
- Phase 3: CLI & Shell (implemented - REPL)
- Phase 4: Advanced Matching (RecursiveInfinigram)
- Phase 5: Performance & Scale
- Phase 6: Ecosystem & Integration
