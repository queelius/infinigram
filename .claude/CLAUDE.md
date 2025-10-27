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
# Run all tests (36 tests)
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=infinigram --cov-report=html

# Run specific test file
pytest tests/test_infinigram.py

# Run specific test class
pytest tests/test_infinigram.py::TestPredict

# Run specific test
pytest tests/test_infinigram.py::TestPredict::test_predict_returns_probabilities
```

### Development
```bash
# Install in development mode (editable)
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Run example demo
python examples/demo.py
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

1. **`infinigram/suffix_array.py`** (232 lines)
   - `SuffixArray` class: Efficient suffix array with binary search
   - **Construction**: O(n log n) time using lexicographic sorting
   - **LCP array**: Longest Common Prefix array for future optimizations
   - **Pattern search**: `search(pattern)` returns all occurrence positions using binary search
   - **Longest suffix**: `find_longest_suffix(query)` finds longest matching suffix

2. **`infinigram/infinigram.py`** (348 lines)
   - `Infinigram` class: Main language model implementation
   - **Key methods**:
     - `longest_suffix(context)`: Returns (position, length) of longest match (delegates to SuffixArray)
     - `continuations(context)`: Returns token counts following the matched pattern
     - `predict(context, top_k)`: Returns probability distribution with Laplace smoothing
     - `confidence(context)`: Returns 0-1 score based on match length and frequency
     - `update(new_tokens)`: Appends tokens and rebuilds suffix array

3. **`infinigram/__init__.py`** (19 lines)
   - Public API exports: `Infinigram`, `create_infinigram`, `SuffixArray`
   - Version: 0.1.0

### Algorithm Flow

When `predict(context)` is called:
1. Find longest matching suffix of `context` in the corpus (via suffix array binary search)
2. If match found: collect all tokens that follow this pattern in the corpus
3. Apply Laplace smoothing: `P(token) = (count + smoothing) / (total + smoothing * vocab_size)`
4. Return top_k predictions as probability distribution

### Design Decisions

- **Token representation**: All input must be integer token IDs (no built-in tokenizer)
- **Smoothing**: Default 0.01 Laplace smoothing ensures all vocab tokens have non-zero probability
- **Updates**: `update()` rebuilds the entire suffix array (incremental updates planned for future)
- **Memory efficiency**: Stores only corpus + suffix array indices = O(n) space vs O(V^n) for hash-based n-grams
- **Speed**: Suffix array binary search provides O(m log n) pattern matching vs O(1) for hash but with better memory

## Test Structure

The test suite (`tests/test_infinigram.py`) contains 36 tests organized into:
- `TestSuffixArray` (5 tests): Suffix array construction and pattern search
- `TestInfinigramCore` (3 tests): Initialization and configuration
- `TestLongestSuffix` (5 tests): Longest suffix matching logic
- `TestContinuations` (4 tests): Continuation token collection
- `TestPredict` (5 tests): Probability prediction with smoothing
- `TestConfidence` (3 tests): Confidence scoring
- `TestUpdate` (3 tests): Dynamic corpus updates
- `TestEdgeCases` (5 tests): Empty inputs, long contexts, single tokens
- `TestIntegration` (3 tests): Realistic scenarios (Wikipedia, code completion)

## Future Roadmap (see ARCHITECTURE.md for details)

### Phase 1: Core API Enhancements (Current)
- Multi-length suffix weighting (combine predictions from multiple suffix lengths)
- Input/output projections (lemmatization, filtering)
- Hierarchical matching with configurable weight functions

### Phase 2: REST API Server
- FastAPI-based server with OpenAI-compatible endpoints
- `/v1/completions` and `/v1/chat/completions` endpoints
- Model management (load/unload/stats)
- Streaming responses

### Phase 3: CLI & Shell
- `infinigram build` - Build corpus from text files
- `infinigram serve` - Start REST API server
- `infinigram predict` - One-shot predictions
- `infinigram shell` - Interactive REPL with state management

### Phase 4: Advanced Features
- Binary search optimization (currently uses optimized binary search in suffix_array.py)
- Memory-mapped corpus files for large datasets
- Compressed suffix arrays
- GPU acceleration for batch inference

## Important Implementation Notes

### Current State
- **Clean separation**: SuffixArray is in its own module, Infinigram uses it via composition
- **No external dependencies**: Removed LangCalc dependencies (was originally part of that project)
- **Binary search**: Fully implemented in `suffix_array.py` with left/right boundary search
- **All tests passing**: 36/36 tests pass after cleanup

### When Modifying Code
- The `predict()` method is the primary public API - changes here affect all downstream users
- Suffix array construction happens in `SuffixArray.__init__()` and `Infinigram.update()` - performance-critical paths
- The `_compare_pattern_at()` method in `suffix_array.py` is subtle - handles both exact and prefix-only matching modes
- `find_longest_suffix()` tries progressively longer suffixes to find the best match

### Key Algorithms
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

### LLM Probability Mixing (Future)
```python
# Ground LLM with domain-specific corpus
llm_probs = llm.predict(context)
corpus_probs = infinigram.predict(context)

# Weighted mixture
final_probs = 0.7 * llm_probs + 0.3 * corpus_probs
```

### Multi-Corpus Models (Planned)
```python
# Load specialized corpora
wiki = Infinigram(wikipedia_corpus)
code = Infinigram(github_corpus)
docs = Infinigram(python_docs_corpus)

# Query different models
wiki.predict(context)
code.predict(context)
```

For comprehensive architectural vision and roadmap, see [ARCHITECTURE.md](../ARCHITECTURE.md).
