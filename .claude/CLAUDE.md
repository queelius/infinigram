# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Infinigram is a **corpus-based language model** using suffix arrays for variable-length n-gram pattern matching.

- **Instant training**: The corpus IS the model (no gradient descent)
- **Variable-length patterns**: Automatically uses longest matching context
- **Exact matching**: Every prediction traces to actual corpus occurrences
- **O(m log n) queries**: Binary search on suffix arrays
- **Byte-level**: UTF-8 bytes (0-255), works with any text

Infinigram provides `predict(context) → probabilities` - a clean interface that can be used standalone or composed with other models via external frameworks.

## Commands

### Testing
```bash
pytest tests/                                    # All tests
pytest tests/ -v                                 # Verbose
pytest tests/ --cov=infinigram --cov-report=html # Coverage
pytest tests/test_infinigram.py                  # Single file
pytest tests/test_infinigram.py::TestPredict     # Single class
pytest tests/test_infinigram.py::TestPredict::test_predict_returns_probabilities  # Single test
pytest tests/ -k "transform"                     # Keyword filter
```

### Development
```bash
pip install -e .              # Editable install
pip install -e .[dev]         # With dev dependencies
python -m infinigram.repl     # Interactive REPL
python -m infinigram.server.api  # REST API server
```

### Code Quality
```bash
black infinigram/ tests/      # Format
flake8 infinigram/ tests/     # Lint
mypy infinigram/              # Type check
```

## Architecture

### Core Classes

**`SuffixArray`** (`suffix_array.py`) - O(n log n) construction, O(m log n) queries
- `search(pattern)` → all occurrence positions
- `find_longest_suffix(query)` → (position, length) of longest match
- `get_context(position, window)` → (before, after) token sequences
- `ngrams(n)` → iterator of (ngram, count) tuples

**`Infinigram`** (`infinigram.py`) - Core language model
- `predict(context, top_k, transforms)` → probability distribution with Laplace smoothing
- `predict_weighted(context, weight_fn)` → hierarchical multi-length matching
- `predict_search(context, search, max_depth, beam_width)` → beam search over transforms
- `longest_suffix(context, transforms)` → (position, length)
- `find_all_suffix_matches(context, transforms)` → all suffix matches at all lengths with positions
- `confidence(context, transforms)` → 0-1 score based on match quality
- `update(new_tokens)` → append and rebuild suffix array
- Runtime transforms: `lowercase`, `uppercase`, `casefold`, `strip`, `normalize_whitespace`

**`TransformationScorer`** (`scoring.py`) - Weights transformed predictions
- Components: match length, match frequency, transformation depth, transformer reliability

### Supporting Modules

| Module | Purpose |
|--------|---------|
| `weighting.py` | Weight functions: `linear_weight`, `quadratic_weight`, `exponential_weight`, `sigmoid_weight` |
| `corpus_utils.py` | Corpus building with augmentations/projections |
| `evaluation.py` | OOD generalization evaluation framework |
| `storage.py` | Hybrid JSONL+SQLite dataset storage |
| `vfs.py` | Unix-like virtual filesystem for datasets |
| `repl.py` | Interactive shell with dataset management |
| `server/api.py` | FastAPI REST server (OpenAI-compatible `/v1/completions`) |

### Projections/Augmentations

Projections are **corpus augmentations** - they add transformed variants of documents to the corpus:

```python
# Original document: "Hello World"
# With lowercase projection, corpus contains both:
# - "Hello World"
# - "hello world"
```

Available projections: `lowercase`, `uppercase`, `title`, `strip`

This enables case-insensitive matching without modifying the query.

### Algorithm Flow

**`predict(context)`**:
1. Binary search suffix array for longest matching suffix
2. Collect all tokens following matched pattern
3. Apply Laplace smoothing: `P(token) = (count + α) / (total + α × vocab_size)`
4. Return top_k as probability distribution

**`predict_weighted(context)`**:
1. Find matches at multiple suffix lengths (1 to max_length)
2. Weight each length's predictions using weight function
3. Combine into single distribution

## Key Implementation Details

- **Byte vocabulary**: Fixed 256 tokens (0-255), smoothing ensures non-zero probabilities
- **Suffix array access**: `model.sa` (not `model.suffix_array`)
- **Transforms**: Runtime query transforms applied at query time, not corpus build time
- **Search**: `predict_search()` performs beam search over transform combinations

### Performance Targets
- Construction: 1M tokens/second
- Query latency: <10ms for 100-token context
- Memory: <10 bytes per corpus token

## Test Structure

Test files with key coverage areas:

| File | Coverage Area |
|------|---------------|
| `test_infinigram.py` | Core model (includes transforms) |
| `test_repl.py` | REPL commands |
| `test_server.py` | REST API |
| `test_evaluation.py` | OOD evaluation |
| `test_vfs.py` | Virtual filesystem |
| `test_storage.py` | Dataset storage |
| `test_weighting.py` | Weight functions |
| `test_scoring.py` | Transformation scoring |

## Usage Examples

### Basic
```python
from infinigram import Infinigram

corpus = list("the cat sat on the mat".encode('utf-8'))
model = Infinigram(corpus, max_length=20)

context = list("the cat".encode('utf-8'))
probs = model.predict(context, top_k=5)
# {32: 1.0, ...}  # space byte has highest probability
```

### With Augmentations
```python
from infinigram import Infinigram
from infinigram.corpus_utils import build_corpus_with_augmentation

docs = ["Hello World", "Goodbye World"]
corpus = build_corpus_with_augmentation(docs, augmentations=[str.lower, str.upper])
# Corpus now contains: original + lowercase + uppercase variants

model = Infinigram(corpus)
```

### With Runtime Transforms
```python
from infinigram import Infinigram

corpus = b"The Cat sat on The Mat"
model = Infinigram(corpus, default_transforms=['lowercase'])

# Handles case mismatch via runtime transform
probs = model.predict(b"the cat")

# Or specify transforms per-call
probs = model.predict(b"THE CAT", transforms=['lowercase', 'strip'])

# Beam search over transform combinations
probs = model.predict_search(b"THE CAT", search=['lowercase', 'casefold'])
```

### Future Features

The following OOD generalization features are planned but deferred due to runtime performance concerns:
- **Synonym transforms**: Corpus-guided word replacement (requires WordNet integration or embedding similarity)
- **Typo correction**: Edit-distance based transforms (would need fuzzy suffix arrays or BK-trees for efficiency)

### REPL
```bash
infinigram> ds demo              # Create dataset
infinigram> add the cat sat      # Add document
infinigram> proj ls              # List projections
infinigram> proj + lowercase     # Enable lowercase augmentation
infinigram> predict the cat      # Get predictions
infinigram> complete the --max 10  # Generate text
infinigram> config               # Show settings
```
