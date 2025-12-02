# Infinigram Documentation

Welcome to the Infinigram documentation! Infinigram is a high-speed, corpus-based language model that uses suffix arrays for variable-length n-gram pattern matching.

## What is Infinigram?

Unlike traditional neural language models or fixed-order n-grams, Infinigram:

- ✅ **Trains instantly**: Models are corpora (no gradient descent needed)
- ✅ **Finds variable-length patterns**: Automatically uses longest matching context
- ✅ **Provides exact matching**: Every prediction traces back to actual corpus occurrences
- ✅ **Runs extremely fast**: Orders of magnitude faster than neural inference
- ✅ **Enables LLM grounding**: Can be mixed with neural LM probabilities for domain adaptation

## Key Features

### Runtime Query Transforms
Handle out-of-distribution (OOD) data through runtime query transformations:

- **Case normalization**: `lowercase`, `uppercase`, `casefold`
- **Whitespace normalization**: `strip`, `normalize_whitespace`
- **Sequential composition**: Apply multiple transforms in order
- **Beam search**: Explore transform combinations with `predict_search()`

### Future OOD Features (Planned)
The following features are planned but deferred due to runtime performance concerns:
- **Typo correction**: Edit distance-based corrections (requires fuzzy suffix arrays)
- **Semantic synonyms**: WordNet-based synonym matching (requires embedding integration)

## Quick Start

```python
from infinigram import Infinigram

# Create model from corpus
corpus = b"the cat sat on the mat"
model = Infinigram(corpus, max_length=10)

# Predict next token
context = b"the cat"
probs = model.predict(context)
print(probs)  # {115: 0.657, 97: 0.330, ...}  # 's' (sat), 'a' (at)
```

### With Runtime Transforms

```python
from infinigram import Infinigram

# Set default transforms at model creation
model = Infinigram(corpus, default_transforms=['lowercase'])

# Handles case variations automatically
context = b"The Cat"  # Uppercase
probs = model.predict(context)

# Or specify transforms per-call
probs = model.predict(b"THE CAT", transforms=['lowercase', 'strip'])

# Beam search over transform combinations
probs = model.predict_search(context, search=['lowercase', 'casefold'])
```

## Documentation Sections

### User Guides
- **[Loading Datasets](guides/LOADING_DATASETS.md)**: Learn how to load and manage datasets
- **[Benchmarks & Performance](guides/BENCHMARKS.md)**: Comprehensive benchmark results and analysis

### Features
- **[Transformation Scoring](features/SCORING_AND_EVALUATION.md)**: Multi-factor scoring system for weighted predictions

### Development
- **[Test Strategy](development/TEST_STRATEGY_REVIEW.md)**: TDD strategy and coverage analysis
- **[Test Summary](development/TEST_REVIEW_SUMMARY.md)**: Executive test coverage summary
- **[Priority Tests](development/PRIORITY_TESTS_TO_ADD.md)**: Ready-to-add test implementations

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=infinigram --cov-report=html

# Run specific test file
pytest tests/test_infinigram.py
```

## Use Cases

### 1. Domain-Specific Grounding
Mix Infinigram probabilities with LLM probabilities to ground outputs in specific corpora:

```python
llm_probs = llm.predict(context)
corpus_probs = infinigram.predict(context)
final_probs = 0.7 * llm_probs + 0.3 * corpus_probs
```

### 2. Fast Pattern Matching
Orders of magnitude faster than neural inference for pattern-based predictions.

### 3. Exact Source Attribution
Every prediction traces back to actual corpus occurrences.

### 4. Zero-Shot Domain Adaptation
No training required - just point at a corpus.

## Architecture

See [Architecture](ARCHITECTURE.md) for detailed system design.

## Contributing

Contributions welcome! See our comprehensive [Test Strategy](development/TEST_STRATEGY_REVIEW.md) for testing guidelines.

## License

[License information here]

## Citation

```bibtex
@software{infinigram2024,
  title={Infinigram: Variable-Length N-gram Language Model},
  author={Towell, Alex},
  year={2024},
  url={https://github.com/queelius/infinigram}
}
```
