# Infinigram: Variable-Length N-gram Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Efficient variable-length n-gram language models using suffix arrays for O(m log n) pattern matching.

## 🎯 Overview

Infinigram is a language model that uses suffix arrays to efficiently find and score variable-length patterns in a corpus. Unlike traditional fixed-order n-grams, Infinigram automatically finds the longest matching context (up to a configurable maximum) and uses it for prediction.

**Key advantages over traditional n-grams:**
- 🚀 **Variable-length patterns**: Automatically uses as much context as available
- 💾 **Memory efficient**: O(n) space vs O(V^n) for hash-based n-grams
- 🎯 **Accurate**: Longer context = better predictions
- 🔄 **Dynamic updates**: Can add new data to the corpus
- ⚡ **Fast queries**: O(m log n) suffix array search

## 🚀 Quick Start

### Installation

```bash
pip install infinigram
```

Or install from source:

```bash
git clone https://github.com/yourusername/infinigram.git
cd infinigram
pip install -e .
```

### Basic Usage

```python
from infinigram import Infinigram

# Create a model from a corpus
corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4]
model = Infinigram(corpus, max_length=10)

# Predict next token
context = [2, 3]
probs = model.predict(context)
print(probs)
# {4: 0.6569, 5: 0.3301, 1: 0.0033, ...}

# Find longest matching suffix
position, length = model.longest_suffix(context)
print(f"Matched {length} tokens at position {position}")

# Get confidence score (0.0 to 1.0)
confidence = model.confidence(context)
print(f"Confidence: {confidence:.4f}")

# Update with new data
model.update([2, 3, 7, 8, 9])
```

### Text Example

```python
from infinigram import Infinigram

# Prepare text corpus
sentences = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat ran on the mat"
]

# Build vocabulary and tokenize
vocab = {}
corpus = []
for sent in sentences:
    for word in sent.split():
        if word not in vocab:
            vocab[word] = len(vocab)
        corpus.append(vocab[word])

# Create model
model = Infinigram(corpus, max_length=5)

# Predict after "the cat"
context = [vocab["the"], vocab["cat"]]
probs = model.predict(context)

# Convert predictions back to words
id_to_word = {v: k for k, v in vocab.items()}
for token_id, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"{id_to_word[token_id]}: {prob:.3f}")
```

## 🔑 Key Features

### Variable-Length Pattern Matching

Infinigram automatically finds the longest matching suffix in the corpus:

```python
corpus = [1, 2, 3, 4, 5, 1, 2, 3, 6, 1, 2, 3, 4, 7]
model = Infinigram(corpus, max_length=10)

# Short context - finds exact match
model.longest_suffix([1, 2])      # Returns (0, 2)

# Longer context - finds longer match
model.longest_suffix([1, 2, 3, 4]) # Returns (0, 4)
```

### Confidence Scoring

Longer matches provide higher confidence:

```python
model.confidence([1])           # Low confidence (~0.07)
model.confidence([1, 2])        # Medium confidence (~0.15)
model.confidence([1, 2, 3, 4])  # High confidence (~0.29)
```

### Dynamic Corpus Updates

Add new data without rebuilding from scratch:

```python
model = Infinigram([1, 2, 3, 4, 5])
print(len(model.corpus))  # 5

model.update([6, 7, 8, 9])
print(len(model.corpus))  # 9

# Predictions automatically reflect new data
```

### Efficient Suffix Arrays

The underlying suffix array provides:
- **O(n log n)** construction time
- **O(m log n)** query time (m = pattern length, n = corpus size)
- **O(n)** space complexity
- Binary search for pattern matching

## 📊 Performance

From benchmarks on various corpus sizes:

| Corpus Size | Construction | Avg Prediction | Avg Suffix Search |
|-------------|--------------|----------------|-------------------|
| 100 tokens  | 0.07 ms      | 0.043 ms       | 0.014 ms          |
| 1K tokens   | 6.09 ms      | 0.390 ms       | 0.184 ms          |
| 10K tokens  | 718 ms       | 4.370 ms       | 2.373 ms          |

**Memory efficiency**: For a 1B token corpus:
- Hash-based 5-gram: ~34 GB
- Infinigram suffix array: ~1 GB
- **34x reduction in memory usage**

## 🛠️ API Reference

### Infinigram Class

```python
class Infinigram:
    def __init__(
        self,
        corpus: List[int],
        max_length: Optional[int] = None,
        min_count: int = 1,
        smoothing: float = 0.01
    )
```

**Parameters:**
- `corpus`: List of integer token IDs
- `max_length`: Maximum pattern length to consider (None = unlimited)
- `min_count`: Minimum occurrences for a pattern to be included
- `smoothing`: Laplace smoothing factor for unseen tokens

### Methods

#### `predict(context, top_k=50)`

Predict probability distribution over next tokens.

**Returns:** `Dict[int, float]` - Token ID to probability mapping

#### `longest_suffix(context)`

Find longest matching suffix in corpus.

**Returns:** `Tuple[int, int]` - (position, length) of match

#### `confidence(context)`

Get confidence score based on match length.

**Returns:** `float` - Confidence score (0.0 to 1.0)

#### `continuations(context, position, length)`

Get all tokens that follow a matched pattern.

**Returns:** `Counter[int]` - Token counts

#### `update(new_tokens)`

Add new tokens to corpus and rebuild suffix array.

**Parameters:**
- `new_tokens`: List of new token IDs to add

## 🧪 Testing

The package includes 36 comprehensive tests:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=infinigram --cov-report=html

# Run specific test categories
pytest tests/test_suffix_array.py    # Suffix array tests
pytest tests/test_infinigram.py      # Infinigram tests
pytest tests/test_integration.py     # Integration tests
```

Test coverage:
- ✅ Suffix array construction and queries
- ✅ Longest suffix matching
- ✅ Continuation probability computation
- ✅ Prediction with smoothing
- ✅ Confidence scoring
- ✅ Dynamic corpus updates
- ✅ Edge cases (empty corpus, long contexts)
- ✅ Integration scenarios (Wikipedia, code completion)

## 📚 Documentation

- [API Reference](docs/API.md) - Complete API documentation
- [Design Document](docs/DESIGN.md) - Architecture and implementation details
- [Examples](examples/) - Usage examples and demos
- [Benchmarks](docs/BENCHMARKS.md) - Performance analysis

## 🔬 Use Cases

### Code Completion

```python
# Train on source code
code_corpus = tokenize_source_files("src/**/*.py")
model = Infinigram(code_corpus, max_length=50)

# Complete code
context = tokenize("def factorial(n):")
suggestions = model.predict(context, top_k=5)
```

### Text Generation

```python
# Train on text corpus
text_corpus = tokenize_books(["book1.txt", "book2.txt"])
model = Infinigram(text_corpus, max_length=20)

# Generate text
context = tokenize("Once upon a time")
next_word = sample_from_distribution(model.predict(context))
```

### Autocomplete

```python
# Train on user queries
query_corpus = tokenize_queries(user_queries)
model = Infinigram(query_corpus, max_length=10)

# Suggest completions
partial_query = tokenize("how to make")
completions = model.predict(partial_query, top_k=10)
```

## 🤝 Integration

Infinigram can be used standalone or integrated with other systems:

### With LangCalc

```python
# Use as a language model in the LangCalc framework
from langcalc.models import LanguageModel
from infinigram import Infinigram

# Infinigram implements the LanguageModel interface
wiki = Infinigram(wikipedia_corpus, max_length=20)
news = Infinigram(news_corpus, max_length=15)

# Compose with other models
model = 0.7 * llm + 0.2 * wiki + 0.1 * news
```

### Custom Tokenization

```python
from infinigram import Infinigram

class TextInfinigram:
    def __init__(self, texts, max_length=20):
        # Build vocabulary
        self.vocab = {}
        corpus = []
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                corpus.append(self.vocab[token])

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.model = Infinigram(corpus, max_length=max_length)

    def tokenize(self, text):
        return text.lower().split()

    def predict_text(self, context_text):
        context = [self.vocab[t] for t in self.tokenize(context_text) if t in self.vocab]
        probs = self.model.predict(context)
        return {self.id_to_token[tid]: p for tid, p in probs.items()}
```

## 🔄 Comparison with Traditional N-grams

| Feature | Traditional N-gram | Infinigram |
|---------|-------------------|------------|
| Pattern Length | Fixed (n) | Variable (1 to max_length) |
| Memory | O(V^n) exponential | O(corpus_size) linear |
| Query Time | O(1) hash lookup | O(m log n) suffix search |
| Context Usage | Last n-1 tokens | Longest matching suffix |
| Updates | Fast (hash insert) | Slow (rebuild suffix array) |
| Best For | Frequent updates, small n | Large patterns, static/batch updates |

## 📈 Roadmap

- [ ] Incremental suffix array updates (avoid full rebuild)
- [ ] Compressed suffix arrays for large corpora
- [ ] Parallel suffix array construction
- [ ] GPU acceleration for batch predictions
- [ ] Integration with popular NLP libraries
- [ ] Pre-trained models for common domains
- [ ] Support for character-level and subword tokenization
- [ ] Streaming corpus updates

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📞 Contact

- GitHub: [yourusername/infinigram](https://github.com/yourusername/infinigram)
- Issues: [GitHub Issues](https://github.com/yourusername/infinigram/issues)
- Email: your.email@example.com

## 🙏 Acknowledgments

Infinigram was originally developed as part of the [LangCalc](https://github.com/queelius/langcalc) project, which explores algebraic frameworks for language model composition.

## 📚 References

- Suffix Arrays: Manber, U., & Myers, G. (1993). "Suffix arrays: a new method for on-line string searches"
- Variable-length n-grams in language modeling
- Efficient text indexing and retrieval

---

**Built with ❤️ for efficient language modeling**
