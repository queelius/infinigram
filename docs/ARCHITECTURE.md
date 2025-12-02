# Infinigram Architecture & Vision

**Version**: 0.2.0 (Post-LangCalc Independence)
**Date**: October 17, 2025
**Status**: Design Phase

## Vision

Infinigram is a **high-speed, corpus-based language model** that leverages suffix arrays for variable-length n-gram matching. Unlike traditional neural LMs, Infinigram provides:

- **Instant training**: Models are corpora (no gradient descent)
- **Exact matching**: Finds actual patterns from training data
- **Explainability**: Every prediction traces back to corpus evidence
- **Speed**: Orders of magnitude faster than neural inference
- **LLM grounding**: Weight mixture with neural LM next-token probabilities

## Core Use Cases

### 1. LLM Fine-tuning via Probability Mixing
```python
# Weighted mixture of neural LM and corpus-based predictions
final_probs = 0.7 * llm.predict(context) + 0.3 * infinigram.predict(context)
```

**Benefits**:
- Ground LLM outputs in specific corpora (technical docs, legal text, etc.)
- Boost domain-specific vocabulary without expensive fine-tuning
- Reduce hallucinations by anchoring to real text
- Real-time adaptation without retraining

### 2. Multi-Corpus Models
```bash
# Load multiple specialized corpora
infinigram serve \
  --corpus wikipedia:/data/wiki.bin \
  --corpus shakespeare:/data/shakespeare.bin \
  --corpus python-docs:/data/python-stdlib.bin \
  --port 8000
```

### 3. Projection-Based Matching
Beyond simple longest suffix matching, support:
- **Input projections**: Transform query context to find better matches (e.g., lemmatization, semantic clustering)
- **Hierarchical matching**: Weight contributions from multiple suffix lengths
- **Output projections**: Map predicted tokens to target vocabulary

## Architectural Principles

### 1. Clean API Layers
```
┌─────────────────────────────────────┐
│         CLI & Shell                 │  User-facing commands + REPL
├─────────────────────────────────────┤
│         REST API                    │  HTTP endpoints (OpenAI-compatible)
├─────────────────────────────────────┤
│      Python API (Core)              │  Core Infinigram class
├─────────────────────────────────────┤
│    Suffix Array Engine              │  Pattern matching primitives
└─────────────────────────────────────┘
```

### 2. REST API Design (OpenAI-Compatible)

#### Completions Endpoint
```http
POST /v1/completions
Content-Type: application/json

{
  "model": "wikipedia",
  "prompt": "The capital of France is",
  "max_tokens": 10,
  "temperature": 1.0,
  "top_k": 50
}

Response:
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1697558400,
  "model": "wikipedia",
  "choices": [{
    "text": " Paris",
    "index": 0,
    "logprobs": {...},
    "finish_reason": "stop",
    "metadata": {
      "match_length": 4,
      "confidence": 0.89,
      "corpus_position": 1234567
    }
  }]
}
```

#### Chat Endpoint
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "python-docs",
  "messages": [
    {"role": "user", "content": "How do I read a file in Python?"}
  ],
  "max_tokens": 100
}
```

#### Models Management
```http
GET /v1/models
POST /v1/models/load
DELETE /v1/models/{model_id}
GET /v1/models/{model_id}/stats
```

### 3. Python API (Enhanced Core)

```python
from infinigram import Infinigram, InfinigramServer

# Basic usage (unchanged for backward compatibility)
model = Infinigram(corpus, max_length=20)
probs = model.predict(context, top_k=10)

# Enhanced: Multi-length matching with weights
probs = model.predict_weighted(
    context,
    min_length=1,
    max_length=10,
    weight_fn=lambda length: length ** 2  # Quadratic weighting
)

# Projection-based matching
probs = model.predict_projected(
    context,
    input_projection="lemmatize",
    output_projection="top_frequent_10k"
)

# Corpus management
model.add_corpus(new_texts, corpus_id="technical_docs")
model.remove_corpus("old_corpus")

# Model serving
server = InfinigramServer(port=8000)
server.add_model("wiki", corpus_path="wiki.bin")
server.add_model("code", corpus_path="github.bin", max_length=50)
server.start()
```

### 4. CLI Design

```bash
# Training (corpus building)
infinigram build wikipedia.txt -o wikipedia.igram --max-length 20
infinigram build *.txt -o combined.igram --merge

# Serving
infinigram serve wikipedia.igram --port 8000
infinigram serve wikipedia.igram code.igram --port 8000

# Interactive shell
infinigram shell wikipedia.igram
> load shakespeare.igram as shakespeare
> predict "to be or not to"
> set max_length 15
> set weight_fn quadratic
> stats wikipedia
> exit

# One-shot predictions
infinigram predict wikipedia.igram "The capital of"
infinigram complete --model wiki --text "Once upon a" --max-tokens 50

# Model inspection
infinigram info wikipedia.igram
infinigram stats wikipedia.igram
infinigram search wikipedia.igram "machine learning"
```

### 5. Shell (Stateful REPL)

```python
# Interactive shell with state management
$ infinigram shell

infinigram> load wikipedia.igram as wiki
Loaded: wiki (125M tokens, max_length=20)

infinigram> load shakespeare.igram as shakespeare
Loaded: shakespeare (884K tokens, max_length=15)

infinigram> models
- wiki: 125M tokens, max_length=20
- shakespeare: 884K tokens, max_length=15

infinigram> use wiki
Active model: wiki

infinigram> predict "The capital of France is"
Top predictions:
  Paris (0.856) ████████████████████████████
  located (0.089) ███
  situated (0.034) █

infinigram> set temperature 0.5
infinigram> set top_k 20

infinigram> match-info "The capital of France is"
Longest match: length=4 ("capital of France is")
Position: 1234567
Context: "...The capital of France is Paris, and it is..."
Confidence: 0.89

infinigram> history
1. predict "The capital of France is"
2. match-info "The capital of France is"

infinigram> export history results.json
infinigram> exit
```

## Advanced Features

### 1. Hierarchical Suffix Weighting

Instead of only using longest match, combine predictions from multiple suffix lengths:

```python
# P(next | context) = Σ w(k) * P(next | suffix_k)
# where suffix_k is the k-length suffix match

def weight_function(match_length, max_length):
    """Weight longer matches more heavily."""
    return (match_length / max_length) ** 2

probs = model.predict_hierarchical(
    context,
    min_length=1,
    max_length=10,
    weight_fn=weight_function
)
```

### 2. Input Projections

Transform input context to find better matches:

```python
class InputProjection:
    """Transform context before suffix matching."""

    def lemmatize(self, tokens: List[int]) -> List[int]:
        """Reduce tokens to lemmas."""
        pass

    def semantic_cluster(self, tokens: List[int]) -> List[int]:
        """Map to semantic cluster IDs."""
        pass

    def drop_stopwords(self, tokens: List[int]) -> List[int]:
        """Remove common stopwords."""
        pass

# Usage
model.predict(context, input_projection="lemmatize")
```

### 3. Output Projections

Filter or transform predicted tokens:

```python
class OutputProjection:
    """Filter/transform output predictions."""

    def top_k_frequent(self, probs: Dict[int, float], k: int) -> Dict[int, float]:
        """Restrict to k most frequent vocabulary tokens."""
        pass

    def domain_filter(self, probs: Dict[int, float], domain: str) -> Dict[int, float]:
        """Only allow domain-specific vocabulary."""
        pass

# Usage
model.predict(context, output_projection="top_frequent_10k")
```

### 4. Multi-Scale Matching

```python
# Combine evidence from different granularities
model = MultiScaleInfinigram([
    ("char", char_corpus, max_length=100),
    ("subword", bpe_corpus, max_length=50),
    ("word", word_corpus, max_length=20)
])

# Automatically blends predictions across scales
probs = model.predict(context, scales=["word", "subword"])
```

### 5. Corpus Versioning & Hot-Swapping

```python
server = InfinigramServer()

# Load initial corpus
server.add_model("v1", corpus_v1)

# Later: hot-swap without downtime
server.update_model("v1", corpus_v2)  # Atomic replacement

# A/B testing
server.add_model("experimental", corpus_exp)
probs_control = server.predict("v1", context)
probs_exp = server.predict("experimental", context)
```

## Implementation Roadmap

### Phase 1: Cleanup & Core API (Current)
- [x] Remove LangCalc dependencies
- [ ] Fix test imports
- [ ] Remove `LanguageModel` ABC (not needed standalone)
- [ ] Add `predict_weighted()` for multi-length matching
- [ ] Comprehensive unit tests for new APIs

### Phase 2: REST API Server
- [ ] FastAPI-based REST server
- [ ] OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`)
- [ ] Model loading/unloading endpoints
- [ ] Authentication & rate limiting
- [ ] Streaming responses
- [ ] Docker container

### Phase 3: CLI & Shell
- [ ] Click-based CLI with subcommands
- [ ] `infinigram build` for corpus creation
- [ ] `infinigram serve` for starting server
- [ ] `infinigram predict` for one-shot inference
- [ ] `infinigram shell` for interactive REPL
- [ ] Tab completion, history, config files

### Phase 4: Advanced Matching
- [ ] Hierarchical suffix weighting
- [ ] Input projections (lemmatization, semantic)
- [ ] Output projections (filtering, mapping)
- [ ] Configurable weight functions
- [ ] Multi-scale matching (char/subword/word)

### Phase 5: Performance & Scale
- [ ] Binary search for suffix array queries (vs current linear scan)
- [ ] Memory-mapped corpus files for large datasets
- [ ] Compressed suffix arrays
- [ ] Parallel construction
- [ ] GPU acceleration for batch inference

### Phase 6: Ecosystem & Integration
- [ ] Pre-built corpus packages (Wikipedia, Common Crawl, etc.)
- [ ] Tokenizer compatibility layer for popular models (GPT, Llama)
- [ ] LangChain/LlamaIndex integration
- [ ] Hugging Face integration
- [ ] Evaluation benchmarks

## File Structure (Target)

```
infinigram/
├── infinigram/
│   ├── __init__.py
│   ├── core/
│   │   ├── infinigram.py           # Core model class
│   │   ├── suffix_array.py         # Suffix array engine
│   │   ├── projections.py          # Input/output projections
│   │   └── weighting.py            # Weighting functions
│   ├── server/
│   │   ├── api.py                  # FastAPI app
│   │   ├── models.py               # Model management
│   │   ├── auth.py                 # Authentication
│   │   └── streaming.py            # Streaming responses
│   ├── cli/
│   │   ├── main.py                 # Click CLI entry point
│   │   ├── build.py                # Corpus building
│   │   ├── serve.py                # Server management
│   │   ├── predict.py              # One-shot inference
│   │   └── shell.py                # Interactive REPL
│   └── utils/
│       ├── tokenizer.py            # Tokenization utilities
│       ├── corpus.py               # Corpus I/O
│       └── serialization.py        # Model serialization
├── tests/
│   ├── test_core/
│   ├── test_server/
│   ├── test_cli/
│   └── test_integration/
├── docs/
└── benchmarks/
```

## Design Principles

### 1. Speed First
Infinigram's killer feature is speed. Every design decision should preserve this:
- Pre-computed suffix arrays (no online construction)
- Memory-mapped corpora for large datasets
- Avoid Python loops in hot paths (use NumPy/Cython)
- Batch operations where possible

### 2. Simplicity & Composability
- Unix philosophy: do one thing well (pattern matching + prediction)
- Easy to compose with other models (mixture weights)
- Clean separation: core logic, server, CLI

### 3. Explainability
Every prediction should be traceable:
- Return corpus positions of matches
- Show actual text context
- Confidence scores based on match quality

### 4. Backward Compatibility
- Maintain existing `Infinigram` API for current users
- Deprecation warnings before breaking changes
- Versioned REST API (`/v1/`, `/v2/`)

## Ideas for Sample Efficiency

### 1. Fuzzy Matching
- Allow 1-2 token substitutions in suffix matching
- Use edit distance to find "close enough" matches
- Weight by similarity score

### 2. Semantic Clustering
- Cluster tokens by embeddings
- Match on cluster IDs instead of exact tokens
- Find longer "semantic suffixes"

### 3. Frequency-Based Fallbacks
- When no long match found, use shorter matches from high-frequency contexts
- Weight by corpus frequency (common phrases matter more)

### 4. Context Expansion
- Look for matches in expanded window (e.g., bag-of-words nearby)
- Find non-contiguous matches

### 5. Hybrid Neural-Symbolic
- Use neural encoder for context → embedding
- Nearest neighbor search in embedding space for similar corpus contexts
- Use those contexts' continuations

## Performance Targets

- **Construction**: 1M tokens/second
- **Query latency**: <10ms for 100-token context
- **Throughput**: 1000+ queries/second on single CPU
- **Memory**: <10 bytes per corpus token
- **Scaling**: 1B+ token corpora

## Success Metrics

1. **API adoption**: Used in 10+ downstream projects
2. **Performance**: 100x faster than neural LM inference
3. **Accuracy**: Competitive perplexity on domain-specific corpora
4. **LLM improvement**: Measurable reduction in hallucinations when mixed with LLMs
5. **Ease of use**: New model trained and deployed in <5 minutes
