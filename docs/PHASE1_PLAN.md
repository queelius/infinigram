# Phase 1: Core API Enhancements - Implementation Plan

**Goal**: Extend Infinigram with advanced matching capabilities while maintaining backward compatibility.

## Features to Implement

### 1. Hierarchical Suffix Weighting
**Motivation**: Currently we only use the longest matching suffix. But shorter suffixes also provide valuable information. Combining predictions from multiple suffix lengths with appropriate weights can improve accuracy.

**API Design**:
```python
# New method on Infinigram class
def predict_weighted(
    self,
    context: List[int],
    min_length: int = 1,
    max_length: Optional[int] = None,
    weight_fn: Optional[Callable[[int], float]] = None,
    top_k: int = 50
) -> Dict[int, float]:
    """
    Predict using weighted combination of multiple suffix lengths.

    Args:
        context: Token sequence
        min_length: Minimum suffix length to consider
        max_length: Maximum suffix length (None = use self.max_length)
        weight_fn: Function mapping suffix_length -> weight
                  Default: lambda k: k (linear weighting)
        top_k: Return top k predictions

    Returns:
        Dict mapping token -> probability
    """
```

**Weight Functions** (in `infinigram/weighting.py`):
```python
def linear_weight(length: int) -> float:
    """w(k) = k"""
    return float(length)

def quadratic_weight(length: int) -> float:
    """w(k) = k^2"""
    return float(length ** 2)

def exponential_weight(base: float = 2.0) -> Callable[[int], float]:
    """w(k) = base^k"""
    def weight(length: int) -> float:
        return base ** length
    return weight

def custom_weight(max_length: int, steepness: float = 1.0) -> Callable[[int], float]:
    """Sigmoid-like weight: w(k) = 1 / (1 + exp(-steepness * (k - max_length/2)))"""
    def weight(length: int) -> float:
        return 1.0 / (1.0 + np.exp(-steepness * (length - max_length / 2)))
    return weight
```

**Tests** (`tests/test_weighted_prediction.py`):
- Test that weighted prediction with only longest match equals regular predict
- Test that shorter suffixes contribute to final prediction
- Test different weight functions produce different distributions
- Test min_length and max_length boundaries
- Test that weights sum correctly
- Test edge case: no matches at any length

### 2. Input Projections
**Motivation**: Transform input context to find better matches. For example, lemmatization might match "running" with "run", or dropping stopwords might find longer content-word matches.

**API Design**:
```python
# New abstract base class
class InputProjection(ABC):
    """Transform context before matching."""

    @abstractmethod
    def project(self, tokens: List[int]) -> List[int]:
        """Transform token sequence."""
        pass

# Concrete implementations
class IdentityProjection(InputProjection):
    """No transformation (default)."""

class SubsampleProjection(InputProjection):
    """Keep every nth token."""
    def __init__(self, stride: int = 2):
        self.stride = stride

class TruncateProjection(InputProjection):
    """Keep last k tokens."""
    def __init__(self, max_tokens: int = 10):
        self.max_tokens = max_tokens

# Update Infinigram
def predict(
    self,
    context: List[int],
    top_k: int = 50,
    input_projection: Optional[InputProjection] = None
) -> Dict[int, float]:
    """
    Added input_projection parameter.
    """
```

**Tests** (`tests/test_input_projections.py`):
- Test identity projection doesn't change behavior
- Test subsample projection finds matches that full context misses
- Test truncate projection limits context length
- Test custom projection can be implemented
- Test projection with weighted prediction

### 3. Output Projections
**Motivation**: Filter or transform the predicted token distribution. For example, restrict to top-k most frequent tokens, or filter to domain-specific vocabulary.

**API Design**:
```python
# New abstract base class
class OutputProjection(ABC):
    """Filter/transform output predictions."""

    @abstractmethod
    def project(self, probs: Dict[int, float]) -> Dict[int, float]:
        """Transform probability distribution."""
        pass

# Concrete implementations
class IdentityOutputProjection(OutputProjection):
    """No transformation (default)."""

class TopKFrequentProjection(OutputProjection):
    """Restrict to k most frequent tokens in corpus."""
    def __init__(self, corpus: List[int], k: int = 1000):
        # Compute top k frequent tokens
        counter = Counter(corpus)
        self.allowed_tokens = set(t for t, _ in counter.most_common(k))

class VocabularyFilterProjection(OutputProjection):
    """Only allow specific vocabulary tokens."""
    def __init__(self, allowed_tokens: Set[int]):
        self.allowed_tokens = allowed_tokens

class ThresholdProjection(OutputProjection):
    """Zero out probabilities below threshold."""
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

# Update Infinigram
def predict(
    self,
    context: List[int],
    top_k: int = 50,
    input_projection: Optional[InputProjection] = None,
    output_projection: Optional[OutputProjection] = None
) -> Dict[int, float]:
    """
    Added output_projection parameter.
    """
```

**Tests** (`tests/test_output_projections.py`):
- Test identity projection doesn't change output
- Test top-k frequent filter restricts vocabulary
- Test vocabulary filter only returns allowed tokens
- Test threshold projection zeros out low probabilities
- Test renormalization after filtering
- Test composition of multiple output projections

## Implementation Order

### Step 1: Weighting Functions (Simplest)
1. Write tests for weight functions
2. Implement `infinigram/weighting.py`
3. Verify tests pass

### Step 2: Hierarchical Prediction
1. Write tests for `predict_weighted()`
2. Implement method in `Infinigram` class
3. Verify tests pass
4. Update documentation

### Step 3: Input Projections
1. Write tests for projection classes
2. Implement `infinigram/projections.py` with `InputProjection` classes
3. Update `predict()` to accept input projection
4. Verify tests pass
5. Update documentation

### Step 4: Output Projections
1. Write tests for output projection classes
2. Implement `OutputProjection` classes in `infinigram/projections.py`
3. Update `predict()` to accept output projection
4. Verify tests pass
5. Update documentation

### Step 5: Integration & Examples
1. Write integration tests combining all features
2. Create example notebook/script demonstrating Phase 1 features
3. Update README with new capabilities
4. Update CLAUDE.md

## Success Criteria

- [ ] All new tests pass (targeting 20+ new tests)
- [ ] All existing tests still pass (backward compatibility)
- [ ] Code coverage remains > 90%
- [ ] Documentation updated
- [ ] Example demonstrating all Phase 1 features
- [ ] No breaking changes to existing API

## File Structure After Phase 1

```
infinigram/
├── infinigram/
│   ├── __init__.py           # Export new classes
│   ├── infinigram.py         # Enhanced with new methods
│   ├── suffix_array.py       # Unchanged
│   ├── weighting.py          # NEW: Weight functions
│   └── projections.py        # NEW: Input/Output projections
├── tests/
│   ├── test_infinigram.py    # Existing tests
│   ├── test_weighting.py     # NEW
│   ├── test_weighted_prediction.py  # NEW
│   ├── test_input_projections.py    # NEW
│   ├── test_output_projections.py   # NEW
│   └── test_phase1_integration.py   # NEW
└── examples/
    └── phase1_demo.py        # NEW: Demonstrate new features
```

## Timeline Estimate

- Step 1 (Weighting): ~30 min
- Step 2 (Hierarchical): ~1 hour
- Step 3 (Input Projections): ~1 hour
- Step 4 (Output Projections): ~1 hour
- Step 5 (Integration): ~30 min

**Total**: ~4 hours for Phase 1

## Notes

- Keep existing `predict()` signature for backward compatibility
- All new parameters should be optional with sensible defaults
- Maintain test coverage above 90%
- Document all new public APIs with examples
