# Transformation Scoring and Evaluation Framework

## Overview

This document describes the transformation scoring system and evaluation framework implemented for RecursiveInfinigram.

## Transformation Scoring System

### Motivation

RecursiveInfinigram generates multiple transformed contexts (e.g., "the cat" → "The cat", "the feline", "teh cat"). Each transformation may produce different predictions. We need to **weight** these predictions based on:

1. How good the suffix match is
2. How frequently the pattern appears in the corpus
3. How reliable the transformations are
4. How many transformations were applied

### Architecture

#### TransformationScorer Class

Location: `infinigram/scoring.py`

```python
class TransformationScorer:
    """
    Scores transformed contexts for weighted prediction combining.

    Considers multiple factors:
    - Match length (longer = better)
    - Match frequency (more occurrences = more confident)
    - Transformation depth (fewer transformations = better)
    - Transformation type (some transformers more reliable)
    """
```

#### Scoring Components

The scorer computes a final score in [0, 1] as a weighted combination of four components:

**1. Match Length Score (default weight: 0.4)**

```python
def _score_match_length(self, match_length: int, context_length: int) -> float:
    """
    Longer matches = more confident predictions.
    Uses sqrt for diminishing returns.
    """
    ratio = match_length / context_length
    return sqrt(ratio)
```

Example:
- 100% match (full context): score = 1.0
- 50% match (half context): score = 0.707
- 25% match (quarter): score = 0.5

**2. Match Frequency Score (default weight: 0.2)**

```python
def _score_match_frequency(self, match_positions: List[int], corpus_size: int) -> float:
    """
    More occurrences = more confident (pattern is common).
    Uses logarithmic scaling.
    """
    num_matches = len(match_positions)
    return log(num_matches + 1) / log(101)
```

Example:
- 1 match: score = 0.15
- 10 matches: score = 0.52
- 100 matches: score = 1.0

**3. Transformation Quality Score (default weight: 0.3)**

```python
def _score_transformations(self, transformations: List[str]) -> float:
    """
    Different transformers have different reliability.
    Multiplies reliability scores.
    """
    cumulative_reliability = 1.0
    for transform in transformations:
        transformer_type = transform.split(':')[0]
        reliability = self.transformer_weights.get(transformer_type, 0.7)
        cumulative_reliability *= reliability
    return cumulative_reliability
```

Transformer reliability weights:
- `case`: 0.99 (case normalization is very safe)
- `typo`: 0.95 (typo corrections usually reliable)
- `edit_distance`: 0.95 (same as typo)
- `synonym`: 0.85 (synonyms might not preserve exact meaning)
- `unknown`: 0.7 (conservative default)

Example:
- No transformations: score = 1.0
- One case transformation: score = 0.99
- Two transformations (case + synonym): score = 0.99 × 0.85 = 0.84

**4. Depth Penalty Score (default weight: 0.1)**

```python
def _score_depth(self, num_transformations: int) -> float:
    """
    Fewer transformations = better (closer to original).
    Uses exponential decay.
    """
    return exp(-num_transformations / 3.0)
```

Example:
- 0 transformations: score = 1.0
- 3 transformations: score = 0.37
- 6 transformations: score = 0.14

#### Final Score Calculation

```python
total_score = (
    0.4 * match_score +
    0.2 * frequency_score +
    0.3 * transformation_score +
    0.1 * depth_score
)
```

### Scorer Presets

Three preset configurations are provided:

**1. Default Scorer**
```python
create_default_scorer()
# Balanced: (0.4, 0.2, 0.3, 0.1)
```
Balanced configuration for general use.

**2. Conservative Scorer**
```python
create_conservative_scorer()
# Match-focused: (0.5, 0.2, 0.1, 0.2)
```
Heavily prefers original context and long matches. Use when corpus coverage is high.

**3. Aggressive Scorer**
```python
create_aggressive_scorer()
# Transformation-friendly: (0.3, 0.3, 0.3, 0.1)
```
More willing to try transformations. Use for OOD scenarios where corpus coverage is low.

### Integration with RecursiveInfinigram

The scorer is integrated into RecursiveInfinigram's prediction process:

```python
class RecursiveInfinigram:
    def __init__(self, corpus: bytes, transformers=None, scorer=None):
        if scorer is None:
            from infinigram.scoring import create_default_scorer
            self.scorer = create_default_scorer()
        else:
            self.scorer = scorer

    def predict(self, context: bytes, top_k: int = 10):
        # Generate all transformations
        all_contexts = self._generate_all_transformations(context)

        weighted_predictions = []
        for transformed_context, transforms in all_contexts:
            # Get predictions from transformed context
            probs = self.model.predict(transformed_context, top_k=top_k)

            if probs:
                # Get match information
                suffix, positions = self._find_best_suffix_match(transformed_context)
                match_len = len(suffix)

                # Score this transformation
                weight = self.scorer.score(
                    context=context,
                    transformed_context=transformed_context,
                    transformations=transforms,
                    match_length=match_len,
                    match_positions=positions,
                    corpus_size=len(self.corpus)
                )

                weighted_predictions.append((probs, weight))

        # Combine weighted predictions
        return self._combine_predictions(weighted_predictions)
```

### Example Scoring Scenarios

**Scenario 1: Original Context with Perfect Match**

```
Context: "the quick brown fox"
Transformed: "the quick brown fox"
Transformations: []
Match length: 19 / 19
Match positions: [0, 10, 20, 30] (4 occurrences)

Scores:
- Match length: sqrt(1.0) = 1.0
- Frequency: log(5)/log(101) = 0.35
- Transformation: 1.0 (no transforms)
- Depth: 1.0 (no transforms)

Final: 0.4*1.0 + 0.2*0.35 + 0.3*1.0 + 0.1*1.0 = 0.87
```

**Scenario 2: Case Normalization with Good Match**

```
Context: "The Quick Brown Fox"
Transformed: "the quick brown fox"
Transformations: ["case:The->the", "case:Quick->quick", ...]
Match length: 19 / 19
Match positions: [0, 10, 20, 30] (4 occurrences)

Scores:
- Match length: 1.0
- Frequency: 0.35
- Transformation: 0.99^4 ≈ 0.96
- Depth: exp(-4/3) ≈ 0.26

Final: 0.4*1.0 + 0.2*0.35 + 0.3*0.96 + 0.1*0.26 = 0.78
```

**Scenario 3: Multiple Transformations**

```
Context: "the BIG cat"
Transformed: "the large cat"
Transformations: ["case:BIG->big", "synonym:big->large"]
Match length: 10 / 13
Match positions: [5] (1 occurrence)

Scores:
- Match length: sqrt(10/13) = 0.88
- Frequency: log(2)/log(101) = 0.15
- Transformation: 0.99 * 0.85 = 0.84
- Depth: exp(-2/3) = 0.51

Final: 0.4*0.88 + 0.2*0.15 + 0.3*0.84 + 0.1*0.51 = 0.66
```

## Evaluation Framework

### Architecture

Location: `infinigram/evaluation.py`

The evaluation framework provides tools to:
1. Evaluate models on test data
2. Create in-distribution and OOD test sets
3. Compare multiple models
4. Generate comprehensive metrics

### Components

#### 1. Evaluator

Evaluates a single model on test data.

```python
evaluator = Evaluator(model, model_name="My Model")
metrics, results = evaluator.evaluate(test_data, top_k=10)
```

Returns:
- `EvaluationMetrics`: Aggregated metrics
- `List[PredictionResult]`: Per-sample detailed results

#### 2. BenchmarkSuite

Creates test datasets and compares models.

```python
suite = BenchmarkSuite(corpus)

# Create test sets
in_dist = suite.create_in_distribution_test(num_samples=200)
ood_case = suite.create_ood_test(['case'], num_samples=200)
ood_typo = suite.create_ood_test(['typo'], num_samples=200)

# Compare models
results = suite.compare_models(
    models={"Vanilla": vanilla, "Recursive": recursive},
    test_datasets={"In-Dist": in_dist, "OOD-Case": ood_case}
)
```

#### 3. Metrics

```python
@dataclass
class EvaluationMetrics:
    # Accuracy metrics
    accuracy: float  # % of correct predictions
    top_k_accuracy: Dict[int, float]  # Top-k accuracy
    mean_rank: float  # Average rank of correct token

    # Coverage metrics
    coverage: float  # % with predictions
    no_match_rate: float  # % with no match

    # Quality metrics
    perplexity: float  # Lower = better
    mean_probability: float  # Avg prob of correct token

    # Performance metrics
    mean_time_ms: float  # Avg prediction time
    total_time_s: float  # Total evaluation time
```

### OOD Test Generation

The framework can automatically create OOD test data:

**1. Case Variations**
```python
ood_case = suite.create_ood_test(['case'], num_samples=100)
# "the quick brown" → "ThE QuIcK BroWN"
```

**2. Typos**
```python
ood_typo = suite.create_ood_test(['typo'], num_samples=100)
# "the quick brown" → "teh qwick brown"
```

**3. Synonyms**
```python
ood_syn = suite.create_ood_test(['synonym'], num_samples=100)
# "the big cat" → "the large cat"
```

**4. Combined**
```python
ood_multi = suite.create_ood_test(['case', 'typo'], num_samples=100)
# "the quick brown" → "Teh QuIcK brwon"
```

### Running Benchmarks

**Command Line:**
```bash
python3 examples/run_benchmark.py
```

**Programmatic:**
```python
from infinigram.infinigram import Infinigram
from infinigram.recursive import RecursiveInfinigram, CaseNormalizer
from infinigram.evaluation import BenchmarkSuite, print_comparison_table

# Create models
corpus = b"your training data"
vanilla = Infinigram(corpus)
recursive = RecursiveInfinigram(corpus, transformers=[CaseNormalizer()])

# Create benchmark suite
suite = BenchmarkSuite(corpus)

# Create test datasets
test_datasets = {
    "In-Dist": suite.create_in_distribution_test(100),
    "OOD-Case": suite.create_ood_test(['case'], 100),
}

# Compare
results = suite.compare_models(
    models={"Vanilla": vanilla, "Recursive": recursive},
    test_datasets=test_datasets
)

# Print results
print_comparison_table(results)
```

## Test Coverage

### Scoring Tests (33 tests)

Location: `tests/test_scoring.py`

- ✅ Score ranges [0, 1]
- ✅ Longer matches score higher
- ✅ More frequent patterns score higher
- ✅ Fewer transformations score higher
- ✅ Transformation quality matters
- ✅ Component scoring (match length, frequency, quality, depth)
- ✅ Batch scoring
- ✅ Custom weights
- ✅ Adaptive scoring
- ✅ Factory functions (default, conservative, aggressive)

### Evaluation Tests (20 tests)

Location: `tests/test_evaluation.py`

- ✅ Evaluator initialization
- ✅ Basic evaluation
- ✅ Accuracy calculation
- ✅ Coverage calculation
- ✅ Top-k accuracy
- ✅ Perplexity calculation
- ✅ Rank calculation
- ✅ In-distribution test creation
- ✅ OOD test creation (case, typo, synonym, multi)
- ✅ Model comparison
- ✅ Synthetic corpus generation
- ✅ Transformation functions

### Total Test Coverage

- **96 tests total** across all recursive components
- All tests passing ✅

## Performance Characteristics

### Time Complexity

**Vanilla Infinigram:**
- Prediction: O(log n) suffix array lookup

**RecursiveInfinigram:**
- Prediction: O(t × log n) where t = number of transformations
- Scoring: O(1) per transformation
- Combination: O(k) where k = vocabulary size

### Space Complexity

**Vanilla Infinigram:**
- O(n) for suffix array

**RecursiveInfinigram:**
- O(n) for suffix array
- O(c) for transformation cache
- O(1) for scorer

where n = corpus size, c = cache size

### Benchmark Results

See `BENCHMARKS.md` for detailed results. Summary:

| Scenario | Accuracy Improvement | Time Overhead |
|----------|---------------------|---------------|
| In-Distribution | +0% | 2.7x |
| OOD: Case | +22% | 78x |
| OOD: Typo | +6% | 21x |
| OOD: Multi | +20.5% | 61x |

## Configuration Guide

### Choosing a Scorer

**Use Default Scorer when:**
- General purpose use
- Balanced accuracy and speed
- Moderate corpus coverage

**Use Conservative Scorer when:**
- High corpus coverage
- Prefer original context
- Speed is important
- Don't expect many transformations needed

**Use Aggressive Scorer when:**
- Low corpus coverage (OOD domain)
- Willing to try many transformations
- Accuracy more important than speed
- Expect typos, variations, paraphrasing

### Custom Scorer Configuration

```python
from infinigram.scoring import TransformationScorer

# Custom weights
scorer = TransformationScorer(
    match_length_weight=0.5,     # Emphasize match quality
    match_frequency_weight=0.1,  # De-emphasize frequency
    transformation_weight=0.3,   # Moderate transformation weight
    depth_weight=0.1             # Low depth penalty
)

# Custom transformer reliabilities
scorer = TransformationScorer(
    transformer_weights={
        'case': 0.99,
        'typo': 0.95,
        'synonym': 0.7,  # Less confident in synonyms
        'custom': 0.8,   # Custom transformer
    }
)

# Use with RecursiveInfinigram
model = RecursiveInfinigram(corpus, transformers=[...], scorer=scorer)
```

### Evaluation Configuration

```python
from infinigram.evaluation import Evaluator

# Adjust top-k
metrics, results = evaluator.evaluate(test_data, top_k=20)

# Verbose mode
metrics, results = evaluator.evaluate(test_data, verbose=True)

# Custom test data
test_data = [
    (b"context 1", b"n"),
    (b"context 2", b"e"),
    # ...
]
```

## Future Enhancements

### Scoring System

1. **Adaptive Weights**: Learn optimal weights from validation data
2. **Context-Aware Scoring**: Use context length and complexity
3. **Confidence Intervals**: Provide uncertainty estimates
4. **Multi-Objective**: Balance accuracy, speed, and confidence

### Evaluation Framework

1. **Cross-Validation**: k-fold evaluation
2. **Statistical Significance**: Hypothesis testing for comparisons
3. **Error Analysis**: Categorize and analyze errors
4. **Visualization**: Plot ROC curves, confusion matrices
5. **Real-World Benchmarks**: Evaluate on actual datasets (Wikipedia, news, etc.)

## References

- Source code: `infinigram/scoring.py`, `infinigram/evaluation.py`
- Tests: `tests/test_scoring.py`, `tests/test_evaluation.py`
- Examples: `examples/run_benchmark.py`
- Benchmarks: `BENCHMARKS.md`
- WordNet integration: `WORDNET_INTEGRATION.md`
