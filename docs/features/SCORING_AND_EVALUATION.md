# Transformation Scoring and Evaluation Framework

## Overview

This document describes the transformation scoring system and evaluation framework for Infinigram's OOD generalization features.

## Transformation Scoring System

### Motivation

When using `predict_search()` to beam search over multiple transform combinations, each transformed context may produce different predictions. We need to **weight** these predictions based on:

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
- Longer matches = more confident predictions
- Uses sqrt for diminishing returns

**2. Match Frequency Score (default weight: 0.2)**
- More occurrences = more confident
- Uses logarithmic scaling

**3. Transformation Quality Score (default weight: 0.3)**
- Different transformers have different reliability
- Current transform reliability weights:
  - `case`: 0.99 (case normalization is very safe)
  - `lowercase`/`uppercase`/`casefold`: 0.99
  - `strip`/`normalize_whitespace`: 0.99

**4. Depth Penalty Score (default weight: 0.1)**
- Fewer transformations = better (closer to original)
- Uses exponential decay

### Scorer Presets

Three preset configurations are provided:

**1. Default Scorer**
```python
create_default_scorer()
# Balanced: (0.4, 0.2, 0.3, 0.1)
```

**2. Conservative Scorer**
```python
create_conservative_scorer()
# Match-focused: (0.5, 0.2, 0.1, 0.2)
```
Use when corpus coverage is high.

**3. Aggressive Scorer**
```python
create_aggressive_scorer()
# Transformation-friendly: (0.3, 0.3, 0.3, 0.1)
```
Use for OOD scenarios where corpus coverage is low.

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

#### 2. BenchmarkSuite

Creates test datasets and compares models.

```python
suite = BenchmarkSuite(corpus)

# Create test sets
in_dist = suite.create_in_distribution_test(num_samples=200)
ood_case = suite.create_ood_test(['case'], num_samples=200)

# Compare models
vanilla = Infinigram(corpus)
with_transforms = Infinigram(corpus, default_transforms=['lowercase'])

results = suite.compare_models(
    models={"Vanilla": vanilla, "WithTransforms": with_transforms},
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
# "the quick brown" -> "ThE QuIcK BroWN"
```

**2. Typos** (for testing purposes)
```python
ood_typo = suite.create_ood_test(['typo'], num_samples=100)
# "the quick brown" -> "teh qwick brown"
```

**3. Combined**
```python
ood_multi = suite.create_ood_test(['case', 'typo'], num_samples=100)
```

### Running Benchmarks

```python
from infinigram.infinigram import Infinigram
from infinigram.evaluation import BenchmarkSuite, print_comparison_table

# Create models with different configurations
corpus = b"your training data"
vanilla = Infinigram(corpus)
with_transforms = Infinigram(corpus, default_transforms=['lowercase'])

# Create benchmark suite
suite = BenchmarkSuite(corpus)

# Create test datasets
test_datasets = {
    "In-Dist": suite.create_in_distribution_test(100),
    "OOD-Case": suite.create_ood_test(['case'], 100),
}

# Compare
results = suite.compare_models(
    models={"Vanilla": vanilla, "WithTransforms": with_transforms},
    test_datasets=test_datasets
)

# Print results
print_comparison_table(results)
```

## Test Coverage

### Scoring Tests

Location: `tests/test_scoring.py`

- Score ranges [0, 1]
- Longer matches score higher
- More frequent patterns score higher
- Fewer transformations score higher
- Factory functions (default, conservative, aggressive)

### Evaluation Tests

Location: `tests/test_evaluation.py`

- Evaluator initialization and basic evaluation
- Accuracy and coverage calculation
- Top-k accuracy and perplexity calculation
- In-distribution and OOD test creation
- Model comparison

## Future Enhancements

### Planned OOD Features (Deferred)

The following features are planned but deferred due to runtime performance concerns:

1. **Synonym transforms**: Corpus-guided word replacement
   - Would require WordNet integration or embedding similarity
   - Runtime cost could be significant

2. **Typo correction**: Edit-distance based transforms
   - Would need fuzzy suffix arrays or BK-trees for efficiency
   - Current implementation only for test data generation

### Scoring System Improvements

1. **Adaptive Weights**: Learn optimal weights from validation data
2. **Context-Aware Scoring**: Use context length and complexity
3. **Confidence Intervals**: Provide uncertainty estimates

### Evaluation Framework Improvements

1. **Cross-Validation**: k-fold evaluation
2. **Statistical Significance**: Hypothesis testing
3. **Error Analysis**: Categorize and analyze errors
4. **Visualization**: Plot ROC curves, confusion matrices

## References

- Source code: `infinigram/scoring.py`, `infinigram/evaluation.py`
- Tests: `tests/test_scoring.py`, `tests/test_evaluation.py`
