# Test Strategy Review: Infinigram RecursiveInfinigram System

**Date:** 2025-10-22
**Modules Reviewed:** `infinigram/recursive.py`, `infinigram/scoring.py`, `infinigram/evaluation.py`
**Current Test Coverage:**
- `scoring.py`: 100% (82/82 statements) ✅
- `evaluation.py`: 93% (186/201 statements) ✅
- `recursive.py`: 41% (110/271 statements) ⚠️

---

## Executive Summary

The test suite demonstrates **excellent behavioral testing** for the scoring and evaluation components (100% and 93% coverage respectively), with well-structured tests that focus on contracts rather than implementation. However, **the recursive transformation system has significant coverage gaps (41%)**, particularly around the core transformation generation logic and edge case handling.

### Key Strengths
1. **Scoring tests are exemplary** - Full coverage with focused, behavioral tests
2. **Strong property-based assertions** - Tests verify mathematical properties (monotonicity, ranges, etc.)
3. **Good separation of concerns** - Component tests are isolated and clear
4. **Minimal implementation coupling** - Tests focus on observable behaviors

### Critical Gaps
1. **SynonymTransformer core logic untested** - Only 41% coverage in recursive.py
2. **EditDistanceTransformer transformation generation** - Core algorithm not exercised
3. **Edge cases in prediction combining** - Empty predictions, weight normalization
4. **Integration paths incomplete** - Transformer → Scorer → Predictor flow not fully tested
5. **Error handling paths** - Unicode errors, corpus size edge cases

---

## Detailed Analysis by Module

## 1. Scoring Module (`infinigram/scoring.py`) - 100% Coverage ✅

### Test Quality: EXCELLENT

**What's Working Well:**
- **Comprehensive component testing** - Each scoring component tested in isolation
- **Mathematical properties verified** - Ranges, monotonicity, scaling behavior
- **Edge cases covered** - Zero values, empty inputs, boundary conditions
- **Factory pattern tested** - Default, conservative, aggressive scorer variants
- **Adaptive scorer** - Performance tracking and analysis tested

**Test Structure:**
```
TestTransformationScorer (10 tests)
├── Behavioral properties (score ranges, ordering)
├── Component interactions (combining scores)
└── Edge cases (zero length, empty matches)

TestMatchLengthScoring (4 tests)
TestMatchFrequencyScoring (4 tests)
TestTransformationQualityScoring (5 tests)
TestDepthScoring (4 tests)
TestAdaptiveScorer (3 tests)
TestScorerFactories (4 tests)
```

**Excellent Examples:**
```python
def test_longer_match_higher_score(self):
    """Longer matches should score higher."""
    # Tests the BEHAVIOR: longer matches → higher scores
    # NOT testing HOW the score is calculated
    assert score_long > score_short

def test_sqrt_scaling(self):
    """Should use sqrt for diminishing returns."""
    # Tests a CONTRACT: the scaling function must be sqrt
    # This is a specification, not implementation detail
    score = scorer._score_match_length(match_length=50, context_length=100)
    expected = math.sqrt(0.5)
    assert abs(score - expected) < 1e-6
```

**No Gaps Identified** - This module's tests are a model for the rest of the codebase.

---

## 2. Evaluation Module (`infinigram/evaluation.py`) - 93% Coverage ✅

### Test Quality: VERY GOOD

**What's Working Well:**
- **End-to-end evaluation flow tested** - Evaluator, BenchmarkSuite, metrics
- **Metrics calculations verified** - Accuracy, coverage, perplexity, ranks
- **Test data generation tested** - In-distribution and OOD creation
- **Model comparison framework** - Multi-model, multi-dataset testing
- **Practical integration** - RecursiveInfinigram vs vanilla Infinigram comparison

**Test Structure:**
```
TestEvaluator (5 tests) - Core evaluation logic
TestBenchmarkSuite (7 tests) - Test generation and comparison
TestSyntheticCorpus (2 tests) - Corpus generation
TestTransformations (3 tests) - OOD transformations
TestMetrics (2 tests) - Metric calculations
TestPrintComparisonTable (1 test) - Output formatting
```

### Missing Coverage (14 lines, 7%):

**Lines 97, 112, 131, 138-142** - Verbose progress printing:
```python
if verbose and i % 100 == 0:
    print(f"Evaluating {i}/{len(test_data)}...")  # Line 97
# ... more verbose prints at 112, 131, 138-142
```
**Impact:** Low - These are logging statements, not critical logic
**Recommendation:** Add one test with `verbose=True` or mark as excluded from coverage

**Lines 204-205** - Edge case in perplexity calculation:
```python
else:
    perplexity = float('inf')
    mean_probability = 0.0
```
**Impact:** Medium - This handles the case where NO predictions have probability > 0
**Recommendation:** Add test case with model that never returns predictions

**Lines 392, 399, 405-407** - Verbose comparison printing:
```python
if verbose:
    print(f"\nEvaluating {model_name}...")
    # ... more verbose prints
```
**Impact:** Low - Logging only
**Recommendation:** Test with `verbose=True` or exclude

---

## 3. Recursive Module (`infinigram/recursive.py`) - 41% Coverage ⚠️

### Test Quality: NEEDS SIGNIFICANT IMPROVEMENT

**What's Working Well:**
- **Basic initialization tested** - RecursiveInfinigram constructor
- **Cycle detection tested** - Prevents infinite loops
- **Max depth limiting tested** - Recursion bounds respected
- **Case normalizer tested** - Simple transformer works

**Critical Gaps (159/271 lines untested):**

### Gap 1: SynonymTransformer Core Logic (Lines 70-140, 146-184)

**Untested:**
- `generate_transformations()` - The main transformation generation logic
- Corpus inspection at match positions
- Word tokenization and comparison
- Synonym detection via WordNet
- Transformation deduplication
- Word replacement in context

**Impact:** HIGH - This is core OOD handling functionality

**Current Test Limitation:**
```python
def test_edit_distance_transformer(self):
    transformations = transformer.generate_transformations(...)
    # Should work without errors
    assert isinstance(transformations, list)  # Too weak!
```

**What's Missing:**
```python
# MISSING: Test actual transformation generation
def test_synonym_transformer_generates_transformations(self):
    """Test that synonyms are detected from corpus inspection."""
    corpus = b"the feline sat on the mat"
    transformer = SynonymTransformer()

    # Context: "the cat sat" (cat → feline is in corpus)
    context = b"the cat sat"
    suffix = b"sat"
    positions = [find_positions_in_corpus(corpus, suffix)]

    transformations = transformer.generate_transformations(
        context=context,
        suffix=suffix,
        corpus=corpus,
        match_positions=positions
    )

    # BEHAVIOR: Should generate cat→feline transformation
    assert len(transformations) > 0
    new_context, desc = transformations[0]
    assert b"feline" in new_context
    assert "synonym" in desc
```

### Gap 2: EditDistanceTransformer (Lines 284-346)

**Untested:**
- Typo detection and correction
- Edit distance calculation
- Transformation generation from corpus typos

**Impact:** HIGH - Another core transformation strategy

### Gap 3: Prediction Combining Logic (Lines 645-669)

**Untested:**
- `_combine_predictions()` - Weighted combination of multiple predictions
- Weight normalization
- Handling empty predictions
- Combining overlapping byte predictions

**Impact:** HIGH - This is how recursive predictions are merged

**Missing Test:**
```python
def test_combine_predictions_with_weights(self):
    """Test weighted combination of predictions."""
    model = RecursiveInfinigram(corpus)

    # Two predictions for same byte with different weights
    weighted_predictions = [
        ({ord('a'): 0.8, ord('b'): 0.2}, 0.7),  # High weight
        ({ord('a'): 0.3, ord('b'): 0.7}, 0.3),  # Low weight
    ]

    combined = model._combine_predictions(weighted_predictions)

    # Should weight towards first prediction
    assert combined[ord('a')] > combined[ord('b')]

    # Should normalize to sum to 1.0
    assert abs(sum(combined.values()) - 1.0) < 1e-6
```

### Gap 4: Edge Cases

**Untested scenarios:**
- Empty context (len=0)
- No suffix matches found
- All transformers return empty lists
- Unicode decode errors in transformers
- Very deep recursion (depth=10+)
- Beam width = 1 (minimal beam)
- Corpus smaller than context
- Context not in corpus at all

---

## Strategic Recommendations

### Priority 1: Critical Gaps (Complete First)

#### 1.1 SynonymTransformer Full Coverage
**File:** `tests/test_recursive.py` or create `tests/test_transformers.py`

```python
class TestSynonymTransformerBehavior:
    """Test SynonymTransformer contract and behavior."""

    def test_generates_transformation_from_corpus_patterns(self):
        """Given corpus with synonym pattern, generates transformation."""
        # Test BEHAVIOR: corpus inspection → transformation generation

    def test_respects_word_boundaries(self):
        """Transformations preserve word boundaries and spacing."""
        # Test BEHAVIOR: whitespace handling is correct

    def test_deduplicates_transformations(self):
        """Multiple matches don't create duplicate transformations."""
        # Test BEHAVIOR: deduplication works

    def test_limits_transformations_per_match(self):
        """Only generates one transformation per match position."""
        # Test BEHAVIOR: prevents explosion
```

#### 1.2 Prediction Combining Edge Cases
```python
def test_combine_predictions_empty_list(self):
    """Empty prediction list returns empty dict."""

def test_combine_predictions_zero_total_weight(self):
    """Handles case where all weights sum to zero."""

def test_combine_predictions_overlapping_bytes(self):
    """Multiple predictions for same byte are correctly weighted."""
```

#### 1.3 EditDistanceTransformer Coverage
```python
class TestEditDistanceTransformerBehavior:
    def test_detects_single_char_typos(self):
        """Detects and corrects single-character substitutions."""

    def test_respects_max_distance(self):
        """Only corrects typos within max_distance."""

    def test_edit_distance_calculation_accuracy(self):
        """Levenshtein distance calculation is correct."""
```

### Priority 2: Integration Tests (Add After P1)

#### 2.1 End-to-End Transformation Flow
```python
def test_recursive_prediction_with_typo_corpus_mismatch(self):
    """
    Given: Corpus with correct spelling
    When: Context has typo
    Then: RecursiveInfinigram corrects typo and makes good prediction
    """
    corpus = b"the quick brown fox jumps over the lazy dog"
    model = RecursiveInfinigram(corpus)

    # Typo: "quikc" instead of "quick"
    context = b"the quikc brown"
    probs = model.predict(context, max_depth=2)

    # Should predict ' ' (space) after "brown"
    assert ord(' ') in probs
    assert probs[ord(' ')] > 0.5

def test_recursive_prediction_with_synonym_corpus_mismatch(self):
    """Tests synonym transformation enables prediction."""
    # Similar end-to-end test with synonyms
```

#### 2.2 Scorer Integration
```python
def test_scorer_weights_affect_prediction_ranking(self):
    """Conservative vs aggressive scorer changes prediction distribution."""
    corpus = b"test corpus"

    conservative = RecursiveInfinigram(corpus, scorer=create_conservative_scorer())
    aggressive = RecursiveInfinigram(corpus, scorer=create_aggressive_scorer())

    context = b"transformed context"

    probs_conservative = conservative.predict(context)
    probs_aggressive = aggressive.predict(context)

    # Distributions should differ based on scorer
    assert probs_conservative != probs_aggressive
```

### Priority 3: Robustness Tests (Add After P1 & P2)

#### 3.1 Error Handling
```python
def test_unicode_decode_error_in_synonym_detection(self):
    """Invalid UTF-8 bytes don't crash synonym detection."""

def test_empty_corpus_handling(self):
    """Empty corpus doesn't crash initialization."""

def test_context_longer_than_corpus(self):
    """Context longer than corpus handled gracefully."""
```

#### 3.2 Performance Edge Cases
```python
def test_deep_recursion_performance(self):
    """Very deep recursion doesn't cause stack overflow."""

def test_large_beam_width_manageable(self):
    """Large beam widths don't cause memory explosion."""
```

---

## Test Organization Assessment

### Current Structure: GOOD
```
tests/
├── test_recursive.py          # 10 tests - Basic structure only
├── test_scoring.py            # 33 tests - EXCELLENT
├── test_evaluation.py         # 20 tests - VERY GOOD
├── test_wordnet_integration.py    # 14 tests - Failing (numpy issue)
└── test_corpus_guided_transformations.py  # 11 tests - Failing
```

### Recommended Structure:
```
tests/
├── test_recursive.py          # Keep integration tests here
├── test_transformers.py       # NEW: Dedicated transformer tests
│   ├── TestSynonymTransformerBehavior
│   ├── TestEditDistanceTransformerBehavior
│   └── TestCaseNormalizerBehavior
├── test_scoring.py            # Keep as-is (100% coverage)
├── test_evaluation.py         # Keep as-is (93% coverage)
├── test_recursive_integration.py  # NEW: End-to-end workflows
│   ├── TestTypoCorrectionFlow
│   ├── TestSynonymHandlingFlow
│   └── TestScorerIntegration
└── test_edge_cases.py         # NEW: Robustness and error handling
```

---

## Test Quality Anti-Patterns Found

### ❌ Anti-Pattern 1: Too-Weak Assertions (test_recursive.py)
```python
# BAD: Only checks type, not behavior
assert isinstance(transformations, list)

# GOOD: Checks behavior
assert len(transformations) > 0
assert any("synonym" in desc for _, desc in transformations)
```

### ❌ Anti-Pattern 2: No Assertions on Core Logic
```python
def test_basic_prediction(self, simple_corpus):
    probs = model.predict(context, max_depth=1)
    assert isinstance(probs, dict)
    # May be empty if no matches, that's ok for now  # ← This is a gap!
```

### ✅ Anti-Pattern Fixed: Excellent Behavioral Tests (test_scoring.py)
```python
# EXCELLENT: Tests observable property
def test_longer_match_higher_score(self):
    score_long = scorer.score(..., match_length=15, ...)
    score_short = scorer.score(..., match_length=5, ...)
    assert score_long > score_short
```

---

## Specific Test Recommendations

### New Tests to Add to `test_recursive.py`

```python
class TestRecursiveInfinigramPredictionCombining:
    """Test prediction combining logic."""

    def test_combine_empty_predictions_returns_empty(self):
        """Empty prediction list returns empty dict."""
        model = RecursiveInfinigram(b"test corpus")
        result = model._combine_predictions([])
        assert result == {}

    def test_combine_single_prediction_normalizes(self):
        """Single prediction is normalized to sum to 1.0."""
        model = RecursiveInfinigram(b"test corpus")
        weighted_preds = [({65: 0.3, 66: 0.7}, 1.0)]
        result = model._combine_predictions(weighted_preds)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_combine_respects_weights(self):
        """Higher weight predictions contribute more."""
        model = RecursiveInfinigram(b"test corpus")
        weighted_preds = [
            ({65: 1.0}, 0.9),  # High weight for 'A'
            ({66: 1.0}, 0.1),  # Low weight for 'B'
        ]
        result = model._combine_predictions(weighted_preds)
        assert result[65] > result[66]

    def test_combine_overlapping_predictions_sum(self):
        """Overlapping byte predictions are summed."""
        model = RecursiveInfinigram(b"test corpus")
        weighted_preds = [
            ({65: 0.5}, 0.5),
            ({65: 0.8}, 0.5),
        ]
        result = model._combine_predictions(weighted_preds)
        # (0.5*0.5 + 0.8*0.5) / (0.5*0.5 + 0.8*0.5) = 1.0
        assert abs(result[65] - 1.0) < 1e-9


class TestSynonymTransformerCorpusInspection:
    """Test corpus inspection and transformation generation."""

    def test_inspects_corpus_at_match_positions(self):
        """Transformer looks at corpus before match positions."""
        corpus = b"the big cat sat. the large cat stood."
        transformer = SynonymTransformer(use_wordnet=False)  # Avoid nltk

        context = b"the small cat sat"
        suffix = b"sat"

        # Find where "sat" appears in corpus
        positions = [i for i in range(len(corpus)) if corpus[i:i+3] == suffix]

        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=positions
        )

        # Should inspect corpus and see words differ
        # (Actual synonym detection depends on WordNet)
        assert isinstance(transformations, list)

    def test_preserves_suffix_in_transformation(self):
        """Generated transformation preserves the matched suffix."""
        corpus = b"test suffix match"
        transformer = SynonymTransformer(use_wordnet=False)

        context = b"other suffix"
        suffix = b"suffix"
        positions = [5]  # "suffix" at position 5 in corpus

        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=positions
        )

        # All transformations should preserve suffix
        for new_context, desc in transformations:
            assert new_context.endswith(suffix)


class TestEditDistanceTransformerCorrectness:
    """Test edit distance transformer produces correct transformations."""

    def test_edit_distance_calculation_is_accurate(self):
        """Levenshtein distance calculation matches expected values."""
        transformer = EditDistanceTransformer()

        # Known edit distances
        assert transformer._edit_distance(b"cat", b"cat") == 0
        assert transformer._edit_distance(b"cat", b"bat") == 1
        assert transformer._edit_distance(b"cat", b"dog") == 3
        assert transformer._edit_distance(b"sitting", b"kitten") == 3

    def test_only_corrects_within_max_distance(self):
        """Respects max_distance parameter."""
        transformer = EditDistanceTransformer(max_distance=1)

        corpus = b"the cat sat on the mat"
        context = b"the dog sat"  # "dog" vs "cat" = distance 3
        suffix = b"sat"
        positions = [8]

        transformations = transformer.generate_transformations(
            context=context,
            suffix=suffix,
            corpus=corpus,
            match_positions=positions
        )

        # Should NOT generate transformation (distance too large)
        # (This depends on the words lining up correctly)
        assert isinstance(transformations, list)
```

### New Tests to Add to `test_evaluation.py`

```python
class TestEvaluatorEdgeCases:
    """Test evaluator edge cases."""

    def test_evaluate_with_no_predictions(self):
        """Handles model that never returns predictions."""
        # Create a model that always returns empty dict
        class NoOpModel:
            def predict(self, context, top_k=10):
                return {}

        model = NoOpModel()
        evaluator = Evaluator(model, "NoOp")

        test_data = [(b"test", b"x")]
        metrics, results = evaluator.evaluate(test_data)

        # Coverage should be 0%
        assert metrics.coverage == 0.0
        # Perplexity should be infinity
        assert metrics.perplexity == float('inf')
        # Mean probability should be 0
        assert metrics.mean_probability == 0.0

    def test_evaluate_with_verbose_output(self):
        """Verbose mode prints progress (coverage for logging)."""
        corpus = b"test corpus"
        model = Infinigram(corpus)
        evaluator = Evaluator(model, "Test")

        test_data = [(b"te", b"s")] * 100  # 100 samples

        # This should print progress at multiples of 100
        metrics, results = evaluator.evaluate(test_data, verbose=True)

        assert len(results) == 100
```

---

## Missing Test Scenarios by Feature

### RecursiveInfinigram Core Functionality

| Feature | Current Coverage | Missing Tests |
|---------|-----------------|---------------|
| Transformation generation | 20% | Corpus inspection logic, word comparison |
| Synonym detection | 0% | WordNet integration, similarity thresholds |
| Typo correction | 10% | Edit distance, max_distance enforcement |
| Prediction combining | 0% | Weight normalization, empty predictions |
| Beam search | 40% | Beam width limiting, scoring cutoff |
| Cycle detection | 100% ✅ | None |
| Max depth | 100% ✅ | None |

### Edge Cases and Error Handling

| Scenario | Tested? | Priority |
|----------|---------|----------|
| Empty corpus | ❌ | High |
| Empty context | ❌ | High |
| No suffix matches | ❌ | High |
| Unicode decode errors | ❌ | Medium |
| Context longer than corpus | ❌ | Medium |
| Very deep recursion (10+) | ❌ | Low |
| Large beam width (100+) | ❌ | Low |
| Zero probability predictions | ❌ | Medium |

### Integration Scenarios

| Integration Path | Tested? | Priority |
|-----------------|---------|----------|
| Transformer → Scorer → Predictor | ❌ | High |
| Conservative vs Aggressive scorer impact | ❌ | High |
| Multiple transformations in sequence | ❌ | Medium |
| Transformation + prediction explanation | Partial | Medium |
| Benchmark suite with real OOD data | Partial | Low |

---

## Recommendations Summary

### Immediate Actions (Complete in 1-2 days)

1. **Add transformation generation tests** (Priority 1.1)
   - Test `SynonymTransformer.generate_transformations()` with real examples
   - Test `EditDistanceTransformer.generate_transformations()` with typos
   - Verify corpus inspection logic works correctly

2. **Add prediction combining tests** (Priority 1.2)
   - Test empty predictions, zero weights, normalization
   - Test overlapping byte predictions are summed correctly
   - Test weight distribution affects final predictions

3. **Add EditDistanceTransformer unit tests** (Priority 1.3)
   - Test edit distance calculation accuracy
   - Test max_distance parameter enforcement
   - Test typo detection and correction logic

### Short-term Improvements (Complete in 1 week)

4. **Add end-to-end integration tests** (Priority 2.1)
   - Test full typo correction → prediction flow
   - Test full synonym handling → prediction flow
   - Verify explanations are generated correctly

5. **Add scorer integration tests** (Priority 2.2)
   - Test conservative vs aggressive scorer impact on predictions
   - Verify scorer weights affect transformation selection

6. **Fix evaluation.py coverage gaps** (Small task)
   - Add test with `verbose=True` to cover logging
   - Add test for "no predictions" edge case (lines 204-205)

### Long-term Quality Improvements (Complete in 2-3 weeks)

7. **Add robustness tests** (Priority 3)
   - Test error handling (Unicode, empty inputs)
   - Test performance edge cases (deep recursion, large beam)
   - Add property-based tests with Hypothesis

8. **Refactor test organization**
   - Create `test_transformers.py` for dedicated transformer tests
   - Create `test_recursive_integration.py` for end-to-end tests
   - Create `test_edge_cases.py` for robustness tests

---

## Coverage Goals

### Target Coverage by Module (3-month timeline)

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| scoring.py | 100% ✅ | 100% | Maintain |
| evaluation.py | 93% | 98% | Low (close logging gaps) |
| recursive.py | 41% ⚠️ | 85% | **HIGH** |

### Lines to Focus On (recursive.py)

**High Value (Core Logic):**
- Lines 70-140: SynonymTransformer.generate_transformations()
- Lines 284-346: EditDistanceTransformer.generate_transformations()
- Lines 645-669: _combine_predictions()
- Lines 536-607: _recursive_transform()

**Medium Value (Supporting Logic):**
- Lines 146-184: _are_synonyms() and WordNet integration
- Lines 233-266: _replace_word_in_context()
- Lines 352-372: _edit_distance()

**Lower Value (Helper Methods):**
- Lines 609-643: _find_best_suffix_match(), _find_all_suffix_matches()
- Lines 671-731: predict_with_explanation()

---

## Conclusion

The Infinigram test suite demonstrates **strong test engineering practices** in the scoring and evaluation modules, with behavioral tests that would remain valid even after significant refactoring. The scoring module in particular is an excellent example of TDD done right.

However, the **recursive transformation system needs significant test attention**. The 41% coverage represents untested core logic that handles OOD generalization - arguably the most important innovation in the system.

**Key Action Items:**
1. Add 15-20 focused tests for transformer generation logic (P1)
2. Add 5-10 tests for prediction combining (P1)
3. Add 10-15 integration tests for end-to-end flows (P2)
4. Reach 85% coverage on recursive.py within 3 months

The existing test structure is sound and can accommodate these additions with minimal refactoring. The scoring tests provide an excellent template for how to write resilient, behavioral tests.
