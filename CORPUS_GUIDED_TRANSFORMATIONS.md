# Corpus-Guided Transformation Generation

## Overview

Implemented corpus-guided transformation generation for the recursive Infinigram system. Transformers now inspect the corpus at match positions to generate intelligent, context-aware transformations.

## What Was Implemented

### 1. Enhanced Transformer Logic

All transformers (`SynonymTransformer`, `EditDistanceTransformer`) now:

- **Inspect corpus at match positions**: When a suffix match is found, transformers examine what comes BEFORE the match in the corpus
- **Compare with input context**: They compare the corpus prefix with the input prefix to identify differences
- **Generate targeted transformations**: Only suggest transformations that would make the input more like patterns seen in the corpus

### 2. Deduplication

- Transformers track unique transformations to avoid generating duplicates
- Even when multiple corpus positions have the same pattern, only one transformation is generated

### 3. Word Replacement Logic

Implemented robust word replacement that:
- Preserves the matched suffix unchanged
- Correctly handles whitespace between prefix and suffix
- Replaces words at specific positions in the tokenized prefix

### 4. Validation

Added validation to ensure:
- Prefixes have the same number of words before comparing
- Corpus positions are valid (position >= prefix_length)
- Edit distances are within specified bounds

## How It Works

### Example: Typo Detection

**Corpus**: `"hello world the cat sleeps"`

**Input**: `"the caat sleeps"`

**Process**:
1. Find suffix match: `" sleeps"` matches at position 19
2. Extract prefixes:
   - Context prefix: `"the caat"` (8 bytes before suffix)
   - Corpus prefix: `"the cat "` (8 bytes before position 19)
3. Tokenize and compare:
   - Context words: `[b'the', b'caat']`
   - Corpus words: `[b'the', b'cat']`
4. Detect difference: `caat` vs `cat`, edit distance = 1
5. Generate transformation: `typo:caat→cat`
6. Return: `b"the cat sleeps"`

### Example: Synonym Detection (Placeholder)

**Corpus**: `"the feline chased the mouse"`

**Input**: `"the cat chased"`

**Process**:
1. Find suffix match: `"chased"` at some position
2. Extract prefixes and compare words
3. Detect `cat` vs `feline`
4. Check if synonyms (currently uses placeholder logic)
5. Generate: `synonym:cat→feline`

(Note: Full WordNet integration pending)

## Test Coverage

Created comprehensive test suite (`tests/test_corpus_guided_transformations.py`) with 11 tests:

### SynonymTransformer Tests
- ✅ Generates transformations from corpus inspection
- ✅ Avoids duplicate transformations
- ✅ Handles different word counts gracefully

### EditDistanceTransformer Tests
- ✅ Detects typos by comparing with corpus
- ✅ Respects max edit distance parameter
- ✅ Avoids duplicate typo corrections

### CaseNormalizer Tests
- ✅ Normalizes case correctly

### Integration Tests
- ✅ End-to-end typo correction in RecursiveInfinigram
- ✅ End-to-end case normalization

### Word Replacement Tests
- ✅ Synonym word replacement preserves suffix
- ✅ Typo correction preserves suffix

## Key Implementation Details

### SynonymTransformer

```python
def generate_transformations(self, context, suffix, corpus, match_positions):
    # Extract what comes BEFORE the suffix in both context and corpus
    prefix_len = len(context) - len(suffix)
    context_prefix = context[:prefix_len]

    # For each match position
    for pos in match_positions[:10]:  # Limit to prevent explosion
        if pos < prefix_len:
            continue  # Not enough corpus before match

        corpus_prefix = corpus[pos - prefix_len:pos]

        # Tokenize and compare
        context_words = context_prefix.split()
        corpus_words = corpus_prefix.split()

        # Find synonyms and generate transformations
        ...
```

### EditDistanceTransformer

```python
def generate_transformations(self, context, suffix, corpus, match_positions):
    # Similar to SynonymTransformer, but uses edit distance
    for i, (ctx_word, corp_word) in enumerate(zip(context_words, corpus_words)):
        if ctx_word != corp_word:
            dist = self._edit_distance(ctx_word, corp_word)
            if 0 < dist <= self.max_distance:
                # Generate typo correction
                new_context = self._replace_word_in_context(...)
                transformations.append((new_context, f"typo:{ctx_word}→{corp_word}"))
```

### Word Replacement

```python
def _replace_word_in_context(self, context, context_prefix, old_word, new_word, position):
    suffix = context[len(context_prefix):]

    # Preserve trailing whitespace from prefix
    prefix_ends_with_space = context_prefix.endswith(b' ')

    words = context_prefix.split()
    words[position] = new_word
    new_prefix = b' '.join(words)

    if prefix_ends_with_space:
        new_prefix = new_prefix + b' '

    return new_prefix + suffix
```

## Limitations and Future Work

### Current Limitations

1. **Synonym detection is placeholder**: Currently uses case-insensitive equality instead of real semantic similarity
2. **Fixed word tokenization**: Uses simple whitespace splitting
3. **No cross-language support**: Assumes English word boundaries

### Next Steps

1. **WordNet Integration**: Replace placeholder `_are_synonyms()` with real WordNet/NLTK integration
2. **Better Match Scoring**: Implement weighted scoring based on:
   - Match length
   - Edit distance
   - Frequency of pattern in corpus
3. **Benchmarks**: Evaluate OOD generalization vs vanilla Infinigram
4. **Performance Optimization**: Cache transformations, parallelize corpus inspection

## Test Results

```
tests/test_corpus_guided_transformations.py::TestSynonymTransformerCorpusInspection::test_generates_transformation_from_corpus_inspection PASSED
tests/test_corpus_guided_transformations.py::TestSynonymTransformerCorpusInspection::test_avoids_duplicate_transformations PASSED
tests/test_corpus_guided_transformations.py::TestSynonymTransformerCorpusInspection::test_handles_different_word_counts PASSED
tests/test_corpus_guided_transformations.py::TestEditDistanceTransformerCorpusInspection::test_detects_typos_from_corpus PASSED
tests/test_corpus_guided_transformations.py::TestEditDistanceTransformerCorpusInspection::test_respects_max_distance PASSED
tests/test_corpus_guided_transformations.py::TestEditDistanceTransformerCorpusInspection::test_avoids_duplicate_typo_corrections PASSED
tests/test_corpus_guided_transformations.py::TestCaseNormalizerCorpusInspection::test_normalizes_case PASSED
tests/test_corpus_guided_transformations.py::TestIntegrationWithRecursiveInfinigram::test_end_to_end_typo_correction PASSED
tests/test_corpus_guided_transformations.py::TestIntegrationWithRecursiveInfinigram::test_end_to_end_case_normalization PASSED
tests/test_corpus_guided_transformations.py::TestTransformerWordReplacement::test_synonym_word_replacement_preserves_suffix PASSED
tests/test_corpus_guided_transformations.py::TestTransformerWordReplacement::test_edit_distance_word_replacement_preserves_suffix PASSED

11 passed in 0.24s
```

## Files Modified

- `infinigram/recursive.py`: Updated transformer implementations
- `tests/test_corpus_guided_transformations.py`: New comprehensive test suite

## Integration with Existing System

The corpus-guided transformations integrate seamlessly with the existing `RecursiveInfinigram` system:

1. `RecursiveInfinigram.predict()` calls `_recursive_transform()`
2. `_recursive_transform()` finds suffix matches using `find_all_suffix_matches()`
3. For each match, transformers generate corpus-guided transformations
4. Transformations are scored and pruned via beam search
5. Each transformed context is recursively processed
6. Predictions from all contexts are weighted and combined

The system is ready for the next phase: **WordNet integration for semantic synonym detection**.
