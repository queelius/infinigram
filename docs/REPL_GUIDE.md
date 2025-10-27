# Infinigram REPL Guide

The Infinigram REPL (Read-Eval-Print Loop) provides an interactive shell for exploring byte-level language models, testing predictions, and training models incrementally.

## Getting Started

### Starting the REPL

```bash
# From Python
python -m infinigram.repl

# Or using the entry point script
./bin/infinigram-repl
```

You'll see:
```
======================================================================
  INFINIGRAM INTERACTIVE REPL
======================================================================

Type '/help' for available commands or '/quit' to exit.

infinigram>
```

## Core Concepts

### Datasets = Models

In Infinigram, **datasets are models**. Each dataset is a trained Infinigram model built from the text you load into it. You can:

- Create multiple named datasets
- Switch between datasets
- Incrementally add training data
- Compare predictions across datasets

### The Prompt

The prompt shows your current dataset:

```
infinigram>              # No dataset selected
infinigram [english]>    # "english" dataset is active
```

## Commands

### Dataset Management

#### `/dataset <name>` - Create or Switch to Dataset
Create a new dataset or switch to an existing one.

```
infinigram> /dataset english
✓ Created dataset: english
✓ Switched to dataset: english
```

#### `/dataset copy <source> <destination>` - Copy Dataset
Create a deep copy of an existing dataset with all its data and configuration.

```
infinigram [english]> /dataset copy english english_backup
✓ Copied dataset 'english' to 'english_backup'
  Size: 1024 bytes
```

This is useful for:
- Creating backups before augmentation
- Experimenting with different projections on the same base data
- A/B testing different model configurations

#### `/datasets` - List All Datasets
See all loaded datasets and their sizes.

```
infinigram [english]> /datasets
Available datasets:
  english: 1024 bytes (current)
  spanish: 2048 bytes
  code: 512 bytes
```

#### `/use <name>` - Switch Dataset
Switch to a different dataset.

```
infinigram [english]> /use spanish
✓ Switched to dataset: spanish
```

### Loading Data

#### `/load <text>` - Load Text
Create a dataset from inline text.

```
infinigram> /dataset demo
infinigram [demo]> /load the cat sat on the mat
✓ Loaded into 'demo': 23 bytes
```

#### `/load --file <path>` - Load from File
Load a text file into the current dataset.

```
infinigram [demo]> /load --file data/shakespeare.txt
Loaded 1048576 characters from data/shakespeare.txt
✓ Loaded into 'demo': 1048576 bytes
```

#### `/load --jsonl <path>` - Load from JSONL
Load documents from a JSONL file. Each line should be a JSON object with a `text` field.

```
infinigram [demo]> /load --jsonl data/documents.jsonl
Loaded 1000 documents from data/documents.jsonl
✓ Loaded into 'demo': 524288 bytes
```

**JSONL Format**:
```jsonl
{"text": "First document content"}
{"text": "Second document content"}
{"text": "Third document content"}
```

Documents are automatically separated with `\n\n` to prevent cross-document patterns.

**Custom Field**:
```bash
/load --jsonl data.jsonl --field content
```

### Incremental Training

#### `/add <text>` - Add Text to Dataset
Add more training examples to the current dataset.

```
infinigram [demo]> /add the dog sat on the log
✓ Added 23 bytes to 'demo'
  Total corpus size: 46 bytes
```

#### `/add --file <path>` - Add File
```
infinigram [demo]> /add --file more_data.txt
✓ Added 2048 bytes to 'demo'
  Total corpus size: 2094 bytes
```

#### `/add --jsonl <path>` - Add JSONL
```
infinigram [demo]> /add --jsonl more_docs.jsonl
Loaded 100 documents from more_docs.jsonl
✓ Added 51200 bytes to 'demo'
  Total corpus size: 53294 bytes
```

### Prediction

#### `/predict <text>` - Show Next-Byte Probabilities
Display the probability distribution for the next byte.

```
infinigram [demo]> /predict the cat
Context: 'the cat' (7 bytes)
Top 50 predictions:

  ' ' (byte 32): 0.853
  's' (byte 115): 0.042
  '.' (byte 46): 0.031
  ...
```

**Show as bytes**:
```
infinigram [demo]> /predict the cat --bytes
Context: 'the cat' (7 bytes)
Top 50 predictions:

  Byte  32 (0x20): 0.853
  Byte 115 (0x73): 0.042
  Byte  46 (0x2E): 0.031
  ...
```

#### `/complete <text>` - Generate Completion
Generate a continuation of the input text.

```
infinigram [demo]> /complete the cat
Context: 'the cat'
Generating up to 50 bytes...

Generated: ' sat on the mat. the dog ran on the log.'
(45 bytes)
```

**Specify length**:
```
infinigram [demo]> /complete the cat --max 20
Generated: ' sat on the warm ma'
(20 bytes)
```

### Configuration

#### `/temperature <value>` - Sampling Temperature
Control randomness in generation (default: 1.0).

- Higher values (>1.0) = more uniform/random
- Lower values (<1.0) = more peaked/deterministic

```
infinigram [demo]> /temperature 0.5
✓ Temperature set to 0.5

infinigram [demo]> /temperature 2.0
✓ Temperature set to 2.0
```

#### `/top_k <n>` - Top-K Display
Set how many predictions to show (default: 50).

```
infinigram [demo]> /top_k 10
✓ top_k set to 10
```

#### `/max_length <n>` - Max Suffix Length
Limit the maximum suffix length for matching (default: unlimited).

```
infinigram [demo]> /max_length 20
✓ max_length set to 20

infinigram [demo]> /max_length none
✓ max_length set to unlimited
```

#### `/weight <function>` - Weight Function
Enable hierarchical weighted prediction.

Options: `none`, `linear`, `quadratic`, `exponential`, `sigmoid`

```
infinigram [demo]> /weight quadratic
✓ Weight function set to quadratic

infinigram [demo]> /weight none
✓ Disabled weighted prediction
```

#### `/config` - Show Configuration
Display current settings.

```
infinigram [demo]> /config
Current Configuration:
  Temperature: 1.0
  Top-k: 50
  Max suffix length: unlimited
  Weight function: none
  Weighted prediction min_length: 1
  Weighted prediction max_length: auto
```

### Information

#### `/info` - Dataset Information
Show current dataset details.

```
infinigram [demo]> /info
Dataset: demo
  Corpus size: 1024 bytes
  Vocabulary size: 256
  Max suffix length: unlimited
  Min count: 1
  Smoothing: 0.01
```

#### `/stats` - Corpus Statistics
Show byte distribution statistics.

```
infinigram [demo]> /stats
Corpus Statistics:
  Total bytes: 1024
  Unique bytes: 52

  Top 10 most frequent bytes:
     32 (' '):   156 (15.23%)
    101 ('e'):   102 (9.96%)
    116 ('t'):    89 (8.69%)
    ...
```

### Augmentation/Projections

Augmentations apply transformations to your dataset, creating variants alongside the original data. This is powerful for case-insensitive models, normalization, and data augmentation.

#### `/augment <projection> [projection...]` - Apply Projections
Apply one or more projections to the current dataset. The original data is preserved, and augmented variants are added.

Available projections:
- `lowercase` - Convert text to lowercase
- `uppercase` - Convert text to UPPERCASE
- `title` - Convert Text To Title Case
- `strip` - Remove leading/trailing whitespace

```
infinigram [demo]> /augment lowercase uppercase
Applying 2 projection(s) to 'demo'...
✓ Applied projections: lowercase, uppercase
  Original size: 100 bytes
  Augmented size: 300 bytes
  Multiplier: 3.00x
```

**How it works**:
- Original data: "Hello World"
- After `/augment lowercase uppercase`:
  - Original: "Hello World"
  - Lowercase variant: "hello world"
  - Uppercase variant: "HELLO WORLD"

This creates a model that can handle multiple case variations.

#### `/projections` - List Active Projections
Show which projections have been applied to the current dataset.

```
infinigram [demo]> /projections
Active projections for 'demo':
  lowercase
  uppercase
```

#### `/projections --available` - List Available Projections
Show all registered projections you can apply.

```
infinigram> /projections --available
Available projections:
  lowercase
  strip
  title
  uppercase
```

### Bash Commands

#### `!<command>` - Execute Bash Command
Run any bash command from within the REPL. Useful for file operations, checking data, or quick system tasks.

```
infinigram> !ls -lh data/
-rw-r--r-- 1 user user 1.2M Oct 17 data.txt

infinigram> !wc -l data/corpus.txt
10000 data/corpus.txt

infinigram> !head -3 data/sample.jsonl
{"text": "First document"}
{"text": "Second document"}
{"text": "Third document"}
```

Commands run with a 30-second timeout and capture both stdout and stderr.

### Other

#### `/help` - Show Help
Display all available commands.

#### `/quit` or `/exit` - Exit REPL
Leave the REPL.

#### `/reset` - Delete Current Dataset
Remove the current dataset from memory.

```
infinigram [demo]> /reset
✓ Deleted dataset: demo
```

## Example Workflows

### Dataset Copying and Augmentation

```bash
# Create a base dataset
/dataset base
/load The quick brown fox jumps over the lazy dog.

# Create a copy for experimentation
/dataset copy base base_augmented

# Apply augmentations to the copy
/use base_augmented
/augment lowercase uppercase

# Compare predictions
/use base
/predict The quick
# Only matches exact case

/use base_augmented
/predict the quick
# Works! Lowercase variant exists

/predict THE QUICK
# Also works! Uppercase variant exists
```

### Building a Multi-Domain Model

```bash
# Create separate datasets for different domains
/dataset code
/load --file python_code.txt

/dataset prose
/load --file shakespeare.txt

/dataset technical
/load --jsonl papers.jsonl

# Compare predictions
/use code
/predict def fibonacci

/use prose
/predict To be or not

/use technical
/predict The algorithm
```

### Incremental Training

```bash
# Start with base knowledge
/dataset assistant
/load Hello! How can I help you today?

# Add more examples as you go
/add I'm happy to assist with your questions.
/add Please let me know if you need anything else.
/add I'll do my best to provide helpful information.

# Test predictions
/predict How can I
```

### Experimenting with Weight Functions

```bash
/load the cat sat on the mat. the cat ran.

# Try different weighting schemes
/weight linear
/predict the cat

/weight quadratic
/predict the cat

/weight exponential
/predict the cat
```

### Loading from JSONL

```bash
# Create a JSONL dataset
/dataset wiki
/load --jsonl wikipedia_articles.jsonl

# Add more articles
/add --jsonl more_articles.jsonl

# Query
/predict The capital of France
```

### Using Bash Commands

```bash
# Check data files before loading
!ls -lh data/
!wc -l data/corpus.txt

# Preview JSONL structure
!head -3 data/documents.jsonl

# Load the data
/dataset docs
/load --jsonl data/documents.jsonl

# Check system resources
!free -h
!df -h

# Quick text processing
!grep -c "keyword" data/corpus.txt
```

## Tips

1. **Start Small**: Begin with small datasets to understand behavior
2. **Multiple Datasets**: Create separate datasets for different tasks
3. **Incremental Learning**: Use `/add` to grow datasets over time
4. **Experiment**: Try different temperatures and weight functions
5. **Check Stats**: Use `/stats` to understand your corpus composition
6. **JSONL for Documents**: Use JSONL format for multi-document corpora
7. **Copy Before Augmenting**: Use `/dataset copy` to preserve original data before applying projections
8. **Bash Integration**: Use `!command` for quick file checks, data inspection, and system tasks
9. **Projection Combinations**: Apply multiple projections together for comprehensive normalization

## Advanced Usage

### Temperature Effects

```bash
# Deterministic (greedy)
/temperature 0.1
/complete the cat
# Output: " sat on the mat. the cat sat on the mat."

# Balanced
/temperature 1.0
/complete the cat
# Output: " ran on the log. the dog sat near"

# Random/Creative
/temperature 2.0
/complete the cat
# Output: " wobbled through mysterious gardens while"
```

### Hierarchical Prediction

```bash
# Linear: w(k) = k
/weight linear
/predict the

# Quadratic: w(k) = k²  (strongly favors longer matches)
/weight quadratic
/predict the

# Exponential: w(k) = 2^k  (very strongly favors longest)
/weight exponential
/predict the
```

## Troubleshooting

**Problem**: "No dataset selected"
**Solution**: Create or select a dataset with `/dataset <name>`

**Problem**: Slow predictions on large corpus
**Solution**: Use `/max_length` to limit suffix search

**Problem**: Repetitive completions
**Solution**: Increase `/temperature` for more variety

**Problem**: Random/incoherent completions
**Solution**: Decrease `/temperature` for more focus
