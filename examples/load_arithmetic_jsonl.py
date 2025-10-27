#!/usr/bin/env python3
"""
Load arithmetic data from JSONL file into the math dataset.
This is more efficient than adding documents one by one.
"""

from infinigram.repl import InfinigramREPL

def load_arithmetic_jsonl():
    """Load arithmetic data from JSONL file."""
    repl = InfinigramREPL()

    print("Loading math dataset...")
    repl.execute('load math')
    print()

    print("Adding arithmetic data from JSONL...")
    repl.execute('add --jsonl data/arithmetic.jsonl')
    print()

    print("Dataset updated!")
    print()

    # Show info
    print("=" * 70)
    repl.execute('ds info')

    print()
    print("Saving to disk...")
    repl.execute('save')

    print()
    print("=" * 70)
    print("Testing arithmetic completions:")
    print("=" * 70)

    test_phrases = [
        ("5 plus 7", 20),
        ("12 times", 20),
        ("100 divided by", 25),
        ("2 to the power", 25),
        ("10 squared", 20),
        ("3 cubed", 20),
        ("one half equals", 20),
        ("50 percent", 20),
    ]

    for phrase, max_bytes in test_phrases:
        print(f"\n▶ Completing: '{phrase}'")
        print("-" * 70)
        repl.cmd_complete([phrase, '--max', str(max_bytes)])

    print()
    print("=" * 70)
    print("Arithmetic data successfully loaded!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    load_arithmetic_jsonl()
