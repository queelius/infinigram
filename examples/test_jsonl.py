#!/usr/bin/env python3
"""
Test JSONL loading functionality.
"""

from infinigram.repl import InfinigramREPL


def test_jsonl():
    """Test loading from JSONL."""
    repl = InfinigramREPL()

    print("Testing JSONL loading...")
    print("=" * 60)

    # Create dataset and load JSONL
    print("\n1. Loading from JSONL...")
    repl.execute("/dataset animals")
    repl.execute("/load --jsonl /tmp/test_data.jsonl")

    print("\n2. Dataset info...")
    repl.execute("/info")

    print("\n3. Predicting 'The quick'...")
    repl.execute("/predict The quick")

    print("\n4. Adding more data via JSONL...")
    # Create another JSONL file
    with open('/tmp/more_data.jsonl', 'w') as f:
        f.write('{"text": "The quick bird flies through the sky."}\n')
        f.write('{"text": "The bird sings a beautiful song."}\n')

    repl.execute("/add --jsonl /tmp/more_data.jsonl")
    repl.execute("/info")

    print("\n5. Predicting 'The bird'...")
    repl.execute("/predict The bird")

    print("\n" + "=" * 60)
    print("✓ JSONL test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_jsonl()
