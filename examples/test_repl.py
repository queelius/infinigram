#!/usr/bin/env python3
"""
Test the REPL functionality programmatically.
"""

from infinigram.repl import InfinigramREPL


def test_repl():
    """Test REPL commands programmatically."""
    repl = InfinigramREPL()

    print("Testing REPL commands...")
    print("=" * 60)

    # Test dataset creation
    print("\n1. Creating datasets...")
    repl.execute("/dataset english")
    repl.execute("/datasets")

    # Test loading text
    print("\n2. Loading text...")
    repl.execute("/load the cat sat on the mat. the cat ran on the mat.")
    repl.execute("/info")

    # Test adding more text
    print("\n3. Adding more text...")
    repl.execute("/add the dog sat on the log.")
    repl.execute("/info")

    # Test prediction
    print("\n4. Testing prediction...")
    repl.execute("/predict the cat")

    # Test completion
    print("\n5. Testing completion...")
    repl.execute("/complete the cat --max 20")

    # Test creating another dataset
    print("\n6. Creating second dataset...")
    repl.execute("/dataset numbers")
    repl.execute("/load one two three four five. one two six seven. one two three eight.")
    repl.execute("/datasets")

    # Test switching between datasets
    print("\n7. Switching datasets...")
    repl.execute("/use english")
    repl.execute("/predict the")

    repl.execute("/use numbers")
    repl.execute("/predict one two")

    # Test configuration
    print("\n8. Testing configuration...")
    repl.execute("/config")
    repl.execute("/temperature 0.5")
    repl.execute("/top_k 10")
    repl.execute("/weight quadratic")
    repl.execute("/config")

    # Test stats
    print("\n9. Dataset statistics...")
    repl.execute("/use english")
    repl.execute("/stats")

    print("\n" + "=" * 60)
    print("✓ All REPL tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_repl()
