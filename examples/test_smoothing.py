#!/usr/bin/env python3
"""
Test smoothing parameter functionality.
"""

from infinigram.repl import InfinigramREPL


def test_smoothing():
    """Test smoothing configuration."""
    repl = InfinigramREPL()

    print("Testing Smoothing Configuration...")
    print("=" * 60)

    # Create dataset
    print("\n1. Creating dataset with training data...")
    repl.execute("/dataset test")
    repl.execute("/add alex towell")
    repl.execute("/add alex is old")
    repl.execute("/add alex has cancer")

    # Test with no smoothing (default)
    print("\n2. Testing with NO smoothing (smoothing=0.0)...")
    repl.execute("/smoothing 0.0")
    repl.execute("/config")
    repl.execute("/predict alex")

    # Test with smoothing
    print("\n3. Testing WITH smoothing (smoothing=0.01)...")
    repl.execute("/smoothing 0.01")
    repl.execute("/config")
    repl.execute("/predict alex")

    # Test with higher smoothing
    print("\n4. Testing with HIGHER smoothing (smoothing=0.1)...")
    repl.execute("/smoothing 0.1")
    repl.execute("/predict alex")

    print("\n" + "=" * 60)
    print("✓ Smoothing test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_smoothing()
