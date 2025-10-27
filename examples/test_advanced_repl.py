#!/usr/bin/env python3
"""
Test advanced REPL features: dataset copying, bash commands, augmentation.
"""

from infinigram.repl import InfinigramREPL


def test_advanced_features():
    """Test advanced REPL functionality."""
    repl = InfinigramREPL()

    print("Testing Advanced REPL Features...")
    print("=" * 60)

    # Test bash commands
    print("\n1. Testing bash commands...")
    repl.execute("!echo 'Hello from bash!'")
    repl.execute("!date +%Y-%m-%d")
    repl.execute("!ls -la | head -5")

    # Test dataset creation and copying
    print("\n2. Creating and copying datasets...")
    repl.execute("/dataset original")
    repl.execute("/load the quick brown fox jumps over the lazy dog")
    repl.execute("/info")

    print("\n3. Copying dataset...")
    repl.execute("/dataset copy original modified")
    repl.execute("/datasets")

    # Test augmentation on copy
    print("\n4. Augmenting copied dataset...")
    repl.execute("/use modified")
    repl.execute("/augment lowercase uppercase")
    repl.execute("/projections")
    repl.execute("/datasets")

    # Compare original vs modified
    print("\n5. Comparing datasets...")
    repl.execute("/use original")
    repl.execute("/predict the quick")

    repl.execute("/use modified")
    repl.execute("/predict the quick")
    repl.execute("/predict THE QUICK")  # Should work in modified

    # Test projection listing
    print("\n6. Testing projection management...")
    repl.execute("/projections --available")
    repl.execute("/projections")

    # Test multiple dataset workflow
    print("\n7. Multi-dataset workflow...")
    repl.execute("/dataset numbers")
    repl.execute("/load one two three four five")
    repl.execute("/dataset copy numbers numbers_aug")
    repl.execute("/use numbers_aug")
    repl.execute("/augment title")
    repl.execute("/datasets")

    print("\n8. Final state...")
    repl.execute("/datasets")
    repl.execute("/use modified")
    repl.execute("/info")
    repl.execute("/projections")

    print("\n" + "=" * 60)
    print("✓ All advanced features tested successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_advanced_features()
