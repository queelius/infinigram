#!/usr/bin/env python3
"""
Test augmentation/projection functionality.
"""

from infinigram.repl import InfinigramREPL


def test_augmentation():
    """Test projection augmentation."""
    repl = InfinigramREPL()

    print("Testing Augmentation...")
    print("=" * 60)

    # Create dataset
    print("\n1. Creating dataset...")
    repl.execute("/dataset demo")
    repl.execute("/load Hello World. Hello Everyone.")

    print("\n2. Initial predictions...")
    repl.execute("/predict Hello")

    print("\n3. Available projections...")
    repl.execute("/projections --available")

    print("\n4. Applying lowercase projection...")
    repl.execute("/augment lowercase")
    repl.execute("/projections")

    print("\n5. Predicting after augmentation...")
    repl.execute("/predict hello")  # Should now work with lowercase

    print("\n6. Applying multiple projections...")
    repl.execute("/dataset demo2")
    repl.execute("/load The Cat Sat")
    repl.execute("/augment lowercase uppercase")
    repl.execute("/info")

    print("\n7. Testing predictions on augmented data...")
    repl.execute("/predict the cat")  # lowercase variant
    repl.execute("/predict THE CAT")  # uppercase variant

    print("\n" + "=" * 60)
    print("✓ Augmentation test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_augmentation()
