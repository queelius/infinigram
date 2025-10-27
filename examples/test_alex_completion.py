#!/usr/bin/env python3
"""
Test completion behavior with alex dataset.
"""

from infinigram.repl import InfinigramREPL


def test_alex():
    """Test alex completion."""
    repl = InfinigramREPL()

    print("Building alex dataset...")
    print("=" * 60)

    repl.execute("/dataset alex")
    repl.execute("/add my name is alex")
    repl.execute("/add alex is old")
    repl.execute("/add alex is 50 years old")
    repl.execute("/add alex was born in 1975")
    repl.execute("/add alex was born 50 years ago in 1975")

    print("\n" + "=" * 60)
    print("Testing completion: 'alex is'")
    print("=" * 60)
    repl.execute("/complete alex is")

    print("\n" + "=" * 60)
    print("Testing completion: 'alex was'")
    print("=" * 60)
    repl.execute("/complete alex was")

    print("\n" + "=" * 60)
    print("Testing prediction: 'alex is 50 years old'")
    print("=" * 60)
    repl.execute("/predict alex is 50 years old")


if __name__ == "__main__":
    test_alex()
