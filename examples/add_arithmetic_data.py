#!/usr/bin/env python3
"""
Generate synthetic arithmetic data for the math dataset.
Creates lots of arithmetic equations with their solutions.
"""

from infinigram.repl import InfinigramREPL
import random

def generate_arithmetic_data():
    """Generate synthetic arithmetic equations and add to math dataset."""
    repl = InfinigramREPL()

    print("Loading math dataset...")
    repl.execute('load math')
    print()

    arithmetic_docs = []

    # Addition facts
    print("Generating addition facts...")
    for i in range(0, 21):
        for j in range(0, 21):
            result = i + j
            arithmetic_docs.append(f"{i} plus {j} equals {result}.")
            if i != j:  # Add commutative version
                arithmetic_docs.append(f"{j} plus {i} equals {result}.")

    # Subtraction facts
    print("Generating subtraction facts...")
    for i in range(0, 31):
        for j in range(0, i + 1):
            result = i - j
            arithmetic_docs.append(f"{i} minus {j} equals {result}.")

    # Multiplication tables (up to 12x12)
    print("Generating multiplication tables...")
    for i in range(0, 13):
        for j in range(0, 13):
            result = i * j
            arithmetic_docs.append(f"{i} times {j} equals {result}.")
            if i != j:  # Add commutative version
                arithmetic_docs.append(f"{j} times {i} equals {result}.")

    # Division facts
    print("Generating division facts...")
    for i in range(1, 13):
        for j in range(1, 13):
            dividend = i * j
            arithmetic_docs.append(f"{dividend} divided by {i} equals {j}.")
            if i != j:
                arithmetic_docs.append(f"{dividend} divided by {j} equals {i}.")

    # Square numbers
    print("Generating square numbers...")
    for i in range(0, 21):
        result = i * i
        arithmetic_docs.append(f"{i} squared equals {result}.")
        arithmetic_docs.append(f"The square of {i} is {result}.")

    # Cube numbers
    print("Generating cube numbers...")
    for i in range(0, 11):
        result = i ** 3
        arithmetic_docs.append(f"{i} cubed equals {result}.")
        arithmetic_docs.append(f"The cube of {i} is {result}.")

    # Powers of 2
    print("Generating powers of 2...")
    for i in range(0, 11):
        result = 2 ** i
        arithmetic_docs.append(f"2 to the power of {i} equals {result}.")

    # Powers of 10
    print("Generating powers of 10...")
    for i in range(0, 7):
        result = 10 ** i
        arithmetic_docs.append(f"10 to the power of {i} equals {result}.")

    # Simple fractions
    print("Generating fraction equivalents...")
    fractions = [
        ("one half", "0.5"),
        ("one third", "0.333"),
        ("one quarter", "0.25"),
        ("one fifth", "0.2"),
        ("one tenth", "0.1"),
        ("two thirds", "0.667"),
        ("three quarters", "0.75"),
        ("two fifths", "0.4"),
        ("three fifths", "0.6"),
        ("four fifths", "0.8"),
    ]
    for fraction, decimal in fractions:
        arithmetic_docs.append(f"{fraction} equals {decimal}.")
        arithmetic_docs.append(f"{decimal} equals {fraction}.")

    # Percentages
    print("Generating percentage conversions...")
    for i in range(0, 101, 5):
        decimal = i / 100
        arithmetic_docs.append(f"{i} percent equals {decimal}.")

    # Simple algebraic identities
    print("Generating algebraic patterns...")
    for i in range(1, 11):
        arithmetic_docs.append(f"{i} plus 0 equals {i}.")
        arithmetic_docs.append(f"{i} times 0 equals 0.")
        arithmetic_docs.append(f"{i} times 1 equals {i}.")
        arithmetic_docs.append(f"0 plus {i} equals {i}.")
        arithmetic_docs.append(f"1 times {i} equals {i}.")

    # Negative numbers
    print("Generating negative number facts...")
    for i in range(1, 11):
        arithmetic_docs.append(f"negative {i} plus {i} equals 0.")
        arithmetic_docs.append(f"{i} plus negative {i} equals 0.")
        arithmetic_docs.append(f"negative {i} times negative {i} equals {i * i}.")

    # Order of operations examples
    print("Generating order of operations examples...")
    examples = [
        ("2 plus 3 times 4", 14),
        ("10 minus 2 times 3", 4),
        ("20 divided by 4 plus 1", 6),
        ("3 times 4 plus 5", 17),
        ("5 plus 3 times 2", 11),
        ("16 divided by 4 times 2", 8),
        ("10 plus 5 minus 3", 12),
        ("20 minus 5 plus 2", 17),
    ]
    for expr, result in examples:
        arithmetic_docs.append(f"{expr} equals {result}.")

    # Common mathematical sequences
    print("Generating number sequences...")
    arithmetic_docs.append("The first 10 natural numbers are 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.")
    arithmetic_docs.append("The first 5 even numbers are 2, 4, 6, 8, 10.")
    arithmetic_docs.append("The first 5 odd numbers are 1, 3, 5, 7, 9.")
    arithmetic_docs.append("The first 10 prime numbers are 2, 3, 5, 7, 11, 13, 17, 19, 23, 29.")
    arithmetic_docs.append("The first 5 square numbers are 1, 4, 9, 16, 25.")
    arithmetic_docs.append("The first 5 cube numbers are 1, 8, 27, 64, 125.")
    arithmetic_docs.append("The first 10 Fibonacci numbers are 0, 1, 1, 2, 3, 5, 8, 13, 21, 34.")

    # Factorizations
    print("Generating factorizations...")
    for i in range(2, 31):
        factors = []
        for j in range(2, i):
            if i % j == 0:
                factors.append(str(j))
        if factors:
            arithmetic_docs.append(f"The factors of {i} are {', '.join(factors)}.")

    print()
    print(f"Generated {len(arithmetic_docs)} synthetic arithmetic statements!")
    print()
    print("Adding to dataset...")

    # Add all documents
    for i, doc in enumerate(arithmetic_docs, 1):
        repl.cmd_add([doc])
        if i % 100 == 0:
            print(f"  Added {i}/{len(arithmetic_docs)} documents...")

    print()
    print("✓ Arithmetic data added!")
    print()

    # Show final stats
    print("=" * 70)
    repl.execute('ds info')

    print()
    print("Saving dataset...")
    repl.execute('save')

    print()
    print("=" * 70)
    print("Testing arithmetic completions:")
    print("=" * 70)

    test_phrases = [
        ("5 plus 7", 20),
        ("12 times", 20),
        ("100 divided by", 25),
        ("The square of 8", 20),
        ("2 to the power", 25),
        ("negative 5 plus 5", 20),
        ("The first 10 prime", 30),
        ("one quarter equals", 20),
    ]

    for phrase, max_bytes in test_phrases:
        print(f"\n▶ Completing: '{phrase}'")
        print("-" * 70)
        repl.cmd_complete([phrase, '--max', str(max_bytes)])

    print()
    print("=" * 70)
    print(f"Math dataset now has extensive arithmetic knowledge!")
    print(f"Total documents: {155 + len(arithmetic_docs)}")
    print("=" * 70)
    print()

if __name__ == "__main__":
    generate_arithmetic_data()
