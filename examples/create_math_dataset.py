#!/usr/bin/env python3
"""
Create a comprehensive math dataset for Infinigram.
"""

from infinigram.repl import InfinigramREPL

def create_math_dataset():
    """Create and populate a math dataset."""
    repl = InfinigramREPL()

    print("Creating math dataset...")
    print()

    # Create dataset
    repl.execute('ds math')

    # Math documents covering various topics
    math_docs = [
        # Basic arithmetic
        "Addition is the process of combining two or more numbers. For example, 2 plus 3 equals 5.",
        "Subtraction is the inverse of addition. When you subtract 5 from 10, you get 5.",
        "Multiplication is repeated addition. 3 times 4 equals 12.",
        "Division splits a number into equal parts. 12 divided by 3 equals 4.",

        # Algebra
        "In algebra, we use variables like x and y to represent unknown values.",
        "A linear equation has the form y equals mx plus b.",
        "To solve for x in 2x plus 5 equals 13, subtract 5 then divide by 2 to get x equals 4.",
        "The quadratic formula solves equations of the form ax squared plus bx plus c equals 0.",
        "Factoring breaks down expressions into simpler components.",

        # Geometry
        "The area of a rectangle equals length times width.",
        "The circumference of a circle is 2 times pi times the radius.",
        "The area of a circle is pi times the radius squared.",
        "The Pythagorean theorem states that a squared plus b squared equals c squared.",
        "The sum of angles in a triangle always equals 180 degrees.",
        "Parallel lines never intersect and maintain the same distance apart.",

        # Calculus
        "The derivative measures the rate of change of a function.",
        "The derivative of x squared is 2x.",
        "The derivative of sine x is cosine x.",
        "Integration is the reverse process of differentiation.",
        "The integral of 2x is x squared plus a constant.",

        # Number theory
        "A prime number is divisible only by 1 and itself. Examples are 2, 3, 5, 7, 11, 13.",
        "An even number is divisible by 2 and ends in 0, 2, 4, 6, or 8.",
        "An odd number is not divisible by 2. Examples include 1, 3, 5, 7, 9.",
        "The greatest common divisor of 12 and 18 is 6.",
        "Two numbers are coprime if their greatest common divisor is 1.",

        # Statistics
        "The mean is the average of numbers. Add all values and divide by the count.",
        "The median is the middle value when numbers are arranged in order.",
        "The mode is the most frequently occurring value in a dataset.",
        "Standard deviation measures how spread out numbers are from the mean.",
        "Probability is a number between 0 and 1, where 0 means impossible and 1 means certain.",

        # Trigonometry
        "Sine, cosine, and tangent are the three basic trigonometric functions.",
        "In a right triangle, sine equals opposite over hypotenuse.",
        "Cosine equals adjacent over hypotenuse.",
        "Tangent equals opposite over adjacent.",
        "The unit circle has radius 1 and is centered at the origin.",

        # Sets
        "A set is a collection of distinct objects.",
        "The union of two sets contains all elements from both sets.",
        "The intersection contains only elements that appear in both sets.",
        "An empty set contains no elements.",

        # Constants
        "Pi is approximately 3.14159 and represents the ratio of circumference to diameter.",
        "Euler's number e is approximately 2.71828 and is the base of natural logarithms.",
        "The golden ratio phi is approximately 1.618.",

        # Functions
        "A matrix is a rectangular array of numbers in rows and columns.",
        "Complex numbers have a real part and an imaginary part.",
        "A function maps each input to exactly one output.",
        "The domain is the set of all possible input values.",
        "The range is the set of all possible output values.",
    ]

    print(f"Adding {len(math_docs)} mathematical documents...")
    for i, doc in enumerate(math_docs, 1):
        # Use programmatic API instead of execute to avoid quote issues
        repl.cmd_add([doc])
        if i % 10 == 0:
            print(f"  Added {i}/{len(math_docs)} documents...")

    print()
    print("Dataset creation complete!")
    print()

    # Show info
    repl.execute('ds info')

    print()
    print("Saving to disk...")
    repl.execute('save')

    print()
    print("=" * 60)
    print("Testing predictions on the math dataset:")
    print("=" * 60)

    test_phrases = [
        ("The area of a circle", 20),
        ("The derivative of", 15),
        ("A prime number", 20),
        ("In algebra", 25),
        ("Pi is approximately", 15),
        ("The mean is", 20),
    ]

    for phrase, max_bytes in test_phrases:
        print(f"\n▶ Completing: '{phrase}'")
        print("-" * 60)
        repl.cmd_complete([phrase, '--max', str(max_bytes)])

    print()
    print("=" * 60)
    print("Math dataset ready!")
    print("=" * 60)
    print()
    print("To use this dataset later:")
    print("  1. Start the REPL: infinigram-repl")
    print("  2. Load the dataset: load math")
    print("  3. Try completions: complete The area of")
    print()

if __name__ == "__main__":
    create_math_dataset()
