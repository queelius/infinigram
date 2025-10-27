#!/usr/bin/env python3
"""
Expand the math dataset with additional documents.
"""

from infinigram.repl import InfinigramREPL

def expand_math_dataset():
    """Add more math documents to the existing dataset."""
    repl = InfinigramREPL()

    print("Loading existing math dataset...")
    repl.execute('load math')
    print()

    # Additional comprehensive math documents
    additional_docs = [
        # More arithmetic
        "Zero is the additive identity. Any number plus zero equals that number.",
        "One is the multiplicative identity. Any number times one equals that number.",
        "Negative numbers are less than zero and lie to the left of zero on the number line.",
        "Positive numbers are greater than zero and lie to the right of zero.",
        "The absolute value of a number is its distance from zero, always non-negative.",
        "Exponentiation is repeated multiplication. 2 to the power of 5 equals 32.",
        "A square number is the product of an integer with itself. 16 is a square number because 4 times 4 equals 16.",
        "A cube number is an integer raised to the power of 3. 27 is a cube because 3 cubed equals 27.",
        "The order of operations is parentheses, exponents, multiplication and division, addition and subtraction.",
        "Fractions represent parts of a whole. One half plus one quarter equals three quarters.",

        # More algebra
        "An equation states that two expressions are equal.",
        "An inequality shows that one expression is greater than or less than another.",
        "A polynomial is a sum of terms with variables raised to non-negative integer powers.",
        "The degree of a polynomial is the highest power of the variable.",
        "A monomial has one term, a binomial has two terms, and a trinomial has three terms.",
        "To expand brackets, multiply each term inside by the term outside.",
        "Like terms have the same variables raised to the same powers and can be combined.",
        "The distributive property states that a times the quantity b plus c equals ab plus ac.",
        "The commutative property means a plus b equals b plus a.",
        "The associative property means a plus the quantity b plus c equals the quantity a plus b plus c.",
        "An exponential function has the form y equals a times b to the power of x.",
        "A logarithm is the inverse of exponentiation. Log base 10 of 100 equals 2.",
        "Natural logarithms use base e. The natural log of e equals 1.",

        # More geometry
        "A polygon is a closed figure with straight sides.",
        "A triangle has 3 sides, a quadrilateral has 4 sides, a pentagon has 5 sides.",
        "An equilateral triangle has all sides equal and all angles equal to 60 degrees.",
        "An isosceles triangle has two equal sides and two equal angles.",
        "A scalene triangle has no equal sides and no equal angles.",
        "The area of a triangle is one half times base times height.",
        "A square has 4 equal sides and 4 right angles.",
        "A rectangle has opposite sides equal and 4 right angles.",
        "The perimeter is the distance around a shape.",
        "The volume of a cube is side length cubed.",
        "The volume of a rectangular prism is length times width times height.",
        "The surface area of a sphere is 4 times pi times radius squared.",
        "The volume of a sphere is four thirds times pi times radius cubed.",
        "A cylinder has two circular bases and a curved surface.",
        "Congruent shapes have the same size and shape.",
        "Similar shapes have the same shape but different sizes.",

        # More calculus
        "The limit describes the behavior of a function as the input approaches a value.",
        "A function is continuous if you can draw it without lifting your pen.",
        "The derivative of a constant is zero.",
        "The derivative of x to the power of n is n times x to the power of n minus 1.",
        "The product rule states that the derivative of uv is u times dv plus v times du.",
        "The quotient rule finds the derivative of one function divided by another.",
        "The chain rule is used to differentiate composite functions.",
        "A critical point occurs where the derivative equals zero or is undefined.",
        "A local maximum is a point higher than nearby points.",
        "A local minimum is a point lower than nearby points.",
        "The second derivative tells us about concavity and inflection points.",
        "An indefinite integral represents a family of antiderivatives.",
        "A definite integral calculates the area under a curve between two points.",
        "The power rule for integration states that the integral of x to the n is x to the n plus 1 divided by n plus 1.",

        # More number theory
        "Composite numbers have factors other than 1 and themselves.",
        "The number 1 is neither prime nor composite.",
        "Every integer greater than 1 is either prime or can be factored into primes.",
        "The least common multiple of two numbers is the smallest number divisible by both.",
        "Perfect numbers equal the sum of their proper divisors. 6 is perfect because 1 plus 2 plus 3 equals 6.",
        "Fibonacci numbers form a sequence where each number is the sum of the two preceding ones.",
        "The Fibonacci sequence starts 0, 1, 1, 2, 3, 5, 8, 13, 21.",
        "Rational numbers can be expressed as a fraction of two integers.",
        "Irrational numbers cannot be expressed as fractions. Pi and the square root of 2 are irrational.",
        "Real numbers include both rational and irrational numbers.",

        # More statistics and probability
        "The range is the difference between the maximum and minimum values.",
        "Variance measures the average squared deviation from the mean.",
        "A percentile indicates the percentage of data below a given value.",
        "The 50th percentile is the same as the median.",
        "Correlation measures the strength of the relationship between two variables.",
        "A positive correlation means variables increase together.",
        "A negative correlation means one variable increases as the other decreases.",
        "Independent events have no effect on each other.",
        "Dependent events influence each other.",
        "The probability of two independent events both occurring is the product of their probabilities.",
        "The probability of either of two mutually exclusive events is the sum of their probabilities.",
        "A random variable assigns numerical values to outcomes of a random process.",
        "Expected value is the average outcome weighted by probabilities.",

        # More trigonometry
        "Radians are another way to measure angles. Pi radians equals 180 degrees.",
        "Sine of 0 degrees equals 0. Sine of 30 degrees equals one half.",
        "Cosine of 0 degrees equals 1. Cosine of 90 degrees equals 0.",
        "The Pythagorean identity states that sine squared plus cosine squared equals 1.",
        "Tangent equals sine divided by cosine.",
        "The amplitude of a sine or cosine function is the height of the wave.",
        "The period is the length of one complete cycle of a periodic function.",
        "Inverse trigonometric functions undo the trigonometric functions.",

        # Sequences and series
        "An arithmetic sequence has a constant difference between consecutive terms.",
        "A geometric sequence has a constant ratio between consecutive terms.",
        "The sum of an arithmetic sequence can be found using the formula n times the quantity first term plus last term divided by 2.",
        "An infinite geometric series converges if the common ratio is between negative 1 and 1.",
        "The sum of a convergent geometric series is the first term divided by 1 minus the common ratio.",

        # Combinatorics
        "Factorials count permutations. 5 factorial equals 5 times 4 times 3 times 2 times 1 which equals 120.",
        "A permutation is an arrangement where order matters.",
        "A combination is a selection where order does not matter.",
        "The binomial coefficient n choose k counts the ways to choose k items from n items.",
        "Pascal's triangle displays binomial coefficients in a triangular array.",

        # Mathematical reasoning
        "A proof demonstrates that a statement is logically true.",
        "A theorem is a statement that has been proven.",
        "A lemma is a preliminary result used to prove a theorem.",
        "A corollary is a result that follows easily from a theorem.",
        "Proof by induction involves proving a base case and an inductive step.",
        "Proof by contradiction assumes the opposite and derives a contradiction.",
        "A counterexample shows that a statement is false.",

        # Linear algebra
        "A vector has both magnitude and direction.",
        "Vectors can be added by adding corresponding components.",
        "Scalar multiplication multiplies each component by a constant.",
        "The dot product of two vectors produces a scalar.",
        "Matrices can be multiplied if the number of columns in the first equals the number of rows in the second.",
        "The determinant of a 2 by 2 matrix ad minus bc determines if the matrix is invertible.",
        "An identity matrix has ones on the diagonal and zeros elsewhere.",
    ]

    print(f"Adding {len(additional_docs)} more mathematical documents...")
    print()

    for i, doc in enumerate(additional_docs, 1):
        repl.cmd_add([doc])
        if i % 20 == 0:
            print(f"  Added {i}/{len(additional_docs)} documents...")

    print()
    print("✓ Dataset expansion complete!")
    print()

    # Show updated info
    print("=" * 60)
    repl.execute('ds info')

    print()
    print("Saving expanded dataset...")
    repl.execute('save')

    print()
    print("=" * 60)
    print("Testing new completions:")
    print("=" * 60)

    test_phrases = [
        ("The Fibonacci sequence", 25),
        ("A proof", 30),
        ("The volume of a sphere", 20),
        ("Factorial", 25),
        ("The limit", 30),
        ("Correlation measures", 25),
        ("An arithmetic sequence", 30),
        ("The binomial", 25),
    ]

    for phrase, max_bytes in test_phrases:
        print(f"\n▶ Completing: '{phrase}'")
        print("-" * 60)
        repl.cmd_complete([phrase, '--max', str(max_bytes)])

    print()
    print("=" * 60)
    print(f"Math dataset now has {len(additional_docs) + 47} documents!")
    print("=" * 60)
    print()

if __name__ == "__main__":
    expand_math_dataset()
