#!/usr/bin/env python3
"""
Test that the REPL entry point works correctly.
"""

import subprocess
import sys


def test_repl_launches():
    """Test that infinigram-repl command exists and launches."""
    # Test 1: Check command exists
    result = subprocess.run(
        ["which", "infinigram-repl"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ infinigram-repl command not found")
        sys.exit(1)

    print(f"✓ infinigram-repl found at: {result.stdout.strip()}")

    # Test 2: Launch REPL with /quit command
    result = subprocess.run(
        ["infinigram-repl"],
        input="/quit\n",
        capture_output=True,
        text=True,
        timeout=5
    )

    # Check for expected banner
    if "INFINIGRAM INTERACTIVE REPL" in result.stdout:
        print("✓ REPL banner displayed")
    else:
        print("❌ REPL banner not found")
        print(f"Output: {result.stdout}")
        sys.exit(1)

    # Test 3: Python module entry point
    result = subprocess.run(
        [sys.executable, "-m", "infinigram.repl"],
        input="/quit\n",
        capture_output=True,
        text=True,
        timeout=5
    )

    if "INFINIGRAM INTERACTIVE REPL" in result.stdout:
        print("✓ Python module entry point works")
    else:
        print("❌ Python module entry point failed")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ All entry point tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_repl_launches()
