#!/usr/bin/env python3
"""
Demonstrate VFS commands on the math dataset.
"""

from infinigram.repl import InfinigramREPL

def demo_vfs():
    """Demo VFS commands on math dataset."""
    repl = InfinigramREPL()

    print("=" * 70)
    print("VFS DEMO: Exploring the Math Dataset")
    print("=" * 70)
    print()

    # Load the math dataset
    print("Loading math dataset...")
    repl.execute('load math')
    print()

    # List all documents
    print("=" * 70)
    print("1. Listing all documents with 'ls':")
    print("=" * 70)
    repl.execute('ls')
    print()

    # View specific documents
    print("=" * 70)
    print("2. Viewing specific documents with 'cat':")
    print("=" * 70)
    print("\nDocument [0]:")
    repl.execute('cat 0')
    print("\nDocument [10]:")
    repl.execute('cat 10')
    print("\nDocument [50]:")
    repl.execute('cat 50')
    print()

    # Test ds cat
    print("=" * 70)
    print("3. Viewing documents from another context with 'ds cat':")
    print("=" * 70)
    repl.execute('ds cat math')
    print()

    # Test proj cat
    print("=" * 70)
    print("4. Exploring projections with 'proj cat':")
    print("=" * 70)
    repl.execute('proj cat uppercase')
    print()

    # List available projections
    print("=" * 70)
    print("5. Listing all available projections:")
    print("=" * 70)
    repl.execute('proj ls -a')
    print()

    # Show dataset info
    print("=" * 70)
    print("6. Dataset information:")
    print("=" * 70)
    repl.execute('ds info')
    print()

    # Test some completions
    print("=" * 70)
    print("7. Testing completions on the math dataset:")
    print("=" * 70)
    print("\nCompletion 1:")
    repl.execute('complete "The area of a" --max 20')
    print("\nCompletion 2:")
    repl.execute('complete "5 plus 7" --max 15')
    print("\nCompletion 3:")
    repl.execute('complete "The first 10" --max 30')
    print()

    print("=" * 70)
    print("VFS DEMO COMPLETE!")
    print("=" * 70)
    print("\nYou can now use these commands in the interactive REPL:")
    print("  ls              - List all documents in the current dataset")
    print("  cat <index>     - View a specific document")
    print("  rm <index>      - Remove a document")
    print("  ds cat <name>   - View documents in any dataset")
    print("  proj cat <name> - View projection details")
    print()

if __name__ == "__main__":
    demo_vfs()
