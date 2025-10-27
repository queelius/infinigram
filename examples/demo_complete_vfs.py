#!/usr/bin/env python3
"""
Complete demonstration of all VFS commands in Infinigram REPL.
Shows the power of treating datasets like a Unix filesystem.
"""

from infinigram.repl import InfinigramREPL

def demo_complete_vfs():
    """Demonstrate all VFS commands."""
    repl = InfinigramREPL()

    print("=" * 70)
    print("COMPLETE VFS DEMONSTRATION")
    print("Infinigram REPL - Treating Datasets Like a Unix Filesystem")
    print("=" * 70)
    print()

    # Load math dataset
    print("Loading the math dataset...")
    repl.execute('load math')
    print()

    # Basic navigation
    print("=" * 70)
    print("1. BASIC NAVIGATION")
    print("=" * 70)
    print("\nCommand: ls (list documents)")
    print("-" * 70)
    repl.execute('ls')
    print()

    print("Command: head 5 (show first 5 documents)")
    print("-" * 70)
    repl.execute('head 5')
    print()

    print("Command: tail 5 (show last 5 documents)")
    print("-" * 70)
    repl.execute('tail 5')
    print()

    # Viewing content
    print("=" * 70)
    print("2. VIEWING CONTENT")
    print("=" * 70)
    print("\nCommand: cat 10 (view document [10])")
    print("-" * 70)
    repl.execute('cat 10')
    print()

    # Searching
    print("=" * 70)
    print("3. SEARCHING")
    print("=" * 70)
    print("\nCommand: find derivative")
    print("-" * 70)
    repl.execute('find derivative')
    print()

    print("Command: grep Fibonacci (grep is alias for find)")
    print("-" * 70)
    repl.execute('grep Fibonacci')
    print()

    # Statistics
    print("=" * 70)
    print("4. STATISTICS & ANALYSIS")
    print("=" * 70)
    print("\nCommand: stat (dataset statistics)")
    print("-" * 70)
    repl.execute('stat')
    print()

    print("Command: stat 105 (document statistics)")
    print("-" * 70)
    repl.execute('stat 105')
    print()

    print("Command: wc (word count for dataset)")
    print("-" * 70)
    repl.execute('wc')
    print()

    print("Command: wc 0 (word count for document)")
    print("-" * 70)
    repl.execute('wc 0')
    print()

    print("Command: du (disk usage by document)")
    print("-" * 70)
    repl.execute('du')
    print()

    # Real workflow example
    print("=" * 70)
    print("5. REAL WORKFLOW EXAMPLE")
    print("=" * 70)
    print("\nScenario: Finding and analyzing documents about calculus")
    print("-" * 70)

    print("\nStep 1: Search for calculus-related documents")
    repl.execute('find integral')

    print("\nStep 2: Check stats for document [97]")
    repl.execute('stat 97')

    print("\nStep 3: View full content")
    repl.execute('cat 97')

    print("\nStep 4: Find related documents")
    repl.execute('find antiderivative')
    print()

    # Comparison with other datasets
    print("=" * 70)
    print("6. CROSS-DATASET OPERATIONS")
    print("=" * 70)
    print("\nCommand: ds ls (list all datasets)")
    print("-" * 70)
    repl.execute('ds ls')
    print()

    print("Command: ds cat math (view documents in math dataset)")
    print("-" * 70)
    print("(This works even from another dataset context)")
    repl.execute('ds cat math')
    print()

    # Projection exploration
    print("=" * 70)
    print("7. PROJECTION SYSTEM")
    print("=" * 70)
    print("\nCommand: proj ls -a (list available projections)")
    print("-" * 70)
    repl.execute('proj ls -a')
    print()

    print("Command: proj cat lowercase (view projection details)")
    print("-" * 70)
    repl.execute('proj cat lowercase')
    print()

    # Data manipulation
    print("=" * 70)
    print("8. DATA MANIPULATION")
    print("=" * 70)
    print("\nCreating a test dataset for manipulation...")
    repl.execute('ds test_vfs')
    repl.execute('add "First test document"')
    repl.execute('add "Second test document with more content"')
    repl.execute('add "Third document to be removed"')
    print()

    print("Command: ls (show all documents)")
    print("-" * 70)
    repl.execute('ls')
    print()

    print("Command: rm 2 (remove document [2])")
    print("-" * 70)
    repl.execute('rm 2')
    print()

    print("Command: ls (verify removal)")
    print("-" * 70)
    repl.execute('ls')
    print()

    # Summary
    print("=" * 70)
    print("VFS COMMANDS SUMMARY")
    print("=" * 70)
    print()
    print("Navigation:")
    print("  ls           - List all documents")
    print("  head/tail    - Preview first/last documents")
    print("  cat <index>  - View full document")
    print()
    print("Search:")
    print("  find/grep    - Search by text or regex")
    print()
    print("Analysis:")
    print("  stat         - Show statistics")
    print("  wc           - Count words/lines/bytes")
    print("  du           - Show disk usage")
    print()
    print("Manipulation:")
    print("  rm <index>   - Remove document")
    print()
    print("Cross-namespace:")
    print("  ds cat       - View other datasets")
    print("  proj cat     - View projection details")
    print()
    print("=" * 70)
    print("All VFS commands work together to provide a complete")
    print("Unix-like interface for exploring and managing datasets!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    demo_complete_vfs()
