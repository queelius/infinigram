#!/usr/bin/env python3
"""
Test the extended VFS commands: find, stat, head, tail, wc, du
"""

from infinigram.repl import InfinigramREPL

def test_extended_vfs():
    """Test all extended VFS commands."""
    repl = InfinigramREPL()

    print("=" * 70)
    print("TESTING EXTENDED VFS COMMANDS")
    print("=" * 70)
    print()

    # Load the math dataset
    print("Loading math dataset...")
    repl.execute('load math')
    print()

    # Test find command
    print("=" * 70)
    print("1. Testing 'find' command (text search):")
    print("=" * 70)
    print("\nSearching for 'derivative':")
    repl.execute('find derivative')
    print()

    print("Searching for 'circle':")
    repl.execute('find circle')
    print()

    print("Searching with regex pattern 'squared?':")
    repl.execute('find squared?')
    print()

    # Test grep alias
    print("=" * 70)
    print("2. Testing 'grep' (alias for find):")
    print("=" * 70)
    repl.execute('grep integral')
    print()

    # Test stat command (dataset-level)
    print("=" * 70)
    print("3. Testing 'stat' command (dataset-level):")
    print("=" * 70)
    repl.execute('stat')
    print()

    # Test stat command (document-level)
    print("=" * 70)
    print("4. Testing 'stat' command (document-level):")
    print("=" * 70)
    print("Stats for document [0]:")
    repl.execute('stat 0')
    print()
    print("Stats for document [50]:")
    repl.execute('stat 50')
    print()

    # Test head command
    print("=" * 70)
    print("5. Testing 'head' command:")
    print("=" * 70)
    print("First 5 documents:")
    repl.execute('head 5')
    print()

    # Test tail command
    print("=" * 70)
    print("6. Testing 'tail' command:")
    print("=" * 70)
    print("Last 5 documents:")
    repl.execute('tail 5')
    print()

    # Test wc command (dataset-level)
    print("=" * 70)
    print("7. Testing 'wc' command (dataset-level):")
    print("=" * 70)
    repl.execute('wc')
    print()

    # Test wc command (document-level)
    print("=" * 70)
    print("8. Testing 'wc' command (document-level):")
    print("=" * 70)
    print("Word count for document [0]:")
    repl.execute('wc 0')
    print("Word count for document [100]:")
    repl.execute('wc 100')
    print()

    # Test du command
    print("=" * 70)
    print("9. Testing 'du' command (disk usage):")
    print("=" * 70)
    repl.execute('du')
    print()

    # Test find with no matches
    print("=" * 70)
    print("10. Testing 'find' with no matches:")
    print("=" * 70)
    repl.execute('find xyzabc123nonexistent')
    print()

    # Test combining commands
    print("=" * 70)
    print("11. Workflow example - Find, then inspect:")
    print("=" * 70)
    print("Step 1: Find documents about 'Fibonacci'")
    repl.execute('find Fibonacci')
    print("\nStep 2: View stats for document [105]")
    repl.execute('stat 105')
    print("\nStep 3: View full content")
    repl.execute('cat 105')
    print()

    # Test on smaller dataset
    print("=" * 70)
    print("12. Testing on a small custom dataset:")
    print("=" * 70)
    repl.execute('ds tiny')
    repl.execute('add "First document is short."')
    repl.execute('add "Second document has more words in it."')
    repl.execute('add "Third document contains the word elephant three times: elephant elephant elephant."')
    print()

    print("Using stat on tiny dataset:")
    repl.execute('stat')
    print()

    print("Using du on tiny dataset:")
    repl.execute('du')
    print()

    print("Finding 'elephant':")
    repl.execute('find elephant')
    print()

    print("=" * 70)
    print("EXTENDED VFS COMMANDS TEST COMPLETE!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    test_extended_vfs()
