#!/usr/bin/env python3
"""
Test the VFS (virtual filesystem) commands in Infinigram REPL.
"""

from infinigram.repl import InfinigramREPL

def test_vfs_commands():
    """Test all VFS commands."""
    repl = InfinigramREPL()

    print("=" * 70)
    print("TESTING VIRTUAL FILESYSTEM COMMANDS")
    print("=" * 70)
    print()

    # Create a test dataset
    print("1. Creating test dataset...")
    print("-" * 70)
    repl.execute('ds test_vfs')
    print()

    # Add some documents
    print("2. Adding test documents...")
    print("-" * 70)
    repl.execute('add "The quick brown fox jumps over the lazy dog."')
    repl.execute('add "Python is a powerful programming language."')
    repl.execute('add "Machine learning models require training data."')
    repl.execute('add "Short doc"')
    repl.execute('add "This is a very long document that will definitely be truncated when displayed in the list view because it exceeds the 60 character preview limit."')
    print()

    # Test ls command
    print("3. Testing 'ls' command (list documents)...")
    print("-" * 70)
    repl.execute('ls')
    print()

    # Test cat command
    print("4. Testing 'cat' command (view document)...")
    print("-" * 70)
    print("Viewing document [0]:")
    repl.execute('cat 0')
    print()
    print("Viewing document [4] (long document):")
    repl.execute('cat 4')
    print()

    # Test cat with invalid index
    print("5. Testing 'cat' with invalid index...")
    print("-" * 70)
    repl.execute('cat 999')
    print()

    # Test ds cat command
    print("6. Testing 'ds cat <dataset>' command...")
    print("-" * 70)
    repl.execute('ds cat test_vfs')
    print()

    # Create another dataset to test ds cat
    print("7. Creating another dataset for ds cat test...")
    print("-" * 70)
    repl.execute('ds another_test')
    repl.execute('add "Document in another dataset"')
    repl.execute('add "Second document here"')
    repl.execute('ds cat another_test')
    print()

    # Switch back to test_vfs
    repl.execute('ds test_vfs')
    print()

    # Test proj cat command
    print("8. Testing 'proj cat <projection>' command...")
    print("-" * 70)
    repl.execute('proj cat lowercase')
    print()

    # Test rm command
    print("9. Testing 'rm' command (remove document)...")
    print("-" * 70)
    print("Before removal:")
    repl.execute('ls')
    print()
    print("Removing document [1]:")
    repl.execute('rm 1')
    print()
    print("After removal:")
    repl.execute('ls')
    print()

    # Test rm with invalid index
    print("10. Testing 'rm' with invalid index...")
    print("-" * 70)
    repl.execute('rm 999')
    print()

    # Test completions after modification
    print("11. Testing completions after document removal...")
    print("-" * 70)
    repl.execute('complete "The quick" --max 20')
    print()

    # Test ls on empty dataset
    print("12. Testing 'ls' on empty dataset...")
    print("-" * 70)
    repl.execute('ds empty_test')
    repl.execute('ls')
    print()

    # Test ls without dataset selected
    print("13. Testing 'ls' without dataset (switch to none)...")
    print("-" * 70)
    # We can't actually deselect, so create a fresh REPL
    fresh_repl = InfinigramREPL()
    fresh_repl.execute('ls')
    print()

    print("=" * 70)
    print("VFS COMMANDS TEST COMPLETE!")
    print("=" * 70)
    print()

if __name__ == "__main__":
    test_vfs_commands()
