#!/usr/bin/env python3
"""
Interactive REPL for Infinigram.

Provides an interactive shell for exploring byte-level language models,
testing predictions, and configuring model parameters.
"""

import sys
import shlex
import json
import subprocess
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from infinigram import Infinigram, IdentityAdapter
from infinigram.corpus_utils import (
    build_corpus_from_documents,
    build_corpus_with_augmentation,
    text_to_bytes,
    bytes_to_text
)
from infinigram.weighting import (
    linear_weight, quadratic_weight, exponential_weight, sigmoid_weight, get_weight_function
)
from infinigram.vfs import VirtualFilesystem


# Registry of available projections/augmentations
PROJECTION_REGISTRY = {
    'lowercase': lambda text: text.lower() if isinstance(text, str) else bytes(text).decode('utf-8', errors='replace').lower(),
    'uppercase': lambda text: text.upper() if isinstance(text, str) else bytes(text).decode('utf-8', errors='replace').upper(),
    'title': lambda text: text.title() if isinstance(text, str) else bytes(text).decode('utf-8', errors='replace').title(),
    'strip': lambda text: text.strip() if isinstance(text, str) else bytes(text).decode('utf-8', errors='replace').strip(),
}


class InfinigramREPL:
    """Interactive REPL for Infinigram."""

    def __init__(self):
        """Initialize REPL."""
        # Dataset/Model management
        self.datasets: Dict[str, Infinigram] = {}  # name -> model
        self.current_dataset: Optional[str] = None
        self.adapter = IdentityAdapter()

        # Document tracking per dataset (for proper separators)
        self.dataset_documents: Dict[str, List[str]] = {}  # dataset_name -> list of document strings

        # Projection/augmentation tracking per dataset
        self.dataset_projections: Dict[str, List[str]] = {}  # dataset_name -> list of projection names

        # Per-dataset configuration (max_length, min_count)
        self.dataset_config: Dict[str, Dict[str, Any]] = {}  # dataset_name -> config dict

        # Model configuration (REPL-level defaults)
        self.temperature = 1.0
        self.top_k = 50
        self.max_length = None
        self.smoothing = 0.0
        self.weight_function = None
        self.min_length = 1
        self.max_weight_length = None

        # Storage directory
        self.storage_dir = Path.home() / ".infinigram" / "datasets"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Virtual filesystem
        self.vfs = VirtualFilesystem(self.storage_dir)

        # Command history
        if PROMPT_TOOLKIT_AVAILABLE:
            self.history = InMemoryHistory()
        else:
            self.history: List[str] = []

        # Commands mapping (new clean structure)
        self.commands = {
            # System
            'help': self.cmd_help,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,

            # VFS - Navigation
            'pwd': self.cmd_pwd,
            'cd': self.cmd_cd,

            # VFS - Current dataset operations
            'ls': self.cmd_ls,
            'cat': self.cmd_cat,
            'rm': self.cmd_rm,
            'find': self.cmd_find,
            'grep': self.cmd_find,  # Alias for find
            'stat': self.cmd_stat,
            'head': self.cmd_head,
            'tail': self.cmd_tail,
            'wc': self.cmd_wc,
            'du': self.cmd_du,

            # Dataset namespace (ds)
            'ds': self.cmd_ds,  # With no args, shows current dataset
            'ds_ls': self.cmd_ds_ls,
            'ds_cat': self.cmd_ds_cat,
            'ds_cp': self.cmd_ds_cp,
            'ds_rm': self.cmd_ds_rm,
            'ds_info': self.cmd_ds_info,
            'ds_stats': self.cmd_ds_stats,

            # Storage namespace (store)
            'save': self.cmd_save,  # Common operation, keep short
            'load': self.cmd_load,  # Common operation, keep short
            'store_ls': self.cmd_store_ls,
            'store_rm': self.cmd_store_rm,

            # Content
            'add': self.cmd_add,

            # Projection namespace (proj)
            'proj': self.cmd_proj,  # Set projections
            'proj_ls': self.cmd_proj_ls,
            'proj_cat': self.cmd_proj_cat,
            'proj_rm': self.cmd_proj_rm,

            # Inference
            'predict': self.cmd_predict,
            'complete': self.cmd_complete,

            # Configuration namespace (set)
            'set_temperature': self.cmd_set_temperature,
            'set_top_k': self.cmd_set_top_k,
            'set_max_length': self.cmd_set_max_length,
            'set_smoothing': self.cmd_set_smoothing,
            'set_weight': self.cmd_set_weight,
            'config': self.cmd_config,
        }

    @property
    def model(self) -> Optional[Infinigram]:
        """Get current model."""
        if self.current_dataset:
            return self.datasets.get(self.current_dataset)
        return None

    def run(self):
        """Run the REPL."""
        print("=" * 70)
        print("  INFINIGRAM INTERACTIVE REPL")
        print("=" * 70)
        print()
        print("Type 'help' for available commands or 'quit' to exit.")
        print()

        while True:
            try:
                # Get input
                # Show VFS path in prompt
                prompt_str = f"infinigram:{self.vfs.cwd}> "

                if PROMPT_TOOLKIT_AVAILABLE:
                    user_input = prompt(
                        prompt_str,
                        history=self.history,
                        auto_suggest=AutoSuggestFromHistory()
                    ).strip()
                else:
                    user_input = input(prompt_str).strip()

                if not user_input:
                    continue

                # Add to history (if not using prompt_toolkit, which does it automatically)
                if not PROMPT_TOOLKIT_AVAILABLE:
                    self.history.append(user_input)

                # Parse and execute command
                self.execute(user_input)
                print()

            except KeyboardInterrupt:
                print("\n(Use 'quit' to exit)")
                print()
            except EOFError:
                print("\nGoodbye!")
                break

    def execute(self, user_input: str):
        """Execute a command or text input."""
        if user_input.startswith('!'):
            # Bash command
            self.execute_bash(user_input[1:])
            return

        # Parse command (no / prefix needed)
        parts = shlex.split(user_input)
        if not parts:
            return

        cmd_name = parts[0]
        args = parts[1:]

        # Handle namespaced commands (e.g., "ds ls", "store save")
        if len(parts) >= 2 and cmd_name in ['ds', 'store', 'proj', 'set']:
            # Check if second part is a known subcommand
            namespace = cmd_name
            potential_subcmd = parts[1]
            full_cmd = f"{namespace}_{potential_subcmd}"

            if full_cmd in self.commands:
                # It's a subcommand, execute it
                args = parts[2:]
                self.commands[full_cmd](args)
            elif cmd_name in self.commands:
                # Not a subcommand, treat as arguments to base command
                args = parts[1:]
                self.commands[cmd_name](args)
            else:
                print(f"Unknown command: {cmd_name}")
                print("Type 'help' for available commands")
        elif cmd_name in self.commands:
            # Simple command
            self.commands[cmd_name](args)
        else:
            print(f"Unknown command: {cmd_name}")
            print("Type 'help' for available commands")

    def execute_bash(self, command: str):
        """Execute a bash command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.stdout:
                print(result.stdout, end='')
            if result.stderr:
                print(result.stderr, end='', file=sys.stderr)
            if result.returncode != 0:
                print(f"(Command exited with code {result.returncode})")
        except subprocess.TimeoutExpired:
            print("Error: Command timed out (30s limit)")
        except Exception as e:
            print(f"Error executing command: {e}")

    # ========================================================================
    # COMMAND IMPLEMENTATIONS
    # ========================================================================

    def cmd_help(self, args: List[str]):
        """Show help information."""
        print("Infinigram REPL - Variable-length n-gram language models")
        print()
        print("VFS Navigation:")
        print("  pwd                       Print working directory")
        print("  cd <path>                 Change directory")
        print("  cd /                      Go to root")
        print("  cd ~                      Go to root (home)")
        print("  cd -                      Go to previous directory")
        print("  cd ..                     Go to parent directory")
        print()
        print("Virtual Filesystem (Current Dataset):")
        print("  ls                        List documents in current dataset")
        print("  cat <index>               View document by index")
        print("  rm <index>                Remove document by index")
        print("  find <pattern>            Search documents by text/regex")
        print("  grep <pattern>            Search documents (alias for find)")
        print("  stat [index]              Show statistics for document or dataset")
        print("  head [n]                  Show first n documents (default: 10)")
        print("  tail [n]                  Show last n documents (default: 10)")
        print("  wc [index]                Count words/lines/bytes")
        print("  du                        Show disk usage per document")
        print()
        print("Dataset Operations (ds):")
        print("  ds <name>                 Create or switch to dataset")
        print("  ds ls                     List datasets in memory")
        print("  ds cat <dataset>          View documents in a dataset")
        print("  ds cp <src> <dst>         Copy dataset")
        print("  ds rm                     Delete current dataset")
        print("  ds info                   Show dataset information")
        print("  ds stats                  Show corpus statistics")
        print()
        print("Storage Operations (store):")
        print("  save [name]               Save dataset to disk")
        print("  load <name>               Load dataset from disk")
        print("  store ls                  List saved datasets")
        print("  store rm <name>           Delete saved dataset")
        print()
        print("Content:")
        print("  add <text>                Add text to current dataset")
        print("  add --file <path>         Add file contents")
        print("  add --jsonl <path>        Add JSONL file")
        print()
        print("Projections (proj):")
        print("  proj <p1> [p2...]         Apply projections (lowercase, uppercase, etc.)")
        print("  proj ls                   List active projections")
        print("  proj ls -a                List available projections")
        print("  proj cat <projection>     View projection details")
        print("  proj rm                   Remove all projections")
        print()
        print("Inference:")
        print("  predict <text>            Show next-byte probabilities")
        print("  predict <text> --bytes    Show as raw bytes")
        print("  complete <text>           Generate completion")
        print("  complete <text> --max N   Generate N bytes")
        print()
        print("Configuration (set):")
        print("  set temperature <val>     Set sampling temperature (default: 1.0)")
        print("  set top_k <n>             Set top-k predictions (default: 50)")
        print("  set max_length <n>        Set max suffix length (default: unlimited)")
        print("  set smoothing <val>       Set smoothing parameter (default: 0.0)")
        print("  set weight <func>         Set weight function (linear, quadratic, ...)")
        print("  config                    Show all configuration")
        print()
        print("System:")
        print("  help                      Show this help")
        print("  quit, exit                Exit REPL")
        print("  !<command>                Execute bash command")
        print()

    # ========================================================================
    # VFS NAVIGATION COMMANDS
    # ========================================================================

    def cmd_pwd(self, args: List[str]):
        """Print current working directory."""
        print(self.vfs.cwd)

    def cmd_cd(self, args: List[str]):
        """Change directory."""
        if not args:
            # cd with no args goes to root
            path = '/'
        else:
            path = args[0]

        try:
            new_dir = self.vfs.change_directory(path)
            print(new_dir)
        except (ValueError, FileNotFoundError) as e:
            print(f"cd: {e}")

    # ========================================================================
    # VFS COMMANDS (Current Dataset Operations)
    # ========================================================================

    def cmd_ls(self, args: List[str]):
        """List documents in the current dataset."""
        if not self.current_dataset:
            print("No dataset selected")
            print("Usage: ds <name> to select a dataset")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])
        if not docs:
            print(f"Dataset '{self.current_dataset}' is empty")
            return

        print(f"Documents in '{self.current_dataset}':")
        for i, doc in enumerate(docs):
            # Truncate long documents for display
            preview = doc[:60] + "..." if len(doc) > 60 else doc
            # Replace newlines with \n for single-line display
            preview = preview.replace('\n', '\\n')
            print(f"  [{i}] {preview}")

        print(f"\nTotal: {len(docs)} documents")

    def cmd_cat(self, args: List[str]):
        """View a document by index."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        if not args:
            print("Usage: cat <index>")
            return

        try:
            index = int(args[0])
        except ValueError:
            print(f"Error: Invalid index '{args[0]}'")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])
        if index < 0 or index >= len(docs):
            print(f"Error: Index {index} out of range (0-{len(docs)-1})")
            return

        print(f"Document [{index}] in '{self.current_dataset}':")
        print("-" * 70)
        print(docs[index])
        print("-" * 70)

    def cmd_rm(self, args: List[str]):
        """Remove a document by index."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        if not args:
            print("Usage: rm <index>")
            return

        try:
            index = int(args[0])
        except ValueError:
            print(f"Error: Invalid index '{args[0]}'")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])
        if index < 0 or index >= len(docs):
            print(f"Error: Index {index} out of range (0-{len(docs)-1})")
            return

        # Show what we're removing
        removed_doc = docs[index]
        preview = removed_doc[:60] + "..." if len(removed_doc) > 60 else removed_doc
        preview = preview.replace('\n', '\\n')

        # Remove from document list
        docs.pop(index)

        # Rebuild corpus
        corpus = build_corpus_from_documents(docs, separator=b"\x00")

        # Rebuild model with projections if any
        projections = self.dataset_projections.get(self.current_dataset, [])
        if projections:
            corpus = build_corpus_with_augmentation(
                docs,
                projections,
                separator=b"\x00"
            )
        else:
            corpus = build_corpus_from_documents(docs, separator=b"\x00")

        # Recreate model
        self.datasets[self.current_dataset] = Infinigram(
            corpus,
            max_length=self.max_length
        )

        print(f"✓ Removed document [{index}]: {preview}")
        print(f"  Remaining documents: {len(docs)}")
        print(f"  Corpus size: {len(corpus)} bytes")

    def cmd_find(self, args: List[str]):
        """Search for documents containing a pattern (text or regex)."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        if not args:
            print("Usage: find <pattern>")
            print("       grep <pattern>")
            return

        import re
        pattern = ' '.join(args)

        docs = self.dataset_documents.get(self.current_dataset, [])
        if not docs:
            print(f"Dataset '{self.current_dataset}' is empty")
            return

        # Try to compile as regex, fall back to plain text search
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            use_regex = True
        except re.error:
            use_regex = False

        matches = []
        for i, doc in enumerate(docs):
            if use_regex:
                if regex.search(doc):
                    matches.append(i)
            else:
                if pattern.lower() in doc.lower():
                    matches.append(i)

        if not matches:
            print(f"No documents found matching: {pattern}")
            return

        print(f"Found {len(matches)} document(s) matching '{pattern}':")
        for i in matches:
            preview = docs[i][:60] + "..." if len(docs[i]) > 60 else docs[i]
            preview = preview.replace('\n', '\\n')
            print(f"  [{i}] {preview}")

    def cmd_stat(self, args: List[str]):
        """Show statistics for a document or the entire dataset."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])

        if not args:
            # Dataset-level stats
            if not docs:
                print(f"Dataset '{self.current_dataset}' is empty")
                return

            total_chars = sum(len(doc) for doc in docs)
            total_words = sum(len(doc.split()) for doc in docs)
            total_lines = sum(doc.count('\n') + 1 for doc in docs)

            avg_chars = total_chars / len(docs) if docs else 0
            avg_words = total_words / len(docs) if docs else 0

            min_len = min(len(doc) for doc in docs) if docs else 0
            max_len = max(len(doc) for doc in docs) if docs else 0

            print(f"Statistics for dataset '{self.current_dataset}':")
            print(f"  Documents: {len(docs)}")
            print(f"  Total characters: {total_chars}")
            print(f"  Total words: {total_words}")
            print(f"  Total lines: {total_lines}")
            print(f"  Average chars/doc: {avg_chars:.1f}")
            print(f"  Average words/doc: {avg_words:.1f}")
            print(f"  Shortest document: {min_len} chars")
            print(f"  Longest document: {max_len} chars")

            if self.model:
                print(f"  Corpus size: {self.model.n} bytes")
                print(f"  Vocabulary size: {len(set(self.model.corpus))}")
        else:
            # Document-level stats
            try:
                index = int(args[0])
            except ValueError:
                print(f"Error: Invalid index '{args[0]}'")
                return

            if index < 0 or index >= len(docs):
                print(f"Error: Index {index} out of range (0-{len(docs)-1})")
                return

            doc = docs[index]
            chars = len(doc)
            words = len(doc.split())
            lines = doc.count('\n') + 1

            print(f"Statistics for document [{index}]:")
            print(f"  Characters: {chars}")
            print(f"  Words: {words}")
            print(f"  Lines: {lines}")
            print(f"  Average word length: {chars/words:.1f}" if words > 0 else "  No words")
            print(f"\nPreview:")
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            print(f"  {preview}")

    def cmd_head(self, args: List[str]):
        """Show first N documents (default 10)."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])
        if not docs:
            print(f"Dataset '{self.current_dataset}' is empty")
            return

        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                print(f"Error: Invalid number '{args[0]}'")
                return

        n = min(n, len(docs))

        print(f"First {n} documents in '{self.current_dataset}':")
        for i in range(n):
            preview = docs[i][:60] + "..." if len(docs[i]) > 60 else docs[i]
            preview = preview.replace('\n', '\\n')
            print(f"  [{i}] {preview}")

    def cmd_tail(self, args: List[str]):
        """Show last N documents (default 10)."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])
        if not docs:
            print(f"Dataset '{self.current_dataset}' is empty")
            return

        n = 10
        if args:
            try:
                n = int(args[0])
            except ValueError:
                print(f"Error: Invalid number '{args[0]}'")
                return

        n = min(n, len(docs))
        start_idx = len(docs) - n

        print(f"Last {n} documents in '{self.current_dataset}':")
        for i in range(start_idx, len(docs)):
            preview = docs[i][:60] + "..." if len(docs[i]) > 60 else docs[i]
            preview = preview.replace('\n', '\\n')
            print(f"  [{i}] {preview}")

    def cmd_wc(self, args: List[str]):
        """Count words, lines, and bytes (like Unix wc)."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])

        if not args:
            # Count for entire dataset
            if not docs:
                print(f"Dataset '{self.current_dataset}' is empty")
                return

            total_lines = sum(doc.count('\n') + 1 for doc in docs)
            total_words = sum(len(doc.split()) for doc in docs)
            total_bytes = sum(len(doc.encode('utf-8')) for doc in docs)

            print(f"  {total_lines:8} {total_words:8} {total_bytes:8} {self.current_dataset}")
        else:
            # Count for specific document
            try:
                index = int(args[0])
            except ValueError:
                print(f"Error: Invalid index '{args[0]}'")
                return

            if index < 0 or index >= len(docs):
                print(f"Error: Index {index} out of range (0-{len(docs)-1})")
                return

            doc = docs[index]
            lines = doc.count('\n') + 1
            words = len(doc.split())
            bytes_count = len(doc.encode('utf-8'))

            print(f"  {lines:8} {words:8} {bytes_count:8} document[{index}]")

    def cmd_du(self, args: List[str]):
        """Show disk usage (size) per document."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        docs = self.dataset_documents.get(self.current_dataset, [])
        if not docs:
            print(f"Dataset '{self.current_dataset}' is empty")
            return

        print(f"Disk usage for '{self.current_dataset}':")

        total_bytes = 0
        doc_sizes = []

        for i, doc in enumerate(docs):
            size = len(doc.encode('utf-8'))
            total_bytes += size
            doc_sizes.append((i, size, doc))

        # Sort by size (descending)
        doc_sizes.sort(key=lambda x: x[1], reverse=True)

        # Show top 20 or all if less
        show_count = min(20, len(doc_sizes))

        for i, size, doc in doc_sizes[:show_count]:
            preview = doc[:40] + "..." if len(doc) > 40 else doc
            preview = preview.replace('\n', '\\n')
            print(f"  {size:6} bytes  [{i:3}] {preview}")

        if len(docs) > show_count:
            print(f"  ... ({len(docs) - show_count} more documents)")

        print(f"\nTotal: {total_bytes} bytes across {len(docs)} documents")

    # ========================================================================
    # DATASET NAMESPACE COMMANDS
    # ========================================================================

    def cmd_ds(self, args: List[str]):
        """Create or switch to a dataset."""
        if not args:
            if self.current_dataset:
                print(f"Current dataset: {self.current_dataset}")
            else:
                print("No dataset selected")
            print("Usage: ds <name>")
            return

        # Create or switch
        name = args[0]

        # Create if doesn't exist
        if name not in self.datasets:
            # Create empty model
            self.datasets[name] = Infinigram(
                [],  # Empty corpus initially
                max_length=self.max_length
            )
            # Initialize document list
            self.dataset_documents[name] = []
            print(f"✓ Created dataset: {name}")

        # Switch to it
        self.current_dataset = name
        print(f"✓ Switched to dataset: {name}")

    def cmd_ds_ls(self, args: List[str]):
        """List all datasets in memory."""
        if not self.datasets:
            print("No datasets loaded")
            return

        print("Datasets in memory:")
        for name, model in self.datasets.items():
            current = " (current)" if name == self.current_dataset else ""
            num_docs = len(self.dataset_documents.get(name, []))
            projs = self.dataset_projections.get(name, [])
            proj_str = f" [{', '.join(projs)}]" if projs else ""
            print(f"  {name}: {model.n} bytes, {num_docs} docs{proj_str}{current}")

    def cmd_ds_cat(self, args: List[str]):
        """View documents in a specific dataset."""
        if not args:
            print("Usage: ds cat <dataset>")
            return

        dataset_name = args[0]

        if dataset_name not in self.datasets:
            print(f"Dataset '{dataset_name}' not found")
            return

        docs = self.dataset_documents.get(dataset_name, [])
        if not docs:
            print(f"Dataset '{dataset_name}' is empty")
            return

        print(f"Documents in '{dataset_name}':")
        for i, doc in enumerate(docs):
            # Truncate long documents for display
            preview = doc[:60] + "..." if len(doc) > 60 else doc
            # Replace newlines with \n for single-line display
            preview = preview.replace('\n', '\\n')
            print(f"  [{i}] {preview}")

        print(f"\nTotal: {len(docs)} documents")

    def cmd_ds_cp(self, args: List[str]):
        """Copy a dataset."""
        if len(args) < 2:
            print("Usage: ds cp <source> <destination>")
            return

        src_name = args[0]
        dst_name = args[1]

        if src_name not in self.datasets:
            print(f"Source dataset '{src_name}' not found")
            return

        if dst_name in self.datasets:
            print(f"Destination dataset '{dst_name}' already exists")
            return

        # Deep copy the model
        src_model = self.datasets[src_name]
        dst_model = Infinigram(
            list(src_model.corpus),  # Copy corpus
            max_length=src_model.max_length,
            min_count=src_model.min_count
        )

        self.datasets[dst_name] = dst_model

        # Copy document tracking
        if src_name in self.dataset_documents:
            self.dataset_documents[dst_name] = list(self.dataset_documents[src_name])

        # Copy projection tracking
        if src_name in self.dataset_projections:
            self.dataset_projections[dst_name] = list(self.dataset_projections[src_name])

        # Copy config
        if src_name in self.dataset_config:
            self.dataset_config[dst_name] = dict(self.dataset_config[src_name])

        print(f"✓ Copied dataset '{src_name}' to '{dst_name}'")
        print(f"  Size: {dst_model.n} bytes")

    def cmd_ds_rm(self, args: List[str]):
        """Delete current dataset."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        name = self.current_dataset
        del self.datasets[name]

        # Clean up associated data
        if name in self.dataset_documents:
            del self.dataset_documents[name]
        if name in self.dataset_projections:
            del self.dataset_projections[name]
        if name in self.dataset_config:
            del self.dataset_config[name]

        self.current_dataset = None

        print(f"✓ Deleted dataset: {name}")

        # Switch to first available dataset if any
        if self.datasets:
            self.current_dataset = list(self.datasets.keys())[0]
            print(f"Switched to: {self.current_dataset}")

    def cmd_ds_info(self, args: List[str]):
        """Show current dataset information."""
        if not self.model:
            print("No dataset selected")
            return

        num_docs = len(self.dataset_documents.get(self.current_dataset, []))
        projs = self.dataset_projections.get(self.current_dataset, [])

        print(f"Dataset: {self.current_dataset}")
        print(f"  Corpus size: {self.model.n} bytes")
        print(f"  Documents: {num_docs}")
        print(f"  Vocabulary size: {self.model.vocab_size}")
        print(f"  Max suffix length: {self.model.max_length or 'unlimited'}")
        print(f"  Min count: {self.model.min_count}")
        if projs:
            print(f"  Active projections: {', '.join(projs)}")

    def cmd_ds_stats(self, args: List[str]):
        """Show corpus statistics."""
        if not self.model:
            print("No dataset selected")
            return

        corpus = self.model.corpus

        print(f"Corpus Statistics for '{self.current_dataset}':")
        print(f"  Total bytes: {len(corpus)}")
        print(f"  Unique bytes: {len(set(corpus))}")

        # Byte frequency
        from collections import Counter
        freq = Counter(corpus)
        most_common = freq.most_common(10)

        print(f"\n  Most common bytes:")
        for byte_val, count in most_common:
            char_repr = chr(byte_val) if 32 <= byte_val <= 126 else f"0x{byte_val:02X}"
            print(f"    {char_repr} (byte {byte_val}): {count} times ({100*count/len(corpus):.1f}%)")

    def cmd_use_OLD(self, args: List[str]):
        """REMOVED - use 'ds <name>' instead."""
        if not args:
            print("Usage: /use <name>")
            return

        name = args[0]
        if name not in self.datasets:
            print(f"Dataset '{name}' not found")
            print("Available datasets:", ', '.join(self.datasets.keys()))
            return

        self.current_dataset = name
        print(f"✓ Switched to dataset: {name}")


    def _load_text_file(self, args: List[str]) -> Optional[str]:
        """Load text from file."""
        if not args:
            print("Usage: /add --file <path>")
            return None

        file_path = args[0]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Loaded {len(text)} characters from {file_path}")
            return text
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def _load_jsonl(self, args: List[str]) -> Optional[str]:
        """Load text from JSONL file."""
        if not args:
            print("Usage: /add --jsonl <path> [--field <field_name>]")
            return None

        file_path = args[0]
        field_name = 'text'  # Default field

        # Check for --field argument
        if '--field' in args:
            idx = args.index('--field')
            if idx + 1 < len(args):
                field_name = args[idx + 1]

        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        obj = json.loads(line.strip())
                        if field_name in obj:
                            documents.append(obj[field_name])
                        else:
                            print(f"Warning: Line {i} missing field '{field_name}'")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Line {i} invalid JSON: {e}")

            print(f"Loaded {len(documents)} documents from {file_path}")

            # Build corpus with document separators
            corpus_bytes = build_corpus_from_documents(documents, separator=b"\x00")
            return bytes_to_text(corpus_bytes)

        except Exception as e:
            print(f"Error loading JSONL: {e}")
            return None

    def cmd_add(self, args: List[str]):
        """Add text to current dataset."""
        if not args:
            print("Usage: /add <text> or /add --file <path> or /add --jsonl <path>")
            return

        # Require explicit dataset selection
        if not self.current_dataset:
            print("No dataset selected. Use /dataset <name> to create or switch to a dataset.")
            return

        # Parse add type
        if args[0] == '--file':
            text = self._load_text_file(args[1:])
        elif args[0] == '--jsonl':
            text = self._load_jsonl(args[1:])
        else:
            # Add from command line
            text = ' '.join(args)

        if text is None:
            return

        # Append to document list
        if self.current_dataset not in self.dataset_documents:
            self.dataset_documents[self.current_dataset] = []
        self.dataset_documents[self.current_dataset].append(text)

        # Check if dataset has active projections
        active_projections = self.dataset_projections.get(self.current_dataset, [])

        if active_projections:
            # Auto-reapply projections to maintain consistency
            aug_functions = [PROJECTION_REGISTRY[name] for name in active_projections]
            corpus = build_corpus_with_augmentation(
                self.dataset_documents[self.current_dataset],
                augmentations=aug_functions,
                separator=b"\x00"  # NULL byte separator
            )
            print(f"✓ Added document to '{self.current_dataset}' (with {len(active_projections)} projection(s))")
        else:
            # No projections, just build corpus normally
            corpus = build_corpus_from_documents(
                self.dataset_documents[self.current_dataset],
                separator=b"\x00"  # NULL byte separator
            )
            print(f"✓ Added document to '{self.current_dataset}'")

        # Rebuild model
        self.datasets[self.current_dataset] = Infinigram(
            corpus,
            max_length=self.max_length
        )

        print(f"  Total corpus size: {len(corpus)} bytes ({len(self.dataset_documents[self.current_dataset])} documents)")

    def cmd_proj(self, args: List[str]):
        """Apply projections to augment current dataset."""
        if not self.model:
            print("No dataset selected. Use ds <name> or load <name> first.")
            return

        if not args:
            print("Usage: proj <projection> [<projection2> ...]")
            print("Available projections:", ', '.join(sorted(PROJECTION_REGISTRY.keys())))
            return

        # Validate projections
        invalid = [p for p in args if p not in PROJECTION_REGISTRY]
        if invalid:
            print(f"Unknown projections: {', '.join(invalid)}")
            print("Available:", ', '.join(sorted(PROJECTION_REGISTRY.keys())))
            return

        print(f"Applying {len(args)} projection(s) to '{self.current_dataset}'...")

        # Get original documents (not the augmented corpus)
        if self.current_dataset not in self.dataset_documents:
            print("Error: No original documents found for this dataset")
            return

        original_documents = self.dataset_documents[self.current_dataset]
        if not original_documents:
            print("Error: Dataset has no documents")
            return

        # Build augmentation functions
        aug_functions = [PROJECTION_REGISTRY[name] for name in args]

        # Apply augmentations to original documents
        augmented_corpus = build_corpus_with_augmentation(
            original_documents,
            augmentations=aug_functions,
            separator=b"\x00"  # NULL byte separator
        )

        # Replace model with augmented version
        original_size = self.model.n
        self.datasets[self.current_dataset] = Infinigram(
            augmented_corpus,
            max_length=self.max_length
        )

        # Track applied projections (replace previous projections)
        self.dataset_projections[self.current_dataset] = list(args)

        print(f"✓ Applied projections: {', '.join(args)}")
        print(f"  Original size: {original_size} bytes")
        print(f"  Augmented size: {self.model.n} bytes")
        if original_size > 0:
            print(f"  Multiplier: {self.model.n / original_size:.2f}x")

    def cmd_proj_ls(self, args: List[str]):
        """List projections."""
        if args and args[0] in ['-a', '--all']:
            # List all available projections
            print("Available projections:")
            for name in sorted(PROJECTION_REGISTRY.keys()):
                print(f"  {name}")
            return

        # List active projections for current dataset
        if not self.current_dataset:
            print("No dataset selected")
            return

        projections = self.dataset_projections.get(self.current_dataset, [])

        if not projections:
            print(f"No active projections for '{self.current_dataset}'")
        else:
            print(f"Active projections for '{self.current_dataset}':")
            for p in projections:
                print(f"  {p}")

    def cmd_proj_cat(self, args: List[str]):
        """View projection details."""
        if not args:
            print("Usage: proj cat <projection>")
            print("\nAvailable projections:")
            for name in sorted(PROJECTION_REGISTRY.keys()):
                print(f"  {name}")
            return

        proj_name = args[0]

        if proj_name not in PROJECTION_REGISTRY:
            print(f"Projection '{proj_name}' not found")
            print("\nAvailable projections:")
            for name in sorted(PROJECTION_REGISTRY.keys()):
                print(f"  {name}")
            return

        print(f"Projection: {proj_name}")
        print("-" * 70)

        # Get the projection function
        proj_func = PROJECTION_REGISTRY[proj_name]

        # Show docstring if available
        if proj_func.__doc__:
            print(f"Description: {proj_func.__doc__}")
        else:
            print("No description available")

        # Show examples
        print("\nExamples:")
        test_inputs = [
            "Hello World",
            "THE QUICK BROWN FOX",
            "python programming",
            "  spaces  around  ",
        ]

        for test_input in test_inputs:
            try:
                result = proj_func(test_input)
                print(f"  '{test_input}' → '{result}'")
            except Exception as e:
                print(f"  '{test_input}' → Error: {e}")

        print("-" * 70)

    def cmd_proj_rm(self, args: List[str]):
        """Remove all projections from current dataset."""
        if not self.current_dataset:
            print("No dataset selected")
            return

        if self.current_dataset not in self.dataset_projections or not self.dataset_projections[self.current_dataset]:
            print("No projections to remove")
            return

        # Clear projections
        self.dataset_projections[self.current_dataset] = []

        # Rebuild corpus without projections
        if self.current_dataset in self.dataset_documents:
            corpus = build_corpus_from_documents(
                self.dataset_documents[self.current_dataset],
                separator=b"\x00"
            )
            self.datasets[self.current_dataset] = Infinigram(
                corpus,
                max_length=self.max_length
            )
            print(f"✓ Removed all projections from '{self.current_dataset}'")
            print(f"  Corpus size: {len(corpus)} bytes")

    def cmd_info_OLD(self, args: List[str]):
        """Show current dataset information."""
        if not self.model:
            print("No dataset selected.")
            return

        print(f"Dataset: {self.current_dataset}")
        print(f"  Corpus size: {self.model.n} bytes")
        num_docs = len(self.dataset_documents.get(self.current_dataset, []))
        if num_docs > 0:
            print(f"  Documents: {num_docs}")
        print(f"  Vocabulary size: {self.model.vocab_size}")
        print(f"  Max suffix length: {self.model.max_length or 'unlimited'}")
        print(f"  Min count: {self.model.min_count}")

    def cmd_stats_OLD(self, args: List[str]):
        """Show corpus statistics."""
        if not self.model:
            print("No model loaded.")
            return

        corpus = self.model.corpus

        # Byte distribution
        byte_counts = {}
        for b in corpus:
            byte_counts[b] = byte_counts.get(b, 0) + 1

        print("Corpus Statistics:")
        print(f"  Total bytes: {len(corpus)}")
        print(f"  Unique bytes: {len(byte_counts)}")
        print()

        # Top 10 most frequent bytes
        top_bytes = sorted(byte_counts.items(), key=lambda x: -x[1])[:10]
        print("  Top 10 most frequent bytes:")
        for byte_val, count in top_bytes:
            char = chr(byte_val) if 32 <= byte_val < 127 else f"0x{byte_val:02X}"
            pct = 100 * count / len(corpus)
            print(f"    {byte_val:3d} ('{char}'): {count:5d} ({pct:5.2f}%)")

    def cmd_predict(self, args: List[str]):
        """Predict next byte probabilities."""
        if not self.model:
            print("No dataset selected. Use /dataset <name> or /add first.")
            return

        if not args:
            print("Usage: /predict <text> [--bytes]")
            return

        # Parse arguments
        show_bytes = '--bytes' in args
        if show_bytes:
            args = [a for a in args if a != '--bytes']

        text = ' '.join(args)
        context = text_to_bytes(text)

        # Get predictions
        if self.weight_function:
            probs = self.model.predict_weighted(
                context,
                min_length=self.min_length,
                max_length=self.max_weight_length or len(context),
                weight_fn=self.weight_function,
                top_k=self.top_k,
                smoothing=self.smoothing
            )
        else:
            probs = self.model.predict(context, top_k=self.top_k, smoothing=self.smoothing)

        # Apply temperature
        if self.temperature != 1.0:
            probs = self._apply_temperature(probs, self.temperature)

        # Display
        print(f"Context: '{text}' ({len(context)} bytes)")
        print(f"Top {min(self.top_k, len(probs))} predictions:")
        print()

        for byte_val, prob in sorted(probs.items(), key=lambda x: -x[1])[:20]:
            if show_bytes:
                print(f"  Byte {byte_val:3d} (0x{byte_val:02X}): {prob:.6f}")
            else:
                # Try to display as character
                if 32 <= byte_val < 127:
                    char = chr(byte_val)
                    print(f"  '{char}' (byte {byte_val}): {prob:.6f}")
                else:
                    print(f"  0x{byte_val:02X} (byte {byte_val}): {prob:.6f}")

    def cmd_complete(self, args: List[str]):
        """Generate completion."""
        if not self.model:
            print("No dataset selected. Use /dataset <name> or /add first.")
            return

        if not args:
            print("Usage: /complete <text> [--max N]")
            return

        # Parse arguments
        max_bytes = 50  # Default
        if '--max' in args:
            idx = args.index('--max')
            if idx + 1 < len(args):
                try:
                    max_bytes = int(args[idx + 1])
                    args = args[:idx] + args[idx+2:]
                except ValueError:
                    print("Invalid --max value")
                    return

        text = ' '.join(args)
        context = text_to_bytes(text)

        print(f"Context: '{text}'")
        print(f"Generating up to {max_bytes} bytes...")
        print()

        # Generate completion
        generated = list(context)  # Copy context

        for i in range(max_bytes):
            # Get predictions
            if self.weight_function:
                probs = self.model.predict_weighted(
                    generated,
                    min_length=self.min_length,
                    max_length=self.max_weight_length or len(generated),
                    weight_fn=self.weight_function,
                    top_k=self.top_k,
                    smoothing=self.smoothing
                )
            else:
                probs = self.model.predict(generated, top_k=self.top_k, smoothing=self.smoothing)

            if not probs:
                break

            # Apply temperature
            if self.temperature != 1.0:
                probs = self._apply_temperature(probs, self.temperature)

            # Sample next byte
            next_byte = self._sample(probs)
            generated.append(next_byte)

            # Stop at NULL byte (end-of-document marker)
            if next_byte == 0:
                break

        # Decode and display
        completion_bytes = generated[len(context):]
        completion_text = bytes_to_text(completion_bytes)

        print(f"Generated: '{completion_text}'")
        print(f"({len(completion_bytes)} bytes)")

    def cmd_set_temperature(self, args: List[str]):
        """Set temperature."""
        if not args:
            print(f"Current temperature: {self.temperature}")
            print("Usage: set temperature <value>")
            print("  Higher values (>1.0) = more uniform distribution")
            print("  Lower values (<1.0) = more peaked distribution")
            return

        try:
            temp = float(args[0])
            if temp <= 0:
                print("Temperature must be positive")
                return
            self.temperature = temp
            print(f"✓ Temperature set to {temp}")
        except ValueError:
            print("Invalid temperature value")

    def cmd_set_top_k(self, args: List[str]):
        """Set top_k."""
        if not args:
            print(f"Current top_k: {self.top_k}")
            print("Usage: set top_k <n>")
            return

        try:
            k = int(args[0])
            if k <= 0:
                print("top_k must be positive")
                return
            self.top_k = k
            print(f"✓ top_k set to {k}")
        except ValueError:
            print("Invalid top_k value")

    def cmd_set_max_length(self, args: List[str]):
        """Set max suffix length."""
        if not args:
            print(f"Current max_length: {self.max_length or 'unlimited'}")
            print("Usage: set max_length <n> or set max_length none")
            return

        if args[0].lower() == 'none':
            self.max_length = None
            print("✓ max_length set to unlimited")
        else:
            try:
                length = int(args[0])
                if length <= 0:
                    print("max_length must be positive")
                    return
                self.max_length = length
                print(f"✓ max_length set to {length}")
            except ValueError:
                print("Invalid max_length value")

    def cmd_set_smoothing(self, args: List[str]):
        """Set smoothing parameter."""
        if not args:
            print(f"Current smoothing: {self.smoothing}")
            print("Usage: set smoothing <value>")
            print("  0.0 = no smoothing (only observed data)")
            print("  >0.0 = add-k smoothing for unseen bytes")
            return

        try:
            smooth = float(args[0])
            if smooth < 0:
                print("Smoothing must be non-negative")
                return
            self.smoothing = smooth
            print(f"✓ Smoothing set to {smooth}")
        except ValueError:
            print("Invalid smoothing value")

    def cmd_set_weight(self, args: List[str]):
        """Set weight function."""
        if not args:
            current = "none" if not self.weight_function else "custom"
            print(f"Current weight function: {current}")
            print("Usage: set weight <function>")
            print("  Options: none, linear, quadratic, exponential, sigmoid")
            return

        func_name = args[0].lower()

        if func_name == 'none':
            self.weight_function = None
            print("✓ Disabled weighted prediction")
        else:
            try:
                self.weight_function = get_weight_function(func_name)
                print(f"✓ Weight function set to {func_name}")
            except ValueError as e:
                print(f"Error: {e}")

    def cmd_config(self, args: List[str]):
        """Show current configuration."""
        print("Current Configuration:")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Max suffix length: {self.max_length or 'unlimited'}")
        print(f"  Smoothing: {self.smoothing}")
        weight_name = "none" if not self.weight_function else "custom"
        print(f"  Weight function: {weight_name}")
        print(f"  Weighted prediction min_length: {self.min_length}")
        print(f"  Weighted prediction max_length: {self.max_weight_length or 'auto'}")

    def cmd_reset_OLD(self, args: List[str]):
        """Delete current dataset."""
        if not self.current_dataset:
            print("No dataset selected.")
            return

        name = self.current_dataset
        del self.datasets[name]
        self.current_dataset = None

        print(f"✓ Deleted dataset: {name}")

        # Switch to first available dataset if any
        if self.datasets:
            self.current_dataset = list(self.datasets.keys())[0]
            print(f"Switched to: {self.current_dataset}")

    def cmd_quit(self, args: List[str]):
        """Quit REPL."""
        print("Goodbye!")
        sys.exit(0)

    def cmd_save(self, args: List[str]):
        """Save dataset to disk."""
        # Determine which dataset to save
        if args:
            dataset_name = args[0]
            if dataset_name not in self.datasets:
                print(f"Dataset '{dataset_name}' not found")
                return
        else:
            if not self.current_dataset:
                print("No dataset selected. Use /save <name> or switch to a dataset first.")
                return
            dataset_name = self.current_dataset

        # Create dataset directory
        dataset_dir = self.storage_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save documents
        documents_file = dataset_dir / "documents.jsonl"
        with open(documents_file, 'w', encoding='utf-8') as f:
            for doc in self.dataset_documents.get(dataset_name, []):
                f.write(json.dumps({"text": doc}) + "\n")

        # Save metadata
        metadata = {
            "name": dataset_name,
            "projections": self.dataset_projections.get(dataset_name, []),
            "config": self.dataset_config.get(dataset_name, {
                "max_length": None,
                "min_count": 1
            }),
            "num_documents": len(self.dataset_documents.get(dataset_name, [])),
            "corpus_size": self.datasets[dataset_name].n if dataset_name in self.datasets else 0
        }

        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved dataset '{dataset_name}' to {dataset_dir}")
        print(f"  Documents: {metadata['num_documents']}")
        print(f"  Corpus size: {metadata['corpus_size']} bytes")
        if metadata['projections']:
            print(f"  Projections: {', '.join(metadata['projections'])}")

    def cmd_load(self, args: List[str]):
        """Load dataset from disk."""
        if not args:
            print("Usage: /load <dataset_name>")
            return

        dataset_name = args[0]
        dataset_dir = self.storage_dir / dataset_name

        if not dataset_dir.exists():
            print(f"Dataset '{dataset_name}' not found in {self.storage_dir}")
            return

        # Load metadata
        metadata_file = dataset_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"Invalid dataset: missing metadata.json")
            return

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load documents
        documents_file = dataset_dir / "documents.jsonl"
        if not documents_file.exists():
            print(f"Invalid dataset: missing documents.jsonl")
            return

        documents = []
        with open(documents_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc_obj = json.loads(line.strip())
                documents.append(doc_obj["text"])

        # Restore dataset state
        self.dataset_documents[dataset_name] = documents
        self.dataset_projections[dataset_name] = metadata.get("projections", [])
        self.dataset_config[dataset_name] = metadata.get("config", {
            "max_length": None,
            "min_count": 1
        })

        # Rebuild corpus with projections
        config = self.dataset_config[dataset_name]
        projections = self.dataset_projections[dataset_name]

        if projections:
            aug_functions = [PROJECTION_REGISTRY[name] for name in projections]
            corpus = build_corpus_with_augmentation(
                documents,
                augmentations=aug_functions,
                separator=b"\x00"
            )
        else:
            corpus = build_corpus_from_documents(
                documents,
                separator=b"\x00"
            )

        # Create model
        self.datasets[dataset_name] = Infinigram(
            corpus,
            max_length=config.get("max_length"),
            min_count=config.get("min_count", 1)
        )

        # Switch to loaded dataset
        self.current_dataset = dataset_name

        print(f"✓ Loaded dataset '{dataset_name}'")
        print(f"  Documents: {len(documents)}")
        print(f"  Corpus size: {self.datasets[dataset_name].n} bytes")
        if projections:
            print(f"  Projections: {', '.join(projections)}")

    def cmd_store_ls(self, args: List[str]):
        """List saved datasets on disk."""
        if not self.storage_dir.exists():
            print("No saved datasets")
            return

        saved_datasets = []
        for dataset_dir in self.storage_dir.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        saved_datasets.append((dataset_dir.name, metadata))
                    except:
                        continue

        if not saved_datasets:
            print("No saved datasets")
            return

        print(f"Saved datasets ({self.storage_dir}):")
        for name, metadata in sorted(saved_datasets):
            loaded = " [loaded]" if name in self.datasets else ""
            print(f"  {name}{loaded}")
            print(f"    Documents: {metadata.get('num_documents', 0)}")
            print(f"    Corpus: {metadata.get('corpus_size', 0)} bytes")
            if metadata.get('projections'):
                print(f"    Projections: {', '.join(metadata['projections'])}")

    def cmd_store_rm(self, args: List[str]):
        """Delete saved dataset from disk."""
        if not args:
            print("Usage: store rm <dataset_name>")
            return

        dataset_name = args[0]
        dataset_dir = self.storage_dir / dataset_name

        if not dataset_dir.exists():
            print(f"Dataset '{dataset_name}' not found in {self.storage_dir}")
            return

        # Delete directory
        import shutil
        shutil.rmtree(dataset_dir)

        print(f"✓ Deleted saved dataset '{dataset_name}'")

        # Also remove from memory if loaded
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
            if self.current_dataset == dataset_name:
                self.current_dataset = None
                if self.datasets:
                    self.current_dataset = list(self.datasets.keys())[0]
                    print(f"Switched to: {self.current_dataset}")
            print(f"  Also removed from memory")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _apply_temperature(self, probs: Dict[int, float], temperature: float) -> Dict[int, float]:
        """Apply temperature scaling to probabilities."""
        # Convert to log space
        log_probs = {k: np.log(v) if v > 0 else -np.inf for k, v in probs.items()}

        # Scale by temperature
        scaled_log_probs = {k: v / temperature for k, v in log_probs.items()}

        # Convert back to probabilities
        max_log = max(scaled_log_probs.values())
        exp_probs = {k: np.exp(v - max_log) for k, v in scaled_log_probs.items()}

        # Normalize
        total = sum(exp_probs.values())
        return {k: v / total for k, v in exp_probs.items()}

    def _sample(self, probs: Dict[int, float]) -> int:
        """Sample a byte from probability distribution."""
        bytes_list = list(probs.keys())
        probs_list = [probs[b] for b in bytes_list]

        # Normalize (in case of floating point errors)
        total = sum(probs_list)
        probs_list = [p / total for p in probs_list]

        # Sample
        return np.random.choice(bytes_list, p=probs_list)


def main():
    """Run the REPL."""
    repl = InfinigramREPL()
    repl.run()


if __name__ == "__main__":
    main()
