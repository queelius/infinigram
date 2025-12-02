#!/usr/bin/env python3
"""
Interactive REPL for Infinigram.

Provides an interactive shell for exploring corpus-based language models,
testing predictions, and managing models.

All models use the unified mmap-backed Infinigram interface.
"""

import sys
import shlex
import json
import random
import math
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from infinigram import Infinigram
from infinigram.weighting import get_weight_function


# Default models directory
DEFAULT_MODELS_DIR = Path.home() / ".infinigram" / "models"


class InfinigramREPL:
    """Interactive REPL for Infinigram corpus-based language models."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize REPL."""
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Current model
        self.model: Optional[Infinigram] = None
        self.model_name: Optional[str] = None

        # Configuration
        self.temperature = 1.0
        self.top_k = 50
        self.max_length = None
        self.smoothing = 0.0
        self.weight_function = None
        self.min_length = 1

        # Command history
        if PROMPT_TOOLKIT_AVAILABLE:
            self.history = InMemoryHistory()
        else:
            self.history: List[str] = []

        # Previous model for cd -
        self.prev_model: Optional[str] = None

        # Commands
        self.commands = {
            # System
            'help': self.cmd_help,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,

            # Navigation (Unix-style)
            'pwd': self.cmd_pwd,
            'cd': self.cmd_cd,
            'ls': self.cmd_ls,

            # Model management
            'model': self.cmd_model,
            'models': self.cmd_models,
            'use': self.cmd_use,
            'build': self.cmd_build,
            'info': self.cmd_info,
            'unload': self.cmd_unload,

            # Queries
            'count': self.cmd_count,
            'search': self.cmd_search,
            'context': self.cmd_context,

            # Inference
            'predict': self.cmd_predict,
            'complete': self.cmd_complete,
            'sample': self.cmd_sample,

            # Configuration
            'config': self.cmd_config,
            'set': self.cmd_set,
        }

    def run(self):
        """Run the REPL."""
        print("=" * 70)
        print("  INFINIGRAM - Corpus-Based Language Model")
        print("=" * 70)
        print()
        print("Type 'help' for commands, 'models' to list available models.")
        print()

        # Show available models
        available = self.list_available_models()
        if available:
            print(f"Available models: {', '.join(available)}")
            print("Use 'use <name>' to load a model.")
        else:
            print(f"No models found in {self.models_dir}")
            print("Use 'build <corpus_file> <name>' to create a model.")
        print()

        while True:
            try:
                # Build prompt
                if self.model_name:
                    prompt_str = f"infinigram[{self.model_name}]> "
                else:
                    prompt_str = "infinigram> "

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

                if not PROMPT_TOOLKIT_AVAILABLE:
                    self.history.append(user_input)

                # Handle shell commands
                if user_input.startswith('!'):
                    import subprocess
                    subprocess.run(user_input[1:], shell=True)
                    continue

                self.execute(user_input)
                print()

            except KeyboardInterrupt:
                print("\n(Use 'quit' to exit)")
                continue
            except EOFError:
                print("\nGoodbye!")
                break

    def execute(self, command_line: str):
        """Execute a command."""
        try:
            parts = shlex.split(command_line)
        except ValueError as e:
            print(f"Parse error: {e}")
            return

        if not parts:
            return

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in self.commands:
            try:
                self.commands[cmd](args)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for available commands.")

    def list_available_models(self) -> List[str]:
        """List models in the models directory."""
        return Infinigram.list_models(self.models_dir)

    # ========================================================================
    # HELP
    # ========================================================================

    def cmd_help(self, args: List[str]):
        """Show help."""
        print("INFINIGRAM COMMANDS")
        print("=" * 70)
        print()
        print("Navigation:")
        print("  pwd                       Show current model context")
        print("  ls                        List available models")
        print("  cd <model>                Switch to a model")
        print("  cd ..                     Unload current model")
        print("  cd -                      Switch to previous model")
        print()
        print("Model Management:")
        print("  models                    List available models (alias: ls)")
        print("  use <name>                Load and use a model (alias: cd)")
        print("  use <path>                Load model from path")
        print("  build <file> <name>       Build model from corpus file")
        print("  build <file> <name> -c N  Chunk size in GB (default: 5)")
        print("  info                      Show current model info")
        print("  unload                    Unload current model (alias: cd ..)")
        print()
        print("Queries:")
        print("  count <pattern>           Count pattern occurrences")
        print("  search <pattern>          Search for pattern")
        print("  search <p> -n N           Limit results (default: 5)")
        print("  search <p> -w N           Context window (default: 60)")
        print("  context <pos>             Show context at position")
        print()
        print("Inference:")
        print("  predict <text>            Predict next byte probabilities")
        print("  predict <text> -n N       Show top N (default: 10)")
        print("  predict <text> -b         Show as bytes")
        print("  complete <text>           Generate text completion")
        print("  complete <text> -n N      Generate N bytes (default: 50)")
        print("  complete <text> -t T      Temperature (default: 1.0)")
        print("  sample <text> -n N        Sample N completions")
        print()
        print("Configuration:")
        print("  config                    Show current settings")
        print("  set temperature <val>     Set sampling temperature")
        print("  set top_k <n>             Set top-k for predictions")
        print("  set max_length <n>        Set max context length")
        print("  set smoothing <val>       Set Laplace smoothing")
        print("  set weight <func>         Weight function (linear, quadratic, ...)")
        print()
        print("System:")
        print("  help                      Show this help")
        print("  quit, exit                Exit REPL")
        print("  !<command>                Execute shell command")
        print()

    def cmd_quit(self, args: List[str]):
        """Exit the REPL."""
        print("Goodbye!")
        sys.exit(0)

    # ========================================================================
    # NAVIGATION
    # ========================================================================

    def cmd_pwd(self, args: List[str]):
        """Show current model context (like Unix pwd)."""
        if self.model_name:
            model_path = self.models_dir / self.model_name
            print(f"/{self.model_name}")
            print(f"  Path: {model_path}")
            print(f"  Size: {self.model.n:,} bytes")
        else:
            print("/")
            print(f"  Models dir: {self.models_dir}")
            print("  No model loaded")

    def cmd_cd(self, args: List[str]):
        """Change to a model (like Unix cd)."""
        if not args:
            # cd with no args - go to root (unload)
            if self.model_name:
                self.prev_model = self.model_name
                self.cmd_unload([])
            else:
                print("/")
            return

        target = args[0]

        # Handle special cases
        if target == '..':
            # Go up (unload model)
            if self.model_name:
                self.prev_model = self.model_name
                self.cmd_unload([])
            else:
                print("Already at root")
            return

        if target == '-':
            # Switch to previous model
            if self.prev_model:
                old_model = self.model_name
                self.cmd_use([self.prev_model])
                self.prev_model = old_model
            else:
                print("No previous model")
            return

        if target == '~' or target == '/':
            # Go to root
            if self.model_name:
                self.prev_model = self.model_name
                self.cmd_unload([])
            return

        # Strip leading / if present
        if target.startswith('/'):
            target = target[1:]

        # Try to load the model
        old_model = self.model_name
        self.cmd_use([target])

        # Update prev_model if load succeeded
        if self.model_name and self.model_name != old_model:
            self.prev_model = old_model

    def cmd_ls(self, args: List[str]):
        """List available models (like Unix ls)."""
        self.cmd_models(args)

    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================

    def cmd_models(self, args: List[str]):
        """List available models."""
        models = self.list_available_models()

        if not models:
            print(f"No models found in {self.models_dir}")
            print("Use 'build <corpus_file> <name>' to create a model.")
            return

        print("Available Models:")
        print("-" * 70)
        for name in models:
            model_path = self.models_dir / name
            meta_file = model_path / "meta.json"

            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                size = meta.get('corpus_size', meta.get('n', 0))
                size_str = self._format_size(size)
                chunks = meta.get('num_chunks', 1)
                chunk_str = f" ({chunks} chunks)" if chunks > 1 else ""
                loaded = " [loaded]" if name == self.model_name else ""
                print(f"  {name}: {size_str}{chunk_str}{loaded}")
            else:
                print(f"  {name}: (metadata not found)")

        print("-" * 70)
        print(f"Total: {len(models)} models")

    def cmd_model(self, args: List[str]):
        """Alias for 'use' command."""
        self.cmd_use(args)

    def cmd_use(self, args: List[str]):
        """Load and use a model."""
        if not args:
            if self.model_name:
                print(f"Current model: {self.model_name}")
                self.cmd_info([])
            else:
                print("No model loaded")
                print("Usage: use <name>")
            return

        name = args[0]

        # Check if it's a path
        path = Path(name)
        if path.exists() and path.is_dir():
            model_path = path
            name = path.name
        else:
            model_path = self.models_dir / name

        if not model_path.exists():
            print(f"Model '{name}' not found")
            print(f"Available models: {', '.join(self.list_available_models())}")
            return

        print(f"Loading model '{name}'...")
        try:
            self.model = Infinigram.load(str(model_path))
            self.model_name = name
            print(f"Loaded: {self.model}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def cmd_build(self, args: List[str]):
        """Build a model from corpus file."""
        if len(args) < 2:
            print("Usage: build <corpus_file> <name> [-c chunk_size_gb]")
            return

        corpus_file = Path(args[0])
        name = args[1]
        chunk_size = 5.0

        # Parse options
        i = 2
        while i < len(args):
            if args[i] == '-c' and i + 1 < len(args):
                chunk_size = float(args[i + 1])
                i += 2
            else:
                i += 1

        if not corpus_file.exists():
            print(f"Corpus file not found: {corpus_file}")
            return

        model_path = self.models_dir / name
        if model_path.exists():
            print(f"Model '{name}' already exists. Delete it first.")
            return

        print(f"Building model '{name}' from {corpus_file}...")
        print(f"Chunk size: {chunk_size} GB")

        try:
            self.model = Infinigram.build(
                str(corpus_file),
                str(model_path),
                chunk_size_gb=chunk_size,
                verbose=True
            )
            self.model_name = name
            print(f"\nModel built: {self.model}")
        except Exception as e:
            print(f"Build failed: {e}")

    def cmd_info(self, args: List[str]):
        """Show current model info."""
        if not self.model:
            print("No model loaded")
            return

        print(f"Model: {self.model_name}")
        print(f"  Corpus size: {self._format_size(self.model.n)}")
        print(f"  Vocabulary: 256 bytes")

        if self.model._is_chunked:
            print(f"  Chunks: {self.model._sa.num_chunks}")

        if self.model.max_length:
            print(f"  Max length: {self.model.max_length}")

        print(f"  Path: {self.model.model_path}")

    def cmd_unload(self, args: List[str]):
        """Unload current model."""
        if self.model:
            self.model.close()
            print(f"Unloaded model '{self.model_name}'")
            self.model = None
            self.model_name = None
        else:
            print("No model loaded")

    # ========================================================================
    # QUERIES
    # ========================================================================

    def cmd_count(self, args: List[str]):
        """Count pattern occurrences."""
        if not self.model:
            print("No model loaded. Use 'use <name>' first.")
            return

        if not args:
            print("Usage: count <pattern>")
            return

        pattern = ' '.join(args)
        pattern_bytes = pattern.encode('utf-8')

        count = self.model.count(pattern_bytes)
        print(f"'{pattern}': {count:,} occurrences")

    def cmd_search(self, args: List[str]):
        """Search for pattern and show context."""
        if not self.model:
            print("No model loaded. Use 'use <name>' first.")
            return

        if not args:
            print("Usage: search <pattern> [-n limit] [-w window]")
            return

        # Parse args
        limit = 5
        window = 60
        pattern_parts = []

        i = 0
        while i < len(args):
            if args[i] == '-n' and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            elif args[i] == '-w' and i + 1 < len(args):
                window = int(args[i + 1])
                i += 2
            else:
                pattern_parts.append(args[i])
                i += 1

        pattern = ' '.join(pattern_parts)
        pattern_bytes = pattern.encode('utf-8')

        results = self.model.search(pattern_bytes)
        total = len(results)

        print(f"Pattern: '{pattern}'")
        print(f"Found: {total:,} occurrences")

        if total == 0:
            return

        print()
        print(f"Showing {min(limit, total)} results (window={window}):")
        print("-" * 70)

        for i, pos in enumerate(results[:limit]):
            if self.model._is_chunked:
                chunk_idx, pos_in_chunk = pos
                ctx = self.model.get_context(pos_in_chunk, window, chunk_idx)
                loc = f"chunk {chunk_idx}, pos {pos_in_chunk:,}"
            else:
                ctx = self.model.get_context(pos, window)
                loc = f"pos {pos:,}"

            # Decode and highlight
            ctx_text = ctx.decode('utf-8', errors='replace')
            # Find pattern in context
            pattern_start = ctx_text.find(pattern)
            if pattern_start >= 0:
                before = ctx_text[:pattern_start]
                match = ctx_text[pattern_start:pattern_start+len(pattern)]
                after = ctx_text[pattern_start+len(pattern):]
                ctx_text = f"{before}[{match}]{after}"

            print(f"[{i+1}] {loc}")
            print(f"    ...{ctx_text}...")
            print()

    def cmd_context(self, args: List[str]):
        """Show context at a position."""
        if not self.model:
            print("No model loaded. Use 'use <name>' first.")
            return

        if not args:
            print("Usage: context <position> [-w window] [-c chunk]")
            return

        position = int(args[0])
        window = 100
        chunk_idx = None

        i = 1
        while i < len(args):
            if args[i] == '-w' and i + 1 < len(args):
                window = int(args[i + 1])
                i += 2
            elif args[i] == '-c' and i + 1 < len(args):
                chunk_idx = int(args[i + 1])
                i += 2
            else:
                i += 1

        if self.model._is_chunked:
            if chunk_idx is None:
                print("Chunked model - use -c to specify chunk index")
                return
            ctx = self.model.get_context(position, window, chunk_idx)
        else:
            ctx = self.model.get_context(position, window)

        print(f"Context at position {position}:")
        print("-" * 70)
        print(ctx.decode('utf-8', errors='replace'))
        print("-" * 70)

    # ========================================================================
    # INFERENCE
    # ========================================================================

    def cmd_predict(self, args: List[str]):
        """Predict next byte probabilities."""
        if not self.model:
            print("No model loaded. Use 'use <name>' first.")
            return

        if not args:
            print("Usage: predict <text> [-n top_k] [-b]")
            return

        # Parse args
        top_k = 10
        show_bytes = False
        text_parts = []

        i = 0
        while i < len(args):
            if args[i] == '-n' and i + 1 < len(args):
                top_k = int(args[i + 1])
                i += 2
            elif args[i] == '-b':
                show_bytes = True
                i += 1
            else:
                text_parts.append(args[i])
                i += 1

        context = ' '.join(text_parts)

        # Get weight function if set
        weight_fn = None
        if self.weight_function:
            weight_fn = get_weight_function(self.weight_function)

        if weight_fn:
            probs = self.model.predict_weighted(
                context,
                min_length=self.min_length,
                max_length=self.max_length,
                weight_fn=weight_fn,
                top_k=top_k,
                smoothing=self.smoothing
            )
        else:
            probs = self.model.predict(
                context, top_k=top_k, smoothing=self.smoothing
            )

        print(f"Context: '{context}'")
        print(f"Predictions (top {top_k}):")
        print("-" * 40)

        for byte_val, prob in probs.items():
            if show_bytes:
                display = f"0x{byte_val:02x}"
            else:
                if 32 <= byte_val < 127:
                    display = f"'{chr(byte_val)}'"
                else:
                    display = f"<0x{byte_val:02x}>"
            print(f"  {display:12s} {prob:.4f} ({prob*100:.1f}%)")

    def cmd_complete(self, args: List[str]):
        """Generate text completion."""
        if not self.model:
            print("No model loaded. Use 'use <name>' first.")
            return

        if not args:
            print("Usage: complete <text> [-n max_tokens] [-t temperature]")
            return

        # Parse args
        max_tokens = 50
        temperature = self.temperature
        text_parts = []

        i = 0
        while i < len(args):
            if args[i] == '-n' and i + 1 < len(args):
                max_tokens = int(args[i + 1])
                i += 2
            elif args[i] == '-t' and i + 1 < len(args):
                temperature = float(args[i + 1])
                i += 2
            else:
                text_parts.append(args[i])
                i += 1

        context = ' '.join(text_parts)
        context_bytes = list(context.encode('utf-8'))

        # Generate completion
        completion = []
        current = context_bytes.copy()

        weight_fn = None
        if self.weight_function:
            weight_fn = get_weight_function(self.weight_function)

        for _ in range(max_tokens):
            if weight_fn:
                probs = self.model.predict_weighted(
                    current,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    weight_fn=weight_fn,
                    top_k=self.top_k,
                    smoothing=self.smoothing
                )
            else:
                probs = self.model.predict(
                    current, top_k=self.top_k, smoothing=self.smoothing
                )

            if not probs:
                break

            # Sample with temperature
            next_byte = self._sample(probs, temperature)
            completion.append(next_byte)
            current.append(next_byte)

        # Decode and display
        completion_text = bytes(completion).decode('utf-8', errors='replace')

        print(f"Context: '{context}'")
        print(f"Completion ({len(completion)} bytes, temp={temperature}):")
        print("-" * 70)
        print(context + completion_text)
        print("-" * 70)

    def cmd_sample(self, args: List[str]):
        """Sample multiple completions."""
        if not self.model:
            print("No model loaded. Use 'use <name>' first.")
            return

        if not args:
            print("Usage: sample <text> [-n samples] [-l length] [-t temp]")
            return

        # Parse args
        num_samples = 5
        length = 30
        temperature = self.temperature
        text_parts = []

        i = 0
        while i < len(args):
            if args[i] == '-n' and i + 1 < len(args):
                num_samples = int(args[i + 1])
                i += 2
            elif args[i] == '-l' and i + 1 < len(args):
                length = int(args[i + 1])
                i += 2
            elif args[i] == '-t' and i + 1 < len(args):
                temperature = float(args[i + 1])
                i += 2
            else:
                text_parts.append(args[i])
                i += 1

        context = ' '.join(text_parts)
        context_bytes = list(context.encode('utf-8'))

        print(f"Context: '{context}'")
        print(f"Sampling {num_samples} completions (length={length}, temp={temperature}):")
        print("-" * 70)

        for s in range(num_samples):
            completion = []
            current = context_bytes.copy()

            for _ in range(length):
                probs = self.model.predict(
                    current, top_k=self.top_k, smoothing=self.smoothing
                )
                if not probs:
                    break
                next_byte = self._sample(probs, temperature)
                completion.append(next_byte)
                current.append(next_byte)

            completion_text = bytes(completion).decode('utf-8', errors='replace')
            print(f"[{s+1}] {context}{completion_text}")

        print("-" * 70)

    def _sample(self, probs: Dict[int, float], temperature: float) -> int:
        """Sample from probability distribution with temperature."""
        if not probs:
            return 0

        if temperature == 0:
            return max(probs.items(), key=lambda x: x[1])[0]

        if temperature != 1.0:
            tokens = list(probs.keys())
            log_probs = [math.log(p + 1e-10) / temperature for p in probs.values()]
            max_log = max(log_probs)
            exp_probs = [math.exp(lp - max_log) for lp in log_probs]
            total = sum(exp_probs)
            probs = {t: p / total for t, p in zip(tokens, exp_probs)}

        r = random.random()
        cumulative = 0.0
        for token, prob in probs.items():
            cumulative += prob
            if r < cumulative:
                return token

        return list(probs.keys())[-1]

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    def cmd_config(self, args: List[str]):
        """Show current configuration."""
        print("Configuration:")
        print("-" * 40)
        print(f"  temperature:    {self.temperature}")
        print(f"  top_k:          {self.top_k}")
        print(f"  max_length:     {self.max_length or 'unlimited'}")
        print(f"  smoothing:      {self.smoothing}")
        print(f"  weight:         {self.weight_function or 'none'}")
        print(f"  min_length:     {self.min_length}")
        print("-" * 40)
        print(f"  models_dir:     {self.models_dir}")
        if self.model_name:
            print(f"  current model:  {self.model_name}")

    def cmd_set(self, args: List[str]):
        """Set configuration value."""
        if len(args) < 2:
            print("Usage: set <parameter> <value>")
            print("Parameters: temperature, top_k, max_length, smoothing, weight, min_length")
            return

        param = args[0].lower()
        value = args[1]

        if param == 'temperature':
            self.temperature = float(value)
            print(f"temperature = {self.temperature}")
        elif param == 'top_k':
            self.top_k = int(value)
            print(f"top_k = {self.top_k}")
        elif param == 'max_length':
            self.max_length = int(value) if value.lower() != 'none' else None
            print(f"max_length = {self.max_length}")
        elif param == 'smoothing':
            self.smoothing = float(value)
            print(f"smoothing = {self.smoothing}")
        elif param == 'weight':
            if value.lower() == 'none':
                self.weight_function = None
            else:
                # Validate
                get_weight_function(value)
                self.weight_function = value
            print(f"weight = {self.weight_function}")
        elif param == 'min_length':
            self.min_length = int(value)
            print(f"min_length = {self.min_length}")
        else:
            print(f"Unknown parameter: {param}")
            print("Parameters: temperature, top_k, max_length, smoothing, weight, min_length")

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _format_size(self, size: int) -> str:
        """Format byte size for display."""
        if size >= 1_000_000_000:
            return f"{size / 1_000_000_000:.2f} GB"
        elif size >= 1_000_000:
            return f"{size / 1_000_000:.2f} MB"
        elif size >= 1_000:
            return f"{size / 1_000:.2f} KB"
        else:
            return f"{size} bytes"


def main():
    """Entry point for REPL."""
    import argparse

    parser = argparse.ArgumentParser(description="Infinigram Interactive REPL")
    parser.add_argument('--models-dir', type=Path, help="Models directory")
    parser.add_argument('--use', type=str, help="Model to load on startup")

    args = parser.parse_args()

    repl = InfinigramREPL(models_dir=args.models_dir)

    if args.use:
        repl.cmd_use([args.use])

    repl.run()


if __name__ == "__main__":
    main()
