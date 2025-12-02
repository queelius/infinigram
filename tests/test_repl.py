"""
Tests for Infinigram REPL.

Tests the unified model-based REPL interface.
"""

import pytest
import tempfile
from pathlib import Path

from infinigram.repl import InfinigramREPL
from infinigram import Infinigram


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_models_dir():
    """Create temporary models directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def repl(temp_models_dir):
    """Create REPL with temporary models directory."""
    return InfinigramREPL(models_dir=temp_models_dir)


@pytest.fixture
def repl_with_model(temp_models_dir):
    """Create REPL with a pre-built model."""
    # Build a test model
    corpus = "The cat sat on the mat. The cat slept."
    model_path = temp_models_dir / "test-model"
    Infinigram.build(corpus, str(model_path), verbose=False)

    repl = InfinigramREPL(models_dir=temp_models_dir)
    repl.cmd_use(["test-model"])
    return repl


# ============================================================================
# REPL Initialization Tests
# ============================================================================

class TestREPLInit:
    """Tests for REPL initialization."""

    def test_init_creates_models_dir(self, temp_models_dir):
        """Test REPL creates models directory."""
        models_dir = temp_models_dir / "subdir" / "models"
        repl = InfinigramREPL(models_dir=models_dir)
        assert models_dir.exists()

    def test_init_no_model_loaded(self, repl):
        """Test REPL starts with no model loaded."""
        assert repl.model is None
        assert repl.model_name is None

    def test_init_default_config(self, repl):
        """Test REPL starts with default configuration."""
        assert repl.temperature == 1.0
        assert repl.top_k == 50
        assert repl.max_length is None
        assert repl.smoothing == 0.0
        assert repl.weight_function is None


# ============================================================================
# Model Management Tests
# ============================================================================

class TestModelManagement:
    """Tests for model management commands."""

    def test_models_empty(self, repl):
        """Test models command with no models."""
        # Should not raise
        repl.cmd_models([])

    def test_models_lists_available(self, repl_with_model):
        """Test models command lists available models."""
        repl_with_model.cmd_models([])

    def test_use_loads_model(self, repl, temp_models_dir):
        """Test use command loads a model."""
        # Build a model first
        corpus = "Hello world"
        model_path = temp_models_dir / "hello"
        Infinigram.build(corpus, str(model_path), verbose=False)

        repl.cmd_use(["hello"])

        assert repl.model is not None
        assert repl.model_name == "hello"

    def test_use_nonexistent_model(self, repl):
        """Test use with nonexistent model shows error."""
        repl.cmd_use(["nonexistent"])
        assert repl.model is None

    def test_use_with_path(self, temp_models_dir):
        """Test use command with full path."""
        # Build model
        corpus = "Test corpus"
        model_path = temp_models_dir / "path-test"
        Infinigram.build(corpus, str(model_path), verbose=False)

        repl = InfinigramREPL(models_dir=temp_models_dir / "other")
        repl.cmd_use([str(model_path)])

        assert repl.model is not None

    def test_info_no_model(self, repl):
        """Test info command with no model loaded."""
        repl.cmd_info([])

    def test_info_with_model(self, repl_with_model):
        """Test info command shows model info."""
        repl_with_model.cmd_info([])

    def test_unload_model(self, repl_with_model):
        """Test unload command removes model."""
        assert repl_with_model.model is not None

        repl_with_model.cmd_unload([])

        assert repl_with_model.model is None
        assert repl_with_model.model_name is None

    def test_build_creates_model(self, repl, temp_models_dir):
        """Test build command creates a model."""
        # Create corpus file
        corpus_file = temp_models_dir / "corpus.txt"
        corpus_file.write_text("Test corpus for building")

        repl.cmd_build([str(corpus_file), "built-model"])

        assert repl.model is not None
        assert repl.model_name == "built-model"
        assert (temp_models_dir / "built-model").exists()


# ============================================================================
# Query Tests
# ============================================================================

class TestQueries:
    """Tests for query commands."""

    def test_count_no_model(self, repl):
        """Test count command with no model loaded."""
        repl.cmd_count(["test"])

    def test_count_pattern(self, repl_with_model):
        """Test count command counts occurrences."""
        repl_with_model.cmd_count(["cat"])

    def test_count_no_args(self, repl_with_model):
        """Test count command with no arguments."""
        repl_with_model.cmd_count([])

    def test_search_no_model(self, repl):
        """Test search command with no model loaded."""
        repl.cmd_search(["test"])

    def test_search_pattern(self, repl_with_model):
        """Test search command finds occurrences."""
        repl_with_model.cmd_search(["cat"])

    def test_search_with_limit(self, repl_with_model):
        """Test search command with limit."""
        repl_with_model.cmd_search(["the", "-n", "2"])

    def test_search_no_matches(self, repl_with_model):
        """Test search command with no matches."""
        repl_with_model.cmd_search(["xyz123"])


# ============================================================================
# Inference Tests
# ============================================================================

class TestInference:
    """Tests for inference commands."""

    def test_predict_no_model(self, repl):
        """Test predict command with no model loaded."""
        repl.cmd_predict(["test"])

    def test_predict_returns_probabilities(self, repl_with_model):
        """Test predict command returns probabilities."""
        repl_with_model.cmd_predict(["The", "cat"])

    def test_predict_with_top_k(self, repl_with_model):
        """Test predict command with custom top_k."""
        repl_with_model.cmd_predict(["The", "-n", "5"])

    def test_predict_bytes_mode(self, repl_with_model):
        """Test predict command with bytes display."""
        repl_with_model.cmd_predict(["cat", "-b"])

    def test_complete_no_model(self, repl):
        """Test complete command with no model loaded."""
        repl.cmd_complete(["test"])

    def test_complete_generates_text(self, repl_with_model):
        """Test complete command generates text."""
        repl_with_model.cmd_complete(["The", "cat"])

    def test_complete_with_max_tokens(self, repl_with_model):
        """Test complete command with max tokens."""
        repl_with_model.cmd_complete(["The", "-n", "10"])

    def test_complete_with_temperature(self, repl_with_model):
        """Test complete command with temperature."""
        repl_with_model.cmd_complete(["cat", "-t", "0.5"])

    def test_sample_no_model(self, repl):
        """Test sample command with no model loaded."""
        repl.cmd_sample(["test"])

    def test_sample_multiple(self, repl_with_model):
        """Test sample command generates multiple completions."""
        repl_with_model.cmd_sample(["The", "-n", "3", "-l", "10"])


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Tests for configuration commands."""

    def test_config_shows_settings(self, repl):
        """Test config command shows current settings."""
        repl.cmd_config([])

    def test_set_temperature(self, repl):
        """Test set temperature command."""
        repl.cmd_set(["temperature", "0.5"])
        assert repl.temperature == 0.5

    def test_set_top_k(self, repl):
        """Test set top_k command."""
        repl.cmd_set(["top_k", "100"])
        assert repl.top_k == 100

    def test_set_max_length(self, repl):
        """Test set max_length command."""
        repl.cmd_set(["max_length", "20"])
        assert repl.max_length == 20

    def test_set_max_length_none(self, repl):
        """Test set max_length to none."""
        repl.cmd_set(["max_length", "20"])
        repl.cmd_set(["max_length", "none"])
        assert repl.max_length is None

    def test_set_smoothing(self, repl):
        """Test set smoothing command."""
        repl.cmd_set(["smoothing", "0.1"])
        assert repl.smoothing == 0.1

    def test_set_weight(self, repl):
        """Test set weight command."""
        repl.cmd_set(["weight", "quadratic"])
        assert repl.weight_function == "quadratic"

    def test_set_weight_none(self, repl):
        """Test set weight to none."""
        repl.cmd_set(["weight", "linear"])
        repl.cmd_set(["weight", "none"])
        assert repl.weight_function is None

    def test_set_unknown_parameter(self, repl):
        """Test set with unknown parameter."""
        repl.cmd_set(["unknown", "value"])


# ============================================================================
# Command Execution Tests
# ============================================================================

class TestCommandExecution:
    """Tests for command execution."""

    def test_execute_unknown_command(self, repl):
        """Test executing unknown command."""
        repl.execute("unknowncommand")

    def test_execute_help(self, repl):
        """Test executing help command."""
        repl.execute("help")

    def test_execute_with_quoted_args(self, repl_with_model):
        """Test executing command with quoted arguments."""
        repl_with_model.execute('count "the cat"')

    def test_execute_empty_command(self, repl):
        """Test executing empty command."""
        repl.execute("")


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for full REPL workflows."""

    def test_build_use_predict_workflow(self, temp_models_dir):
        """Test complete workflow: build, use, predict."""
        repl = InfinigramREPL(models_dir=temp_models_dir)

        # Create corpus file
        corpus_file = temp_models_dir / "corpus.txt"
        corpus_file.write_text("Hello world. Hello there.")

        # Build model
        repl.cmd_build([str(corpus_file), "workflow-test"])
        assert repl.model is not None

        # Predict
        repl.cmd_predict(["Hello"])

        # Complete
        repl.cmd_complete(["Hello", "-n", "10"])

    def test_multiple_models(self, temp_models_dir):
        """Test working with multiple models."""
        repl = InfinigramREPL(models_dir=temp_models_dir)

        # Build two models
        corpus1 = temp_models_dir / "c1.txt"
        corpus1.write_text("First corpus text")
        corpus2 = temp_models_dir / "c2.txt"
        corpus2.write_text("Second corpus text")

        repl.cmd_build([str(corpus1), "model1"])
        repl.cmd_unload([])

        repl.cmd_build([str(corpus2), "model2"])

        # Switch between models
        repl.cmd_use(["model1"])
        assert repl.model_name == "model1"

        repl.cmd_use(["model2"])
        assert repl.model_name == "model2"

    def test_config_affects_inference(self, repl_with_model):
        """Test configuration affects inference."""
        # Set low temperature (deterministic)
        repl_with_model.cmd_set(["temperature", "0"])

        # Multiple completes should give same result
        repl_with_model.cmd_complete(["The", "-n", "5"])


# ============================================================================
# Navigation Tests
# ============================================================================

class TestNavigation:
    """Tests for Unix-style navigation commands (pwd, cd, ls)."""

    def test_pwd_no_model(self, repl):
        """Test pwd with no model loaded shows root."""
        repl.cmd_pwd([])
        # Should not raise

    def test_pwd_with_model(self, repl_with_model):
        """Test pwd with model loaded shows model info."""
        repl_with_model.cmd_pwd([])
        assert repl_with_model.model_name == "test-model"

    def test_ls_is_alias_for_models(self, repl):
        """Test ls is an alias for models."""
        repl.cmd_ls([])
        # Should not raise

    def test_cd_loads_model(self, repl, temp_models_dir):
        """Test cd <model> loads a model."""
        corpus = "Hello world"
        model_path = temp_models_dir / "nav-test"
        Infinigram.build(corpus, str(model_path), verbose=False)

        repl.cmd_cd(["nav-test"])

        assert repl.model is not None
        assert repl.model_name == "nav-test"

    def test_cd_with_leading_slash(self, repl, temp_models_dir):
        """Test cd /model strips the leading slash."""
        corpus = "Hello world"
        model_path = temp_models_dir / "slash-test"
        Infinigram.build(corpus, str(model_path), verbose=False)

        repl.cmd_cd(["/slash-test"])

        assert repl.model_name == "slash-test"

    def test_cd_dotdot_unloads_model(self, repl_with_model):
        """Test cd .. unloads the current model."""
        assert repl_with_model.model is not None

        repl_with_model.cmd_cd([".."])

        assert repl_with_model.model is None
        assert repl_with_model.model_name is None

    def test_cd_dotdot_at_root(self, repl):
        """Test cd .. at root shows message."""
        repl.cmd_cd([".."])
        # Should not raise, model still None
        assert repl.model is None

    def test_cd_tilde_unloads_model(self, repl_with_model):
        """Test cd ~ unloads the current model."""
        assert repl_with_model.model is not None

        repl_with_model.cmd_cd(["~"])

        assert repl_with_model.model is None

    def test_cd_root_unloads_model(self, repl_with_model):
        """Test cd / unloads the current model."""
        assert repl_with_model.model is not None

        repl_with_model.cmd_cd(["/"])

        assert repl_with_model.model is None

    def test_cd_no_args_unloads_model(self, repl_with_model):
        """Test cd with no args unloads the current model."""
        assert repl_with_model.model is not None

        repl_with_model.cmd_cd([])

        assert repl_with_model.model is None

    def test_cd_dash_switches_to_previous(self, temp_models_dir):
        """Test cd - switches to the previous model."""
        # Build two models
        corpus1 = "First corpus"
        corpus2 = "Second corpus"
        Infinigram.build(corpus1, str(temp_models_dir / "model1"), verbose=False)
        Infinigram.build(corpus2, str(temp_models_dir / "model2"), verbose=False)

        repl = InfinigramREPL(models_dir=temp_models_dir)

        # Load first model
        repl.cmd_cd(["model1"])
        assert repl.model_name == "model1"

        # Switch to second model
        repl.cmd_cd(["model2"])
        assert repl.model_name == "model2"
        assert repl.prev_model == "model1"

        # Switch back with cd -
        repl.cmd_cd(["-"])
        assert repl.model_name == "model1"
        assert repl.prev_model == "model2"

        # Switch again
        repl.cmd_cd(["-"])
        assert repl.model_name == "model2"
        assert repl.prev_model == "model1"

    def test_cd_dash_no_previous(self, repl):
        """Test cd - with no previous model."""
        repl.cmd_cd(["-"])
        # Should not raise, just print message
        assert repl.model is None

    def test_cd_nonexistent_model(self, repl):
        """Test cd to nonexistent model."""
        repl.cmd_cd(["nonexistent"])
        assert repl.model is None

    def test_prev_model_set_on_unload(self, repl_with_model):
        """Test prev_model is set when unloading."""
        model_name = repl_with_model.model_name

        repl_with_model.cmd_cd([".."])

        assert repl_with_model.prev_model == model_name

    def test_navigation_workflow(self, temp_models_dir):
        """Test complete navigation workflow."""
        # Build models
        Infinigram.build("First", str(temp_models_dir / "first"), verbose=False)
        Infinigram.build("Second", str(temp_models_dir / "second"), verbose=False)

        repl = InfinigramREPL(models_dir=temp_models_dir)

        # pwd at root
        repl.cmd_pwd([])
        assert repl.model is None

        # ls to see models
        repl.cmd_ls([])

        # cd to first
        repl.cmd_cd(["first"])
        assert repl.model_name == "first"

        # pwd shows model
        repl.cmd_pwd([])

        # cd to second
        repl.cmd_cd(["second"])
        assert repl.model_name == "second"

        # cd - back to first
        repl.cmd_cd(["-"])
        assert repl.model_name == "first"

        # cd .. to unload
        repl.cmd_cd([".."])
        assert repl.model is None

        # cd - back to first
        repl.cmd_cd(["-"])
        assert repl.model_name == "first"
