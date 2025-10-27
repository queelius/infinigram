#!/usr/bin/env python3
"""
Tests for REPL navigation commands (pwd, cd).
"""

import pytest
import tempfile
from pathlib import Path
from infinigram.repl import InfinigramREPL


@pytest.fixture
def repl():
    """Create a temporary REPL for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test REPL with temporary storage
        test_repl = InfinigramREPL()
        test_repl.storage_dir = Path(tmpdir) / "datasets"
        test_repl.storage_dir.mkdir(parents=True, exist_ok=True)

        # Re-initialize VFS with temp storage
        from infinigram.vfs import VirtualFilesystem
        test_repl.vfs = VirtualFilesystem(test_repl.storage_dir)

        # Create test datasets
        math = test_repl.vfs.create_dataset("math")
        math.add_document("Addition is combining numbers")
        math.add_document("Subtraction is inverse of addition")

        another = test_repl.vfs.create_dataset("another")
        another.add_document("Test document")

        yield test_repl
        test_repl.vfs.close_all()


class TestPwd:
    """Test pwd command."""

    def test_pwd_at_root(self, repl, capsys):
        """Test pwd at root directory."""
        repl.cmd_pwd([])
        captured = capsys.readouterr()
        assert captured.out.strip() == "/"

    def test_pwd_after_cd(self, repl, capsys):
        """Test pwd after changing directory."""
        repl.cmd_cd(["/proj"])
        repl.cmd_pwd([])
        captured = capsys.readouterr()
        assert "/proj" in captured.out


class TestCd:
    """Test cd command."""

    def test_cd_to_root(self, repl):
        """Test cd to root."""
        repl.cmd_cd(["/proj"])  # Go somewhere else first
        repl.cmd_cd(["/"])
        assert repl.vfs.cwd == "/"

    def test_cd_to_proj(self, repl):
        """Test cd to projections directory."""
        repl.cmd_cd(["/proj"])
        assert repl.vfs.cwd == "/proj"

    def test_cd_to_dataset(self, repl):
        """Test cd to a dataset."""
        repl.cmd_cd(["/math"])
        assert repl.vfs.cwd == "/math"

    def test_cd_with_no_args(self, repl):
        """Test cd with no arguments goes to root."""
        repl.cmd_cd(["/proj"])
        repl.cmd_cd([])
        assert repl.vfs.cwd == "/"

    def test_cd_parent(self, repl):
        """Test cd to parent directory."""
        repl.cmd_cd(["/math"])
        repl.cmd_cd([".."])
        assert repl.vfs.cwd == "/"

    def test_cd_home(self, repl):
        """Test cd to home (~)."""
        repl.cmd_cd(["/proj"])
        repl.cmd_cd(["~"])
        assert repl.vfs.cwd == "/"

    def test_cd_previous(self, repl):
        """Test cd to previous directory (-)."""
        repl.cmd_cd(["/math"])
        repl.cmd_cd(["/another"])
        repl.cmd_cd(["-"])
        assert repl.vfs.cwd == "/math"

    def test_cd_relative_path(self, repl):
        """Test cd with relative path."""
        repl.cmd_cd(["/"])
        repl.cmd_cd(["math"])
        assert repl.vfs.cwd == "/math"

    def test_cd_to_nonexistent_dataset(self, repl, capsys):
        """Test cd to nonexistent dataset shows error."""
        repl.cmd_cd(["/nonexistent"])
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "nonexistent" in captured.out.lower()

    def test_cd_to_document_fails(self, repl, capsys):
        """Test that cd to a document fails."""
        repl.cmd_cd(["/math/0"])
        captured = capsys.readouterr()
        assert "not a directory" in captured.out.lower()


class TestPrompt:
    """Test prompt shows current directory."""

    def test_prompt_shows_cwd(self, repl):
        """Test that prompt string includes current directory."""
        assert repl.vfs.cwd == "/"

        repl.cmd_cd(["/proj"])
        assert repl.vfs.cwd == "/proj"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
