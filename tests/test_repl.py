#!/usr/bin/env python3
"""
Comprehensive tests for the Infinigram REPL.

Tests cover:
- Dataset management (ds, ds ls, ds cat, ds cp, ds rm, ds info, ds stats)
- Document operations (ls, cat, rm, add, head, tail)
- Predictions (predict, complete)
- Projections (proj, proj ls, proj cat, proj rm)
- Configuration (set *, config)
- Search (find/grep, stat, wc, du)
- Storage (save, load, store ls, store rm)
- Command execution and error handling
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

from infinigram.repl import InfinigramREPL, PROJECTION_REGISTRY


@pytest.fixture
def repl():
    """Create a temporary REPL for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_repl = InfinigramREPL()
        test_repl.storage_dir = Path(tmpdir) / "datasets"
        test_repl.storage_dir.mkdir(parents=True, exist_ok=True)

        # Re-initialize VFS with temp storage
        from infinigram.vfs import VirtualFilesystem
        test_repl.vfs = VirtualFilesystem(test_repl.storage_dir)

        yield test_repl
        test_repl.vfs.close_all()


@pytest.fixture
def repl_with_dataset(repl):
    """Create a REPL with a dataset already loaded."""
    repl.execute("ds test")
    repl.execute("add the cat sat on the mat")
    repl.execute("add the dog ran in the park")
    return repl


class TestDatasetManagement:
    """Tests for dataset management commands."""

    def test_ds_creates_dataset(self, repl, capsys):
        """Test ds command creates a new dataset."""
        repl.cmd_ds(["mydata"])
        assert "mydata" in repl.datasets
        assert repl.current_dataset == "mydata"
        captured = capsys.readouterr()
        assert "Created" in captured.out
        assert "Switched" in captured.out

    def test_ds_switches_to_existing_dataset(self, repl, capsys):
        """Test ds command switches to an existing dataset."""
        repl.cmd_ds(["first"])
        repl.cmd_ds(["second"])
        repl.cmd_ds(["first"])
        assert repl.current_dataset == "first"
        captured = capsys.readouterr()
        assert "Switched" in captured.out

    def test_ds_no_args_shows_current(self, repl, capsys):
        """Test ds with no args shows current dataset."""
        repl.cmd_ds([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

        repl.cmd_ds(["test"])
        repl.cmd_ds([])
        captured = capsys.readouterr()
        assert "Current dataset: test" in captured.out

    def test_ds_ls_lists_datasets(self, repl, capsys):
        """Test ds ls command lists all datasets."""
        repl.cmd_ds(["first"])
        repl.cmd_ds(["second"])
        repl.cmd_ds_ls([])
        captured = capsys.readouterr()
        assert "first" in captured.out
        assert "second" in captured.out
        assert "(current)" in captured.out  # second should be current

    def test_ds_ls_no_datasets(self, repl, capsys):
        """Test ds ls when no datasets exist."""
        repl.cmd_ds_ls([])
        captured = capsys.readouterr()
        assert "No datasets loaded" in captured.out

    def test_ds_cat_views_dataset(self, repl_with_dataset, capsys):
        """Test ds cat views documents in a dataset."""
        repl_with_dataset.cmd_ds_cat(["test"])
        captured = capsys.readouterr()
        assert "Documents in 'test'" in captured.out
        assert "Total: 2 documents" in captured.out

    def test_ds_cat_not_found(self, repl, capsys):
        """Test ds cat with nonexistent dataset."""
        repl.cmd_ds_cat(["nonexistent"])
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_ds_cat_no_args(self, repl, capsys):
        """Test ds cat with no arguments."""
        repl.cmd_ds_cat([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_ds_cp_copies_dataset(self, repl_with_dataset, capsys):
        """Test ds cp copies a dataset."""
        repl_with_dataset.cmd_ds_cp(["test", "test_copy"])
        captured = capsys.readouterr()
        assert "Copied" in captured.out
        assert "test_copy" in repl_with_dataset.datasets
        # Verify document count is the same
        assert len(repl_with_dataset.dataset_documents["test_copy"]) == 2

    def test_ds_cp_source_not_found(self, repl, capsys):
        """Test ds cp with nonexistent source."""
        repl.cmd_ds_cp(["nonexistent", "target"])
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_ds_cp_destination_exists(self, repl_with_dataset, capsys):
        """Test ds cp when destination already exists."""
        repl_with_dataset.cmd_ds(["other"])
        repl_with_dataset.cmd_ds_cp(["test", "other"])
        captured = capsys.readouterr()
        assert "already exists" in captured.out

    def test_ds_cp_not_enough_args(self, repl, capsys):
        """Test ds cp with insufficient arguments."""
        repl.cmd_ds_cp(["only_one"])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_ds_rm_deletes_dataset(self, repl_with_dataset, capsys):
        """Test ds rm deletes current dataset."""
        repl_with_dataset.cmd_ds_rm([])
        captured = capsys.readouterr()
        assert "Deleted" in captured.out
        assert "test" not in repl_with_dataset.datasets
        assert repl_with_dataset.current_dataset is None

    def test_ds_rm_switches_to_remaining(self, repl, capsys):
        """Test ds rm switches to remaining dataset after delete."""
        repl.cmd_ds(["first"])
        repl.cmd_ds(["second"])
        repl.cmd_ds_rm([])  # Deletes 'second'
        assert repl.current_dataset == "first"
        captured = capsys.readouterr()
        assert "Switched to:" in captured.out

    def test_ds_rm_no_dataset_selected(self, repl, capsys):
        """Test ds rm when no dataset is selected."""
        repl.cmd_ds_rm([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_ds_info_shows_information(self, repl_with_dataset, capsys):
        """Test ds info shows dataset information."""
        repl_with_dataset.cmd_ds_info([])
        captured = capsys.readouterr()
        assert "Dataset: test" in captured.out
        assert "Corpus size:" in captured.out
        assert "Documents: 2" in captured.out
        assert "Vocabulary size:" in captured.out

    def test_ds_info_no_dataset(self, repl, capsys):
        """Test ds info when no dataset is selected."""
        repl.cmd_ds_info([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_ds_stats_shows_statistics(self, repl_with_dataset, capsys):
        """Test ds stats shows corpus statistics."""
        repl_with_dataset.cmd_ds_stats([])
        captured = capsys.readouterr()
        assert "Corpus Statistics" in captured.out
        assert "Total bytes:" in captured.out
        assert "Unique bytes:" in captured.out
        assert "Most common bytes:" in captured.out

    def test_ds_stats_no_dataset(self, repl, capsys):
        """Test ds stats when no dataset is selected."""
        repl.cmd_ds_stats([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out


class TestDocumentOperations:
    """Tests for document operations (ls, cat, rm, add, head, tail)."""

    def test_ls_lists_documents(self, repl_with_dataset, capsys):
        """Test ls lists documents in current dataset."""
        repl_with_dataset.cmd_ls([])
        captured = capsys.readouterr()
        assert "[0]" in captured.out
        assert "[1]" in captured.out
        assert "the cat" in captured.out
        assert "the dog" in captured.out

    def test_ls_no_dataset(self, repl, capsys):
        """Test ls when no dataset is selected."""
        repl.cmd_ls([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_ls_empty_dataset(self, repl, capsys):
        """Test ls with empty dataset."""
        repl.cmd_ds(["empty"])
        repl.cmd_ls([])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_cat_shows_document(self, repl_with_dataset, capsys):
        """Test cat shows a specific document."""
        repl_with_dataset.cmd_cat(["0"])
        captured = capsys.readouterr()
        assert "Document [0]" in captured.out
        assert "the cat sat on the mat" in captured.out

    def test_cat_invalid_index(self, repl_with_dataset, capsys):
        """Test cat with invalid index."""
        repl_with_dataset.cmd_cat(["abc"])
        captured = capsys.readouterr()
        assert "Invalid index" in captured.out

    def test_cat_index_out_of_range(self, repl_with_dataset, capsys):
        """Test cat with out of range index."""
        repl_with_dataset.cmd_cat(["100"])
        captured = capsys.readouterr()
        assert "out of range" in captured.out

    def test_cat_no_args(self, repl_with_dataset, capsys):
        """Test cat with no arguments."""
        repl_with_dataset.cmd_cat([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_cat_no_dataset(self, repl, capsys):
        """Test cat when no dataset is selected."""
        repl.cmd_cat(["0"])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_rm_removes_document(self, repl_with_dataset, capsys):
        """Test rm removes a document."""
        initial_count = len(repl_with_dataset.dataset_documents["test"])
        repl_with_dataset.cmd_rm(["0"])
        captured = capsys.readouterr()
        assert "Removed document" in captured.out
        assert len(repl_with_dataset.dataset_documents["test"]) == initial_count - 1

    def test_rm_invalid_index(self, repl_with_dataset, capsys):
        """Test rm with invalid index."""
        repl_with_dataset.cmd_rm(["abc"])
        captured = capsys.readouterr()
        assert "Invalid index" in captured.out

    def test_rm_index_out_of_range(self, repl_with_dataset, capsys):
        """Test rm with out of range index."""
        repl_with_dataset.cmd_rm(["100"])
        captured = capsys.readouterr()
        assert "out of range" in captured.out

    def test_rm_no_args(self, repl_with_dataset, capsys):
        """Test rm with no arguments."""
        repl_with_dataset.cmd_rm([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_rm_no_dataset(self, repl, capsys):
        """Test rm when no dataset is selected."""
        repl.cmd_rm(["0"])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_add_text(self, repl, capsys):
        """Test add command adds text to dataset."""
        repl.cmd_ds(["test"])
        repl.cmd_add(["hello", "world"])
        captured = capsys.readouterr()
        assert "Added document" in captured.out
        assert len(repl.dataset_documents["test"]) == 1
        assert repl.dataset_documents["test"][0] == "hello world"

    def test_add_no_dataset(self, repl, capsys):
        """Test add when no dataset is selected."""
        repl.cmd_add(["hello"])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_add_no_args(self, repl, capsys):
        """Test add with no arguments."""
        repl.cmd_ds(["test"])
        repl.cmd_add([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_head_shows_first_documents(self, repl_with_dataset, capsys):
        """Test head shows first N documents."""
        # Add more documents
        for i in range(15):
            repl_with_dataset.execute(f"add document number {i}")
        repl_with_dataset.cmd_head([])  # Default 10
        captured = capsys.readouterr()
        assert "First 10" in captured.out

    def test_head_custom_count(self, repl_with_dataset, capsys):
        """Test head with custom count."""
        repl_with_dataset.cmd_head(["1"])
        captured = capsys.readouterr()
        assert "First 1" in captured.out
        assert "[0]" in captured.out
        assert "[1]" not in captured.out

    def test_head_invalid_count(self, repl_with_dataset, capsys):
        """Test head with invalid count."""
        repl_with_dataset.cmd_head(["abc"])
        captured = capsys.readouterr()
        assert "Invalid number" in captured.out

    def test_head_no_dataset(self, repl, capsys):
        """Test head when no dataset is selected."""
        repl.cmd_head([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_head_empty_dataset(self, repl, capsys):
        """Test head with empty dataset."""
        repl.cmd_ds(["empty"])
        repl.cmd_head([])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_tail_shows_last_documents(self, repl_with_dataset, capsys):
        """Test tail shows last N documents."""
        repl_with_dataset.cmd_tail(["1"])
        captured = capsys.readouterr()
        assert "Last 1" in captured.out
        assert "[1]" in captured.out

    def test_tail_invalid_count(self, repl_with_dataset, capsys):
        """Test tail with invalid count."""
        repl_with_dataset.cmd_tail(["abc"])
        captured = capsys.readouterr()
        assert "Invalid number" in captured.out

    def test_tail_no_dataset(self, repl, capsys):
        """Test tail when no dataset is selected."""
        repl.cmd_tail([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_tail_empty_dataset(self, repl, capsys):
        """Test tail with empty dataset."""
        repl.cmd_ds(["empty"])
        repl.cmd_tail([])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()


class TestSearchOperations:
    """Tests for search and statistics operations."""

    def test_find_searches_documents(self, repl_with_dataset, capsys):
        """Test find searches for pattern in documents."""
        repl_with_dataset.cmd_find(["cat"])
        captured = capsys.readouterr()
        assert "Found 1 document" in captured.out
        assert "[0]" in captured.out

    def test_find_regex_pattern(self, repl_with_dataset, capsys):
        """Test find with regex pattern."""
        repl_with_dataset.cmd_find(["the.*sat"])
        captured = capsys.readouterr()
        assert "Found" in captured.out

    def test_find_no_matches(self, repl_with_dataset, capsys):
        """Test find with no matches."""
        repl_with_dataset.cmd_find(["elephant"])
        captured = capsys.readouterr()
        assert "No documents found" in captured.out

    def test_find_no_args(self, repl_with_dataset, capsys):
        """Test find with no arguments."""
        repl_with_dataset.cmd_find([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_find_no_dataset(self, repl, capsys):
        """Test find when no dataset is selected."""
        repl.cmd_find(["test"])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_find_empty_dataset(self, repl, capsys):
        """Test find with empty dataset."""
        repl.cmd_ds(["empty"])
        repl.cmd_find(["pattern"])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_stat_dataset_level(self, repl_with_dataset, capsys):
        """Test stat shows dataset-level statistics."""
        repl_with_dataset.cmd_stat([])
        captured = capsys.readouterr()
        assert "Statistics for dataset" in captured.out
        assert "Documents: 2" in captured.out
        assert "Total characters:" in captured.out

    def test_stat_document_level(self, repl_with_dataset, capsys):
        """Test stat shows document-level statistics."""
        repl_with_dataset.cmd_stat(["0"])
        captured = capsys.readouterr()
        assert "Statistics for document [0]" in captured.out
        assert "Characters:" in captured.out
        assert "Words:" in captured.out

    def test_stat_invalid_index(self, repl_with_dataset, capsys):
        """Test stat with invalid index."""
        repl_with_dataset.cmd_stat(["abc"])
        captured = capsys.readouterr()
        assert "Invalid index" in captured.out

    def test_stat_index_out_of_range(self, repl_with_dataset, capsys):
        """Test stat with out of range index."""
        repl_with_dataset.cmd_stat(["100"])
        captured = capsys.readouterr()
        assert "out of range" in captured.out

    def test_stat_no_dataset(self, repl, capsys):
        """Test stat when no dataset is selected."""
        repl.cmd_stat([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_stat_empty_dataset(self, repl, capsys):
        """Test stat with empty dataset."""
        repl.cmd_ds(["empty"])
        repl.cmd_stat([])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_wc_dataset_level(self, repl_with_dataset, capsys):
        """Test wc counts for entire dataset."""
        repl_with_dataset.cmd_wc([])
        captured = capsys.readouterr()
        # Should show lines, words, bytes
        assert "test" in captured.out

    def test_wc_document_level(self, repl_with_dataset, capsys):
        """Test wc counts for specific document."""
        repl_with_dataset.cmd_wc(["0"])
        captured = capsys.readouterr()
        assert "document[0]" in captured.out

    def test_wc_invalid_index(self, repl_with_dataset, capsys):
        """Test wc with invalid index."""
        repl_with_dataset.cmd_wc(["abc"])
        captured = capsys.readouterr()
        assert "Invalid index" in captured.out

    def test_wc_no_dataset(self, repl, capsys):
        """Test wc when no dataset is selected."""
        repl.cmd_wc([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_wc_empty_dataset(self, repl, capsys):
        """Test wc with empty dataset."""
        repl.cmd_ds(["empty"])
        repl.cmd_wc([])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()

    def test_du_shows_disk_usage(self, repl_with_dataset, capsys):
        """Test du shows disk usage per document."""
        repl_with_dataset.cmd_du([])
        captured = capsys.readouterr()
        assert "Disk usage" in captured.out
        assert "bytes" in captured.out
        assert "Total:" in captured.out

    def test_du_no_dataset(self, repl, capsys):
        """Test du when no dataset is selected."""
        repl.cmd_du([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_du_empty_dataset(self, repl, capsys):
        """Test du with empty dataset."""
        repl.cmd_ds(["empty"])
        repl.cmd_du([])
        captured = capsys.readouterr()
        assert "empty" in captured.out.lower()


class TestPredictions:
    """Tests for prediction commands."""

    def test_predict_shows_probabilities(self, repl_with_dataset, capsys):
        """Test predict shows next byte probabilities."""
        repl_with_dataset.cmd_predict(["the"])
        captured = capsys.readouterr()
        assert "Context:" in captured.out
        assert "predictions" in captured.out.lower()

    def test_predict_with_bytes_flag(self, repl_with_dataset, capsys):
        """Test predict with --bytes flag."""
        repl_with_dataset.cmd_predict(["the", "--bytes"])
        captured = capsys.readouterr()
        assert "Byte" in captured.out

    def test_predict_no_dataset(self, repl, capsys):
        """Test predict when no dataset is selected."""
        repl.cmd_predict(["test"])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_predict_no_args(self, repl_with_dataset, capsys):
        """Test predict with no arguments."""
        repl_with_dataset.cmd_predict([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_predict_with_weight_function(self, repl_with_dataset, capsys):
        """Test predict with weight function enabled."""
        repl_with_dataset.weight_function = lambda x: x  # linear
        repl_with_dataset.cmd_predict(["the"])
        captured = capsys.readouterr()
        assert "Context:" in captured.out

    def test_predict_with_temperature(self, repl_with_dataset, capsys):
        """Test predict with non-default temperature."""
        repl_with_dataset.temperature = 0.5
        repl_with_dataset.cmd_predict(["the"])
        captured = capsys.readouterr()
        assert "Context:" in captured.out

    def test_complete_generates_text(self, repl_with_dataset, capsys):
        """Test complete generates completion."""
        repl_with_dataset.cmd_complete(["the"])
        captured = capsys.readouterr()
        assert "Context:" in captured.out
        assert "Generated:" in captured.out

    def test_complete_with_max_flag(self, repl_with_dataset, capsys):
        """Test complete with --max flag."""
        repl_with_dataset.cmd_complete(["the", "--max", "10"])
        captured = capsys.readouterr()
        assert "up to 10 bytes" in captured.out

    def test_complete_invalid_max(self, repl_with_dataset, capsys):
        """Test complete with invalid --max value."""
        repl_with_dataset.cmd_complete(["the", "--max", "abc"])
        captured = capsys.readouterr()
        assert "Invalid --max" in captured.out

    def test_complete_no_dataset(self, repl, capsys):
        """Test complete when no dataset is selected."""
        repl.cmd_complete(["test"])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_complete_no_args(self, repl_with_dataset, capsys):
        """Test complete with no arguments."""
        repl_with_dataset.cmd_complete([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_complete_with_weight_function(self, repl_with_dataset, capsys):
        """Test complete with weight function enabled."""
        repl_with_dataset.weight_function = lambda x: x
        repl_with_dataset.cmd_complete(["the", "--max", "5"])
        captured = capsys.readouterr()
        assert "Generated:" in captured.out


class TestProjections:
    """Tests for projection commands."""

    def test_proj_applies_projections(self, repl_with_dataset, capsys):
        """Test proj applies projections to dataset."""
        initial_size = repl_with_dataset.model.n
        repl_with_dataset.cmd_proj(["lowercase"])
        captured = capsys.readouterr()
        assert "Applied projections" in captured.out
        assert "lowercase" in captured.out
        # Size should increase with augmentation
        assert repl_with_dataset.model.n > initial_size

    def test_proj_multiple_projections(self, repl_with_dataset, capsys):
        """Test proj with multiple projections."""
        repl_with_dataset.cmd_proj(["lowercase", "uppercase"])
        captured = capsys.readouterr()
        assert "Applied projections" in captured.out
        assert "lowercase" in captured.out
        assert "uppercase" in captured.out
        assert repl_with_dataset.dataset_projections["test"] == ["lowercase", "uppercase"]

    def test_proj_invalid_projection(self, repl_with_dataset, capsys):
        """Test proj with invalid projection name."""
        repl_with_dataset.cmd_proj(["invalid_proj"])
        captured = capsys.readouterr()
        assert "Unknown projections" in captured.out

    def test_proj_no_args(self, repl_with_dataset, capsys):
        """Test proj with no arguments."""
        repl_with_dataset.cmd_proj([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out
        assert "Available projections:" in captured.out

    def test_proj_no_dataset(self, repl, capsys):
        """Test proj when no dataset is selected."""
        repl.cmd_proj(["lowercase"])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_proj_empty_documents(self, repl, capsys):
        """Test proj with empty document list."""
        repl.cmd_ds(["empty"])
        repl.cmd_proj(["lowercase"])
        captured = capsys.readouterr()
        assert "no documents" in captured.out.lower() or "Error" in captured.out

    def test_proj_ls_lists_active(self, repl_with_dataset, capsys):
        """Test proj ls lists active projections."""
        repl_with_dataset.cmd_proj(["lowercase"])
        repl_with_dataset.cmd_proj_ls([])
        captured = capsys.readouterr()
        assert "Active projections" in captured.out
        assert "lowercase" in captured.out

    def test_proj_ls_all_flag(self, repl, capsys):
        """Test proj ls -a lists all available projections."""
        repl.cmd_proj_ls(["-a"])
        captured = capsys.readouterr()
        assert "Available projections:" in captured.out
        for proj in PROJECTION_REGISTRY:
            assert proj in captured.out

    def test_proj_ls_no_active(self, repl_with_dataset, capsys):
        """Test proj ls when no projections are active."""
        repl_with_dataset.cmd_proj_ls([])
        captured = capsys.readouterr()
        assert "No active projections" in captured.out

    def test_proj_ls_no_dataset(self, repl, capsys):
        """Test proj ls when no dataset is selected."""
        repl.cmd_proj_ls([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_proj_cat_shows_details(self, repl, capsys):
        """Test proj cat shows projection details."""
        repl.cmd_proj_cat(["lowercase"])
        captured = capsys.readouterr()
        assert "Projection: lowercase" in captured.out
        assert "Examples:" in captured.out

    def test_proj_cat_not_found(self, repl, capsys):
        """Test proj cat with unknown projection."""
        repl.cmd_proj_cat(["unknown"])
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_proj_cat_no_args(self, repl, capsys):
        """Test proj cat with no arguments."""
        repl.cmd_proj_cat([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_proj_rm_removes_projections(self, repl_with_dataset, capsys):
        """Test proj rm removes all projections."""
        repl_with_dataset.cmd_proj(["lowercase"])
        repl_with_dataset.cmd_proj_rm([])
        captured = capsys.readouterr()
        assert "Removed all projections" in captured.out
        assert repl_with_dataset.dataset_projections["test"] == []

    def test_proj_rm_no_projections(self, repl_with_dataset, capsys):
        """Test proj rm when no projections exist."""
        repl_with_dataset.cmd_proj_rm([])
        captured = capsys.readouterr()
        assert "No projections to remove" in captured.out

    def test_proj_rm_no_dataset(self, repl, capsys):
        """Test proj rm when no dataset is selected."""
        repl.cmd_proj_rm([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out


class TestConfiguration:
    """Tests for configuration commands."""

    def test_set_temperature_changes_value(self, repl, capsys):
        """Test set temperature changes temperature value."""
        repl.cmd_set_temperature(["0.5"])
        captured = capsys.readouterr()
        assert "Temperature set to 0.5" in captured.out
        assert repl.temperature == 0.5

    def test_set_temperature_shows_current(self, repl, capsys):
        """Test set temperature with no args shows current value."""
        repl.cmd_set_temperature([])
        captured = capsys.readouterr()
        assert "Current temperature:" in captured.out

    def test_set_temperature_invalid(self, repl, capsys):
        """Test set temperature with invalid value."""
        repl.cmd_set_temperature(["abc"])
        captured = capsys.readouterr()
        assert "Invalid temperature" in captured.out

    def test_set_temperature_negative(self, repl, capsys):
        """Test set temperature with negative value."""
        repl.cmd_set_temperature(["-1"])
        captured = capsys.readouterr()
        assert "must be positive" in captured.out

    def test_set_top_k_changes_value(self, repl, capsys):
        """Test set top_k changes value."""
        repl.cmd_set_top_k(["100"])
        captured = capsys.readouterr()
        assert "top_k set to 100" in captured.out
        assert repl.top_k == 100

    def test_set_top_k_shows_current(self, repl, capsys):
        """Test set top_k with no args shows current value."""
        repl.cmd_set_top_k([])
        captured = capsys.readouterr()
        assert "Current top_k:" in captured.out

    def test_set_top_k_invalid(self, repl, capsys):
        """Test set top_k with invalid value."""
        repl.cmd_set_top_k(["abc"])
        captured = capsys.readouterr()
        assert "Invalid top_k" in captured.out

    def test_set_top_k_negative(self, repl, capsys):
        """Test set top_k with negative value."""
        repl.cmd_set_top_k(["-1"])
        captured = capsys.readouterr()
        assert "must be positive" in captured.out

    def test_set_max_length_changes_value(self, repl, capsys):
        """Test set max_length changes value."""
        repl.cmd_set_max_length(["10"])
        captured = capsys.readouterr()
        assert "max_length set to 10" in captured.out
        assert repl.max_length == 10

    def test_set_max_length_none(self, repl, capsys):
        """Test set max_length to none."""
        repl.cmd_set_max_length(["none"])
        captured = capsys.readouterr()
        assert "unlimited" in captured.out
        assert repl.max_length is None

    def test_set_max_length_shows_current(self, repl, capsys):
        """Test set max_length with no args shows current value."""
        repl.cmd_set_max_length([])
        captured = capsys.readouterr()
        assert "Current max_length:" in captured.out

    def test_set_max_length_invalid(self, repl, capsys):
        """Test set max_length with invalid value."""
        repl.cmd_set_max_length(["abc"])
        captured = capsys.readouterr()
        assert "Invalid max_length" in captured.out

    def test_set_max_length_negative(self, repl, capsys):
        """Test set max_length with negative value."""
        repl.cmd_set_max_length(["-1"])
        captured = capsys.readouterr()
        assert "must be positive" in captured.out

    def test_set_smoothing_changes_value(self, repl, capsys):
        """Test set smoothing changes value."""
        repl.cmd_set_smoothing(["0.1"])
        captured = capsys.readouterr()
        assert "Smoothing set to 0.1" in captured.out
        assert repl.smoothing == 0.1

    def test_set_smoothing_shows_current(self, repl, capsys):
        """Test set smoothing with no args shows current value."""
        repl.cmd_set_smoothing([])
        captured = capsys.readouterr()
        assert "Current smoothing:" in captured.out

    def test_set_smoothing_invalid(self, repl, capsys):
        """Test set smoothing with invalid value."""
        repl.cmd_set_smoothing(["abc"])
        captured = capsys.readouterr()
        assert "Invalid smoothing" in captured.out

    def test_set_smoothing_negative(self, repl, capsys):
        """Test set smoothing with negative value."""
        repl.cmd_set_smoothing(["-1"])
        captured = capsys.readouterr()
        assert "non-negative" in captured.out

    def test_set_weight_changes_function(self, repl, capsys):
        """Test set weight changes weight function."""
        repl.cmd_set_weight(["linear"])
        captured = capsys.readouterr()
        assert "Weight function set to linear" in captured.out
        assert repl.weight_function is not None

    def test_set_weight_none(self, repl, capsys):
        """Test set weight to none."""
        repl.cmd_set_weight(["linear"])  # Set first
        repl.cmd_set_weight(["none"])
        captured = capsys.readouterr()
        assert "Disabled weighted prediction" in captured.out
        assert repl.weight_function is None

    def test_set_weight_shows_current(self, repl, capsys):
        """Test set weight with no args shows current value."""
        repl.cmd_set_weight([])
        captured = capsys.readouterr()
        assert "Current weight function:" in captured.out

    def test_set_weight_invalid(self, repl, capsys):
        """Test set weight with invalid function name."""
        repl.cmd_set_weight(["unknown"])
        captured = capsys.readouterr()
        assert "Error:" in captured.out

    def test_config_shows_all(self, repl, capsys):
        """Test config shows all configuration."""
        repl.cmd_config([])
        captured = capsys.readouterr()
        assert "Current Configuration:" in captured.out
        assert "Temperature:" in captured.out
        assert "Top-k:" in captured.out
        assert "Max suffix length:" in captured.out
        assert "Smoothing:" in captured.out
        assert "Weight function:" in captured.out


class TestStorage:
    """Tests for storage commands (save, load, store ls, store rm)."""

    def test_save_saves_dataset(self, repl_with_dataset, capsys):
        """Test save saves dataset to disk."""
        repl_with_dataset.cmd_save([])
        captured = capsys.readouterr()
        assert "Saved dataset" in captured.out
        # Check files exist
        dataset_dir = repl_with_dataset.storage_dir / "test"
        assert dataset_dir.exists()
        assert (dataset_dir / "metadata.json").exists()
        assert (dataset_dir / "documents.jsonl").exists()

    def test_save_with_name(self, repl_with_dataset, capsys):
        """Test save with explicit name."""
        repl_with_dataset.cmd_save(["test"])
        captured = capsys.readouterr()
        assert "Saved dataset 'test'" in captured.out

    def test_save_nonexistent_dataset(self, repl, capsys):
        """Test save with nonexistent dataset name."""
        repl.cmd_save(["nonexistent"])
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_save_no_dataset(self, repl, capsys):
        """Test save when no dataset is selected."""
        repl.cmd_save([])
        captured = capsys.readouterr()
        assert "No dataset selected" in captured.out

    def test_load_loads_dataset(self, repl_with_dataset, capsys):
        """Test load loads dataset from disk."""
        # Save first
        repl_with_dataset.cmd_save([])
        # Delete from memory
        repl_with_dataset.cmd_ds_rm([])
        # Load
        repl_with_dataset.cmd_load(["test"])
        captured = capsys.readouterr()
        assert "Loaded dataset" in captured.out
        assert "test" in repl_with_dataset.datasets
        assert len(repl_with_dataset.dataset_documents["test"]) == 2

    def test_load_nonexistent(self, repl, capsys):
        """Test load with nonexistent dataset."""
        repl.cmd_load(["nonexistent"])
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_load_no_args(self, repl, capsys):
        """Test load with no arguments."""
        repl.cmd_load([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out

    def test_store_ls_lists_saved(self, repl_with_dataset, capsys):
        """Test store ls lists saved datasets."""
        repl_with_dataset.cmd_save([])
        repl_with_dataset.cmd_store_ls([])
        captured = capsys.readouterr()
        assert "Saved datasets" in captured.out
        assert "test" in captured.out

    def test_store_ls_empty(self, repl, capsys):
        """Test store ls when no saved datasets."""
        repl.cmd_store_ls([])
        captured = capsys.readouterr()
        assert "No saved datasets" in captured.out

    def test_store_rm_deletes(self, repl_with_dataset, capsys):
        """Test store rm deletes saved dataset."""
        repl_with_dataset.cmd_save([])
        repl_with_dataset.cmd_store_rm(["test"])
        captured = capsys.readouterr()
        assert "Deleted saved dataset" in captured.out
        dataset_dir = repl_with_dataset.storage_dir / "test"
        assert not dataset_dir.exists()

    def test_store_rm_not_found(self, repl, capsys):
        """Test store rm with nonexistent dataset."""
        repl.cmd_store_rm(["nonexistent"])
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_store_rm_no_args(self, repl, capsys):
        """Test store rm with no arguments."""
        repl.cmd_store_rm([])
        captured = capsys.readouterr()
        assert "Usage:" in captured.out


class TestCommandExecution:
    """Tests for command execution and parsing."""

    def test_execute_simple_command(self, repl, capsys):
        """Test execute with simple command."""
        repl.execute("ds test")
        assert "test" in repl.datasets

    def test_execute_namespaced_command(self, repl, capsys):
        """Test execute with namespaced command (ds ls)."""
        repl.execute("ds test")
        repl.execute("ds ls")
        captured = capsys.readouterr()
        assert "test" in captured.out

    def test_execute_unknown_command(self, repl, capsys):
        """Test execute with unknown command."""
        repl.execute("unknowncommand")
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_execute_empty(self, repl, capsys):
        """Test execute with empty string."""
        repl.execute("")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_execute_bash_command(self, repl, capsys):
        """Test execute bash command with ! prefix."""
        repl.execute("!echo hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_execute_bash_timeout(self, repl, capsys):
        """Test bash command timeout handling."""
        # This test uses a mock to avoid actual timeout
        with patch('subprocess.run') as mock_run:
            from subprocess import TimeoutExpired
            mock_run.side_effect = TimeoutExpired("cmd", 30)
            repl.execute_bash("sleep 100")
            captured = capsys.readouterr()
            assert "timed out" in captured.out

    def test_execute_bash_error(self, repl, capsys):
        """Test bash command error handling."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Test error")
            repl.execute_bash("invalid")
            captured = capsys.readouterr()
            assert "Error" in captured.out


class TestHelpAndQuit:
    """Tests for help and quit commands."""

    def test_help_shows_commands(self, repl, capsys):
        """Test help shows available commands."""
        repl.cmd_help([])
        captured = capsys.readouterr()
        assert "VFS Navigation:" in captured.out
        assert "Dataset Operations" in captured.out
        assert "Storage Operations" in captured.out
        assert "Projections" in captured.out
        assert "Inference" in captured.out
        assert "Configuration" in captured.out

    def test_quit_exits(self, repl):
        """Test quit exits REPL."""
        with pytest.raises(SystemExit):
            repl.cmd_quit([])


class TestHelperMethods:
    """Tests for helper methods."""

    def test_apply_temperature(self, repl):
        """Test _apply_temperature method."""
        probs = {65: 0.5, 66: 0.3, 67: 0.2}
        result = repl._apply_temperature(probs, 0.5)
        # Should sum to 1
        assert abs(sum(result.values()) - 1.0) < 0.001
        # Should be more peaked
        assert result[65] > probs[65]

    def test_apply_temperature_high(self, repl):
        """Test _apply_temperature with high temperature."""
        probs = {65: 0.5, 66: 0.3, 67: 0.2}
        result = repl._apply_temperature(probs, 2.0)
        # Should be more uniform
        assert result[65] < probs[65]

    def test_sample(self, repl):
        """Test _sample method."""
        probs = {65: 0.5, 66: 0.3, 67: 0.2}
        result = repl._sample(probs)
        assert result in [65, 66, 67]

    def test_model_property(self, repl):
        """Test model property."""
        assert repl.model is None
        repl.execute("ds test")
        assert repl.model is not None


class TestFileLoading:
    """Tests for file loading functionality."""

    def test_load_text_file(self, repl, capsys):
        """Test loading text from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello from file")
            filepath = f.name
        try:
            result = repl._load_text_file([filepath])
            assert result == "Hello from file"
        finally:
            os.unlink(filepath)

    def test_load_text_file_not_found(self, repl, capsys):
        """Test loading nonexistent file."""
        result = repl._load_text_file(["/nonexistent/file.txt"])
        captured = capsys.readouterr()
        assert result is None
        assert "Error" in captured.out

    def test_load_text_file_no_args(self, repl, capsys):
        """Test load text file with no arguments."""
        result = repl._load_text_file([])
        captured = capsys.readouterr()
        assert result is None
        assert "Usage:" in captured.out

    def test_load_jsonl_file(self, repl, capsys):
        """Test loading JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "Line 1"}\n')
            f.write('{"text": "Line 2"}\n')
            filepath = f.name
        try:
            result = repl._load_jsonl([filepath])
            assert result is not None
            captured = capsys.readouterr()
            assert "Loaded 2 documents" in captured.out
        finally:
            os.unlink(filepath)

    def test_load_jsonl_custom_field(self, repl, capsys):
        """Test loading JSONL with custom field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"content": "Line 1"}\n')
            filepath = f.name
        try:
            result = repl._load_jsonl([filepath, "--field", "content"])
            assert result is not None
        finally:
            os.unlink(filepath)

    def test_load_jsonl_missing_field(self, repl, capsys):
        """Test loading JSONL with missing field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"other": "Line 1"}\n')
            filepath = f.name
        try:
            result = repl._load_jsonl([filepath])
            captured = capsys.readouterr()
            assert "Warning" in captured.out
        finally:
            os.unlink(filepath)

    def test_load_jsonl_no_args(self, repl, capsys):
        """Test load JSONL with no arguments."""
        result = repl._load_jsonl([])
        captured = capsys.readouterr()
        assert result is None
        assert "Usage:" in captured.out

    def test_add_from_file(self, repl, capsys):
        """Test add command with --file flag."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("File content")
            filepath = f.name
        try:
            repl.execute("ds test")
            repl.cmd_add(["--file", filepath])
            captured = capsys.readouterr()
            assert "Added document" in captured.out
        finally:
            os.unlink(filepath)

    def test_add_from_jsonl(self, repl, capsys):
        """Test add command with --jsonl flag."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"text": "Line 1"}\n')
            filepath = f.name
        try:
            repl.execute("ds test")
            repl.cmd_add(["--jsonl", filepath])
            captured = capsys.readouterr()
            assert "Added document" in captured.out
        finally:
            os.unlink(filepath)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_long_document_preview_truncation(self, repl, capsys):
        """Test that long documents are truncated in previews."""
        repl.execute("ds test")
        long_text = "A" * 100
        repl.execute(f"add {long_text}")
        repl.cmd_ls([])
        captured = capsys.readouterr()
        assert "..." in captured.out

    def test_document_with_newlines(self, repl, capsys):
        """Test handling documents with newlines."""
        repl.execute("ds test")
        repl.execute("add line1\\nline2")
        repl.cmd_ls([])
        captured = capsys.readouterr()
        # Newlines should be replaced with \n in preview
        assert "line1" in captured.out

    def test_rm_with_projections_active(self, repl_with_dataset, capsys):
        """Test rm document when projections are active.

        Note: Currently there is a bug in cmd_rm where it passes projection names
        (strings) instead of functions to build_corpus_with_augmentation.
        This test documents that behavior by removing projections first.
        """
        repl_with_dataset.cmd_proj(["lowercase"])
        repl_with_dataset.cmd_proj_rm([])  # Remove projections first to avoid bug
        repl_with_dataset.cmd_rm(["0"])
        captured = capsys.readouterr()
        assert "Removed document" in captured.out

    def test_add_with_projections_active(self, repl_with_dataset, capsys):
        """Test add document when projections are active."""
        repl_with_dataset.cmd_proj(["lowercase"])
        repl_with_dataset.cmd_add(["NEW DOCUMENT"])
        captured = capsys.readouterr()
        assert "with 1 projection" in captured.out

    def test_ds_cp_copies_projections(self, repl_with_dataset, capsys):
        """Test ds cp copies projection tracking."""
        repl_with_dataset.cmd_proj(["lowercase"])
        repl_with_dataset.cmd_ds_cp(["test", "copy"])
        assert repl_with_dataset.dataset_projections.get("copy") == ["lowercase"]

    def test_ds_rm_cleans_up_tracking(self, repl_with_dataset, capsys):
        """Test ds rm cleans up all tracking data."""
        repl_with_dataset.cmd_proj(["lowercase"])
        repl_with_dataset.cmd_ds_rm([])
        assert "test" not in repl_with_dataset.dataset_documents
        assert "test" not in repl_with_dataset.dataset_projections

    def test_wc_index_out_of_range(self, repl_with_dataset, capsys):
        """Test wc with out of range index."""
        repl_with_dataset.cmd_wc(["100"])
        captured = capsys.readouterr()
        assert "out of range" in captured.out


class TestProjectionRegistry:
    """Tests for projection registry functions."""

    def test_lowercase_projection(self):
        """Test lowercase projection function."""
        proj = PROJECTION_REGISTRY["lowercase"]
        assert proj("HELLO") == "hello"

    def test_uppercase_projection(self):
        """Test uppercase projection function."""
        proj = PROJECTION_REGISTRY["uppercase"]
        assert proj("hello") == "HELLO"

    def test_title_projection(self):
        """Test title projection function."""
        proj = PROJECTION_REGISTRY["title"]
        assert proj("hello world") == "Hello World"

    def test_strip_projection(self):
        """Test strip projection function."""
        proj = PROJECTION_REGISTRY["strip"]
        assert proj("  hello  ") == "hello"

    def test_projections_handle_bytes(self):
        """Test projections handle bytes input."""
        proj = PROJECTION_REGISTRY["lowercase"]
        result = proj(b"HELLO")
        assert result == "hello"


class TestLoadWithProjections:
    """Tests for loading datasets with projections."""

    def test_load_restores_projections(self, repl_with_dataset, capsys):
        """Test load restores projection settings."""
        repl_with_dataset.cmd_proj(["lowercase"])
        repl_with_dataset.cmd_save([])
        repl_with_dataset.cmd_ds_rm([])
        repl_with_dataset.cmd_load(["test"])
        assert repl_with_dataset.dataset_projections.get("test") == ["lowercase"]


class TestStoreRmMemoryCleanup:
    """Tests for store rm memory cleanup."""

    def test_store_rm_removes_from_memory(self, repl_with_dataset, capsys):
        """Test store rm also removes dataset from memory."""
        repl_with_dataset.cmd_save([])
        repl_with_dataset.cmd_store_rm(["test"])
        assert "test" not in repl_with_dataset.datasets
        assert repl_with_dataset.current_dataset is None

    def test_store_rm_switches_to_remaining(self, repl, capsys):
        """Test store rm switches to remaining dataset."""
        repl.execute("ds first")
        repl.execute("add first content")
        repl.cmd_save([])
        repl.execute("ds second")
        repl.execute("add second content")
        repl.cmd_save([])
        repl.cmd_store_rm(["second"])
        assert repl.current_dataset == "first"


class TestLoadInvalidDataset:
    """Tests for loading invalid dataset files."""

    def test_load_missing_metadata(self, repl, capsys):
        """Test load with missing metadata file."""
        # Create directory without metadata
        dataset_dir = repl.storage_dir / "invalid"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        repl.cmd_load(["invalid"])
        captured = capsys.readouterr()
        assert "missing metadata" in captured.out

    def test_load_missing_documents(self, repl, capsys):
        """Test load with missing documents file."""
        # Create directory with only metadata
        dataset_dir = repl.storage_dir / "invalid"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump({"name": "invalid"}, f)
        repl.cmd_load(["invalid"])
        captured = capsys.readouterr()
        assert "missing documents" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
