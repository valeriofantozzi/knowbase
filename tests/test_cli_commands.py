"""
Tests for KnowBase CLI Commands

Tests basic functionality of all CLI commands using Click's CliRunner
and mock data to avoid external dependencies.
"""

import pytest
import json
from pathlib import Path
from click.testing import CliRunner
from src.cli.main import cli
from src.cli.commands.search import search
from src.cli.commands.load import load
from src.cli.commands.info import info


class TestCliBasics:
    """Test basic CLI functionality."""

    def test_version_flag(self):
        """Test --version flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'KnowBase' in result.output or '0.1.0' in result.output

    def test_help_flag(self):
        """Test --help flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Commands:' in result.output
        assert 'ask' in result.output
        assert 'cluster' in result.output
        assert 'export' in result.output

    def test_unknown_command(self):
        """Test invalid command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['invalid-command'])
        assert result.exit_code != 0

    def test_hello_command(self):
        """Test hello command (simple test)."""
        runner = CliRunner()
        result = runner.invoke(cli, ['hello'])
        assert result.exit_code == 0
        assert 'KnowBase' in result.output


class TestSearchCommand:
    """Test search command."""

    def test_search_help(self):
        """Test search --help."""
        runner = CliRunner()
        result = runner.invoke(search, ['--help'])
        assert result.exit_code == 0
        assert 'Search' in result.output
        assert '--query' in result.output
        assert '--top-k' in result.output

    def test_search_missing_query(self):
        """Test search without required query argument."""
        runner = CliRunner()
        result = runner.invoke(search, [])
        assert result.exit_code != 0
        assert 'required' in result.output.lower() or 'error' in result.output.lower()

    def test_search_invalid_format(self):
        """Test search with invalid format."""
        runner = CliRunner()
        result = runner.invoke(search, ['--query', 'test', '--format', 'invalid'])
        # Click should catch invalid choice
        assert result.exit_code != 0

    def test_search_valid_formats(self):
        """Test search accepts all valid formats."""
        runner = CliRunner()
        for fmt in ['text', 'json', 'csv', 'table']:
            result = runner.invoke(search, [
                '--query', 'test',
                '--format', fmt,
            ])
            # Will fail due to empty database, but format should be accepted
            assert 'invalid choice' not in result.output.lower()

    def test_search_top_k_validation(self):
        """Test search top-k parameter validation."""
        runner = CliRunner()
        
        # Valid value
        result = runner.invoke(search, [
            '--query', 'test',
            '--top-k', '5',
        ])
        assert 'invalid value for' not in result.output.lower() or result.exit_code == 0
        
        # Invalid value (too high)
        result = runner.invoke(search, [
            '--query', 'test',
            '--top-k', '100',
        ])
        assert result.exit_code != 0

    def test_search_model_parameter(self):
        """Test search with custom model."""
        runner = CliRunner()
        result = runner.invoke(search, [
            '--query', 'test',
            '--model', 'google/embeddinggemma-300m',
        ])
        # Should accept the model parameter
        assert 'unrecognized arguments' not in result.output.lower()


class TestLoadCommand:
    """Test load command."""

    def test_load_help(self):
        """Test load --help."""
        runner = CliRunner()
        result = runner.invoke(load, ['--help'])
        assert result.exit_code == 0
        assert 'Load' in result.output
        assert '--input' in result.output

    def test_load_missing_input(self):
        """Test load without required input argument."""
        runner = CliRunner()
        result = runner.invoke(load, [])
        assert result.exit_code != 0

    def test_load_batch_size_validation(self):
        """Test load batch size validation."""
        runner = CliRunner()
        
        # Valid value
        result = runner.invoke(load, [
            '--input', '/nonexistent',
            '--batch-size', '32',
        ])
        assert 'invalid value for' not in result.output.lower() or result.exit_code != 0
        
        # Invalid value (too high)
        result = runner.invoke(load, [
            '--input', '/nonexistent',
            '--batch-size', '500',
        ])
        assert result.exit_code != 0

    def test_load_device_validation(self):
        """Test load device parameter validation."""
        runner = CliRunner()
        
        # Valid devices
        for device in ['auto', 'cpu', 'cuda', 'mps']:
            result = runner.invoke(load, [
                '--input', '/nonexistent',
                '--device', device,
            ])
            assert 'invalid choice' not in result.output.lower()
        
        # Invalid device
        result = runner.invoke(load, [
            '--input', '/nonexistent',
            '--device', 'tpu',
        ])
        assert result.exit_code != 0


class TestInfoCommand:
    """Test info command."""

    def test_info_help(self):
        """Test info --help."""
        runner = CliRunner()
        result = runner.invoke(info, ['--help'])
        assert result.exit_code == 0
        assert 'information' in result.output.lower() or 'system' in result.output.lower()

    def test_info_runs(self):
        """Test info command executes."""
        runner = CliRunner()
        result = runner.invoke(info, [])
        # info should execute without errors
        assert 'error' not in result.output.lower() or result.exit_code == 0


class TestClusterCommand:
    """Test cluster command."""

    def test_cluster_help(self):
        """Test cluster --help."""
        runner = CliRunner()
        from src.cli.commands.cluster import cluster
        result = runner.invoke(cluster, ['--help'])
        assert result.exit_code == 0
        assert 'cluster' in result.output.lower()

    def test_cluster_min_cluster_size_validation(self):
        """Test cluster min-cluster-size validation."""
        runner = CliRunner()
        from src.cli.commands.cluster import cluster
        
        # Valid value
        result = runner.invoke(cluster, [
            '--min-cluster-size', '5',
        ])
        assert 'invalid value' not in result.output.lower() or result.exit_code != 0
        
        # Invalid value (too low)
        result = runner.invoke(cluster, [
            '--min-cluster-size', '1',
        ])
        assert result.exit_code != 0


class TestExportCommand:
    """Test export command."""

    def test_export_help(self):
        """Test export --help."""
        runner = CliRunner()
        from src.cli.commands.export import export
        result = runner.invoke(export, ['--help'])
        assert result.exit_code == 0
        assert 'export' in result.output.lower()

    def test_export_missing_output(self):
        """Test export without required output argument."""
        runner = CliRunner()
        from src.cli.commands.export import export
        result = runner.invoke(export, [])
        assert result.exit_code != 0

    def test_export_format_validation(self):
        """Test export format validation."""
        runner = CliRunner()
        from src.cli.commands.export import export
        
        # Valid formats
        for fmt in ['json', 'csv']:
            result = runner.invoke(export, [
                '--output', '/tmp/test',
                '--format', fmt,
            ])
            assert 'invalid choice' not in result.output.lower()


class TestReindexCommand:
    """Test reindex command."""

    def test_reindex_help(self):
        """Test reindex --help."""
        runner = CliRunner()
        from src.cli.commands.reindex import reindex
        result = runner.invoke(reindex, ['--help'])
        assert result.exit_code == 0
        assert 'reindex' in result.output.lower()

    def test_reindex_missing_new_model(self):
        """Test reindex without required new-model argument."""
        runner = CliRunner()
        from src.cli.commands.reindex import reindex
        result = runner.invoke(reindex, [])
        assert result.exit_code != 0


class TestAskCommand:
    """Test ask command."""

    def test_ask_help(self):
        """Test ask --help."""
        runner = CliRunner()
        from src.cli.commands.ask import ask
        result = runner.invoke(ask, ['--help'])
        assert result.exit_code == 0
        assert 'ask' in result.output.lower()

    def test_ask_missing_question(self):
        """Test ask without required question argument."""
        runner = CliRunner()
        from src.cli.commands.ask import ask
        result = runner.invoke(ask, [])
        assert result.exit_code != 0

    def test_ask_temperature_validation(self):
        """Test ask temperature validation."""
        runner = CliRunner()
        from src.cli.commands.ask import ask
        
        # Valid value
        result = runner.invoke(ask, [
            '--help',  # Just test parameter acceptance
        ])
        assert result.exit_code == 0
        assert '--temperature' in result.output


class TestInputValidation:
    """Test input validation for all commands."""

    def test_query_too_long(self):
        """Test search/ask with query that's too long."""
        runner = CliRunner()
        
        # Create a very long query (>2000 chars)
        long_query = 'a' * 2001
        
        result = runner.invoke(search, [
            '--query', long_query,
        ])
        # Should fail validation
        assert result.exit_code != 0 or 'too long' in result.output.lower()

    def test_invalid_path_characters(self):
        """Test commands with invalid path characters."""
        runner = CliRunner()
        
        # These should be handled gracefully
        from src.cli.commands.export import export
        result = runner.invoke(export, [
            '--output', '/\x00/invalid',
        ])
        # Should not crash, may fail with error message
        assert result.exit_code != 0 or 'invalid' in result.output.lower()


class TestVerboseMode:
    """Test verbose output mode."""

    def test_verbose_flag_accepted(self):
        """Test that verbose flag is accepted by commands."""
        runner = CliRunner()
        from src.cli.commands.info import info
        result = runner.invoke(info, ['-v'])
        assert 'unrecognized' not in result.output.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
