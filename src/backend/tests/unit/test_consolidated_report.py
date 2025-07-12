"""Unit tests for consolidated report generation in self-optimization script."""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add the scripts directory to the Python path
scripts_dir = Path(__file__).parent.parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from self_optimize_negotiation_with_logging import _generate_consolidated_report


class TestConsolidatedReport:
    """Test consolidated report generation functionality."""

    def test_generate_consolidated_report_with_list_outputs(self):
        """Test report generation with outputs as list format."""
        test_history = [
            {
                "run_id": 1,
                "outputs": [
                    {"name": "price", "value": 100},
                    {"name": "quantity", "value": 5}
                ],
                "utilities": {
                    "TestAgent": 0.5
                },
                "log_dir": "/tmp/test_logs/run_001"
            },
            {
                "run_id": 2,
                "outputs": [
                    {"name": "price", "value": 110},
                    {"name": "quantity", "value": 6}
                ],
                "utilities": {
                    "TestAgent": 0.7
                },
                "log_dir": "/tmp/test_logs/run_002"
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Should not raise AttributeError
            _generate_consolidated_report(
                output_dir,
                test_history,
                "test_config"
            )
            
            # Check that report was created
            report_path = output_dir / "consolidated_report.html"
            assert report_path.exists()
            
            # Check that visualizations directory was created
            viz_dir = output_dir / "consolidated_visualizations"
            assert viz_dir.exists()
            
            # Check that some plots were created
            plot_files = list(viz_dir.glob("*.png"))
            assert len(plot_files) > 0

    def test_generate_consolidated_report_with_dict_outputs(self):
        """Test report generation with outputs as dict format."""
        test_history = [
            {
                "run_id": 1,
                "outputs": {
                    "price": 100,
                    "quantity": 5
                },
                "utilities": {
                    "TestAgent": 0.5
                },
                "log_dir": "/tmp/test_logs/run_001"
            },
            {
                "run_id": 2,
                "outputs": {
                    "price": 110,
                    "quantity": 6
                },
                "utilities": {
                    "TestAgent": 0.7
                },
                "log_dir": "/tmp/test_logs/run_002"
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Should handle dict format correctly
            _generate_consolidated_report(
                output_dir,
                test_history,
                "test_config"
            )
            
            # Check that report was created
            report_path = output_dir / "consolidated_report.html"
            assert report_path.exists()

    def test_generate_consolidated_report_with_mixed_data(self):
        """Test report generation with mixed data types and edge cases."""
        test_history = [
            {
                "run_id": 1,
                "outputs": [
                    {"name": "price", "value": 100},
                    {"name": "success", "value": True}
                ],
                "utilities": {
                    "Agent1": 0.5,
                    "Agent2": 0.3
                },
                "log_dir": "/tmp/test_logs/run_001"
            },
            {
                "run_id": 2,
                "outputs": [
                    {"name": "price", "value": "high"},  # Non-numeric value
                    {"name": "success", "value": False}
                ],
                "utilities": {
                    "Agent1": "not_a_number",  # Non-numeric utility
                    "Agent2": 0.8
                },
                "log_dir": "/tmp/test_logs/run_002"
            },
            {
                "run_id": 3,
                "outputs": [],  # Empty outputs
                "utilities": {
                    "Agent2": 0.8
                },
                "log_dir": "/tmp/test_logs/run_003"
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Should handle edge cases gracefully
            _generate_consolidated_report(
                output_dir,
                test_history,
                "test_config"
            )
            
            # Check that report was created despite problematic data
            report_path = output_dir / "consolidated_report.html"
            assert report_path.exists()

    def test_generate_consolidated_report_empty_history(self):
        """Test report generation with empty history."""
        test_history = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Should handle empty history gracefully
            _generate_consolidated_report(
                output_dir,
                test_history,
                "test_config"
            )
            
            # Check that report was created (even if empty)
            report_path = output_dir / "consolidated_report.html"
            assert report_path.exists()
            
            # Read report content to verify it contains appropriate message
            with open(report_path, 'r') as f:
                content = f.read()
                assert "No data available" in content or "empty" in content.lower()

    def test_generate_consolidated_report_html_content(self):
        """Test that generated HTML contains expected elements."""
        test_history = [
            {
                "run_id": 1,
                "outputs": [
                    {"name": "price", "value": 100},
                    {"name": "quantity", "value": 5}
                ],
                "utilities": {
                    "TestAgent": 0.5
                },
                "log_dir": "/tmp/test_logs/run_001"
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            _generate_consolidated_report(
                output_dir,
                test_history,
                "test_config"
            )
            
            report_path = output_dir / "consolidated_report.html"
            with open(report_path, 'r') as f:
                html_content = f.read()
            
            # Check for expected HTML elements
            assert "<html>" in html_content
            assert "Consolidated Simulation Report" in html_content
            assert "test_config" in html_content
            assert "<table" in html_content
            assert "TestAgent" in html_content
            assert "price" in html_content or "100" in html_content  # Price could be in table header or data
            assert "quantity" in html_content or "5" in html_content  # Quantity could be in table header or data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])