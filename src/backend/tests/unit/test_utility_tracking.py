"""Unit tests for utility tracking and extraction."""

import unittest
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

from src.backend.agents import UtilityAgent, BuyerAgent, SellerAgent
from src.backend.logging_framework.core import SimulationLogger


class TestUtilityTracking(unittest.TestCase):
    """Test utility tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = SimulationLogger(self.temp_dir)
        
    def test_extract_utility_value_dict_format(self):
        """Test extracting utility value from dict format output_variables."""
        buyer = BuyerAgent(
            name="TestBuyer",
            system_prompt="Test buyer",
            strategy={"max_price": 100}
        )
        
        environment = {
            "runs": [{
                "output_variables": {
                    "utility": {"TestBuyer": 0.75}
                }
            }]
        }
        
        utility = buyer.extract_utility_value(environment)
        self.assertEqual(utility, 0.75)
        
    def test_extract_utility_value_list_format(self):
        """Test extracting utility value from list format output_variables."""
        seller = SellerAgent(
            name="TestSeller",
            system_prompt="Test seller",
            strategy={"target_price": 50}
        )
        
        environment = {
            "runs": [{
                "output_variables": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 60},
                    {"name": "utility", "value": {"TestSeller": 0.8}}
                ]
            }]
        }
        
        utility = seller.extract_utility_value(environment)
        self.assertEqual(utility, 0.8)
        
    def test_extract_utility_value_no_runs(self):
        """Test extracting utility when no runs exist."""
        buyer = BuyerAgent(
            name="TestBuyer",
            system_prompt="Test buyer",
            strategy={"max_price": 100}
        )
        
        # Empty environment
        utility = buyer.extract_utility_value({})
        self.assertEqual(utility, 0.0)
        
        # No runs
        utility = buyer.extract_utility_value({"runs": []})
        self.assertEqual(utility, 0.0)
        
    def test_extract_utility_value_missing_utility(self):
        """Test extracting utility when utility is not in output_variables."""
        buyer = BuyerAgent(
            name="TestBuyer", 
            system_prompt="Test buyer",
            strategy={"max_price": 100}
        )
        
        environment = {
            "runs": [{
                "output_variables": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 60}
                ]
            }]
        }
        
        utility = buyer.extract_utility_value(environment)
        self.assertEqual(utility, 0.0)
        
    def test_buyer_compute_utility_deal_reached(self):
        """Test buyer utility calculation when deal is reached."""
        buyer = BuyerAgent(
            name="TestBuyer",
            system_prompt="Test buyer", 
            strategy={"max_price": 100}
        )
        
        environment = {
            "runs": [{
                "output_variables": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 75}
                ]
            }]
        }
        
        # compute_utility modifies environment
        updated_env = buyer.compute_utility(environment)
        
        # Extract the utility
        utility = buyer.extract_utility_value(updated_env)
        
        # Buyer saved 25 out of 100, so utility = 0.25
        self.assertEqual(utility, 0.25)
        
    def test_buyer_compute_utility_no_deal(self):
        """Test buyer utility calculation when no deal is reached."""
        buyer = BuyerAgent(
            name="TestBuyer",
            system_prompt="Test buyer",
            strategy={"max_price": 100}
        )
        
        environment = {
            "runs": [{
                "output_variables": [
                    {"name": "deal_reached", "value": False},
                    {"name": "final_price", "value": 0}
                ]
            }]
        }
        
        updated_env = buyer.compute_utility(environment)
        utility = buyer.extract_utility_value(updated_env)
        
        # No deal, so utility = 0
        self.assertEqual(utility, 0.0)
        
    def test_seller_compute_utility_deal_reached(self):
        """Test seller utility calculation when deal is reached."""
        seller = SellerAgent(
            name="TestSeller",
            system_prompt="Test seller",
            strategy={"target_price": 50}
        )
        
        environment = {
            "runs": [{
                "output_variables": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 60}
                ]
            }]
        }
        
        updated_env = seller.compute_utility(environment)
        utility = seller.extract_utility_value(updated_env)
        
        # Seller got 60 with target 50, so utility = min(60/50, 1.0) = 1.0
        self.assertAlmostEqual(utility, 1.0, places=2)
        
    def test_utility_logging_integration(self):
        """Test that utilities are properly logged to agent logger."""
        buyer = BuyerAgent(
            name="TestBuyer",
            system_prompt="Test buyer",
            strategy={"max_price": 100}
        )
        
        # Attach logger
        agent_logger = self.logger.get_agent_logger("TestBuyer")
        buyer._logger = agent_logger
        
        # Log some utilities
        buyer._logger.log_utility(1, 0.25, {"test": "data1"})
        buyer._logger.log_utility(2, 0.50, {"test": "data2"})
        buyer._logger.log_utility(3, 0.75, {"test": "data3"})
        
        # Check utility history
        history = agent_logger.utility_history
        self.assertEqual(len(history), 3)
        
        # Check values
        self.assertEqual(history[0].utility_value, 0.25)
        self.assertEqual(history[1].utility_value, 0.50)
        self.assertEqual(history[2].utility_value, 0.75)
        
        # Check that timestamps exist (they are datetime objects, not integers)
        self.assertIsNotNone(history[0].timestamp)
        self.assertIsNotNone(history[1].timestamp)
        self.assertIsNotNone(history[2].timestamp)
        
    def test_consolidated_report_utility_data(self):
        """Test that utility data is properly formatted for consolidated report."""
        from scripts.self_optimize_negotiation_with_logging import _generate_consolidated_report
        
        # Create test history with utility values
        history = [
            {
                "run_id": 1,
                "outputs": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 80}
                ],
                "utilities": {"Buyer": 0.2, "Seller": 0.8},
                "log_dir": "/tmp/run1"
            },
            {
                "run_id": 2,
                "outputs": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 70}
                ],
                "utilities": {"Buyer": 0.3, "Seller": 0.7},
                "log_dir": "/tmp/run2"
            },
            {
                "run_id": 3,
                "outputs": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 60}
                ],
                "utilities": {"Buyer": 0.4, "Seller": 0.6},
                "log_dir": "/tmp/run3"
            }
        ]
        
        # Create output directory
        output_dir = Path(self.temp_dir) / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        # Generate consolidated report
        _generate_consolidated_report(output_dir, history, "test_config.json")
        
        # Check that utility evolution plot was created
        utility_plot = output_dir / "consolidated_visualizations" / "utility_evolution.png"
        self.assertTrue(utility_plot.exists())
        
        # Check that report was created
        report_path = output_dir / "consolidated_report.html"
        self.assertTrue(report_path.exists())
        
        # Read report and check it contains utility data
        with open(report_path) as f:
            report_content = f.read()
            # Check for utility visualization in report
            self.assertIn("Utility Evolution", report_content)
            self.assertIn("utility_evolution.png", report_content)


if __name__ == "__main__":
    unittest.main()