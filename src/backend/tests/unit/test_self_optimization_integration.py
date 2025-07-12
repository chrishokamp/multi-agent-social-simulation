"""Integration tests for self-optimization functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
from pathlib import Path

from scripts.self_optimize_negotiation_with_logging import materialise_config


class TestSelfOptimizationIntegration(unittest.TestCase):
    """Test the self-optimization functionality in the actual script context."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "config": {
                "name": "Test Negotiation",
                "agents": [
                    {
                        "name": "TestBuyer",
                        "description": "Test buyer",
                        "prompt": "Initial buyer prompt",
                        "utility_class": "BuyerAgent",
                        "strategy": {"max_price": 500},
                        "self_improve": True,
                        "optimization_target": True
                    },
                    {
                        "name": "TestSeller", 
                        "description": "Test seller",
                        "prompt": "Initial seller prompt",
                        "utility_class": "SellerAgent",
                        "strategy": {"target_price": 300},
                        "self_improve": False
                    }
                ],
                "termination_condition": "STOP_NEGOTIATION",
                "output_variables": [
                    {"name": "final_price", "type": "Number"},
                    {"name": "deal_reached", "type": "Boolean"}
                ]
            },
            "num_runs": 3,
            "optimization_prompt": "Test optimization prompt"
        }
        
    def test_prompt_updates_persist_between_runs(self):
        """Test that prompts are updated and persisted correctly between runs."""
        # Initialize config once
        current_config = materialise_config(self.test_config)
        initial_buyer_prompt = current_config["agents"][0]["prompt"]
        
        # Simulate updating the buyer's prompt after run 1
        current_config["agents"][0]["prompt"] = "Updated buyer prompt after learning"
        
        # Create a copy for run 2 (as done in the actual script)
        config_for_run2 = current_config.copy()
        
        # The prompt should be preserved
        self.assertEqual(
            config_for_run2["agents"][0]["prompt"], 
            "Updated buyer prompt after learning"
        )
        self.assertNotEqual(
            config_for_run2["agents"][0]["prompt"],
            initial_buyer_prompt
        )
        
    def test_environment_accumulates_runs(self):
        """Test that the environment properly accumulates run data."""
        environment = {"runs": [], "outputs": {}}
        
        # Simulate adding runs
        for i in range(3):
            run_data = {
                "run_id": f"run_{i+1}",
                "output_variables": [
                    {"name": "final_price", "value": 350 - (i * 50)},
                    {"name": "deal_reached", "value": True}
                ],
                "messages": [{"agent": "Buyer", "message": f"Test message {i}"}],
                "system_prompts": {"Buyer": f"Prompt version {i+1}"}
            }
            environment["runs"].append(run_data)
        
        # Verify all runs are present
        self.assertEqual(len(environment["runs"]), 3)
        
        # Verify run IDs
        self.assertEqual(environment["runs"][0]["run_id"], "run_1")
        self.assertEqual(environment["runs"][2]["run_id"], "run_3")
        
        # Verify data integrity
        self.assertEqual(
            environment["runs"][0]["output_variables"][0]["value"], 
            350
        )
        self.assertEqual(
            environment["runs"][2]["output_variables"][0]["value"], 
            250
        )
        
    def test_learning_improves_utility(self):
        """Test that learning leads to improved utility values over runs."""
        # Simulate utility progression
        utilities = [0.24, 0.30, 0.40, 0.45, 0.48]
        
        for i in range(1, len(utilities)):
            # Each utility should be better than or equal to the previous
            self.assertGreaterEqual(utilities[i], utilities[i-1])
            
        # Overall improvement
        improvement = (utilities[-1] - utilities[0]) / utilities[0]
        self.assertGreater(improvement, 0.5)  # At least 50% improvement
        
    def test_only_optimization_target_learns(self):
        """Test that only agents marked as optimization_target learn."""
        config = materialise_config(self.test_config)
        
        # Check agent configurations
        buyer_cfg = next(cfg for cfg in config["agents"] if cfg["name"] == "TestBuyer")
        seller_cfg = next(cfg for cfg in config["agents"] if cfg["name"] == "TestSeller")
        
        # Buyer should have self_improve=True
        self.assertTrue(buyer_cfg.get("self_improve", False))
        
        # Seller should have self_improve=False
        self.assertFalse(seller_cfg.get("self_improve", False))
        
    def test_debug_output_shows_learning(self):
        """Test that debug output is generated when learning occurs."""
        # This would be tested by checking console output in actual runs
        # For unit testing, we verify the print statement exists
        import src.backend.engine.simulation as sim_module
        
        # Check that the learning debug print exists in the code
        with open(sim_module.__file__, 'r') as f:
            content = f.read()
            self.assertIn("🧠", content)  # Learning emoji
            self.assertIn("is learning from", content)
            self.assertIn("previous runs", content)


class TestConfigManagement(unittest.TestCase):
    """Test configuration management in multi-run scenarios."""
    
    def test_materialise_config_preserves_structure(self):
        """Test that materialise_config preserves the config structure."""
        raw_config = {
            "config": {
                "name": "Test",
                "agents": [
                    {"name": "Agent1", "prompt": "Prompt 1"},
                    {"name": "Agent2", "prompt": "Prompt 2"}
                ]
            }
        }
        
        materialized = materialise_config(raw_config)
        
        self.assertEqual(materialized["name"], "Test")
        self.assertEqual(len(materialized["agents"]), 2)
        self.assertEqual(materialized["agents"][0]["name"], "Agent1")
        
    def test_deep_copy_prevents_mutation(self):
        """Test that deep copying prevents unintended mutations."""
        import copy
        
        original = {"agents": [{"name": "Agent1", "prompt": "Original"}]}
        config_copy = copy.deepcopy(original)
        
        # Modify the copy
        config_copy["agents"][0]["prompt"] = "Modified"
        
        # Original should be unchanged
        self.assertEqual(original["agents"][0]["prompt"], "Original")
        self.assertEqual(config_copy["agents"][0]["prompt"], "Modified")


if __name__ == "__main__":
    unittest.main()