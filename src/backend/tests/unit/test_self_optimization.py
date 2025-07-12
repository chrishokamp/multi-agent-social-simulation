"""Unit tests for self-optimization functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import asyncio
from pathlib import Path
import copy

from src.backend.engine.simulation import SelectorGCSimulation
from src.backend.agents import BuyerAgent, SellerAgent


class TestSelfOptimization(unittest.TestCase):
    """Test the self-optimization functionality for agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            "name": "Test Negotiation",
            "agents": [
                {
                    "name": "Buyer",
                    "description": "Test buyer",
                    "prompt": "Initial buyer prompt",
                    "utility_class": "BuyerAgent",
                    "strategy": {"max_price": 500},
                    "self_improve": True,
                    "optimization_target": True
                },
                {
                    "name": "Seller", 
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
        }
        
        self.environment = {"runs": [], "outputs": {}}
        
    def test_environment_state_preserved_between_runs(self):
        """Test that environment state accumulates runs properly."""
        env = {"runs": [], "outputs": {}}
        
        # Simulate first run
        first_run = {
            "run_id": "run_1",
            "output_variables": [
                {"name": "final_price", "value": 380},
                {"name": "deal_reached", "value": True}
            ],
            "messages": [{"agent": "Buyer", "message": "Test"}],
            "system_prompts": {"Buyer": "Initial prompt"}
        }
        env["runs"].append(first_run)
        
        self.assertEqual(len(env["runs"]), 1)
        self.assertEqual(env["runs"][0]["run_id"], "run_1")
        
        # Simulate second run
        second_run = {
            "run_id": "run_2",
            "output_variables": [
                {"name": "final_price", "value": 350},
                {"name": "deal_reached", "value": True}
            ],
            "messages": [{"agent": "Buyer", "message": "Test 2"}],
            "system_prompts": {"Buyer": "Updated prompt"}
        }
        env["runs"].append(second_run)
        
        self.assertEqual(len(env["runs"]), 2)
        self.assertEqual(env["runs"][1]["run_id"], "run_2")
        
    def test_learn_from_feedback_logic(self):
        """Test the logic for when learn_from_feedback should be called."""
        # Test case 1: No runs in environment - should not learn
        env_no_runs = {"runs": []}
        self.assertFalse(env_no_runs.get("runs"))  # No learning should happen
        
        # Test case 2: Has runs but self_improve=False - should not learn
        env_with_runs = {"runs": [{"run_id": "1"}]}
        agent_cfg_no_improve = {"self_improve": False}
        should_learn = env_with_runs.get("runs") and agent_cfg_no_improve.get("self_improve", False)
        self.assertFalse(should_learn)
        
        # Test case 3: Has runs and self_improve=True - should learn
        agent_cfg_improve = {"self_improve": True}
        should_learn = env_with_runs.get("runs") and agent_cfg_improve.get("self_improve", False)
        self.assertTrue(should_learn)
        
    def test_prompt_persistence_in_config(self):
        """Test that prompts are correctly persisted in the config between runs."""
        # Simulate the self_optimize_negotiation_with_logging.py behavior
        current_config = copy.deepcopy(self.base_config)
        
        # Simulate first run
        agent_prompts = {
            "Buyer": "Updated buyer prompt after learning",
            "Seller": "Initial seller prompt"  # Shouldn't change
        }
        
        # Persist improved prompts
        for agent_name, new_prompt in agent_prompts.items():
            for cfg in current_config["agents"]:
                if cfg["name"] == agent_name:
                    cfg["prompt"] = new_prompt
                    
        # Verify prompts were updated
        buyer_cfg = next(cfg for cfg in current_config["agents"] if cfg["name"] == "Buyer")
        seller_cfg = next(cfg for cfg in current_config["agents"] if cfg["name"] == "Seller")
        
        self.assertEqual(buyer_cfg["prompt"], "Updated buyer prompt after learning")
        self.assertEqual(seller_cfg["prompt"], "Initial seller prompt")
        
                    
    def test_utility_calculation_and_storage(self):
        """Test that utilities are calculated and stored correctly."""
        buyer = BuyerAgent(
            name="Buyer",
            system_prompt="Test",
            strategy={"max_price": 500}
        )
        
        environment = {
            "runs": [{
                "output_variables": [
                    {"name": "deal_reached", "value": True},
                    {"name": "final_price", "value": 300}
                ]
            }]
        }
        
        # Calculate utility
        updated_env = buyer.compute_utility(environment)
        
        # Extract utility value
        utility = buyer.extract_utility_value(updated_env)
        
        # Utility should be 1 - (300/500) = 0.4
        self.assertEqual(utility, 0.4)
        
    def test_optimization_prompt_usage(self):
        """Test that optimization_prompt is used when provided."""
        config_with_opt_prompt = copy.deepcopy(self.base_config)
        config_with_opt_prompt["agents"][0]["optimization_prompt"] = "Custom optimization instructions"
        
        with patch('src.backend.agents.client_for_endpoint') as mock_client:
            mock_llm = MagicMock()
            mock_llm.chat.completions.create.return_value.choices = [
                MagicMock(message=MagicMock(content="Optimized prompt"))
            ]
            mock_client.return_value = (mock_llm, "test-model")
            
            buyer = BuyerAgent(
                name="Buyer",
                system_prompt="Initial",
                strategy={"max_price": 500},
                optimization_prompt="Custom optimization instructions"
            )
            
            # Add environment with previous run
            environment = {
                "runs": [{
                    "run_id": "run_1",
                    "output_variables": [
                        {"name": "utility", "value": {"Buyer": 0.3}}
                    ],
                    "messages": [],
                    "system_prompts": {"Buyer": "Initial"}
                }]
            }
            
            buyer.learn_from_feedback(environment)
            
            # Check that the LLM was called with the optimization prompt
            call_args = mock_llm.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            
            # Should have system message with optimization prompt
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[0]["content"], "Custom optimization instructions")


class TestMultiRunSimulation(unittest.TestCase):
    """Test multi-run simulation behavior."""
    
    def test_config_updates_between_runs(self):
        """Test that config is properly updated between simulation runs."""
        from scripts.self_optimize_negotiation_with_logging import materialise_config
        
        raw_config = {
            "config": {
                "name": "Test",
                "agents": [
                    {"name": "Agent1", "prompt": "Initial prompt 1"},
                    {"name": "Agent2", "prompt": "Initial prompt 2"}
                ]
            }
        }
        
        # Initialize config once
        current_config = materialise_config(raw_config)
        
        # Simulate updating prompts after first run
        for agent in current_config["agents"]:
            if agent["name"] == "Agent1":
                agent["prompt"] = "Updated prompt 1"
                
        # Create a copy for the next run (as done in the script)
        config_for_run2 = copy.deepcopy(current_config)
        
        # Verify the prompt was preserved
        agent1_cfg = next(cfg for cfg in config_for_run2["agents"] if cfg["name"] == "Agent1")
        self.assertEqual(agent1_cfg["prompt"], "Updated prompt 1")
        
        # Original should not be affected
        original_agent1 = next(cfg for cfg in raw_config["config"]["agents"] if cfg["name"] == "Agent1")
        self.assertEqual(original_agent1["prompt"], "Initial prompt 1")


if __name__ == "__main__":
    unittest.main()