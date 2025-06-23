"""
Integration tests for logged simulation.
"""
import asyncio
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from engine.logged_simulation import LoggedSelectorGCSimulation
from agents import BuyerAgent, SellerAgent


class TestLoggedSimulation:
    """Integration tests for logged simulation."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "model": "gpt-4o",
            "agents": [
                {
                    "name": "Buyer",
                    "utility_class": "BuyerAgent",
                    "description": "A buyer agent",
                    "prompt": "You are a buyer. Try to get the best price.",
                    "strategy": {"max_price": 1000},
                    "self_improve": False
                },
                {
                    "name": "Seller", 
                    "utility_class": "SellerAgent",
                    "description": "A seller agent",
                    "prompt": "You are a seller. Try to maximize profit.",
                    "strategy": {"target_price": 800},
                    "self_improve": False
                }
            ],
            "output_variables": [
                {"name": "final_price", "type": "Number"},
                {"name": "deal_reached", "type": "Boolean"}
            ],
            "termination_condition": "Agreement reached or negotiation failed"
        }
    
    @pytest.fixture
    def sample_environment(self):
        """Sample environment for testing."""
        return {"runs": [], "outputs": {}}
    
    def test_simulation_initialization(self, sample_config, sample_environment):
        """Test that logged simulation initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            sim = LoggedSelectorGCSimulation(
                sample_config,
                sample_environment,
                max_messages=5,
                min_messages=1,
                log_dir=log_dir
            )
            
            # Check that simulation was initialized
            assert sim.logger is not None
            assert sim.logger.simulation_id == sim.run_id
            assert len(sim.agents) == 3  # 2 main agents + 1 InformationReturnAgent
            
            # Check that log directory was created
            assert sim.logger.log_dir.exists()
            
            # Check that agent loggers were created
            assert len(sim.logger.agent_loggers) == 3  # Includes InformationReturnAgent
            assert "Buyer" in sim.logger.agent_loggers
            assert "Seller" in sim.logger.agent_loggers
            
            # Check that simulation info was saved
            info_file = sim.logger.log_dir / "simulation_info.json"
            assert info_file.exists()
            
            with open(info_file) as f:
                info = json.load(f)
                assert info["simulation_id"] == sim.run_id
                assert info["max_messages"] == 5
    
    def test_agent_logger_setup(self, sample_config, sample_environment):
        """Test that agent loggers are properly set up."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            sim = LoggedSelectorGCSimulation(
                sample_config,
                sample_environment,
                log_dir=log_dir
            )
            
            # Check that agents have loggers attached
            for agent in sim.agents:
                assert hasattr(agent, '_logger')
                assert agent._logger.agent_name == agent.name
                assert agent._logger.simulation_id == sim.run_id
                
                # Check that initial utility was logged (if agent has compute_utility)
                if hasattr(agent, 'compute_utility'):
                    assert len(agent._logger.utility_history) >= 1
                    
                # Check that initialization was logged
                assert len(agent._logger.actions) >= 1
                init_actions = agent._logger.get_actions_by_type('initialization')
                assert len(init_actions) >= 1
    
    @patch('autogen.ConversableAgent.a_initiate_chat')
    @pytest.mark.asyncio
    async def test_simulation_run_with_logging(self, mock_chat, sample_config, sample_environment):
        """Test that simulation run generates proper logs."""
        # Mock chat result
        mock_chat_result = Mock()
        mock_chat_result.chat_history = [
            {"name": "Buyer", "content": "I'd like to buy this bike for $500"},
            {"name": "Seller", "content": "I can sell it for $700"},
            {"name": "Buyer", "content": "How about $600?"},
            {"name": "Seller", "content": "Deal! $600 it is."},
            {"name": "InformationReturnAgent", "content": '{"final_price": 600, "deal_reached": true}'}
        ]
        mock_chat.return_value = mock_chat_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            sim = LoggedSelectorGCSimulation(
                sample_config,
                sample_environment,
                max_messages=10,
                min_messages=3,
                log_dir=log_dir
            )
            
            # Run simulation
            result = await sim.run()
            
            # Check that result was returned
            assert result is not None
            assert len(result["messages"]) == 5
            assert len(result["output_variables"]) == 2
            
            # Check that messages were logged
            assert len(sim.logger.messages) == 5
            
            # Check that agent actions were logged
            buyer_logger = sim.logger.get_agent_logger("Buyer")
            seller_logger = sim.logger.get_agent_logger("Seller")
            
            buyer_messages = buyer_logger.get_actions_by_type('message')
            seller_messages = seller_logger.get_actions_by_type('message')
            
            assert len(buyer_messages) >= 2  # Two buyer messages
            assert len(seller_messages) >= 2  # Two seller messages
            
            # Check that final utilities were logged
            assert len(buyer_logger.utility_history) >= 2  # Initial + final
            assert len(seller_logger.utility_history) >= 2  # Initial + final
            
            # Check that logs were saved to files
            assert (log_dir / "messages.json").exists()
            assert (log_dir / "agent_Buyer.json").exists()
            assert (log_dir / "agent_Seller.json").exists()
            assert (log_dir / "metrics.json").exists()
    
    def test_utility_calculation_and_logging(self, sample_config, sample_environment):
        """Test that utility calculations are properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            sim = LoggedSelectorGCSimulation(
                sample_config,
                sample_environment,
                log_dir=log_dir
            )
            
            # Get the buyer agent (should be BuyerAgent)
            buyer_agent = None
            for agent in sim.agents:
                if agent.name == "Buyer":
                    buyer_agent = agent
                    break
            
            assert buyer_agent is not None
            assert isinstance(buyer_agent, BuyerAgent)
            
            # Test utility calculation
            test_environment = {
                "outputs": {
                    "final_price": 600,
                    "deal_reached": True
                }
            }
            
            utility = buyer_agent.compute_utility(test_environment)
            
            # For buyer: utility = 1.0 - (final_price / max_price)
            # utility = 1.0 - (600 / 1000) = 0.4
            expected_utility = 1.0 - (600 / 1000)
            assert abs(utility - expected_utility) < 0.001
            
            # Log the utility update
            sim.logger.log_utility_update("Buyer", utility, test_environment)
            
            # Check that it was logged
            buyer_logger = sim.logger.get_agent_logger("Buyer")
            utility_history = buyer_logger.utility_history
            
            # Should have initial utility + the one we just logged
            assert len(utility_history) >= 2
            assert utility_history[-1].utility_value == utility
    
    def test_metrics_collection(self, sample_config, sample_environment):
        """Test that metrics are properly collected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            sim = LoggedSelectorGCSimulation(
                sample_config,
                sample_environment,
                log_dir=log_dir
            )
            
            # Record some metrics
            sim.logger.metrics.record("final_price", 600)
            sim.logger.metrics.record("final_price", 700)
            sim.logger.metrics.record("final_price", 550)
            sim.logger.metrics.record("deal_reached", True)
            
            # Get summary
            summary = sim.logger.metrics.get_summary()
            
            assert "final_price" in summary
            assert "deal_reached" in summary
            
            price_stats = summary["final_price"]
            assert price_stats["count"] == 3
            assert price_stats["mean"] == (600 + 700 + 550) / 3
            assert price_stats["min"] == 550
            assert price_stats["max"] == 700
            
            deal_stats = summary["deal_reached"]
            assert deal_stats["count"] == 1
            assert deal_stats["last"] == True


class TestLoggedSimulationIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def full_config(self):
        """Full configuration with InformationReturnAgent included."""
        return {
            "model": "gpt-4o",
            "agents": [
                {
                    "name": "Buyer",
                    "utility_class": "BuyerAgent", 
                    "description": "A buyer looking for a good deal",
                    "prompt": "You are a buyer negotiating for a bike. Your maximum budget is $1000. Try to get the best price possible.",
                    "strategy": {"max_price": 1000},
                    "self_improve": False
                },
                {
                    "name": "Seller",
                    "utility_class": "SellerAgent",
                    "description": "A seller trying to maximize profit", 
                    "prompt": "You are a seller with a bike. Your target price is $800. Try to get the best price possible.",
                    "strategy": {"target_price": 800},
                    "self_improve": False
                }
            ],
            "output_variables": [
                {"name": "final_price", "type": "Number"},
                {"name": "deal_reached", "type": "Boolean"}
            ],
            "termination_condition": "Agreement reached or negotiation failed"
        }
    
    @patch('autogen.ConversableAgent.a_initiate_chat')
    @pytest.mark.asyncio
    async def test_full_simulation_cycle(self, mock_chat, full_config):
        """Test a full simulation cycle with all logging components.""" 
        # Mock a realistic chat history
        mock_chat_result = Mock()
        mock_chat_result.chat_history = [
            {"name": "Buyer", "content": "Hi, I'm interested in buying your bike. What's your asking price?"},
            {"name": "Seller", "content": "Hello! I'm selling this bike for $900. It's in excellent condition."},
            {"name": "Buyer", "content": "That's a bit high for my budget. Would you consider $650?"},
            {"name": "Seller", "content": "I can't go that low. How about $800? That's my bottom line."},
            {"name": "Buyer", "content": "I can do $750. That's the most I can afford."},
            {"name": "Seller", "content": "Alright, $750 works for me. We have a deal!"},
            {"name": "InformationReturnAgent", "content": '{\n  "final_price": 750,\n  "deal_reached": true\n}'}
        ]
        mock_chat.return_value = mock_chat_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            
            # Create and run simulation
            sim = LoggedSelectorGCSimulation(
                full_config,
                environment={"runs": [], "outputs": {}},
                max_messages=10,
                min_messages=5,
                log_dir=log_dir
            )
            
            result = await sim.run()
            
            # Verify result structure
            assert result is not None
            assert "messages" in result
            assert "output_variables" in result
            assert len(result["messages"]) == 7
            assert len(result["output_variables"]) == 2
            
            # Verify output variables
            outputs = {var["name"]: var["value"] for var in result["output_variables"]}
            assert outputs["final_price"] == 750
            assert outputs["deal_reached"] == True
            
            # Verify log files were created
            expected_files = [
                "simulation.log",
                "messages.json",
                "agent_Buyer.json", 
                "agent_Seller.json",
                "metrics.json",
                "simulation_info.json"
            ]
            
            for filename in expected_files:
                assert (log_dir / filename).exists(), f"Missing log file: {filename}"
            
            # Verify message logging
            with open(log_dir / "messages.json") as f:
                logged_messages = json.load(f)
                assert len(logged_messages) == 7
                assert logged_messages[0]["agent"] == "Buyer"
                assert "bike" in logged_messages[0]["message"].lower()
            
            # Verify agent-specific logging
            with open(log_dir / "agent_Buyer.json") as f:
                buyer_data = json.load(f)
                assert buyer_data["agent_name"] == "Buyer"
                assert len(buyer_data["actions"]) >= 3  # init + messages
                assert len(buyer_data["utility_history"]) >= 2  # initial + final
                
                # Check that final utility makes sense for buyer
                final_utility = buyer_data["utility_history"][-1]["utility_value"]
                # Buyer utility = 1.0 - (750 / 1000) = 0.25
                expected_utility = 1.0 - (750 / 1000)
                assert abs(final_utility - expected_utility) < 0.001
            
            with open(log_dir / "agent_Seller.json") as f:
                seller_data = json.load(f)
                assert seller_data["agent_name"] == "Seller"
                assert len(seller_data["actions"]) >= 3  # init + messages
                assert len(seller_data["utility_history"]) >= 2  # initial + final
                
                # Check that final utility makes sense for seller
                final_utility = seller_data["utility_history"][-1]["utility_value"]
                # Seller utility = min(750 / 800, 1.0) = 0.9375
                expected_utility = min(750 / 800, 1.0)
                assert abs(final_utility - expected_utility) < 0.001
            
            # Verify metrics collection
            with open(log_dir / "metrics.json") as f:
                metrics = json.load(f)
                assert "Buyer_utility" in metrics
                assert "Seller_utility" in metrics
                assert "final_price" in metrics
                assert "deal_reached" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])