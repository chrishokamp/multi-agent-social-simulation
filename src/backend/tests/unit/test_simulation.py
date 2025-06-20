"""Unit tests for simulation engine."""
import pytest
import json
import uuid
from unittest.mock import patch, Mock, MagicMock, mock_open

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestSelectorGCSimulation:
    """Test the SelectorGCSimulation class."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample simulation configuration."""
        return {
            "name": "Test Simulation",
            "agents": [
                {
                    "name": "TestAgent1",
                    "description": "First test agent",
                    "prompt": "You are agent 1",
                    "utility_class": "UtilityAgent",
                    "strategy": {"param": 1}
                },
                {
                    "name": "TestAgent2", 
                    "description": "Second test agent",
                    "prompt": "You are agent 2",
                    "utility_class": "BuyerAgent",
                    "strategy": {"max_price": 50000},
                    "self_improve": True
                }
            ],
            "termination_condition": "Task completed",
            "output_variables": [
                {"name": "result", "type": "String"},
                {"name": "score", "type": "Number"}
            ]
        }
    
    @pytest.fixture
    def mock_information_return_agent(self):
        """Mock InformationReturnAgent configuration."""
        return {
            "name": "InformationReturnAgent",
            "description": "Returns structured output",
            "prompt": "Return output in format: {output_variables_str} when {termination_condition}"
        }
    
    @patch('engine.simulation.get_autogen_client')
    @patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "Format: {output_variables_str} when {termination_condition}"}')
    @patch('engine.simulation.UtilityAgent')
    @patch('engine.simulation.BuyerAgent')
    def test_simulation_initialization(self, mock_buyer_agent, mock_utility_agent, mock_file, mock_client, sample_config):
        """Test simulation initialization."""
        from engine.simulation import SelectorGCSimulation
        
        mock_client.return_value = Mock()
        mock_agent_instance = Mock()
        mock_agent_instance.compute_utility.return_value = 0.5
        mock_agent_instance.system_prompt = "Updated prompt"
        mock_agent_instance.learn_from_feedback = Mock()
        mock_utility_agent.return_value = mock_agent_instance
        mock_buyer_agent.return_value = mock_agent_instance
        
        sim = SelectorGCSimulation(sample_config, environment={})
        
        assert sim.config == sample_config
        assert len(sim.config["agents"]) == 3  # 2 original + InformationReturnAgent
        assert sim.config["agents"][-1]["name"] == "InformationReturnAgent"
        assert isinstance(sim.run_id, str)
    
    def test_process_result_success(self):
        """Test successful result processing."""
        from engine.simulation import SelectorGCSimulation
        
        # Mock simulation result
        mock_result = Mock()
        mock_result.messages = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi there"),
            Mock(source="Agent1", content="How are you?"),
            Mock(source="Agent2", content="I'm good"),
            Mock(source="InformationReturnAgent", content='Final result: {"result": "success", "score": 85}')
        ]
        
        with patch('engine.simulation.get_autogen_client'), \
             patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent'):
            
            sim = SelectorGCSimulation({
                "agents": [
                    {"name": "Agent1", "description": "Test agent 1", "prompt": "You are agent 1"},
                    {"name": "Agent2", "description": "Test agent 2", "prompt": "You are agent 2"}
                ], 
                "output_variables": [], 
                "termination_condition": "done"
            }, {})
            result = sim._process_result(mock_result)
        
        assert result is not None
        assert len(result["messages"]) == 5
        assert result["messages"][0]["agent"] == "Agent1"
        assert result["messages"][0]["message"] == "Hello"
        assert len(result["output_variables"]) == 2
        assert result["output_variables"][0]["name"] == "result"
        assert result["output_variables"][0]["value"] == "success"
    
    def test_process_result_too_few_messages(self):
        """Test result processing with too few messages."""
        from engine.simulation import SelectorGCSimulation
        
        mock_result = Mock()
        mock_result.messages = [
            Mock(source="Agent1", content="Hello")
        ]
        
        with patch('engine.simulation.get_autogen_client'), \
             patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent'):
            
            sim = SelectorGCSimulation({
                "agents": [
                    {"name": "Agent1", "description": "Test agent 1", "prompt": "You are agent 1"},
                    {"name": "Agent2", "description": "Test agent 2", "prompt": "You are agent 2"}
                ], 
                "output_variables": [], 
                "termination_condition": "done"
            }, {})
            sim.min_messages = 5
            result = sim._process_result(mock_result)
        
        assert result is None
    
    def test_process_result_invalid_json(self):
        """Test result processing with invalid JSON."""
        from engine.simulation import SelectorGCSimulation
        
        mock_result = Mock()
        mock_result.messages = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi"),
            Mock(source="Agent3", content="More chat"), 
            Mock(source="Agent4", content="Even more"),
            Mock(source="InformationReturnAgent", content="Invalid JSON: {result: incomplete")
        ]
        
        with patch('engine.simulation.get_autogen_client'), \
             patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent'):
            
            sim = SelectorGCSimulation({
                "agents": [
                    {"name": "Agent1", "description": "Test agent 1", "prompt": "You are agent 1"},
                    {"name": "Agent2", "description": "Test agent 2", "prompt": "You are agent 2"}
                ], 
                "output_variables": [], 
                "termination_condition": "done"
            }, {})
            result = sim._process_result(mock_result)
        
        assert result is None
    
    def test_process_result_handles_none_values(self):
        """Test result processing handles None and Unspecified values."""
        from engine.simulation import SelectorGCSimulation
        
        mock_result = Mock()
        mock_result.messages = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi"),
            Mock(source="Agent3", content="More"),
            Mock(source="Agent4", content="Even more"),
            Mock(source="InformationReturnAgent", content='{"result": null, "score": "Unspecified", "final": "done"}')
        ]
        
        with patch('engine.simulation.get_autogen_client'), \
             patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent'):
            
            sim = SelectorGCSimulation({
                "agents": [
                    {"name": "Agent1", "description": "Test agent 1", "prompt": "You are agent 1"},
                    {"name": "Agent2", "description": "Test agent 2", "prompt": "You are agent 2"}
                ], 
                "output_variables": [], 
                "termination_condition": "done"
            }, {})
            result = sim._process_result(mock_result)
        
        assert result is not None
        assert len(result["output_variables"]) == 3
        # Check that None and "Unspecified" are both converted to "Unspecified"
        values = [var["value"] for var in result["output_variables"]]
        assert "Unspecified" in values
        assert None not in values
    
    @pytest.mark.asyncio
    @patch('engine.simulation.Console')
    async def test_run_simulation(self, mock_console):
        """Test running a simulation."""
        from engine.simulation import SelectorGCSimulation
        
        # Mock the console and simulation result
        mock_simulation_result = Mock()
        mock_simulation_result.messages = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi"),
            Mock(source="Agent3", content="More"),
            Mock(source="Agent4", content="Even more"),
            Mock(source="InformationReturnAgent", content='{"result": "success"}')
        ]
        
        mock_console.return_value = mock_simulation_result
        
        with patch('engine.simulation.get_autogen_client'), \
             patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent'), \
             patch('engine.simulation.SelectorGroupChat'):
            
            sim = SelectorGCSimulation({"agents": [], "output_variables": [], "termination_condition": "done"}, {})
            result = await sim.run()
        
        assert result is not None
        assert "messages" in result
        assert "output_variables" in result


class TestUtilityClassRegistry:
    """Test the utility class registry."""
    
    def test_registry_contains_expected_classes(self):
        """Test that the registry contains expected utility agent classes."""
        from engine.simulation import _utility_class_registry
        from agents import UtilityAgent, BuyerAgent, SellerAgent
        
        assert "UtilityAgent" in _utility_class_registry
        assert "BuyerAgent" in _utility_class_registry
        assert "SellerAgent" in _utility_class_registry
        
        assert _utility_class_registry["UtilityAgent"] is UtilityAgent
        assert _utility_class_registry["BuyerAgent"] is BuyerAgent
        assert _utility_class_registry["SellerAgent"] is SellerAgent