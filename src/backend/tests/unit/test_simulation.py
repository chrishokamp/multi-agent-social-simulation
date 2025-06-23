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
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "Format: {output_variables_str} when {termination_condition}"}')
    @patch('engine.simulation.UtilityAgent')
    @patch('engine.simulation.BuyerAgent')
    def test_simulation_initialization(self, mock_buyer_agent, mock_utility_agent, mock_file, sample_config):
        """Test simulation initialization."""
        from engine.simulation import SelectorGCSimulation
        
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
        mock_result.chat_history = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi there"),
            Mock(source="Agent1", content="How are you?"),
            Mock(source="Agent2", content="I'm good"),
            Mock(source="InformationReturnAgent", content='Final result: {"result": "success", "score": 85}')
        ]
        
        with patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
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
        mock_result.chat_history = [
            Mock(source="Agent1", content="Hello")
        ]
        
        with patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
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
        mock_result.chat_history = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi"),
            Mock(source="Agent3", content="More chat"),
            Mock(source="Agent4", content="Even more"),
            Mock(source="InformationReturnAgent", content="Invalid JSON: {result: incomplete")
        ]
        
        with patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
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
        mock_result.chat_history = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi"),
            Mock(source="Agent3", content="More"),
            Mock(source="Agent4", content="Even more"),
            Mock(source="InformationReturnAgent", content='{"result": null, "score": "Unspecified", "final": "done"}')
        ]
        
        with patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
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
    @patch('engine.simulation.ConversableAgent.a_initiate_chat')
    async def test_run_simulation(self, mock_initiate):
        """Test running a simulation."""
        from engine.simulation import SelectorGCSimulation
        
        # Mock the console and simulation result
        mock_simulation_result = Mock()
        mock_simulation_result.chat_history = [
            Mock(source="Agent1", content="Hello"),
            Mock(source="Agent2", content="Hi"),
            Mock(source="Agent3", content="More"),
            Mock(source="Agent4", content="Even more"),
            Mock(source="InformationReturnAgent", content='{"result": "success"}')
        ]
        mock_initiate.return_value = mock_simulation_result
        
        with patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent'), \
             patch('engine.simulation.GroupChat') as mock_group_chat, \
             patch('engine.simulation.GroupChatManager'):
            
            sim = SelectorGCSimulation({"agents": [], "output_variables": [], "termination_condition": "done"}, {})
            
            # Mock the group chat messages to match the mocked chat history
            sim.group_chat.messages = [
                {"name": "Agent1", "content": "Hello"},
                {"name": "Agent2", "content": "Hi"},
                {"name": "Agent3", "content": "More"},
                {"name": "Agent4", "content": "Even more"},
                {"name": "InformationReturnAgent", "content": '{"result": "success"}'}
            ]
            
            result = await sim.run()
        
        assert result is not None
        assert "messages" in result
        assert "output_variables" in result

    @pytest.mark.asyncio  
    @patch('engine.simulation.ConversableAgent')
    async def test_run_uses_group_chat_messages(self, mock_conversable_agent):
        """Test that run() uses group chat messages instead of starter-manager conversation."""
        from engine.simulation import SelectorGCSimulation
        
        # Mock the starter agent
        mock_starter_instance = Mock()
        mock_conversable_agent.return_value = mock_starter_instance
        
        # Mock the initiate_chat result (this would only have starter-manager conversation)
        mock_chat_result = Mock()
        mock_chat_result.chat_history = [
            {"name": "starter", "content": "Begin"}
        ]
        # Make a_initiate_chat return an async mock
        async def mock_initiate_chat(*args, **kwargs):
            return mock_chat_result
        mock_starter_instance.a_initiate_chat = mock_initiate_chat
        
        # Create simulation with mocked components
        with patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent') as mock_utility_agent, \
             patch('engine.simulation.GroupChat') as mock_group_chat, \
             patch('engine.simulation.GroupChatManager'):
            
            # Mock agent instances
            mock_agent = Mock()
            mock_agent.compute_utility.return_value = 0.5
            mock_agent.system_prompt = "Test prompt"
            mock_utility_agent.return_value = mock_agent
            
            # Create simulation
            sim = SelectorGCSimulation({
                "agents": [
                    {"name": "Buyer", "description": "Test buyer", "prompt": "Buy something"},
                    {"name": "Seller", "description": "Test seller", "prompt": "Sell something"}
                ],
                "output_variables": [{"name": "price", "type": "Number"}],
                "termination_condition": "DONE"
            }, {})
            
            # Mock the group chat messages (this contains the full conversation)
            sim.group_chat.messages = [
                {"name": "starter", "content": "Begin"},
                {"name": "Buyer", "content": "I want to buy"},
                {"name": "Seller", "content": "I have something to sell"},
                {"name": "Buyer", "content": "Deal at 100"},
                {"name": "InformationReturnAgent", "content": '{"price": 100}\nTERMINATE'}
            ]
            
            # Run simulation
            result = await sim.run()
            
            # Verify that the result contains all group chat messages
            assert result is not None
            assert len(result["messages"]) == 5
            assert result["messages"][0]["agent"] == "starter"
            assert result["messages"][1]["agent"] == "Buyer"
            assert result["messages"][2]["agent"] == "Seller"
            assert result["messages"][3]["agent"] == "Buyer"
            assert result["messages"][4]["agent"] == "InformationReturnAgent"
            
            # Verify output variables were extracted correctly
            assert len(result["output_variables"]) == 1
            assert result["output_variables"][0]["name"] == "price"
            assert result["output_variables"][0]["value"] == 100

    def test_process_result_with_group_chat_result(self):
        """Test that _process_result works with GroupChatResult (the fix for the bug)."""
        from engine.simulation import SelectorGCSimulation
        
        # Create a GroupChatResult-like object (as created in the fixed run() method)
        class GroupChatResult:
            def __init__(self, messages):
                self.chat_history = messages
        
        # Mock group chat messages including InformationReturnAgent
        group_messages = [
            {"name": "starter", "content": "Begin"},
            {"name": "Buyer", "content": "I want to buy a bike"},
            {"name": "Seller", "content": "I have one for 400"},
            {"name": "Buyer", "content": "How about 350?"},
            {"name": "Seller", "content": "Let's meet at 375"},
            {"name": "Buyer", "content": "Deal! STOP_NEGOTIATION"},
            {"name": "InformationReturnAgent", "content": '{\n"final_price": 375,\n"deal_reached": true,\n"negotiation_rounds": 3\n}\nTERMINATE'}
        ]
        
        mock_result = GroupChatResult(group_messages)
        
        with patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "test"}'), \
             patch('engine.simulation.UtilityAgent'):
            
            sim = SelectorGCSimulation({
                "agents": [
                    {"name": "Buyer", "description": "Test buyer", "prompt": "Buy something"},
                    {"name": "Seller", "description": "Test seller", "prompt": "Sell something"}
                ],
                "output_variables": [
                    {"name": "final_price", "type": "Number"},
                    {"name": "deal_reached", "type": "Boolean"},
                    {"name": "negotiation_rounds", "type": "Number"}
                ],
                "termination_condition": "STOP_NEGOTIATION"
            }, {})
            
            result = sim._process_result(mock_result)
            
            # Verify the result is processed correctly
            assert result is not None
            assert len(result["messages"]) == 7
            
            # Verify InformationReturnAgent message was found and processed
            assert result["messages"][-1]["agent"] == "InformationReturnAgent"
            assert "TERMINATE" in result["messages"][-1]["message"]
            
            # Verify output variables were extracted
            assert len(result["output_variables"]) == 3
            output_dict = {var["name"]: var["value"] for var in result["output_variables"]}
            assert output_dict["final_price"] == 375
            assert output_dict["deal_reached"] is True
            assert output_dict["negotiation_rounds"] == 3

    @patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "Format: {output_variables_str} when {termination_condition}"}')
    @patch('engine.simulation.UtilityAgent')
    def test_termination_condition_from_config(self, mock_utility_agent, mock_file):
        """Test that termination_condition is correctly read from config."""
        from engine.simulation import SelectorGCSimulation
        
        mock_agent_instance = Mock()
        mock_agent_instance.compute_utility.return_value = 0.5
        mock_agent_instance.system_prompt = "Test prompt"
        mock_utility_agent.return_value = mock_agent_instance
        
        # Test with termination_condition in config
        config_with_termination = {
            "agents": [
                {"name": "TestAgent", "description": "Test", "prompt": "Test agent"}
            ],
            "termination_condition": "CUSTOM_TERMINATION",
            "output_variables": [{"name": "result", "type": "String"}]
        }
        
        sim = SelectorGCSimulation(config_with_termination, environment={})
        
        # Check that the InformationReturnAgent was added with the correct termination condition
        assert sim.config["agents"][-1]["name"] == "InformationReturnAgent"
        assert "CUSTOM_TERMINATION" in sim.config["agents"][-1]["prompt"]
        
    @patch('builtins.open', new_callable=mock_open, read_data='{"name": "InformationReturnAgent", "description": "Test", "prompt": "Format: {output_variables_str} when {termination_condition}"}')
    @patch('engine.simulation.UtilityAgent')
    def test_termination_condition_default(self, mock_utility_agent, mock_file):
        """Test that termination_condition defaults to TERMINATE when not in config."""
        from engine.simulation import SelectorGCSimulation
        
        mock_agent_instance = Mock()
        mock_agent_instance.compute_utility.return_value = 0.5
        mock_agent_instance.system_prompt = "Test prompt"
        mock_utility_agent.return_value = mock_agent_instance
        
        # Test without termination_condition in config
        config_without_termination = {
            "agents": [
                {"name": "TestAgent", "description": "Test", "prompt": "Test agent"}
            ],
            "output_variables": [{"name": "result", "type": "String"}]
        }
        
        sim = SelectorGCSimulation(config_without_termination, environment={})
        
        # Check that the InformationReturnAgent was added with the default termination condition
        assert sim.config["agents"][-1]["name"] == "InformationReturnAgent"
        assert "TERMINATE" in sim.config["agents"][-1]["prompt"]


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