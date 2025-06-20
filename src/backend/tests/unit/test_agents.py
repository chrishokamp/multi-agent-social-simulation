"""Unit tests for utility agents."""
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from agents import UtilityAgent, BuyerAgent, SellerAgent


class TestUtilityAgent:
    """Test the base UtilityAgent class."""
    
    @patch('agents.client_for_endpoint')
    def test_utility_agent_initialization(self, mock_client_func):
        """Test UtilityAgent initialization."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = UtilityAgent(
            system_prompt="You are a test agent",
            name="TestAgent",
            description="Test agent",
            model_client=mock_client,
            strategy={"test_param": 100}
        )
        
        assert agent.name == "TestAgent"
        assert agent.system_prompt == "You are a test agent"
        assert agent.strategy == {"test_param": 100}
        assert agent._utility_history == []
    
    def test_compute_utility_default(self):
        """Test default utility computation returns 0."""
        with patch('agents.client_for_endpoint') as mock_client_func:
            mock_client = Mock()
            mock_client_func.return_value = (mock_client, "gpt-4o")
            
            agent = UtilityAgent(
                system_prompt="Test prompt",
                name="TestAgent",
                description="Test agent",
                model_client=mock_client
            )
            
            utility = agent.compute_utility()
            assert utility == 0.0
    
    @patch('agents.client_for_endpoint')
    def test_learn_from_feedback_no_environment(self, mock_client_func):
        """Test learn_from_feedback with no environment."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = UtilityAgent(
            system_prompt="Original prompt",
            name="TestAgent",
            description="Test agent",
            model_client=mock_client
        )
        
        # Should return early with no environment
        agent.learn_from_feedback(0.5, None)
        assert agent.system_prompt == "Original prompt"


class TestBuyerAgent:
    """Test the BuyerAgent class."""
    
    @patch('agents.client_for_endpoint')
    def test_buyer_utility_calculation(self, mock_client_func):
        """Test buyer utility calculation."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = BuyerAgent(
            system_prompt="You are a buyer",
            name="Buyer",
            description="Car buyer",
            model_client=mock_client,
            strategy={"max_price": 50000}
        )
        
        environment = {
            "outputs": {
                "final_price": 40000,
                "deal_reached": True
            }
        }
        
        utility = agent.compute_utility(environment)
        # Utility = 1.0 - (40000 / 50000) = 0.2
        assert abs(utility - 0.2) < 0.001
    
    @patch('agents.client_for_endpoint')
    def test_buyer_no_deal_utility(self, mock_client_func):
        """Test buyer utility when no deal is reached."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = BuyerAgent(
            system_prompt="You are a buyer",
            name="Buyer",
            description="Car buyer",
            model_client=mock_client,
            strategy={"max_price": 50000}
        )
        
        environment = {
            "outputs": {
                "final_price": 45000,
                "deal_reached": False
            }
        }
        
        utility = agent.compute_utility(environment)
        assert utility == 0.0


class TestSellerAgent:
    """Test the SellerAgent class."""
    
    @patch('agents.client_for_endpoint')
    def test_seller_utility_calculation(self, mock_client_func):
        """Test seller utility calculation."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = SellerAgent(
            system_prompt="You are a seller",
            name="Seller",
            description="Car seller",
            model_client=mock_client,
            strategy={"target_price": 45000}
        )
        
        environment = {
            "outputs": {
                "final_price": 48000,
                "deal_reached": True
            }
        }
        
        utility = agent.compute_utility(environment)
        # Utility = min(48000 / 45000, 1.0) = 1.0 (capped)
        assert utility == 1.0
    
    @patch('agents.client_for_endpoint')
    def test_seller_below_target_utility(self, mock_client_func):
        """Test seller utility when selling below target."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = SellerAgent(
            system_prompt="You are a seller",
            name="Seller",
            description="Car seller",
            model_client=mock_client,
            strategy={"target_price": 45000}
        )
        
        environment = {
            "outputs": {
                "final_price": 40000,
                "deal_reached": True  
            }
        }
        
        utility = agent.compute_utility(environment)
        # Utility = 40000 / 45000 = 0.889
        assert abs(utility - 0.8889) < 0.001
    
    @patch('agents.client_for_endpoint')
    def test_seller_no_deal_utility(self, mock_client_func):
        """Test seller utility when no deal is reached.""" 
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = SellerAgent(
            system_prompt="You are a seller",
            name="Seller",
            description="Car seller",
            model_client=mock_client,
            strategy={"target_price": 45000}
        )
        
        environment = {
            "outputs": {
                "final_price": 50000,
                "deal_reached": False
            }
        }
        
        utility = agent.compute_utility(environment)
        assert utility == 0.0