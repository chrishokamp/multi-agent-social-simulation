"""Test agent utility calculations with list-based output variables."""
import pytest
from unittest.mock import Mock, patch
from agents import (
    BuyerAgent, SellerAgent, 
    NegotiationCoachBuyerAgent, NegotiationCoachSellerAgent
)


class TestAgentUtilityWithListOutputVars:
    """Test agent utility calculations when output_variables is a list."""
    
    @patch('agents.client_for_endpoint')
    def test_buyer_utility_with_list_output_vars_deal_reached(self, mock_client_func):
        """Test buyer utility when deal is reached (list format)."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = BuyerAgent(
            system_prompt="You are a buyer",
            name="Buyer",
            description="Bike buyer",
            strategy={"max_price": 400}
        )

        environment = {
            "runs": [{
                "run_id": "test-run",
                "output_variables": [
                    {"name": "final_price", "value": "350"},
                    {"name": "deal_reached", "value": "TRUE"},
                    {"name": "negotiation_rounds", "value": "3"}
                ]
            }]
        }
        
        result = agent.compute_utility(environment)
        
        # Find utility in output variables
        utility_var = None
        for var in result["runs"][0]["output_variables"]:
            if var["name"] == "utility":
                utility_var = var
                break
        
        assert utility_var is not None
        assert "Buyer" in utility_var["value"]
        # Utility should be 1 - (350/400) = 0.125
        assert utility_var["value"]["Buyer"] == 0.125

    @patch('agents.client_for_endpoint')
    def test_buyer_utility_with_list_output_vars_no_deal(self, mock_client_func):
        """Test buyer utility when no deal is reached (list format)."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = BuyerAgent(
            system_prompt="You are a buyer",
            name="Buyer",
            description="Bike buyer",
            strategy={"max_price": 400}
        )

        environment = {
            "runs": [{
                "run_id": "test-run",
                "output_variables": [
                    {"name": "final_price", "value": "450"},
                    {"name": "deal_reached", "value": "FALSE"},
                    {"name": "negotiation_rounds", "value": "5"}
                ]
            }]
        }
        
        result = agent.compute_utility(environment)
        
        # Find utility in output variables
        utility_var = None
        for var in result["runs"][0]["output_variables"]:
            if var["name"] == "utility":
                utility_var = var
                break
        
        assert utility_var is not None
        assert "Buyer" in utility_var["value"]
        assert utility_var["value"]["Buyer"] == 0.0

    @patch('agents.client_for_endpoint')
    def test_seller_utility_with_list_output_vars(self, mock_client_func):
        """Test seller utility calculation (list format)."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = SellerAgent(
            system_prompt="You are a seller",
            name="Seller",
            description="Bike seller",
            strategy={"target_price": 400}
        )

        environment = {
            "runs": [{
                "run_id": "test-run",
                "output_variables": [
                    {"name": "final_price", "value": "380"},
                    {"name": "deal_reached", "value": "TRUE"}
                ]
            }]
        }
        
        result = agent.compute_utility(environment)
        
        # Find utility in output variables
        utility_var = None
        for var in result["runs"][0]["output_variables"]:
            if var["name"] == "utility":
                utility_var = var
                break
        
        assert utility_var is not None
        assert "Seller" in utility_var["value"]
        # Utility should be min(380/400, 1.0) = 0.95
        assert utility_var["value"]["Seller"] == 0.95

    @patch('agents.client_for_endpoint')
    def test_negotiation_coach_buyer_utility(self, mock_client_func):
        """Test NegotiationCoachBuyerAgent utility calculation."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = NegotiationCoachBuyerAgent(
            system_prompt="You are a buyer",
            name="CoachBuyer",
            description="Buyer with coach",
            strategy={"budget": 500}
        )

        environment = {
            "runs": [{
                "run_id": "test-run",
                "output_variables": [
                    {"name": "final_price", "value": "400"},
                    {"name": "deal_reached", "value": "TRUE"}
                ]
            }]
        }
        
        result = agent.compute_utility(environment)
        
        # Find utility in output variables
        utility_var = None
        for var in result["runs"][0]["output_variables"]:
            if var["name"] == "utility":
                utility_var = var
                break
        
        assert utility_var is not None
        assert "CoachBuyer" in utility_var["value"]
        # Utility should be (500-400)/500 = 0.2
        assert utility_var["value"]["CoachBuyer"] == 0.2

    @patch('agents.client_for_endpoint')
    def test_negotiation_coach_seller_utility(self, mock_client_func):
        """Test NegotiationCoachSellerAgent utility calculation."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = NegotiationCoachSellerAgent(
            system_prompt="You are a seller",
            name="CoachSeller",
            description="Seller with coach",
            strategy={"asking_price": 500}
        )

        environment = {
            "runs": [{
                "run_id": "test-run",
                "output_variables": [
                    {"name": "final_price", "value": "450"},
                    {"name": "deal_reached", "value": "TRUE"}
                ]
            }]
        }
        
        result = agent.compute_utility(environment)
        
        # Find utility in output variables
        utility_var = None
        for var in result["runs"][0]["output_variables"]:
            if var["name"] == "utility":
                utility_var = var
                break
        
        assert utility_var is not None
        assert "CoachSeller" in utility_var["value"]
        # With no seller_floor set, denominator is 500-500=0, so utility is 0.0
        assert utility_var["value"]["CoachSeller"] == 0.0

    @patch('agents.client_for_endpoint')
    def test_backward_compatibility_dict_format(self, mock_client_func):
        """Test that dict format is converted to list format."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = BuyerAgent(
            system_prompt="You are a buyer",
            name="Buyer",
            description="Bike buyer",
            strategy={"max_price": 400}
        )

        # Old dict format
        environment = {
            "runs": [{
                "run_id": "test-run",
                "output_variables": {
                    "final_price": 350,
                    "deal_reached": True
                }
            }]
        }
        
        result = agent.compute_utility(environment)
        
        # Should be converted to list format
        assert isinstance(result["runs"][0]["output_variables"], list)
        
        # Find utility in output variables
        utility_var = None
        for var in result["runs"][0]["output_variables"]:
            if var["name"] == "utility":
                utility_var = var
                break
        
        assert utility_var is not None
        assert "Buyer" in utility_var["value"]
        assert utility_var["value"]["Buyer"] == 0.125

    @patch('agents.client_for_endpoint')
    def test_utility_with_string_boolean_values(self, mock_client_func):
        """Test handling of string boolean values in output variables."""
        mock_client = Mock()
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = BuyerAgent(
            system_prompt="You are a buyer",
            name="Buyer",
            description="Bike buyer",
            strategy={"max_price": 400}
        )

        environment = {
            "runs": [{
                "run_id": "test-run",
                "output_variables": [
                    {"name": "final_price", "value": "350"},
                    {"name": "deal_reached", "value": "true"},  # lowercase
                    {"name": "negotiation_rounds", "value": "3"}
                ]
            }]
        }
        
        result = agent.compute_utility(environment)
        
        # Find utility in output variables
        utility_var = None
        for var in result["runs"][0]["output_variables"]:
            if var["name"] == "utility":
                utility_var = var
                break
        
        assert utility_var is not None
        assert "Buyer" in utility_var["value"]
        # Should handle "true" string correctly
        assert utility_var["value"]["Buyer"] == 0.125

    @patch('agents.client_for_endpoint')
    def test_learn_from_feedback_with_list_format(self, mock_client_func):
        """Test learn_from_feedback with list-based output variables."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "Updated prompt"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = Mock(choices=[Mock(message=mock_response)])
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        agent = BuyerAgent(
            system_prompt="You are a buyer",
            name="Buyer",
            description="Bike buyer",
            strategy={"max_price": 400}
        )
        agent.self_improve = True  # Set after initialization

        environment = {
            "runs": [{
                "run_id": "test-run",
                "messages": [
                    {"agent": "Buyer", "message": "I'll offer 300"},
                    {"agent": "Seller", "message": "I can do 350"},
                    {"agent": "Buyer", "message": "Deal!"}
                ],
                "output_variables": [
                    {"name": "final_price", "value": "350"},
                    {"name": "deal_reached", "value": "TRUE"},
                    {"name": "utility", "value": {"Buyer": 0.5}}
                ],
                "system_prompts": {"Buyer": "You are a buyer"}
            }]
        }
        
        # Mock the _client attribute to avoid real API calls
        agent._client = mock_client
        
        # Should not raise an error
        agent.learn_from_feedback(environment)
        
        # Verify the API was called
        assert mock_client.chat.completions.create.called