import pytest
from agents import NegotiationCoachBuyerAgent, NegotiationCoachSellerAgent


def get_utility_from_output_vars(output_vars, agent_name):
    """Extract utility value for agent from output_variables (handles both dict and list formats)."""
    if isinstance(output_vars, dict):
        # Old format
        return output_vars.get('utility', {}).get(agent_name, 0)
    else:
        # New format - find utility in list
        for var in output_vars:
            if var.get('name') == 'utility':
                utility_value = var.get('value', {})
                if isinstance(utility_value, dict):
                    return utility_value.get(agent_name, 0)
        return 0

@pytest.fixture
def base_environment():
    return {
        "runs": [{
            "run_id": 1,
            "output_variables": {
                "deal_reached": True,
                "final_price": 80,
                "utility": {}
            },
            "messages": []
        }]
    }


def test_buyer_utility_success(base_environment):
    agent = NegotiationCoachBuyerAgent(name="Buyer", system_prompt="Buyer", strategy={"budget": 100})
    updated_env = agent.compute_utility(base_environment.copy())
    output_vars = updated_env["runs"][-1]["output_variables"]
    utility = get_utility_from_output_vars(output_vars, "Buyer")
    assert utility == 0.2 

def test_buyer_utility_no_deal(base_environment):
    base_environment["runs"][-1]["output_variables"]["deal_reached"] = False
    agent = NegotiationCoachBuyerAgent(name="Buyer", system_prompt="Buyer", strategy={"budget": 100})
    updated_env = agent.compute_utility(base_environment.copy())
    output_vars = updated_env["runs"][-1]["output_variables"]
    utility = get_utility_from_output_vars(output_vars, "Buyer")
    assert utility == 0.0

def test_seller_utility_zero_denominator(base_environment):
    agent = NegotiationCoachSellerAgent(name="Seller", system_prompt="Seller", strategy={"asking_price": 80, "seller_floor": 80})
    updated_env = agent.compute_utility(base_environment.copy())
    output_vars = updated_env["runs"][-1]["output_variables"]
    utility = get_utility_from_output_vars(output_vars, "Seller")
    assert utility == 0.0

from unittest.mock import patch, Mock

def test_strategies_are_isolated(base_environment):
    buyer = NegotiationCoachBuyerAgent(name="Buyer", system_prompt="Buyer", strategy={"budget": 100})
    seller = NegotiationCoachSellerAgent(name="Seller", system_prompt="Seller", strategy={"asking_price": 100, "seller_floor": 80})
    env = base_environment.copy()
    env = buyer.compute_utility(env)
    env = seller.compute_utility(env)
    with patch.object(buyer._client.chat.completions, "create") as mock_buyer, \
         patch.object(seller._client.chat.completions, "create") as mock_seller:
        mock_buyer.return_value = Mock(choices=[Mock(message=Mock(content="Buyer unique strategy"))])
        mock_seller.return_value = Mock(choices=[Mock(message=Mock(content="Seller unique strategy"))])
        buyer.learn_from_feedback(env)
        seller.learn_from_feedback(env)
    assert "strategies_buyer" in env
    assert "strategies_seller" in env
    assert env["strategies_buyer"] != env["strategies_seller"]

def test_prompt_is_updated(base_environment):
    agent = NegotiationCoachBuyerAgent(name="Buyer", system_prompt="Buyer", strategy={"budget": 100})
    updated_env = agent.compute_utility(base_environment.copy())
    with patch.object(agent._client.chat.completions, "create") as mock_create:
        mock_create.return_value = Mock(choices=[Mock(message=Mock(content="Strategy for prompt"))])
        agent.learn_from_feedback(updated_env)
    assert "Negotiation strategies" in agent.system_prompt