"""Pytest configuration and fixtures."""
import pytest
import os
import tempfile
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def temp_env():
    """Temporary environment variables for testing."""
    with patch.dict(os.environ, {
        "DB_CONNECTION_STRING": "mongodb://localhost:27017/test_db",
        "OPENAI_API_KEY": "test-openai-key"
    }):
        yield


@pytest.fixture
def sample_simulation_config():
    """Sample simulation configuration for testing."""
    return {
        "name": "Test Simulation",
        "agents": [
            {
                "name": "TestBuyer",
                "description": "Test buyer agent",
                "prompt": "You are a test buyer",
                "utility_class": "BuyerAgent",
                "strategy": {"max_price": 50000}
            },
            {
                "name": "TestSeller", 
                "description": "Test seller agent",
                "prompt": "You are a test seller",
                "utility_class": "SellerAgent",
                "strategy": {"target_price": 45000}
            }
        ],
        "termination_condition": "Deal reached or rejected",
        "output_variables": [
            {"name": "final_price", "type": "Number"},
            {"name": "deal_reached", "type": "Boolean"}
        ]
    }


@pytest.fixture
def sample_environment():
    """Sample environment data for testing."""
    return {
        "runs": [
            {
                "run_id": 1,
                "messages": [
                    {"agent": "TestBuyer", "message": "I'd like to buy this car for $40,000"},
                    {"agent": "TestSeller", "message": "I can do $48,000"},
                    {"agent": "TestBuyer", "message": "How about $45,000?"},
                    {"agent": "TestSeller", "message": "Deal!"}
                ]
            }
        ],
        "output_variables": {
            "final_price": 45000,
            "deal_reached": True
        }
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    from unittest.mock import Mock
    
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = '''
    {
        "num_runs": 5,
        "config": {
            "name": "Generated Simulation",
            "agents": [
                {
                    "name": "GeneratedAgent",
                    "description": "An AI-generated agent",
                    "prompt": "You are a generated agent"
                }
            ],
            "termination_condition": "Task complete",
            "output_variables": [
                {"name": "result", "type": "String"}
            ]
        }
    }
    '''
    return response


@pytest.fixture
def temporary_config_file():
    """Create a temporary config file for testing."""
    config_data = {
        "name": "InformationReturnAgent",
        "description": "Extracts structured information",
        "prompt": "Extract information in format: {output_variables_str} when {termination_condition}"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset imports between tests to avoid module caching issues."""
    import sys
    modules_to_reset = [m for m in sys.modules.keys() if m.startswith(('utils', 'agents', 'engine', 'api'))]
    for module in modules_to_reset:
        if module in sys.modules:
            del sys.modules[module]
    yield