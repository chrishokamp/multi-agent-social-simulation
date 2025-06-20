"""Unit tests for API routes."""
import pytest
import json
from unittest.mock import patch, Mock, MagicMock
from flask import Flask

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    # Mock MongoDB to avoid connection requirements
    with patch('pymongo.MongoClient'), \
         patch.dict(os.environ, {
             "DB_CONNECTION_STRING": "mongodb://localhost:27017/test_db",
             "OPENAI_API_KEY": "test-openai-key"
         }):
        from api.app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client


class TestGenConfigRoute:
    """Test the gen_config API route."""
    
    @patch('api.routes.gen_config.client_for_endpoint')
    def test_generate_config_success(self, mock_client_func, client):
        """Test successful config generation."""
        # Mock the OpenAI client
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = '''
        {
            "num_runs": 5,
            "config": {
                "name": "Test Simulation",
                "agents": [
                    {
                        "name": "Agent1",
                        "description": "Test agent",
                        "prompt": "You are a test agent"
                    }
                ],
                "termination_condition": "Test completed",
                "output_variables": [
                    {
                        "name": "result",
                        "type": "String"
                    }
                ]
            }
        }
        '''
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        response = client.post('/sim/gen_config', 
                            json={
                                "desc": "Create a simple test simulation",
                                "temperature": 0.7,
                                "top_p": 0.9
                            })
        
        assert response.status_code == 200
        data = response.get_json()
        assert "config" in data
        assert data["config"]["name"] == "Test Simulation"
    
    def test_generate_config_missing_desc(self, client):
        """Test config generation with missing description."""
        response = client.post('/sim/gen_config', json={})
        
        assert response.status_code == 400
        data = response.get_json()
        assert "desc not given" in data["message"]
    
    @patch('api.routes.gen_config.client_for_endpoint')
    def test_generate_config_client_failure(self, mock_client_func, client):
        """Test config generation when client fails."""
        mock_client_func.return_value = (None, None)
        
        with patch('api.routes.gen_config.run_sim', return_value=None):
            response = client.post('/sim/gen_config',
                                json={"desc": "Test simulation"})
            
            assert response.status_code == 503
            data = response.get_json()
            assert "Failed to generate configuration" in data["error"]


class TestGetConfigRoute:
    """Test the get_config API route."""
    
    @patch('api.routes.gen_config.db')
    def test_get_config_success(self, mock_db, client):
        """Test successful config retrieval."""
        mock_collection = Mock()
        mock_collection.find_one.return_value = {
            "simulation_id": "test-id",
            "config": {"name": "Test Config"}
        }
        mock_db.__getitem__.return_value = mock_collection
        
        response = client.get('/sim/config?id=test-id')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "Test Config"
    
    def test_get_config_missing_id(self, client):
        """Test config retrieval with missing ID."""
        response = client.get('/sim/config')
        
        assert response.status_code == 400
        data = response.get_json()
        assert "id not given" in data["message"]
    
    @patch('api.routes.gen_config.db')
    def test_get_config_not_found(self, mock_db, client):
        """Test config retrieval when config not found."""
        mock_collection = Mock()
        mock_collection.find_one.return_value = None
        mock_db.__getitem__.return_value = mock_collection
        
        response = client.get('/sim/config?id=nonexistent-id')
        
        assert response.status_code == 404
        data = response.get_json()
        assert "config not found" in data["message"]


class TestStopRoute:
    """Test the stop simulation API route."""
    
    @patch('api.routes.stop.simulation_queue')
    def test_stop_simulation_success(self, mock_queue, client):
        """Test successful simulation stop."""
        mock_queue.delete.return_value = True
        
        response = client.post('/sim/stop', json={"id": "test-sim-id"})
        
        assert response.status_code == 200
        data = response.get_json()
        assert "Cancelled simulation test-sim-id" in data["message"]
    
    def test_stop_simulation_missing_id(self, client):
        """Test stop simulation with missing ID."""
        response = client.post('/sim/stop', json={})
        
        assert response.status_code == 400
        data = response.get_json()
        assert "Missing simulation id" in data["error"]
    
    @patch('api.routes.stop.simulation_queue')
    def test_stop_simulation_not_found(self, mock_queue, client):
        """Test stop simulation when simulation not found."""
        mock_queue.delete.return_value = False
        
        response = client.post('/sim/stop', json={"id": "nonexistent-id"})
        
        assert response.status_code == 404
        data = response.get_json()
        assert "No pending simulation found" in data["message"]


class TestRunSimFunction:
    """Test the run_sim helper function."""
    
    @patch('api.routes.gen_config.client_for_endpoint')
    def test_run_sim_success(self, mock_client_func):
        """Test successful run_sim execution."""
        from api.routes.gen_config import run_sim
        
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = '{"test": "value"}'
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        result = run_sim("system prompt", "json prompt", "user prompt")
        
        assert result == '{"test": "value"}'
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('api.routes.gen_config.client_for_endpoint')
    def test_run_sim_retry_logic(self, mock_client_func):
        """Test run_sim retry logic on failures."""
        from api.routes.gen_config import run_sim
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = [
            Exception("Connection error"),
            Exception("Connection error"), 
            Mock(choices=[Mock(message=Mock(content='{"success": true}'))])
        ]
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        result = run_sim("system", "json", "user")
        
        assert result == '{"success": true}'
        assert mock_client.chat.completions.create.call_count == 3
    
    @patch('api.routes.gen_config.client_for_endpoint')
    def test_run_sim_max_retries_exceeded(self, mock_client_func):
        """Test run_sim when max retries are exceeded."""
        from api.routes.gen_config import run_sim
        
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Persistent error")
        mock_client_func.return_value = (mock_client, "gpt-4o")
        
        result = run_sim("system", "json", "user")
        
        assert result is None
        assert mock_client.chat.completions.create.call_count == 5