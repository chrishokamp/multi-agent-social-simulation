"""Unit tests for SimulationQueue."""
import pytest
from unittest.mock import Mock, patch, call
import time
import uuid

from db.simulation_queue import SimulationQueue


class TestSimulationQueue:
    """Test the SimulationQueue database operations."""
    
    @pytest.fixture
    def mock_mongo_client(self):
        """Create a mock MongoDB client."""
        client = Mock()
        db = Mock()
        queue_collection = Mock()
        configs_collection = Mock()
        
        # Use configure_mock or direct assignment for magic methods
        client.__getitem__ = Mock(return_value=db)
        db.__getitem__ = Mock(side_effect=lambda name: {
            "queue": queue_collection,
            "configs": configs_collection
        }.get(name, Mock()))
        
        return client, queue_collection, configs_collection
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid simulation configuration."""
        return {
            "name": "Test Simulation",
            "agents": [
                {
                    "name": "Agent 1",
                    "description": "First agent",
                    "prompt": "You are agent 1"
                },
                {
                    "name": "Agent 2",
                    "description": "Second agent",
                    "prompt": "You are agent 2"
                }
            ],
            "termination_condition": "max_messages",
            "output_variables": [
                {"name": "result", "type": "string"}
            ]
        }
    
    def test_init(self, mock_mongo_client):
        """Test SimulationQueue initialization."""
        client, queue_collection, _ = mock_mongo_client
        
        sim_queue = SimulationQueue(client)
        
        assert sim_queue.queue_collection == queue_collection
        client.__getitem__.assert_called_with("simulation_gym")
    
    @patch('db.simulation_queue.uuid.uuid4')
    @patch('db.simulation_queue.time.time')
    def test_insert_valid_config(self, mock_time, mock_uuid, mock_mongo_client, valid_config):
        """Test inserting a valid configuration."""
        client, queue_collection, configs_collection = mock_mongo_client
        mock_uuid.return_value = "test-uuid-1234"
        mock_time.return_value = 1234567890
        
        sim_queue = SimulationQueue(client)
        result = sim_queue.insert(valid_config, 5)
        
        assert result == "test-uui"
        
        # Verify queue insertion
        queue_collection.insert_one.assert_called_once()
        inserted_doc = queue_collection.insert_one.call_args[0][0]
        assert inserted_doc["simulation_id"] == "test-uui"
        assert inserted_doc["timestamp"] == 1234567890
        assert inserted_doc["remaining_runs"] == 5
        assert inserted_doc["config"]["agents"][0]["name"] == "Agent_1"  # Spaces replaced
        assert inserted_doc["config"]["agents"][1]["name"] == "Agent_2"
        
        # Verify config storage
        configs_collection.update_one.assert_called_once_with(
            {"simulation_id": "test-uui"},
            {"$set": {"config": inserted_doc["config"]}},
            upsert=True
        )
    
    def test_insert_invalid_config_missing_name(self, mock_mongo_client):
        """Test inserting a configuration without name."""
        client, _, _ = mock_mongo_client
        sim_queue = SimulationQueue(client)
        
        invalid_config = {"agents": [], "termination_condition": "max", "output_variables": []}
        result = sim_queue.insert(invalid_config, 5)
        
        assert result is None
    
    def test_insert_invalid_config_insufficient_agents(self, mock_mongo_client, valid_config):
        """Test inserting a configuration with less than 2 agents."""
        client, _, _ = mock_mongo_client
        sim_queue = SimulationQueue(client)
        
        valid_config["agents"] = [valid_config["agents"][0]]  # Only one agent
        result = sim_queue.insert(valid_config, 5)
        
        assert result is None
    
    def test_insert_invalid_config_missing_agent_fields(self, mock_mongo_client, valid_config):
        """Test inserting a configuration with incomplete agent."""
        client, _, _ = mock_mongo_client
        sim_queue = SimulationQueue(client)
        
        valid_config["agents"][0].pop("prompt")  # Remove required field
        result = sim_queue.insert(valid_config, 5)
        
        assert result is None
    
    def test_insert_invalid_num_runs(self, mock_mongo_client, valid_config):
        """Test inserting with invalid number of runs."""
        client, _, _ = mock_mongo_client
        sim_queue = SimulationQueue(client)
        
        result = sim_queue.insert(valid_config, 0)
        
        assert result is None
    
    def test_retrieve_next_empty_queue(self, mock_mongo_client):
        """Test retrieving from empty queue."""
        client, queue_collection, _ = mock_mongo_client
        queue_collection.find_one.return_value = None
        
        sim_queue = SimulationQueue(client)
        result = sim_queue.retrieve_next()
        
        assert result is None
        queue_collection.find_one.assert_called_once_with(sort=[("timestamp", 1)])
    
    def test_retrieve_next_with_remaining_runs(self, mock_mongo_client, valid_config):
        """Test retrieving next simulation with remaining runs."""
        client, queue_collection, _ = mock_mongo_client
        queue_collection.find_one.return_value = {
            "simulation_id": "test-123",
            "timestamp": 1234567890,
            "remaining_runs": 3,
            "config": valid_config
        }
        
        sim_queue = SimulationQueue(client)
        sim_id, config = sim_queue.retrieve_next()
        
        assert sim_id == "test-123"
        assert config == valid_config
        
        # Verify decrement
        queue_collection.update_one.assert_called_once_with(
            {"simulation_id": "test-123"},
            {"$inc": {"remaining_runs": -1}}
        )
        # Should not delete
        queue_collection.delete_one.assert_not_called()
    
    def test_retrieve_next_last_run(self, mock_mongo_client, valid_config):
        """Test retrieving the last run of a simulation."""
        client, queue_collection, _ = mock_mongo_client
        queue_collection.find_one.return_value = {
            "simulation_id": "test-123",
            "timestamp": 1234567890,
            "remaining_runs": 1,
            "config": valid_config
        }
        
        sim_queue = SimulationQueue(client)
        sim_id, config = sim_queue.retrieve_next()
        
        assert sim_id == "test-123"
        assert config == valid_config
        
        # Verify decrement and deletion
        queue_collection.update_one.assert_called_once_with(
            {"simulation_id": "test-123"},
            {"$inc": {"remaining_runs": -1}}
        )
        queue_collection.delete_one.assert_called_once_with(
            {"simulation_id": "test-123"}
        )
    
    def test_retrieve_full_job_empty_queue(self, mock_mongo_client):
        """Test retrieving full job from empty queue."""
        client, queue_collection, _ = mock_mongo_client
        queue_collection.find_one.return_value = None
        
        sim_queue = SimulationQueue(client)
        result = sim_queue.retrieve_full_job()
        
        assert result is None
        queue_collection.find_one.assert_called_once_with(sort=[("timestamp", 1)])
    
    def test_retrieve_full_job_success(self, mock_mongo_client, valid_config):
        """Test successfully retrieving a full job."""
        client, queue_collection, _ = mock_mongo_client
        queue_collection.find_one.return_value = {
            "simulation_id": "test-456",
            "timestamp": 1234567890,
            "remaining_runs": 10,
            "config": valid_config
        }
        
        sim_queue = SimulationQueue(client)
        sim_id, config, num_runs = sim_queue.retrieve_full_job()
        
        assert sim_id == "test-456"
        assert config == valid_config
        assert num_runs == 10
        
        # Verify job was deleted
        queue_collection.delete_one.assert_called_once_with(
            {"simulation_id": "test-456"}
        )
    
    def test_delete_existing(self, mock_mongo_client):
        """Test deleting an existing simulation."""
        client, queue_collection, _ = mock_mongo_client
        queue_collection.delete_one.return_value = Mock(deleted_count=1)
        
        sim_queue = SimulationQueue(client)
        result = sim_queue.delete("test-789")
        
        assert result is True
        queue_collection.delete_one.assert_called_once_with(
            {"simulation_id": "test-789"}
        )
    
    def test_delete_non_existing(self, mock_mongo_client):
        """Test deleting a non-existing simulation."""
        client, queue_collection, _ = mock_mongo_client
        queue_collection.delete_one.return_value = Mock(deleted_count=0)
        
        sim_queue = SimulationQueue(client)
        result = sim_queue.delete("test-999")
        
        assert result is False
        queue_collection.delete_one.assert_called_once_with(
            {"simulation_id": "test-999"}
        )
    
    @patch('db.simulation_queue.time.time')
    def test_insert_with_id(self, mock_time, mock_mongo_client, valid_config):
        """Test inserting with a specific ID."""
        client, queue_collection, configs_collection = mock_mongo_client
        mock_time.return_value = 1234567890
        
        sim_queue = SimulationQueue(client)
        result = sim_queue.insert_with_id("custom-id", valid_config, 3)
        
        assert result == "custom-id"
        
        # Verify queue insertion
        queue_collection.insert_one.assert_called_once()
        inserted_doc = queue_collection.insert_one.call_args[0][0]
        assert inserted_doc["simulation_id"] == "custom-id"
        assert inserted_doc["timestamp"] == 1234567890
        assert inserted_doc["remaining_runs"] == 3