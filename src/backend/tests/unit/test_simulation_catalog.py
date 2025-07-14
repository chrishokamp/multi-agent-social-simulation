"""Unit tests for SimulationCatalog class."""
import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from db.simulation_catalog import SimulationCatalog


class TestSimulationCatalog(unittest.TestCase):
    """Test cases for SimulationCatalog."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock MongoDB client
        self.mock_client = MagicMock()
        self.mock_db = MagicMock()
        self.mock_collection = MagicMock()
        
        # Configure the mock client
        self.mock_client.__getitem__.return_value = self.mock_db
        self.mock_db.__getitem__.return_value = self.mock_collection
        
        # Create SimulationCatalog instance
        self.catalog = SimulationCatalog(self.mock_client)
        self.catalog.catalog_collection = self.mock_collection
        
    def test_find_by_id_success(self):
        """Test find_by_id returns document when found."""
        # Mock data
        simulation_id = "test-sim-123"
        expected_doc = {
            "simulation_id": simulation_id,
            "name": "Test Simulation",
            "expected_runs": 5,
            "progress_percentage": 40,
            "status": "running"
        }
        
        # Configure mock to return document
        self.mock_collection.find_one.return_value = expected_doc
        
        # Call method
        result = self.catalog.find_by_id(simulation_id)
        
        # Assertions
        self.assertEqual(result, expected_doc)
        self.mock_collection.find_one.assert_called_once_with({"simulation_id": simulation_id})
        
    def test_find_by_id_not_found(self):
        """Test find_by_id returns None when document not found."""
        # Configure mock to return None
        self.mock_collection.find_one.return_value = None
        
        # Call method
        result = self.catalog.find_by_id("non-existent-id")
        
        # Assertions
        self.assertIsNone(result)
        self.mock_collection.find_one.assert_called_once_with({"simulation_id": "non-existent-id"})
        
    def test_find_by_id_handles_exception(self):
        """Test find_by_id handles database exceptions gracefully."""
        # Configure mock to raise exception
        self.mock_collection.find_one.side_effect = Exception("Database error")
        
        # Call method
        result = self.catalog.find_by_id("test-id")
        
        # Assertions
        self.assertIsNone(result)
        self.mock_collection.find_one.assert_called_once()
        
    def test_insert_creates_document(self):
        """Test insert method creates a new document."""
        simulation_id = "new-sim-456"
        name = "New Simulation"
        num_runs = 10
        
        # Call method
        result = self.catalog.insert(simulation_id, name, num_runs)
        
        # Assertions
        self.assertEqual(result, simulation_id)
        # Check that insert_one was called
        self.mock_collection.insert_one.assert_called_once()
        # Get the actual call arguments
        call_args = self.mock_collection.insert_one.call_args[0][0]
        # Check all expected fields
        self.assertEqual(call_args["simulation_id"], simulation_id)
        self.assertEqual(call_args["name"], name)
        self.assertEqual(call_args["expected_runs"], num_runs)
        self.assertEqual(call_args["progress_percentage"], 0)
        self.assertEqual(call_args["status"], "queued")
        # created_at should be present
        self.assertIn("created_at", call_args)
        
    def test_insert_validates_parameters(self):
        """Test insert method validates input parameters."""
        # Test with invalid parameters
        self.assertIsNone(self.catalog.insert("", "Name", 5))
        self.assertIsNone(self.catalog.insert("id", "", 5))
        self.assertIsNone(self.catalog.insert("id", "Name", 0))
        self.assertIsNone(self.catalog.insert("id", "Name", -1))
        
        # Verify insert_one was not called
        self.mock_collection.insert_one.assert_not_called()
        
    def test_exists_returns_true_when_found(self):
        """Test exists returns True when simulation exists."""
        self.mock_collection.find_one.return_value = {"simulation_id": "test-id"}
        
        result = self.catalog.exists("test-id")
        
        self.assertTrue(result)
        self.mock_collection.find_one.assert_called_once_with({"simulation_id": "test-id"})
        
    def test_exists_returns_false_when_not_found(self):
        """Test exists returns False when simulation doesn't exist."""
        self.mock_collection.find_one.return_value = None
        
        result = self.catalog.exists("test-id")
        
        self.assertFalse(result)
        self.mock_collection.find_one.assert_called_once_with({"simulation_id": "test-id"})
        
    def test_delete_removes_document(self):
        """Test delete removes a document and returns count."""
        mock_result = Mock()
        mock_result.deleted_count = 1
        self.mock_collection.delete_one.return_value = mock_result
        
        result = self.catalog.delete("test-id")
        
        self.assertEqual(result, 1)
        self.mock_collection.delete_one.assert_called_once_with({"simulation_id": "test-id"})
        
    def test_get_all_returns_catalog_list(self):
        """Test get_all returns formatted catalog list."""
        from datetime import datetime
        mock_data = [
            {
                "simulation_id": "sim1",
                "name": "Simulation 1",
                "expected_runs": 5,
                "progress_percentage": 100,
                "status": "completed",
                "created_at": datetime(2025, 1, 1, 12, 0, 0)
            },
            {
                "simulation_id": "sim2",
                "name": "Simulation 2", 
                "expected_runs": 3,
                "progress_percentage": 66
                # No status field - should default to "unknown"
            }
        ]
        
        # Create a mock cursor that returns our data
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter(mock_data)
        self.mock_collection.find.return_value.sort.return_value = mock_cursor
        
        result = self.catalog.get_all()
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["simulation_id"], "sim1")
        self.assertEqual(result[0]["status"], "completed")
        self.assertEqual(result[0]["created_at"], "2025-01-01T12:00:00")
        self.assertEqual(result[1]["progress_percentage"], 66)
        self.assertEqual(result[1]["status"], "unknown")  # Default value
        self.assertIsNone(result[1]["created_at"])  # No created_at in mock data
        
    def test_update_progress_calculates_percentage(self):
        """Test update_progress calculates and updates percentage."""
        simulation_id = "test-sim"
        
        # Mock catalog entry
        self.mock_collection.find_one.return_value = {"expected_runs": 10}
        
        # Mock simulation results
        mock_results = Mock()
        mock_results.retrieve.return_value = [None] * 7  # 7 completed runs
        self.catalog.simulation_results = mock_results
        
        result = self.catalog.update_progress(simulation_id)
        
        self.assertEqual(result, 70)  # 7/10 * 100 = 70%
        self.mock_collection.update_one.assert_called_once_with(
            {"simulation_id": simulation_id},
            {"$set": {"progress_percentage": 70}}
        )
    
    def test_update_progress_sets_completed_status(self):
        """Test update_progress sets status to completed at 100%."""
        simulation_id = "test-sim"
        
        # Mock catalog entry
        self.mock_collection.find_one.return_value = {"expected_runs": 5}
        
        # Mock simulation results - all runs completed
        mock_results = Mock()
        mock_results.retrieve.return_value = [None] * 5  # 5 completed runs
        self.catalog.simulation_results = mock_results
        
        result = self.catalog.update_progress(simulation_id)
        
        self.assertEqual(result, 100)  # 5/5 * 100 = 100%
        self.mock_collection.update_one.assert_called_once_with(
            {"simulation_id": simulation_id},
            {"$set": {"progress_percentage": 100, "status": "completed"}}
        )
    
    def test_update_progress_handles_missing_document(self):
        """Test update_progress handles missing document gracefully."""
        self.mock_collection.find_one.return_value = None
        
        result = self.catalog.update_progress("non-existent")
        
        self.assertEqual(result, 0)
        self.mock_collection.update_one.assert_not_called()
    
    def test_update_status(self):
        """Test update_status method."""
        simulation_id = "test-sim"
        new_status = "running"
        
        result = self.catalog.update_status(simulation_id, new_status)
        
        self.assertTrue(result)
        self.mock_collection.update_one.assert_called_once_with(
            {"simulation_id": simulation_id},
            {"$set": {"status": new_status}}
        )


if __name__ == '__main__':
    unittest.main()