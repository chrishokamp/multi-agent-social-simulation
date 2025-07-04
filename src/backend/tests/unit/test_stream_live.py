"""Unit tests for the enhanced streaming API with live file streaming."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pytest

# Add parent directory to match the imports in stream_live.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.stream_live import find_stream_files, tail_file, stream_live_bp
from flask import Flask


class TestStreamingUtilities:
    """Test utility functions for streaming."""
    
    def test_find_stream_files_searches_file_contents(self):
        """Test that find_stream_files checks file contents for simulation_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create logs directory
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()
            
            # Create subdirectories for different runs
            run1_dir = logs_dir / "run-001"
            run1_dir.mkdir()
            run2_dir = logs_dir / "run-002"
            run2_dir.mkdir()
            
            # Create stream files with different simulation IDs
            stream1 = run1_dir / "stream_run-001.jsonl"
            stream1.write_text(json.dumps({
                "type": "status",
                "simulation_id": "test-sim-123",
                "run_id": "run-001"
            }) + "\n")
            
            stream2 = run2_dir / "stream_run-002.jsonl"
            stream2.write_text(json.dumps({
                "type": "status",
                "simulation_id": "test-sim-456",
                "run_id": "run-002"
            }) + "\n")
            
            # Non-stream file should not be included
            other_file = logs_dir / "other.txt"
            other_file.write_text("not a stream file")
            
            # Change to temp directory for test
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Find files for test-sim-123
                files = find_stream_files("test-sim-123")
                assert len(files) == 1
                assert files[0].name == "stream_run-001.jsonl"
                
                # Find files for test-sim-456
                files = find_stream_files("test-sim-456")
                assert len(files) == 1
                assert files[0].name == "stream_run-002.jsonl"
                
                # Non-existent simulation should return empty
                files = find_stream_files("non-existent")
                assert len(files) == 0
                
            finally:
                os.chdir(original_cwd)
    
    def test_find_stream_files_handles_errors(self):
        """Test that find_stream_files handles file read errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()
            
            # Create a file that will cause read error
            bad_file = logs_dir / "stream_bad.jsonl"
            bad_file.write_text("not json")
            
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Should not crash on bad files
                files = find_stream_files("test-sim")
                assert len(files) == 0
                
            finally:
                os.chdir(original_cwd)
    
    def test_tail_file_reads_from_position(self):
        """Test tail_file reads from specified position."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Write test content
            for i in range(5):
                f.write(f"Line {i}\n")
            f.flush()
            
            try:
                # Read from beginning
                lines, new_pos = tail_file(f.name, 0)
                assert len(lines) == 5
                assert new_pos == 5
                assert lines[0].strip() == "Line 0"
                assert lines[4].strip() == "Line 4"
                
                # Read from middle
                lines, new_pos = tail_file(f.name, 3)
                assert len(lines) == 2
                assert new_pos == 5
                assert lines[0].strip() == "Line 3"
                assert lines[1].strip() == "Line 4"
                
                # Read from end (no new lines)
                lines, new_pos = tail_file(f.name, 5)
                assert len(lines) == 0
                assert new_pos == 5
                
            finally:
                os.unlink(f.name)
    
    def test_tail_file_handles_growing_file(self):
        """Test tail_file handles files that grow between reads."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Initial content
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.flush()
            
            try:
                # First read
                lines, pos = tail_file(f.name, 0)
                assert len(lines) == 2
                assert pos == 2
                
                # Add more content
                with open(f.name, 'a') as append_f:
                    append_f.write("Line 3\n")
                    append_f.write("Line 4\n")
                
                # Read new content only
                lines, new_pos = tail_file(f.name, pos)
                assert len(lines) == 2
                assert new_pos == 4
                assert lines[0].strip() == "Line 3"
                assert lines[1].strip() == "Line 4"
                
            finally:
                os.unlink(f.name)


class TestStreamingEndpoint:
    """Test the streaming endpoint behavior."""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app."""
        # Import here to avoid circular imports
        from api.app import app as flask_app
        flask_app.config['TESTING'] = True
        return flask_app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    def test_stream_requires_simulation_id(self, client):
        """Test that stream endpoint requires simulation ID parameter."""
        response = client.get('/sim/stream')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Simulation ID required' in data['error']
    
    @patch('api.routes.stream_live.get_simulation_results')
    @patch('api.routes.stream_live.SimulationCatalog')
    @patch('api.routes.stream_live.MongoClient')
    def test_stream_sends_connected_event(self, mock_mongo, mock_catalog, mock_results, client):
        """Test that stream sends initial connected event."""
        # Mock empty results
        mock_results.return_value = None
        mock_catalog_instance = MagicMock()
        mock_catalog_instance.find_by_id.return_value = None
        mock_catalog.return_value = mock_catalog_instance
        
        response = client.get('/sim/stream?id=test-sim-123')
        assert response.status_code == 200
        
        # Parse first event
        data = response.data.decode('utf-8')
        lines = data.split('\n')
        
        # Find first data line
        for line in lines:
            if line.startswith('data: '):
                event = json.loads(line[6:])
                assert event['type'] == 'connected'
                assert event['simulation_id'] == 'test-sim-123'
                break
    
    def test_stream_response_format(self, client):
        """Test that stream returns proper SSE format."""
        # We can't easily mock the database, so just test the response format
        response = client.get('/sim/stream?id=test-123')
        
        # Should return SSE content type
        assert response.content_type == 'text/event-stream; charset=utf-8'
        
        # Should contain SSE formatted data
        data = response.data.decode('utf-8')
        assert 'data: ' in data
        assert '{"type":' in data
    
    def test_stream_handles_missing_simulation(self, client):
        """Test that stream handles missing simulation gracefully."""
        # Request a non-existent simulation
        response = client.get('/sim/stream?id=non-existent-sim')
        data = response.data.decode('utf-8')
        
        # Should still return valid SSE
        assert response.status_code == 200
        assert 'data: ' in data
        
        # Should indicate no data
        assert 'no_data' in data or 'complete' in data
    
    def test_stream_status_endpoint_exists(self, client):
        """Test that status endpoint exists and returns JSON."""
        # Test with non-existent simulation
        response = client.get('/sim/stream/status?id=test-sim')
        
        # Should return JSON
        assert response.content_type == 'application/json'
        
        # If not found, should return 404
        if response.status_code == 404:
            assert response.json['status'] == 'not_found'
        else:
            # If found (unlikely in test), should have expected fields
            data = response.json
            assert 'status' in data
            assert 'has_stream_files' in data
    
    def test_stream_status_not_found(self, client):
        """Test status endpoint returns 404 for non-existent simulation."""
        with patch('api.routes.stream_live.SimulationCatalog') as mock_catalog, \
             patch('api.routes.stream_live.MongoClient'):
            
            mock_catalog_instance = MagicMock()
            mock_catalog_instance.find_by_id.return_value = None
            mock_catalog.return_value = mock_catalog_instance
            
            response = client.get('/sim/stream/status?id=non-existent')
            assert response.status_code == 404
            assert response.json['status'] == 'not_found'




class TestStreamingIntegration:
    """Integration tests for streaming functionality."""
    
    def test_find_stream_files_integration(self):
        """Test that find_stream_files works with real file structure."""
        # Create test directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()
            
            # Create run directories
            run1_dir = logs_dir / "run-001"
            run1_dir.mkdir()
            run2_dir = logs_dir / "run-002" 
            run2_dir.mkdir()
            
            # Create stream files with simulation IDs in content
            sim_id = "test-sim-xyz"
            
            stream1 = run1_dir / "stream_run-001.jsonl"
            stream1.write_text(json.dumps({
                "type": "status",
                "simulation_id": sim_id,
                "run_id": "run-001"
            }) + "\n")
            
            stream2 = run2_dir / "stream_run-002.jsonl"
            stream2.write_text(json.dumps({
                "type": "status",
                "simulation_id": "different-sim",
                "run_id": "run-002"
            }) + "\n")
            
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                # Find files for our simulation
                from api.routes.stream_live import find_stream_files
                files = find_stream_files(sim_id)
                
                # Should find only the file with matching simulation_id
                assert len(files) == 1
                assert files[0].name == "stream_run-001.jsonl"
                
            finally:
                os.chdir(original_cwd)
    
    def test_tail_file_integration(self):
        """Test tail_file with real file operations."""
        from api.routes.stream_live import tail_file
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Write initial content
            f.write("Line 1\n")
            f.write("Line 2\n")
            f.flush()
            
            # Read from beginning
            lines, pos = tail_file(f.name, 0)
            assert len(lines) == 2
            assert lines[0].strip() == "Line 1"
            
            # Append more content
            with open(f.name, 'a') as append_f:
                append_f.write("Line 3\n")
            
            # Read only new content
            lines, new_pos = tail_file(f.name, pos)
            assert len(lines) == 1
            assert lines[0].strip() == "Line 3"
            assert new_pos > pos
            
            os.unlink(f.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])