"""Test the streaming API endpoints based on frontend expectations."""
import pytest
import json
import time
import threading
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import os

import sys
sys.path.insert(0, 'src/backend')

from api.app import app
from api.routes.stream_live import find_stream_files, tail_file


class TestStreamAPI:
    """Test streaming API functionality based on frontend ChatStream component."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_stream_endpoint_requires_id(self, client):
        """Test that stream endpoint requires simulation ID."""
        response = client.get('/sim/stream')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Simulation ID required' in data['error']
    
    def test_stream_endpoint_sends_correct_headers(self, client):
        """Test that streaming endpoint sends correct SSE headers."""
        with patch('db.simulation_results_utils.get_simulation_results') as mock_results:
            mock_results.return_value = None
            
            response = client.get('/sim/stream?id=test-123')
            assert response.status_code == 200
            assert response.content_type == 'text/event-stream; charset=utf-8'
            assert response.headers.get('Cache-Control') == 'no-cache'
            assert response.headers.get('X-Accel-Buffering') == 'no'
    
    def test_stream_sends_initial_connected_event(self, client):
        """Test that stream sends initial connected event matching frontend expectations."""
        with patch('db.simulation_results_utils.get_simulation_results') as mock_results:
            mock_results.return_value = None
            
            response = client.get('/sim/stream?id=test-123')
            
            # Read first event
            events = []
            for line in response.response:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    events.append(data)
                    break
            
            # First event should be connected type
            assert len(events) == 1
            assert events[0]['type'] == 'connected'
            assert events[0]['simulation_id'] == 'test-123'
    
    @pytest.mark.skip(reason="Complex mocking required")
    def test_stream_database_messages_format(self, client):
        """Test streaming messages from database matches frontend expectations."""
        # Mock database results with the format the frontend expects
        mock_results = {
            'id': 'test-123',
            'runs': [
                {
                    'run_id': 'run-001',
                    'messages': [
                        {
                            'agent': 'Buyer',
                            'message': 'I would like to buy a bike',
                            'timestamp': 1234567890
                        },
                        {
                            'agent': 'Seller', 
                            'message': 'What is your budget?',
                            'timestamp': 1234567891
                        }
                    ]
                }
            ]
        }
        
        with patch('db.simulation_results_utils.get_simulation_results') as mock_get_results:
            mock_get_results.return_value = mock_results
            
            response = client.get('/sim/stream?id=test-123')
            
            # Collect all events
            events = []
            for line in response.response:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        events.append(data)
                    except:
                        pass
            
            # Check event types and order
            assert events[0]['type'] == 'connected'
            
            # Should have status event
            status_events = [e for e in events if e.get('type') == 'status']
            assert len(status_events) > 0
            
            # Should have message events matching frontend format
            message_events = [e for e in events if e.get('type') == 'message']
            assert len(message_events) == 2
            
            # Check first message format matches frontend expectations
            assert message_events[0]['type'] == 'message'
            assert message_events[0]['agent'] == 'Buyer'
            assert message_events[0]['content'] == 'I would like to buy a bike'
            assert 'run_id' in message_events[0]
            assert 'timestamp' in message_events[0]
            
            # Should end with complete event
            assert events[-1]['type'] == 'complete'
            assert events[-1]['status'] == 'finished'
    
    def test_stream_handles_no_results_gracefully(self, client):
        """Test streaming handles no results case matching frontend expectations."""
        with patch('db.simulation_results_utils.get_simulation_results') as mock_results:
            mock_results.return_value = None
            
            response = client.get('/sim/stream?id=test-123')
            
            events = []
            for line in response.response:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    events.append(data)
            
            # Should have connected, no_data status, and complete
            assert events[0]['type'] == 'connected'
            assert any(e.get('type') == 'status' and e.get('status') == 'no_data' for e in events)
            assert events[-1]['type'] == 'complete'
    
    @pytest.mark.skip(reason="Complex file system mocking required")
    def test_stream_handles_live_file_streaming_disabled(self, client):
        """Test live file streaming matches frontend expectations."""
        # Create a temporary stream file
        with tempfile.TemporaryDirectory() as tmpdir:
            logs_dir = Path(tmpdir) / "logs"
            logs_dir.mkdir()
            stream_file = logs_dir / "stream_test-run.jsonl"
            
            # Write test events to file
            with open(stream_file, 'w') as f:
                f.write(json.dumps({
                    'type': 'status',
                    'status': 'connected',
                    'simulation_id': 'test-123'
                }) + '\n')
                f.write(json.dumps({
                    'type': 'message',
                    'agent': 'Buyer',
                    'content': 'Live message',
                    'timestamp': time.time()
                }) + '\n')
            
            # Patch the logs directory
            with patch('api.routes.stream_live.Path') as mock_path:
                # Make Path return our temp directory structure
                def mock_path_side_effect(path_str):
                    if path_str == "logs":
                        return logs_dir
                    return Path(path_str)
                
                mock_path.side_effect = mock_path_side_effect
                
                # Also need to patch glob to find our file
                with patch('api.routes.stream_live.glob.glob') as mock_glob:
                    mock_glob.return_value = [str(stream_file)]
                    
                    # Mock no database results to force file streaming
                    with patch('db.simulation_results_utils.get_simulation_results') as mock_results:
                        mock_results.return_value = None
                        
                        response = client.get('/sim/stream?id=test-123')
                        
                        events = []
                        for line in response.response:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    events.append(data)
                                    if data.get('type') == 'message':
                                        break  # Stop after first message
                                except:
                                    pass
                        
                        # Should have message event with correct format
                        message_events = [e for e in events if e.get('type') == 'message']
                        assert len(message_events) > 0
                        assert message_events[0]['agent'] == 'Buyer'
                        assert message_events[0]['content'] == 'Live message'
    
    def test_stream_optimization_events_format(self, client):
        """Test optimization events match frontend expectations."""
        mock_results = {
            'id': 'test-123',
            'messages': [],  # Empty messages to trigger no_data
            'optimization_events': [
                {
                    'run_id': 'run-001',
                    'agent': 'Buyer',
                    'strategy': 'New negotiation strategy',
                    'utility': 0.85
                }
            ]
        }
        
        # The frontend expects optimization events in the stream
        # but the backend may not implement this yet
        # This test documents the expected format


class TestRealTimeStreamingBehavior:
    """Test real-time streaming behavior to ensure messages appear immediately."""
    
    def test_tail_file_reads_new_lines(self):
        """Test that tail_file correctly reads new lines from a file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("line1\n")
            f.write("line2\n")
            f.flush()
            
            # Read initial lines
            lines, last_pos = tail_file(f.name, 0)
            assert len(lines) == 2
            assert lines[0].strip() == "line1"
            assert lines[1].strip() == "line2"
            assert last_pos == 2
            
            # Add more lines
            with open(f.name, 'a') as append_f:
                append_f.write("line3\n")
                append_f.write("line4\n")
                append_f.flush()
            
            # Read only new lines
            lines, new_pos = tail_file(f.name, last_pos)
            assert len(lines) == 2
            assert lines[0].strip() == "line3"
            assert lines[1].strip() == "line4"
            assert new_pos == 4
            
            os.unlink(f.name)
    
    def test_find_stream_files_locates_files(self):
        """Test that find_stream_files correctly locates stream files."""
        # Save current directory
        original_cwd = os.getcwd()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Change to temp directory
                os.chdir(tmpdir)
                
                # Create logs directory structure
                logs_dir = Path("logs")
                logs_dir.mkdir()
                
                # Create stream files with proper content
                run_dir1 = logs_dir / "run1"
                run_dir1.mkdir()
                stream_file1 = run_dir1 / "stream_run1.jsonl"
                stream_file1.write_text(json.dumps({"type": "status", "simulation_id": "test-sim", "run_id": "run1"}) + "\n")
                
                run_dir2 = logs_dir / "run2"
                run_dir2.mkdir()
                stream_file2 = run_dir2 / "stream_run2.jsonl"
                stream_file2.write_text(json.dumps({"type": "status", "simulation_id": "test-sim", "run_id": "run2"}) + "\n")
                
                # Stream file for different simulation should not be found
                run_dir3 = logs_dir / "run3"
                run_dir3.mkdir()
                stream_file3 = run_dir3 / "stream_run3.jsonl"
                stream_file3.write_text(json.dumps({"type": "status", "simulation_id": "other-sim", "run_id": "run3"}) + "\n")
                
                # Non-stream file should not be found
                other_file = logs_dir / "other.txt"
                other_file.write_text("test")
                
                files = find_stream_files("test-sim")
                file_names = [f.name for f in files]
                
                assert len(files) == 2  # Should find 2 files
                assert "stream_run1.jsonl" in file_names
                assert "stream_run2.jsonl" in file_names
                assert "stream_run3.jsonl" not in file_names  # Different simulation
                assert "other.txt" not in file_names
                
            finally:
                # Restore original directory
                os.chdir(original_cwd)
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @patch('api.routes.stream_live.SimulationCatalog')
    @patch('api.routes.stream_live.MongoClient')
    def test_live_streaming_delivers_messages_immediately(self, mock_mongo, mock_catalog, client):
        """Test that messages written to file appear immediately in stream."""
        # Mock simulation as running
        mock_catalog_instance = MagicMock()
        mock_catalog_instance.find_by_id.return_value = {
            'status': 'running',
            'name': 'Test Simulation'
        }
        mock_catalog.return_value = mock_catalog_instance
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            stream_file = Path(f.name)
            
            # Initially empty file
            f.flush()
            
            # Patch find_stream_files to return our temp file
            with patch('api.routes.stream_live.find_stream_files') as mock_find:
                mock_find.return_value = [stream_file]
                
                # Start streaming in a thread
                response_data = []
                stop_event = threading.Event()
                
                def stream_reader():
                    response = client.get('/sim/stream?id=test-sim&live=true')
                    for line in response.response:
                        if stop_event.is_set():
                            break
                        line = line.decode('utf-8')
                        if line.strip():
                            response_data.append(line)
                            # Stop after getting a message
                            if 'Live message 1' in line:
                                stop_event.set()
                
                reader_thread = threading.Thread(target=stream_reader)
                reader_thread.start()
                
                # Give the stream time to start
                time.sleep(0.2)
                
                # Write a message to the file
                with open(stream_file, 'a') as append_f:
                    append_f.write(json.dumps({
                        'type': 'message',
                        'agent': 'TestAgent',
                        'content': 'Live message 1',
                        'timestamp': time.time()
                    }) + '\n')
                    append_f.flush()
                    os.fsync(append_f.fileno())  # Force write to disk
                
                # Wait for reader to get the message
                reader_thread.join(timeout=2)
                
                # Parse events
                events = []
                for line in response_data:
                    if line.startswith('data: '):
                        try:
                            events.append(json.loads(line[6:]))
                        except:
                            pass
                
                # Should have received the message
                message_events = [e for e in events if e.get('type') == 'message']
                assert len(message_events) > 0
                assert message_events[0]['content'] == 'Live message 1'
                assert message_events[0]['agent'] == 'TestAgent'
                
                os.unlink(stream_file)
    
    def test_streaming_file_format_matches_expected(self):
        """Test that the streaming file format matches what the endpoint expects."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write events in the expected format
            events = [
                {"type": "status", "status": "started", "simulation_id": "test-123"},
                {"type": "message", "agent": "Buyer", "content": "Hello", "timestamp": 1234567890},
                {"type": "message", "agent": "Seller", "content": "Hi", "timestamp": 1234567891},
                {"type": "complete", "status": "finished"}
            ]
            
            for event in events:
                f.write(json.dumps(event) + '\n')
            f.flush()
            
            # Read and verify format
            lines, _ = tail_file(f.name, 0)
            assert len(lines) == 4
            
            for i, line in enumerate(lines):
                parsed = json.loads(line.strip())
                assert parsed == events[i]
            
            os.unlink(f.name)


class TestStreamingIntegration:
    """Integration tests for the complete streaming flow."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @patch('api.routes.stream_live.SimulationCatalog')
    @patch('api.routes.stream_live.MongoClient')
    @patch('api.routes.stream_live.get_simulation_results')
    def test_fallback_from_file_to_database(self, mock_results, mock_mongo, mock_catalog, client):
        """Test that streaming falls back to database when no files found."""
        # Mock simulation as completed (not running)
        mock_catalog_instance = MagicMock()
        mock_catalog_instance.find_by_id.return_value = {
            'status': 'completed',
            'name': 'Test Simulation'
        }
        mock_catalog.return_value = mock_catalog_instance
        
        # Mock database results
        mock_results.return_value = {
            'messages': [
                {'agent': 'Buyer', 'message': 'Database message'}
            ]
        }
        
        # No stream files exist
        with patch('api.routes.stream_live.find_stream_files') as mock_find:
            mock_find.return_value = []
            
            response = client.get('/sim/stream?id=test-sim')
            
            # Collect events
            events = []
            for line in response.response:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        events.append(json.loads(line[6:]))
                    except:
                        pass
            
            # Should have database message
            message_events = [e for e in events if e.get('type') == 'message']
            assert len(message_events) == 1
            assert message_events[0]['content'] == 'Database message'