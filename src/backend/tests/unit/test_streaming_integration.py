"""Integration test for real-time streaming functionality."""
import json
import time
import os
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, 'src/backend')

from engine.message_hook import FileStreamHook, MessageHookManager


def test_streaming_integration():
    """Test the complete streaming integration flow."""
    
    # Create a temporary directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to temp directory to isolate test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            
            # Create logs directory
            logs_dir = Path("logs")
            run_id = "test-run-123"
            simulation_id = "test-sim-456"
            
            # Create the hook manager and file stream hook
            hook_manager = MessageHookManager()
            log_dir = logs_dir / run_id
            file_hook = FileStreamHook(log_dir)
            hook_manager.add_hook(file_hook)
            
            # Start simulation
            hook_manager.on_simulation_start(simulation_id, run_id)
            
            # Send some messages
            messages = [
                ("Agent1", "Hello, I want to buy a bike"),
                ("Agent2", "What's your budget?"),
                ("Agent1", "Around $500"),
                ("Agent2", "I have a great bike for $450")
            ]
            
            for i, (agent, content) in enumerate(messages):
                hook_manager.on_message(agent, content, {
                    "message_index": i,
                    "simulation_id": simulation_id,
                    "run_id": run_id,
                    "timestamp": time.time()
                })
            
            # End simulation
            hook_manager.on_simulation_end(simulation_id, run_id, {"status": "completed"})
            
            # Verify the stream file was created
            stream_file = log_dir / f"stream_{run_id}.jsonl"
            assert stream_file.exists(), f"Stream file {stream_file} should exist"
            
            # Read and verify contents
            with open(stream_file, 'r') as f:
                lines = f.readlines()
            
            # Should have: 1 start event + 4 messages + 1 end event = 6 lines
            assert len(lines) == 6, f"Expected 6 lines, got {len(lines)}"
            
            # Parse and verify each event
            events = [json.loads(line) for line in lines]
            
            # First event should be status
            assert events[0]["type"] == "status"
            assert events[0]["status"] == "started"
            assert events[0]["simulation_id"] == simulation_id
            
            # Next 4 should be messages
            for i in range(1, 5):
                assert events[i]["type"] == "message"
                assert events[i]["agent"] == messages[i-1][0]
                assert events[i]["content"] == messages[i-1][1]
            
            # Last event should be complete
            assert events[-1]["type"] == "complete"
            
            print("✅ Streaming integration test passed!")
            
            # Test that find_stream_files can locate this file
            from api.routes.stream_live import find_stream_files
            
            found_files = find_stream_files(simulation_id)
            assert len(found_files) == 1, f"Should find 1 file, found {len(found_files)}"
            assert found_files[0].name == f"stream_{run_id}.jsonl"
            
            print("✅ Stream file discovery test passed!")
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_streaming_integration()