"""Real-time message streaming for simulations."""
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import queue


class RealtimeStreamer:
    """Manages real-time streaming of simulation messages."""
    
    # Class-level storage for active streams
    _active_streams: Dict[str, 'SimulationStream'] = {}
    _lock = threading.Lock()
    
    @classmethod
    def register_simulation(cls, simulation_id: str, run_id: str) -> 'SimulationStream':
        """Register a new simulation for streaming."""
        with cls._lock:
            stream_key = f"{simulation_id}_{run_id}"
            if stream_key not in cls._active_streams:
                cls._active_streams[stream_key] = SimulationStream(simulation_id, run_id)
            return cls._active_streams[stream_key]
    
    @classmethod
    def get_stream(cls, simulation_id: str, run_id: Optional[str] = None) -> Optional['SimulationStream']:
        """Get stream for a simulation."""
        with cls._lock:
            if run_id:
                return cls._active_streams.get(f"{simulation_id}_{run_id}")
            
            # Find any stream for this simulation
            for key, stream in cls._active_streams.items():
                if key.startswith(f"{simulation_id}_"):
                    return stream
            return None
    
    @classmethod
    def unregister_simulation(cls, simulation_id: str, run_id: str):
        """Remove a simulation from active streams."""
        with cls._lock:
            stream_key = f"{simulation_id}_{run_id}"
            if stream_key in cls._active_streams:
                cls._active_streams[stream_key].close()
                del cls._active_streams[stream_key]


class SimulationStream:
    """Handles streaming for a single simulation."""
    
    def __init__(self, simulation_id: str, run_id: str):
        self.simulation_id = simulation_id
        self.run_id = run_id
        self.messages = queue.Queue()
        self.is_active = True
        self.start_time = time.time()
        
    def add_message(self, agent_name: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the stream."""
        if not self.is_active:
            return
            
        message = {
            "type": "message",
            "agent": agent_name,
            "content": content,
            "timestamp": time.time(),
            "run_id": self.run_id,
            "metadata": metadata or {}
        }
        self.messages.put(message)
        
    def add_status(self, status: str, details: Optional[Dict] = None):
        """Add a status update to the stream."""
        if not self.is_active:
            return
            
        status_msg = {
            "type": "status",
            "status": status,
            "timestamp": time.time(),
            "details": details or {}
        }
        self.messages.put(status_msg)
        
    def get_messages(self, timeout: float = 0.1) -> List[Dict]:
        """Get all pending messages."""
        messages = []
        try:
            while True:
                msg = self.messages.get(timeout=timeout)
                messages.append(msg)
        except queue.Empty:
            pass
        return messages
    
    def mark_complete(self):
        """Mark the stream as complete."""
        self.add_status("complete", {"duration": time.time() - self.start_time})
        self.is_active = False
        
    def close(self):
        """Close the stream."""
        self.is_active = False


class StreamingGroupChatManager:
    """GroupChatManager wrapper that streams messages in real-time."""
    
    def __init__(self, original_manager, simulation_id: str, run_id: str):
        self.original_manager = original_manager
        self.simulation_id = simulation_id
        self.run_id = run_id
        self.stream = RealtimeStreamer.register_simulation(simulation_id, run_id)
        
        # Wrap the original manager's methods
        self._wrap_manager()
        
    def _wrap_manager(self):
        """Wrap key methods to intercept messages."""
        # Store original methods
        self._original_receive = self.original_manager.receive
        self._original_send = self.original_manager.send
        
        # Replace with streaming versions
        self.original_manager.receive = self._streaming_receive
        self.original_manager.send = self._streaming_send
        
    def _streaming_receive(self, message, sender, request_reply=False, silent=False):
        """Intercept received messages for streaming."""
        # Stream the message
        if isinstance(message, dict):
            agent_name = getattr(sender, 'name', 'Unknown')
            content = message.get('content', '')
            self.stream.add_message(agent_name, content, {"action": "receive"})
        
        # Call original method
        return self._original_receive(message, sender, request_reply, silent)
        
    def _streaming_send(self, message, recipient, request_reply=False, silent=False):
        """Intercept sent messages for streaming."""
        # Stream the message
        if isinstance(message, dict):
            agent_name = self.original_manager.name
            content = message.get('content', '')
            self.stream.add_message(agent_name, content, {"action": "send"})
        
        # Call original method
        return self._original_send(message, recipient, request_reply, silent)
    
    def __getattr__(self, name):
        """Proxy all other attributes to the original manager."""
        return getattr(self.original_manager, name)