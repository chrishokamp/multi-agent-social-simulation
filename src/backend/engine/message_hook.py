"""Message hook system for intercepting autogen messages in real-time."""
import json
import time
import os
from typing import Dict, List, Optional, Callable
from pathlib import Path
import threading
from datetime import datetime


class MessageHook:
    """Base class for message hooks."""
    
    def on_message(self, agent_name: str, content: str, metadata: Dict):
        """Called when a message is sent/received."""
        pass
    
    def on_simulation_start(self, simulation_id: str, run_id: str):
        """Called when simulation starts."""
        pass
    
    def on_simulation_end(self, simulation_id: str, run_id: str, results: Dict):
        """Called when simulation ends."""
        pass
    
    def on_optimization(self, agent_name: str, optimization_data: Dict):
        """Called when an agent optimizes its prompt."""
        pass


class FileStreamHook(MessageHook):
    """Writes messages to a streaming file that can be tailed."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.stream_file = None
        self.lock = threading.Lock()
        
    def on_simulation_start(self, simulation_id: str, run_id: str):
        """Initialize the stream file."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stream_file = self.log_dir / f"stream_{run_id}.jsonl"
        # Write initial status
        self._write_event({
            "type": "status",
            "status": "started",
            "simulation_id": simulation_id,
            "run_id": run_id,
            "timestamp": time.time()
        })
        
    def on_message(self, agent_name: str, content: str, metadata: Dict):
        """Write message to stream file."""
        self._write_event({
            "type": "message",
            "agent": agent_name,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata
        })
        
    def on_simulation_end(self, simulation_id: str, run_id: str, results: Dict):
        """Write completion event."""
        self._write_event({
            "type": "complete",
            "status": "finished",
            "simulation_id": simulation_id,
            "run_id": run_id,
            "timestamp": time.time(),
            "results_summary": {
                "num_messages": len(results.get("messages", [])),
                "output_variables": len(results.get("output_variables", []))
            }
        })
    
    def on_optimization(self, agent_name: str, optimization_data: Dict):
        """Write optimization event."""
        self._write_event({
            "type": "optimization",
            "agent": agent_name,
            "optimization_data": optimization_data,
            "timestamp": time.time()
        })
        
    def _write_event(self, event: Dict):
        """Write an event to the stream file."""
        if not self.stream_file:
            return
            
        with self.lock:
            with open(self.stream_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
                f.flush()  # Ensure immediate write


class MessageHookManager:
    """Manages multiple message hooks."""
    
    def __init__(self):
        self.hooks: List[MessageHook] = []
        
    def add_hook(self, hook: MessageHook):
        """Add a message hook."""
        self.hooks.append(hook)
        
    def remove_hook(self, hook: MessageHook):
        """Remove a message hook."""
        if hook in self.hooks:
            self.hooks.remove(hook)
            
    def on_message(self, agent_name: str, content: str, metadata: Dict = None):
        """Notify all hooks of a message."""
        metadata = metadata or {}
        for hook in self.hooks:
            try:
                hook.on_message(agent_name, content, metadata)
            except Exception as e:
                print(f"Error in message hook: {e}")
                
    def on_simulation_start(self, simulation_id: str, run_id: str):
        """Notify all hooks of simulation start."""
        for hook in self.hooks:
            try:
                hook.on_simulation_start(simulation_id, run_id)
            except Exception as e:
                print(f"Error in simulation start hook: {e}")
                
    def on_simulation_end(self, simulation_id: str, run_id: str, results: Dict):
        """Notify all hooks of simulation end."""
        for hook in self.hooks:
            try:
                hook.on_simulation_end(simulation_id, run_id, results)
            except Exception as e:
                print(f"Error in simulation end hook: {e}")
    
    def on_optimization(self, agent_name: str, optimization_data: Dict):
        """Notify all hooks of agent optimization."""
        for hook in self.hooks:
            try:
                hook.on_optimization(agent_name, optimization_data)
            except Exception as e:
                print(f"Error in optimization hook: {e}")


# Global hook manager instance
_hook_manager = MessageHookManager()


def get_hook_manager() -> MessageHookManager:
    """Get the global hook manager."""
    return _hook_manager


class HookedConversableAgent:
    """Wrapper for ConversableAgent that intercepts messages."""
    
    def __init__(self, agent, simulation_id: str, run_id: str):
        self.agent = agent
        self.simulation_id = simulation_id
        self.run_id = run_id
        self.hook_manager = get_hook_manager()
        
        # Store original methods
        self._original_send = agent.send
        self._original_receive = agent.receive
        self._original_generate_reply = agent.generate_reply
        
        # Replace with hooked versions
        agent.send = self._hooked_send
        agent.receive = self._hooked_receive
        agent.generate_reply = self._hooked_generate_reply
        
    def _hooked_send(self, message, recipient, request_reply=False, silent=False):
        """Intercept sent messages."""
        # Extract message content
        if isinstance(message, dict) and "content" in message:
            self.hook_manager.on_message(
                self.agent.name, 
                message["content"],
                {
                    "action": "send",
                    "recipient": getattr(recipient, "name", "Unknown"),
                    "simulation_id": self.simulation_id,
                    "run_id": self.run_id,
                    "timestamp": time.time()
                }
            )
        
        # Call original method
        return self._original_send(message, recipient, request_reply, silent)
        
    def _hooked_receive(self, message, sender, request_reply=False, silent=False):
        """Intercept received messages."""
        # Extract message content
        if isinstance(message, dict) and "content" in message:
            self.hook_manager.on_message(
                getattr(sender, "name", "Unknown"),
                message["content"],
                {
                    "action": "receive",
                    "recipient": self.agent.name,
                    "simulation_id": self.simulation_id,
                    "run_id": self.run_id
                }
            )
        
        # Call original method
        return self._original_receive(message, sender, request_reply, silent)
        
    def _hooked_generate_reply(self, messages=None, sender=None, **kwargs):
        """Intercept generated replies."""
        # Call original method
        reply = self._original_generate_reply(messages, sender, **kwargs)
        
        # Log the generated reply
        if isinstance(reply, dict) and "content" in reply:
            self.hook_manager.on_message(
                self.agent.name,
                reply["content"],
                {
                    "action": "generate",
                    "simulation_id": self.simulation_id,
                    "run_id": self.run_id
                }
            )
        
        return reply
        
    def unwrap(self):
        """Restore original methods."""
        self.agent.send = self._original_send
        self.agent.receive = self._original_receive
        self.agent.generate_reply = self._original_generate_reply

