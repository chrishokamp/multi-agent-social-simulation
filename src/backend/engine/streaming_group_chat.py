"""
Custom GroupChat implementation that streams messages in real-time.
"""
import time
from typing import Dict, List, Optional, Union, Callable, Any
from autogen import GroupChat, Agent, GroupChatManager
from engine.message_hook import get_hook_manager


class StreamingGroupChatManager(GroupChatManager):
    """GroupChatManager that intercepts and streams messages."""
    
    def __init__(self, groupchat: "StreamingGroupChat", **kwargs):
        super().__init__(groupchat, **kwargs)
        self.hook_manager = get_hook_manager()
        self._original_run_chat = self.run_chat
        print(f"[StreamingGroupChatManager] Initialized")
    
    async def a_run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Any:
        """Override async run_chat to intercept messages."""
        print(f"[StreamingGroupChatManager] a_run_chat called")
        result = await super().a_run_chat(messages, sender, config)
        return result
    
    def run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Any:
        """Override run_chat to intercept messages."""
        print(f"[StreamingGroupChatManager] run_chat called")
        result = super().run_chat(messages, sender, config)
        return result


class StreamingGroupChat(GroupChat):
    """GroupChat that streams messages as they're sent."""
    
    def __init__(self, agents: List[Agent], messages: List[Dict], 
                 simulation_id: str, run_id: str, **kwargs):
        super().__init__(agents, messages, **kwargs)
        self.simulation_id = simulation_id
        self.run_id = run_id
        self.hook_manager = get_hook_manager()
        self._message_count = 0
        
    def append(self, message: Dict[str, str], speaker: Agent):
        """Override append to stream messages."""
        # Call parent append
        super().append(message, speaker)
        
        # Stream the message
        if isinstance(message, dict):
            agent_name = message.get("name", getattr(speaker, "name", "Unknown"))
            content = message.get("content", "")
            
            # Send to hooks for streaming
            self.hook_manager.on_message(agent_name, content, {
                "message_index": self._message_count,
                "simulation_id": self.simulation_id,
                "run_id": self.run_id,
                "timestamp": time.time()
            })
            
            self._message_count += 1
            
    def messages_to_string(self, messages: List[Dict[str, str]] = None) -> str:
        """Override to track when messages are accessed."""
        return super().messages_to_string(messages)