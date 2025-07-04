"""Configuration for real-time streaming."""
import os
from typing import Dict, Any


class StreamingConfig:
    """Configuration for real-time message streaming."""
    
    # Environment variable to enable/disable streaming
    ENABLE_STREAMING_ENV = "ENABLE_REALTIME_STREAMING"
    
    # Default streaming settings
    DEFAULTS = {
        "enabled": True,
        "stream_to_file": True,
        "stream_buffer_size": 100,
        "stream_timeout": 0.1
    }
    
    @classmethod
    def is_streaming_enabled(cls) -> bool:
        """Check if streaming is enabled."""
        # Check environment variable first
        env_value = os.environ.get(cls.ENABLE_STREAMING_ENV, "").lower()
        if env_value in ["false", "0", "no", "off"]:
            return False
        elif env_value in ["true", "1", "yes", "on"]:
            return True
        
        # Default to enabled
        return cls.DEFAULTS["enabled"]
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get streaming configuration."""
        config = cls.DEFAULTS.copy()
        config["enabled"] = cls.is_streaming_enabled()
        return config