"""Unit tests for utility functions."""
import os
import pytest
from unittest.mock import patch, MagicMock
from openai import OpenAI, AzureOpenAI

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils import create_logger, client_for_endpoint, get_autogen_client


class TestCreateLogger:
    """Test logger creation utility."""
    
    def test_create_logger_basic(self):
        """Test basic logger creation."""
        logger = create_logger("test_logger")
        assert logger.name == "test_logger"
        assert logger.level == 20  # INFO level
        assert len(logger.handlers) == 1
    
    def test_create_logger_no_propagate(self):
        """Test logger doesn't propagate to parent."""
        logger = create_logger("test_no_propagate")
        assert logger.propagate is False


class TestClientForEndpoint:
    """Test client creation for different endpoints."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_client_creation(self):
        """Test OpenAI client creation when no other env vars set."""
        client, model_name = client_for_endpoint()
        assert isinstance(client, OpenAI)
        assert model_name == "gpt-4o"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_client_with_model_override(self):
        """Model parameter overrides default model name."""
        client, model_name = client_for_endpoint(model="gpt-3.5-turbo")
        assert isinstance(client, OpenAI)
        assert model_name == "gpt-3.5-turbo"
    
    @patch.dict(os.environ, {"OLLAMA_MODEL": "qwen3:4b"}, clear=True)
    def test_ollama_client_creation(self):
        """Test Ollama client creation."""
        client, model_name = client_for_endpoint()
        assert isinstance(client, OpenAI)
        assert model_name == "qwen3:4b"
        assert str(client.base_url).rstrip('/') == "http://localhost:11434/v1"
    
    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01"
    }, clear=True)
    def test_azure_client_creation(self):
        """Test Azure OpenAI client creation."""
        client, model_name = client_for_endpoint()
        assert isinstance(client, AzureOpenAI)
        assert model_name == "2024-02-01"


class TestGetAutogenClient:
    """Test autogen client creation."""
    
    @patch.dict(os.environ, {}, clear=True)
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_autogen_client(self):
        """Test OpenAI autogen client creation."""
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        
        client = get_autogen_client()
        assert isinstance(client, OpenAIChatCompletionClient)
    
    @patch.dict(os.environ, {"OLLAMA_MODEL": "qwen3:4b"}, clear=True)
    def test_ollama_autogen_client(self):
        """Test Ollama autogen client creation."""
        from autogen_ext.models.ollama import OllamaChatCompletionClient
        
        client = get_autogen_client()
        assert isinstance(client, OllamaChatCompletionClient)
    
    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01"
    }, clear=True)
    def test_azure_autogen_client(self):
        """Test Azure OpenAI autogen client creation."""
        from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
        
        client = get_autogen_client()
        assert isinstance(client, AzureOpenAIChatCompletionClient)


class TestClientForEndpointLegacy:
    """Test legacy client_for_endpoint function with parameters."""
    
    def test_azure_endpoint_detection(self):
        """Test Azure endpoint detection."""
        endpoint = "https://test.openai.azure.com/openai/deployments/gpt-4"
        api_key = "test-key"
        
        client = client_for_endpoint(endpoint, api_key)
        assert isinstance(client, AzureOpenAI)
    
    def test_ollama_endpoint_detection(self):
        """Test Ollama/local endpoint detection."""
        endpoint = "http://localhost:11434/v1"
        
        client = client_for_endpoint(endpoint)
        assert isinstance(client, OpenAI)
        assert str(client.base_url).rstrip('/') == endpoint
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_openai_endpoint_fallback(self):
        """Test OpenAI endpoint fallback."""
        endpoint = "https://api.openai.com/v1"
        
        client = client_for_endpoint(endpoint)
        assert isinstance(client, OpenAI)