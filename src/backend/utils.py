
import logging
import sys
import os
from openai import OpenAI, AzureOpenAI
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = create_logger(__name__)

def client_for_endpoint(endpoint: str = None, api_key: str = None):
    """Create client for endpoint. Returns (client, model_name) if no args, otherwise just client."""
    CLIENT_TIMEOUT = 6000
    
    # If no arguments, use environment-based selection (returns tuple)
    if endpoint is None:
        if os.environ.get("OLLAMA_MODEL"):
            endpoint = "http://localhost:11434/v1"
            logger.info("Using Ollama-style local server client.")
            client = OpenAI(
                base_url=endpoint, 
                api_key="ollama", 
                timeout=CLIENT_TIMEOUT
            )
            model_name = os.environ.get("OLLAMA_MODEL")
        elif os.environ.get("AZURE_OPENAI_API_KEY"):
            logger.info("Using *Azure* OpenAI client.")
            client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ["AZURE_OPENAI_ENDPOINT"].split("api-version=")[-1],
                timeout=CLIENT_TIMEOUT,
            )
            model_name = os.environ["AZURE_OPENAI_ENDPOINT"].split("api-version=")[-1]
        else:
            logger.info("Using OpenAI client.")
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"], 
                timeout=CLIENT_TIMEOUT
            )
            model_name = None
        return client, model_name
    
    # If endpoint provided, use legacy behavior (returns just client)
    logger.info(f"Creating client for endpoint: {endpoint}")
    if "azure" in endpoint:
        logger.info("Using *Azure* OpenAI client.")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=endpoint.split("api-version=")[-1],
            timeout=CLIENT_TIMEOUT,
        )
    elif "local" in endpoint or "11434" in endpoint:
        logger.info("Using Ollama-style local server client.")
        return OpenAI(
            base_url=endpoint, api_key="ollama", timeout=CLIENT_TIMEOUT
        )
    else:
        logger.info("Using OpenAI client.")
        return OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"], timeout=CLIENT_TIMEOUT
        )
    
def get_autogen_client():
        if os.environ.get("OLLAMA_MODEL"):
            logger.info("Using Ollama client for simulation.")
            return OllamaChatCompletionClient(
                model=os.environ.get("OLLAMA_MODEL"),
                options={} # TODO: make configurable
            )
        elif os.environ.get("AZURE_OPENAI_API_KEY"):
            logger.info("Using Azure OpenAI client for simulation.")
            return AzureOpenAIChatCompletionClient(
                model="gpt-4o", # NOTE: this is unused, but it needs to be a valid model for Azure OpenAI
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ["AZURE_OPENAI_ENDPOINT"].split("api-version=")[-1]
            )
        else:
            logger.info("Using OpenAI client for simulation.")
            return OpenAIChatCompletionClient(
                model="gpt-4o", 
                api_key=os.environ["OPENAI_API_KEY"]
            )