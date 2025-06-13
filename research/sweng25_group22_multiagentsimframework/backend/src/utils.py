
import logging
import sys
import os
from openai import OpenAI, AzureOpenAI

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

def client_for_endpoint(endpoint: str, api_key: str | None = None):
    CLIENT_TIMEOUT = 6000
    logger.info(f"Creating client for endpoint: {endpoint}")
    if "azure" in endpoint:
        logger.info("Using *Azure* OpenAI client.")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key or os.environ["AZURE_OPENAI_API_KEY"],
            api_version=endpoint.split("api-version=")[-1],
            timeout=CLIENT_TIMEOUT,
        )
    elif "local" in endpoint:
        logger.info("Using Ollama-style local server client.")
        return OpenAI(
            base_url=endpoint, api_key="ollama", timeout=CLIENT_TIMEOUT
        )
    else:
        logger.info("Using OpenAI client.")
        return OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"], timeout=CLIENT_TIMEOUT
        )