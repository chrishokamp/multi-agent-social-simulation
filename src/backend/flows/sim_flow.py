import os
import sys
import promptlayer
import openai
from dotenv import load_dotenv
from prefect import flow, task, get_client
import asyncio

# Load environment variables
load_dotenv()

# Append your custom modules path
sys.path.append("sweng25/sweng25_group22_multiagentsimframework/backend/src")

# Set OpenAI and PromptLayer API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
promptlayer.api_key = os.getenv("PROMPTLAYER_API_KEY")

# Use PromptLayer's wrapper for OpenAI calls
# This automatically tracks the prompts used with PromptLayer.
#promptlayer.openai.api_key = os.getenv("PROMPTLAYER_API_KEY")

# Define tasks and flow
from OLD.util.config import SimConfigLoader
from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    ConversableAgent,
    LLMConfig,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Custom subclass to inject PromptLayer tags
class PromptLayerOpenAIClient(OpenAIChatCompletionClient):
    def generate(self, messages, **kwargs):
        # Inject PromptLayer tags automatically for each request
        return promptlayer.openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            promptlayer_tags=["simulation_flow"],
            **kwargs
        )

# Initialize the GPT, Together, and Ollama model clients
@task
def init_gpt_model_client():
    return PromptLayerOpenAIClient(model="gpt-4o", api_key=openai.api_key)

@task
def init_together_model_client():
    return PromptLayerOpenAIClient(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "llama",
        },
    )

@task
def init_ollama_model_client():
    return PromptLayerOpenAIClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434/v1",
        api_key="placeholder",  # You can use a real key here
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "llama",
        },
    )

# Load simulation configuration
@task
def load_simulation_config(sim_config_file_name: str):
    return SimConfigLoader(sim_config_file_name).load()

# Setup agents for the simulation
@task
def setup_agents(config, agent_model_client, info_model_client, gc_model_client):
    agents = [
        AssistantAgent(
            agent["name"],
            description=agent["description"],
            model_client=agent_model_client,
            system_message=agent["system_message"]
        ) for agent in config["agents"]
    ]
    
    output_variables_str = "{\n" + "\n".join([f"\"{v['name']}\": <{v['type']}>" for v in config['output_variables']]) + "\n}"
    
    information_return_agent = AssistantAgent(
        "InformationReturnAgent",
        description="Returns information about the conversation when a termination condition is met.",
        model_client=info_model_client,
        system_message=(f"Do not act like a human.\n"
                        f"You are a system that extracts the following information from the conversation when the termination condition: \"{config['termination_condition']}\" is satisfied:\n\n"
                        f"{output_variables_str}\n\n"
                        f"Make sure the output is valid JSON and units are correct.\n\n"
                        f"After this, send 'TERMINATE'\n")
    )
    agents.append(information_return_agent)
    
    gc = SelectorGroupChat(
        agents,
        model_client=gc_model_client,
        termination_condition=(TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=25))
    )
    
    return gc

@flow
def simulation_flow(sim_config_file_name: str):
    print("Starting simulation flow...")

    # Load configuration for the simulation
    config = load_simulation_config(sim_config_file_name)
    print("Config loaded:", config)

    # Initialize model clients
    gpt_model = init_gpt_model_client()
    together_model = init_together_model_client()
    ollama_model = init_ollama_model_client()

    # Select model client based on the configuration
    agent_model_client = (
        gpt_model if "GPT" in config["models"].values()
        else together_model if "Together" in config["models"].values()
        else ollama_model
    )

    info_model_client = agent_model_client
    gc_model_client = agent_model_client

    print("Setting up agents...")
    gc = setup_agents(config, agent_model_client, info_model_client, gc_model_client)
    print("Agents set up!")

    print("Running the conversation...")
    return Console(gc.run_stream())

# Optional: Monitor Prefect flow runs
async def get_flow_runs():
    async with get_client() as client:
        flow_runs = await client.read_flow_runs()
        for flow_run in flow_runs:
            print(flow_run)

# Run if executed directly
if __name__ == "__main__":
    simulation_flow("sim_config.json")
    asyncio.run(get_flow_runs())