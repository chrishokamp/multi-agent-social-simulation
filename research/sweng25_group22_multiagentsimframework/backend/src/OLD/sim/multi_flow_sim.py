import asyncio
from prefect import flow, task, get_client
from multiprocessing import Process
from time import sleep, time
from flows.api_flow import api_flow  # Importing the API flow
from flows.sim_flow import simulation_flow  # Importing the simulation flow
import openai
import os
from dotenv import load_dotenv
from prefect.artifacts import create_markdown_artifact

# Load environment variables
load_dotenv()

# Fetch the API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key and PromptLayer API key
openai.api_key = OPENAI_API_KEY

# Function to fetch task stats from Prefect client
async def fetch_task_stats():
    async with get_client() as client:
        flow_runs = await client.read_flow_runs()
        num_tasks = len(flow_runs)
        total_runtime = sum(run.total_run_time for run in flow_runs if run.total_run_time)
        print(f"Total Tasks Run: {num_tasks}")
        print(f"Total Running Time: {total_runtime} seconds")

# Function to run the simulation flow
def run_simulation_flow():
    sample_config = {
        "name": "Test Simulation",
        "agents": [{"name": "Agent1", "description": "First agent", "prompt": "Start"}],
        "termination_condition": "Condition Met",
        "output_variables": [{"name": "var1", "type": "Number"}]
    }
    num_runs = 5
    result = simulation_flow(config=sample_config, num_runs=num_runs)
    print("Simulation Flow Result:", result)

@task
def log_token_artifact(usage):
    content = f"""
    ### Token Usage
    - Prompt Tokens: {usage['prompt_tokens']}
    - Completion Tokens: {usage['completion_tokens']}
    - Total Tokens: {usage['total_tokens']}
    """
    create_markdown_artifact(content=content, key="token-usage")

# Function to run the API flow
def run_api_flow():
    api_flow()

# Main flow that manages the execution of other flows
@flow
def main_flow():
    print("Starting all flows...")
    start_time = time()

    # Run simulation in a separate process
    sim_process = Process(target=run_simulation_flow)
    sim_process.start()

    sleep(5)  # Wait for a bit before running API flow
    run_api_flow()

    sim_process.join()  # Wait for the simulation process to finish

    end_time = time()
    print(f"Total Execution Time: {end_time - start_time} seconds")

    # Fetch task statistics from Prefect
    asyncio.run(fetch_task_stats())

    print("All flows executed.")

if __name__ == "__main__":
    main_flow()