import sys
import os
import requests
import promptlayer
from dotenv import load_dotenv

load_dotenv()

# Setup PromptLayer API key
promptlayer.api_key = os.environ["PROMPTLAYER_API_KEY"]

# Change this if needed
API_URL = "http://localhost:5000/report/output"

def get_sim_output(sim_id, exclude_chat_log=True):
    params = {"id": sim_id}
    if not exclude_chat_log:
        params["log"] = "no"
    response = requests.get(API_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")
    
    data = response.json()
    if "error" in data:
        raise Exception(f"Error fetching data for simulation ID {sim_id}: {data['error']}")
    
    return data

def extract_promptlayer_ids(run):
    ids = []
    for agent_data in run.get("agents", {}).values():
        for message in agent_data.get("messages", []):
            pl_id = message.get("promptlayer_request_id")
            if pl_id:
                ids.append(pl_id)
    return ids

def get_token_usage(pl_ids):
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for pl_id in pl_ids:
        try:
            pl_data = promptlayer.get_request(pl_id)
            usage = pl_data.get("usage", {})
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
        except Exception as e:
            print(f"Error fetching PromptLayer ID {pl_id}: {e}")

    return total_prompt_tokens, total_completion_tokens

def main(sim_id):
    output_data = get_sim_output(sim_id)
    runs = output_data.get("runs", [])

    for i, run in enumerate(runs):
        pl_ids = extract_promptlayer_ids(run)
        prompt_tokens, completion_tokens = get_token_usage(pl_ids)
        print(f"Run {i}:")
        print(f"  Prompt Tokens:     {prompt_tokens}")
        print(f"  Completion Tokens: {completion_tokens}")
        print(f"  Total Tokens:      {prompt_tokens + completion_tokens}")
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python perform_api.py <simulation_id>")
        sys.exit(1)

    sim_id = sys.argv[1]
    main(sim_id)
