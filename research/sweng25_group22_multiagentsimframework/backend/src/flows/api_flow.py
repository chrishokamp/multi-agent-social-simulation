import requests
from prefect import task, flow
from dotenv import load_dotenv
import subprocess
import time
import sys
from prefect.artifacts import create_markdown_artifact

load_dotenv()

sys.path.append("sweng25/sweng25_group22_multiagentsimframework/backend/src")

BASE_URL = "http://127.0.0.1:4200"

# Start Flask app in a separate process
@task
def start_api():
    print("Starting API...")
    process = subprocess.Popen([sys.executable, "OLD/api/app.py"])  
    time.sleep(10)  # Give it time to start
    return process

# Stop Flask process
@task
def stop_api(process):
    print("Stopping API...")
    process.terminate()

# Fetch config report
@task(retries=3, retry_delay_seconds=2)
def fetch_config(sim_id):
    url = f"{BASE_URL}/report/config?id={sim_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Failed to fetch config: {response.text}")

# Fetch output report
@task(retries=3, retry_delay_seconds=2)
def fetch_output(sim_id, i=None, log="yes"):
    url = f"{BASE_URL}/report/output?id={sim_id}&log={log}"
    if i is not None:
        url += f"&i={i}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(f"Failed to fetch output: {response.text}")
    
@task
def log_token_artifact(usage):
    content = f"""
    ### Token Usage
    - Prompt Tokens: {usage['prompt_tokens']}
    - Completion Tokens: {usage['completion_tokens']}
    - Total Tokens: {usage['total_tokens']}
    """
    create_markdown_artifact(content=content, key="token-usage")

def get_valid_sim_id():
    response = requests.get(f"{BASE_URL}/list_sim_ids")
    if response.status_code == 200:
        sim_ids = response.json()
        return sim_ids[0] if sim_ids else None
    return None

# Main Prefect Flow
@flow
def api_flow(i=None, log="yes"):
    process = start_api()
    try:
        sim_id = get_valid_sim_id()
        if not sim_id:
            print("No valid sim_id found.")
            return

        config = fetch_config(sim_id)
        print("Config Data:", config)

        output = fetch_output(sim_id, i, log)
        print("Output Data:", output)

    finally:
        stop_api(process)

if __name__ == "__main__":
    api_flow()