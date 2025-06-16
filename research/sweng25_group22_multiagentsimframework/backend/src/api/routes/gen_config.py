import os
import re
import json
from pymongo import MongoClient
from flask import Blueprint, request, jsonify
from openai import AsyncOpenAI, AsyncAzureOpenAI
from utils import create_logger, client_for_endpoint

logger = create_logger(__name__)

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])

gen_config_bp = Blueprint("gen_config", __name__)

def run_sim(def_prompt, json_prompt, desc_prompt):
    if os.environ.get("OLLAMA_MODEL"):
        client = client_for_endpoint(endpoint="http://localhost:11434/v1")
        model_name = os.environ.get("OLLAMA_MODEL")
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        client = client_for_endpoint(endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"), api_key=os.environ.get("AZURE_OPENAI_API_KEY"))
        model_name = os.environ["AZURE_OPENAI_ENDPOINT"].split("api-version=")[-1]
    else:
        client = client_for_endpoint(api_key=os.environ.get("OPENAI_API_KEY"))
        model_name = None
    

    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name or "gpt-4o",
                messages=[
                    {"role": "system", "content": def_prompt},
                    {"role": "user", "content": json_prompt},
                    {"role": "user", "content": desc_prompt},
                ],
            )

            response_str = (
                response.choices[0].message.content
                .replace("\n", "")
                .replace("```json", "")
                .replace("```", "")
            )

            json_match = re.search(r"\{.*\}", response_str, re.DOTALL)
            if not json_match:
                raise ValueError("no JSON found")

            parsed_json = json.loads(json_match.group(0))
            if any(v is None for v in parsed_json.values()):
                raise ValueError("null values in JSON")

            return json_match.group(0)          # success
        except Exception as exc:
            logger.error("run_sim retry %d failed: %s", retries + 1, exc)
            retries += 1

    return None  # all retries exhausted

    
def parse_config_str(s):
    """Removes <think> tags and leading/trailing newlines from the config string."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = s.strip("\n")
    return s

@gen_config_bp.route("/gen_config", methods=["POST"])
def generate_config():
    request_json = request.get_json()

    desc = request_json["desc"]
    if not desc:
        return jsonify({"message": "desc not given"}), 400
    
    JSON_CONFIG_PROMPT = """
        This is the config file output template:
        {
            "num_runs": 999,
            "config": {
                "name": "Name Of Simulation",
                "agents": [
                 {
                    "name": "First AI Agents Name",
                    "description": "Description of the First AI Agent",
                    "prompt": "Prompt to be given to the First AI Agent, this should tell them who they are and what they want"
                 }
                ],
                "termination_condition": "The event that triggers the end of the simulation",
                "output_variables": [
                 {
                    "name": "First Output Variable Name",
                    "type": "First Output Variable Type (e.g. String, Number, Boolean)"
                 },
                 {
                    "name": "placeholder2",
                    "type": "Number"
                 }
                ]
            }
        }
        Please replace all values in the template and do not add any keys not already written into the template. You may add as many agents or output variables as needed.
        """
    
    DEFAULT_PROMPT = """
        You are an AI Assistent who converts a plain text description of a multi-agent AI simulation into a valid JSON config file.
        You should insert one agent into the config for each person or sentient entity in the description. All humans should be given a unique name such as "Teacher" or "Sarah", but not "Agent 3".
        You should also include as many output variables as needed to determine the result of the simulation.
        Output variables should be concrete and not vague, the name alone should describe the units, range or any other information such that all simulations will fill the variables with comparable data.
        If an output variable is a percentage that should be mentioned in the name explicitly. Do not write output variables in snake case, pascal case, etc.
        You should never have only 0 or 1 agents, or no output variables
    """

    prompt = "\n\nThis is your simulation description: " + desc + "\n\nDo not write anything except the JSON" 

    config_str = run_sim(DEFAULT_PROMPT, JSON_CONFIG_PROMPT, prompt)
    config_str = parse_config_str(config_str)  

    return jsonify(json.loads(config_str))
