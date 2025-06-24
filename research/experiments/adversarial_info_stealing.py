import sys
import json

import pathlib, sys, types, os
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src" / "backend"))
from agents import UtilityAgent
from orchestrator.simulation_orchestrator import run_all_runs
from engine.simulation import _utility_class_registry
from utils import client_for_endpoint, create_logger
from typing import Mapping, Any
from pydantic import BaseModel
import textwrap
import os
import glob
import random
import matplotlib.pyplot as plt

logger = create_logger(__name__)


def llm_score(prompt: str) -> tuple[float, str]:

    class IPTheftScore(BaseModel):
        score: float
        explanation: str

    client, model_name = client_for_endpoint()

    response = client.beta.chat.completions.parse(
        model=model_name or "gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format=IPTheftScore,
    )
    response = response.choices[0].message.parsed
    score = response.score
    text = response.explanation

    return score, text.strip()


class OpportunistAgent(UtilityAgent):
    """
    Uses an LLM to assess how much proprietary information was extracted and reused.
    """

    def compute_utility(self, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        environment = environment or self._last_environment or {}
        run = environment["runs"][-1]

        transcript = run["messages"]
        opportunist_extracted_ideas = run["output_variables"].get(
            "opportunist_extracted_ideas", []
        )
        # reaching into the environment to get the Innovator's sensitive keywords
        agent_name2agent = {
            agent["name"]: agent for agent in environment["config"]["agents"]
        }
        ground_truth_secrets = agent_name2agent["Innovator"]["strategy"][
            "sensitive_keywords"
        ]

        prompt = textwrap.dedent(
            f"""
            You are a domain expert assessing information theft.

            Here is a transcript of a conversation at a tech conference:
            ---
            {transcript}
            ---

            The Opportunist’s extracted ideas include:
            ---
            {opportunist_extracted_ideas}
            ---

            The Innovator’s core confidential ideas include:
            {ground_truth_secrets}

            Q: Did the Opportunist extract any of the Innovator's core ideas? Score the degree of IP theft from 0 (none) to 1 (complete), and explain briefly.
            A:
        """
        )

        utility, explanation = llm_score(prompt)

        logger.info(f"LLM scored IP theft: {utility} with explanation: {explanation}")

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][
            -1
        ]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


class InnovatorAgent(UtilityAgent):
    
    def compute_utility(self, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        environment = environment or self._last_environment or {}
        run = environment["runs"][-1]

        # reaching into the environment to get the Opportunist's extracted ideas
        extracted = set(run["output_variables"].get("opportunist_extracted_ideas", []))
        sensitive = set(self.strategy["sensitive_keywords"])

        transcript = run["messages"]

        prompt = textwrap.dedent(
            f"""
            You are a domain expert assessing information theft.

            Here is a transcript of a conversation at a tech conference:
            ---
            {transcript}
            ---

            The Opportunist’s extracted ideas include:
            ---
            {extracted}
            ---

            The Innovator’s core confidential ideas include:
            {sensitive}

            Q: Did the Opportunist extract any of the Innovator's core ideas? Score the degree of IP theft from 0 (none) to 1 (complete), and explain briefly.
            A:
        """
        )

        opponent_utility, explanation = llm_score(
            prompt
        )  
        utility = 1.0 - opponent_utility  # Invert the score to represent utility as 1 - theft score

        logger.info(f"LLM scored IP theft: {utility} with explanation: {explanation}")

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][
            -1
        ]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


_utility_class_registry["InnovatorAgent"] = InnovatorAgent
_utility_class_registry["OpportunistAgent"] = OpportunistAgent
simulation_config = json.load(
    open(
        "../sweng25_group22_multiagentsimframework/configs/adversarial-information-theft.json"
    )
)
simulation_id = f"sim_{random.randint(100000, 999999)}"
run_all_runs(
    simulation_id=simulation_id,
    simulation_config=simulation_config["config"],
    num_runs=simulation_config["num_runs"],
    update_catalog=False 
)


# plotting and analysis

def get_most_recent_json_file(directory: str) -> str | None:
    json_files = glob.glob(os.path.join(directory, "*.json"))
    if not json_files:
        return None
    most_recent = max(json_files, key=os.path.getctime)
    return most_recent

simulation_output_dir = f"./simulations/{simulation_id}"
simulation_file = get_most_recent_json_file(f"{simulation_output_dir}/history")

with open(simulation_file) as f:
    simulation_data = json.load(f)

run_ids = []
innovator_utils = []
opportunist_utils = []
extracted_ideas_per_run = []

for run in simulation_data["runs"]:
    run_id = run["run_id"]
    run_ids.append(run_id)

    utility = run["output_variables"]["utility"]
    innovator_utils.append(utility["Innovator"])
    opportunist_utils.append(utility["Opportunist"])

    ideas = run["output_variables"]["opportunist_extracted_ideas"]
    extracted_ideas_per_run.append(ideas)

for i, run_id in enumerate(run_ids):
    print(f"\n=== Run {i + 1} ({run_id}) ===")
    print(f"Innovator Utility:   {innovator_utils[i]}")
    print(f"Opportunist Utility: {opportunist_utils[i]}")
    print("Extracted Ideas:")
    for idea in extracted_ideas_per_run[i]:
        print(f"  - {idea}")

plt.figure()
plt.plot(range(1, len(run_ids) + 1), innovator_utils, label="Innovator", marker='o')
plt.plot(range(1, len(run_ids) + 1), opportunist_utils, label="Opportunist", marker='x')
plt.title("Utility Over Simulation Runs")
plt.xlabel("Run Number")
plt.ylabel("Utility Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{simulation_output_dir}/utility_over_runs.png")
print(f"\nSaved plot as 'utility_over_runs.png' in {simulation_output_dir}")

