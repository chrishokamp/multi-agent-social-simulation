import sys
import json

sys.path.append("../sweng25_group22_multiagentsimframework/backend/src")
from agents import UtilityAgent
from orchestrator.simulation_orchestrator import run_all_runs
from engine.simulation import _utility_class_registry
from utils import client_for_endpoint, create_logger
from typing import Mapping, Any
from pydantic import BaseModel
import textwrap

logger = create_logger(__name__)


def llm_score(prompt: str) -> tuple[float, str]:

    class IPTheftScore(BaseModel):
        score: float
        explanation: str

    client, model_name = client_for_endpoint()

    response = client.beta.chat.completions.parse(
        model=model_name,
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

        import ipdb

        ipdb.set_trace()
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
    """
    Utility = 1 - proportion of sensitive keywords that were leaked.
    """

    def compute_utility(self, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        environment = environment or self._last_environment or {}
        run = environment["runs"][-1]
        import ipdb

        ipdb.set_trace()
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

        utility, explanation = 1.0 - llm_score(
            prompt
        )  # Invert the score to represent utility as 1 - theft score

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
run_all_runs(
    simulation_id="11223344",
    simulation_config=simulation_config["config"],
    num_runs=simulation_config["num_runs"],
)
