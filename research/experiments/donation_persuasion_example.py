import sys
import json

sys.path.append("../sweng25_group22_multiagentsimframework/backend/src")
from agents import UtilityAgent
from orchestrator.simulation_orchestrator import run_all_runs
from engine.simulation import _utility_class_registry
from utils import client_for_endpoint, create_logger
from typing import Mapping, Any

logger = create_logger(__name__)


class PersuaderAgent(UtilityAgent):
    """
    Utility measures whether the agent manages to persuade the donor to donate.
    """

    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}
        most_recent_run = environment["runs"][-1]

        if most_recent_run["output_variables"]["donor_agreed_to_donate"] is True:
            amount = float(most_recent_run["output_variables"].get("donation_amount", 0.0))
            utility = amount
        else:
            utility = 0.0

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


class DonorAgent(UtilityAgent):
    """
    Utility measures whether the agent manages to avoid being persuaded to donate.
    """

    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}
        most_recent_run = environment["runs"][-1]

        if most_recent_run["output_variables"]["donor_agreed_to_donate"] is True:
            utility = 0.0
        else:
            utility = 1.0

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


_utility_class_registry["PersuaderAgent"] = PersuaderAgent
_utility_class_registry["DonorAgent"] = DonorAgent

simulation_config = json.load(
    open(
        "../sweng25_group22_multiagentsimframework/configs/donation-persuasion.json",
    )
)
run_all_runs(
    simulation_id="55667788",
    simulation_config=simulation_config["config"],
    num_runs=simulation_config["num_runs"],
)