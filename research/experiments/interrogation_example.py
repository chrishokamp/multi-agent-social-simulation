import sys
import json

sys.path.append("../sweng25_group22_multiagentsimframework/backend/src")
from agents import UtilityAgent
from orchestrator.simulation_orchestrator import run_all_runs
from engine.simulation import _utility_class_registry
from utils import client_for_endpoint, create_logger
from typing import Mapping, Any
from pydantic import BaseModel

logger = create_logger(__name__)


class DetectiveAgent(UtilityAgent):
    
    """
    Utility measure whether agent manages to extract sensitive information from the suspect.
    """

    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}
        most_recent_run = environment["runs"][-1]

        if most_recent_run["output_variables"]["suspect_admitted_theft"] is True:
            # No deal reached, so no utility
            utility = 1.0
        else:
            # The detective did not manage to extract sensitive information
            utility = 0.0

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


class SuspectAgent(UtilityAgent):
    """
    Utility measure whether agent manages to keep sensitive information from the detective.
    It is the opposite of the DetectiveAgent.
    """

    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}
        most_recent_run = environment["runs"][-1]

        if most_recent_run["output_variables"]["suspect_admitted_theft"] is True:
            # No deal reached, so no utility
            utility = 0.0
        else:
            # The detective did not manage to extract sensitive information
            utility = 1.0

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


_utility_class_registry["DetectiveAgent"] = DetectiveAgent
_utility_class_registry["SuspectAgent"] = SuspectAgent
simulation_config = json.load(
    open(
        "../sweng25_group22_multiagentsimframework/configs/interrogation.json",
    )
)
run_all_runs(
    simulation_id="11223344",
    simulation_config=simulation_config["config"],
    num_runs=simulation_config["num_runs"],
)
