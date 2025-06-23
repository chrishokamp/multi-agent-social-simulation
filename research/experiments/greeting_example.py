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


"""
Mock interaction for debuggin that is designed to be quick and not need many turns.
"""

class GreeterAgent(UtilityAgent):


    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}
        most_recent_run = environment["runs"][-1]

        if most_recent_run["output_variables"]["Politeness of Response (1-5)"] is True:
            utility = 1.0
        else:
            utility = 0.0

        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


class ResponderAgent(UtilityAgent):
    
    def compute_utility(
        self,
        environment: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if environment is None:
            environment = self._last_environment or {}
        most_recent_run = environment["runs"][-1]
        # static
        utility = 1.0
        environment["runs"][-1]["output_variables"]["utility"] = environment["runs"][-1]["output_variables"].get("utility", {})
        environment["runs"][-1]["output_variables"]["utility"][self.name] = utility
        self._last_environment = environment
        return environment


_utility_class_registry["GreeterAgent"] = GreeterAgent
_utility_class_registry["ResponderAgent"] = ResponderAgent
simulation_config = json.load(
    open(
        "../sweng25_group22_multiagentsimframework/configs/quick-greeting.json",
    )
)
run_all_runs(
    simulation_id="11223344",
    simulation_config=simulation_config["config"],
    num_runs=simulation_config["num_runs"],
)
