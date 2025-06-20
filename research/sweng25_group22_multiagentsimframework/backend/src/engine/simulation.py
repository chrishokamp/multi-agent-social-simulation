import re
import os
import json

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from agents import UtilityAgent, BuyerAgent, SellerAgent
from autogen_agentchat.ui import Console
import uuid

from utils import create_logger, get_autogen_client
logger = create_logger(__name__)

_utility_class_registry = {
    "UtilityAgent": UtilityAgent,
    "BuyerAgent":   BuyerAgent,
    "SellerAgent":  SellerAgent
}

class SelectorGCSimulation:
    def __init__(self, config: dict, environment: dict, max_messages=25, min_messages=5):
        self.model_client = get_autogen_client()
        self.config = config
        self.min_messages = min_messages
        self.run_id = str(uuid.uuid4())

        logger.info(f"Initializing SelectorGCSimulation with config: {self.config}")

        self.config_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
        
        type_mapping = {
            "String": "STRING",
            "Number": "NUMBER",
            "Boolean": "BOOLEAN",
            "Float": "FLOAT",
            "Date": "DATE",
        }

        # inject InformationReturnAgent into config
        if not any(agent["name"] == "InformationReturnAgent" for agent in self.config["agents"]):
            # check if InformationReturnAgent is there already
            with open(os.path.join(self.config_directory, "InformationReturnAgent.json"), "r", encoding="utf-8") as file:
                information_return_agent = json.load(file)
                information_return_agent["prompt"] = information_return_agent["prompt"].format(
                    output_variables_str = (
                        '{\n' + ",\n".join([
                            f'"{v["name"]}": "{type_mapping.get(v["type"], "UNKNOWN_TYPE")}"'
                            for v in self.config["output_variables"]
                        ]) + '\n}'
                    ),
                    termination_condition=self.config["termination_condition"]
                )
                self.config["agents"].append(information_return_agent)

        # initialize agents
        self.agents = []
        for agent_config in self.config["agents"]:
            cls_name   = agent_config.get("utility_class", "UtilityAgent")
            AgentClass = _utility_class_registry[cls_name]

            ag = AgentClass(
                system_prompt=agent_config["prompt"],
                name=agent_config["name"],
                description=agent_config["description"],
                model_client=self.model_client,
                system_message=agent_config["prompt"],
                strategy=agent_config.get("strategy"),
            )


            # (+ self-improvement)
            utility = ag.compute_utility(environment)
            if agent_config.get("self_improve", False):
                ag.learn_from_feedback(utility, environment)

            # re-init agent with update
            ag = AgentClass(
                system_prompt=agent_config["prompt"],
                name=agent_config["name"],
                description=agent_config["description"],
                model_client=self.model_client,
                system_message=ag.system_prompt,
                strategy=agent_config.get("strategy")
            )

            self.agents.append(ag)


        # initialize group chat
        with open(os.path.join(self.config_directory, "selector_prompt.txt"), "r", encoding="utf-8") as file:
            selector_prompt = file.read()

        self.group_chat = SelectorGroupChat(
            self.agents,
            model_client=self.model_client,
            selector_prompt=selector_prompt,
            termination_condition=(
                TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=max_messages)
            ),
            emit_team_events=True,
        )   

    def _process_result(self, simulation_result):
        if len(simulation_result.messages) < self.min_messages:
            logger.warning(f"Simulation result has too few messages: {len(simulation_result.messages)} < {self.min_messages}")
            return None

        messages = []
        for message in simulation_result.messages:
            messages.append({"agent": message.source, "message": message.content})

        output_variables = []
        information_return_agent_message = messages[-1]["message"]
        json_match = re.search(r'\{.*\}', information_return_agent_message, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
                for variable in parsed_json:
                    # Handle both None and "Unspecified" values
                    value = parsed_json[variable]
                    if value is None or value == "Unspecified":
                        value = "Unspecified"
                    output_variables.append({"name": variable, "value": value})
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from message: {information_return_agent_message}")
                return None
        else:
            logger.error(f"No JSON found in message: {information_return_agent_message}")
            return None

        return {"messages": messages, "output_variables": output_variables}


    async def run(self):

        running_history: list[dict[str, str]] = []

        async for event in self.group_chat.run_stream():


            if event.type == "message":
                print(f"{event.source}: {event.content}", flush=True)
            else:                       # join/leave/system events
                print(event, flush=True)

            running_history.append({"agent": event.source, "msg": event.content})

            for ag in self.agents:
                if ag.name == event.source:
                    # Skip reflection
                    if ag.name != "InformationReturnAgent":
                        await ag.think_and_print(running_history)
                    break

        processed = self._process_result(self.group_chat.result)

        if processed:
            processed["private_reflections"] = [
                {"agent": h["agent"], "thought": h.get("thought", "")}
                for h in running_history
                if "thought" in h and h["agent"] != "InformationReturnAgent"
            ]
        return processed
