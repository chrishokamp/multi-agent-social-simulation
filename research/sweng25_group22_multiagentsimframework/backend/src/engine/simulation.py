import re
import os
import json
import textwrap

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from agents import UtilityAgent, BuyerAgent, SellerAgent
from autogen_agentchat.ui import Console
import autogen_agentchat
import uuid

from utils import create_logger, get_autogen_client, client_for_endpoint
logger = create_logger(__name__)

_utility_class_registry = {
    "UtilityAgent": UtilityAgent,
    "BuyerAgent":   BuyerAgent,
    "SellerAgent":  SellerAgent
}

class SelectorGCSimulation:
    def __init__(self, config: dict, environment: dict):
        self.model_client = get_autogen_client()
        self.config = config
        self.min_messages = config.get("min_messages", 2)
        self.max_messages = config.get("max_messages", 25)
        self.run_id = str(uuid.uuid4())
        self.environment = environment
        self.environment["config"] = self.config

        logger.info(f"Initializing SelectorGCSimulation with config: {self.config}")

        self.config_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
        
        type_mapping = {
            "String": "STRING",
            "Number": "NUMBER",
            "Boolean": "BOOLEAN",
            "Float": "FLOAT",
            "Date": "DATE",
            "List": "LIST",
            "Dict": "DICT",
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

            if len(self.environment["runs"]) > 0:
                # we have >=1 runs to learn from
                if agent_config.get("self_improve", False):
                    ag.learn_from_feedback(self.environment)

            self.agents.append(ag)


        # initialize group chat
        with open(os.path.join(self.config_directory, "selector_prompt.txt"), "r", encoding="utf-8") as file:
            selector_prompt = file.read()

        self.group_chat = SelectorGroupChat(
            self.agents,
            model_client=self.model_client,
            selector_prompt=selector_prompt,
            termination_condition=(
                TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=self.max_messages)
            ),
            emit_team_events=True,
        )   

    def force_info_return(
        self,
        messages: dict,
        output_variables: list[str],
    ) -> dict:
        """Call the InfoReturn agent post-hoc to populate missing output variables."""
        client, model_name = client_for_endpoint()

        full_transcript = "\n".join(f"{m['agent']}: {m['message']}" for m in messages)
        
        prompt = textwrap.dedent(f"""
        You are the Information Return Agent for a simulation. The conversation below ended prematurely.
        Your job is to extract structured information from the dialogue.

        --- Conversation Transcript ---
        {full_transcript}
        -------------------------------

        Please output the following output_variables
        {output_variables}

        Return only a valid JSON object, using the "name" fields as keys in your output.
        """)

        response = client.chat.completions.create(
            model=model_name or "gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        parsed = json.loads(response.choices[0].message.content.replace("```json", "").replace("```", "").strip())
        output_variables = [{"name": k, "value": v if v is not None else "Unspecified"} for k, v in parsed.items()]

        return output_variables

    def parse_ira_message(self, information_return_agent_message: str, output_variables: list[dict[str, str]]) -> None:
        json_match = re.search(r'\{.*\}', information_return_agent_message, re.DOTALL)
        parsed_json = json.loads(json_match.group(0))
        for variable in parsed_json:
            # Handle both None and "Unspecified" values
            value = parsed_json[variable]
            if value is None or value == "Unspecified":
                value = "Unspecified"
            output_variables.append({"name": variable, "value": value})
        return output_variables


    def _process_result(self, simulation_result):
        if len(simulation_result.messages) < self.min_messages:
            logger.warning(f"Simulation result has too few messages: {len(simulation_result.messages)} < {self.min_messages}")
            return None

        systems_prompts = {ag.name: ag.system_prompt for ag in self.agents}

        messages = []
        for message in simulation_result.messages:
            messages.append({"agent": message.source, "message": message.content})

        output_variables = []
        last_message = messages[-1]
        if last_message["agent"] == "InformationReturnAgent":
            output_variables = self.parse_ira_message(last_message["message"], output_variables)
        else:
            logger.warning("Last message is not from InformationReturnAgent, forcing information return.")
            output_variables = self.force_info_return(
                messages=messages,
                output_variables=self.config["output_variables"]
            )

        logger.info(f"Processed output variables: {output_variables}")
        
        processed = {
            "run_id": self.run_id,
            "system_prompts": systems_prompts,
            "messages": messages,
            "output_variables": {
                v["name"]: v["value"] 
                for v in output_variables
            }
        }

        return processed

    def calculate_utility(self) -> None:
        for ag in self.agents:
            if len(self.environment["runs"]) > 0:
                # we have >=1 runs to learn from
                self.environment = ag.compute_utility(self.environment)

    async def run(self):

        running_history: list[dict[str, str]] = []

        async for event in self.group_chat.run_stream():

            if type(event) == autogen_agentchat.base._task.TaskResult:
                task_result = event
                continue

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

        processed = self._process_result(task_result)
        
        self.environment["runs"].append(processed)
        self.calculate_utility()

        if processed:
            processed["private_reflections"] = [
                {"agent": h["agent"], "thought": h.get("thought", "")}
                for h in running_history
                if "thought" in h and h["agent"] != "InformationReturnAgent"
            ]
        return processed
