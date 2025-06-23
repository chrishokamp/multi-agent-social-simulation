import re
import os
import json

from autogen import (
    GroupChat,
    GroupChatManager,
    ConversableAgent,
    LLMConfig,
)
from agents import UtilityAgent, BuyerAgent, SellerAgent
import uuid

from utils import create_logger
logger = create_logger(__name__)

_utility_class_registry = {
    "UtilityAgent": UtilityAgent,
    "BuyerAgent":   BuyerAgent,
    "SellerAgent":  SellerAgent,
}

class SelectorGCSimulation:
    def __init__(self, config: dict, environment: dict, max_messages=25, min_messages=5, model: str | None = None):
        model_name = model or config.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.llm_config = LLMConfig(api_type="openai", model=model_name)
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
                llm_config=self.llm_config,
                strategy=agent_config.get("strategy"),
                model=model or self.config.get("model")
            )

            # (+ self-improvement)
            utility = ag.compute_utility(environment)
            if agent_config.get("self_improve", False):
                ag.learn_from_feedback(utility, environment)

            # re-init agent with update
            ag = AgentClass(
                system_prompt=ag.system_prompt,
                name=agent_config["name"],
                description=agent_config["description"],
                llm_config=self.llm_config,
                strategy=agent_config.get("strategy"),
                model=model or self.config.get("model")
            )

            self.agents.append(ag)

        # initialize group chat
        with open(os.path.join(self.config_directory, "selector_prompt.txt"), "r", encoding="utf-8") as file:
            selector_prompt = file.read()

        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=max_messages,
            speaker_selection_method="auto",
        )
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content", "").upper()),
        )

    def _process_result(self, chat_result):
        if len(chat_result.chat_history) < self.min_messages:
            return None

        messages = []
        for message in chat_result.chat_history:
            if isinstance(message, dict):
                agent_name = message.get("name", message.get("role"))
                content = message.get("content", "")
            else:
                agent_name = getattr(message, "source", getattr(message, "name", getattr(message, "role", "")))
                content = getattr(message, "content", "")
            messages.append({"agent": agent_name, "message": content})

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
                return None
        else:
            return None

        return {"messages": messages, "output_variables": output_variables}


    async def run(self):
        starter = ConversableAgent(
            "starter", llm_config=self.llm_config, human_input_mode="NEVER"
        )
        chat_result = await starter.a_initiate_chat(
            recipient=self.manager,
            message="Begin",
            max_turns=1,
            silent=True,
        )
        return self._process_result(chat_result)
