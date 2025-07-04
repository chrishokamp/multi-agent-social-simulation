import re
import os
import json
import textwrap
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from pydantic import BaseModel, ValidationError
from autogen import (
    GroupChat,
    GroupChatManager,
    ConversableAgent,
    LLMConfig,
)
from engine.streaming_group_chat import StreamingGroupChat, StreamingGroupChatManager
from agents import UtilityAgent, BuyerAgent, SellerAgent, NegotiationCoachAgent, NegotiationCoachBuyerAgent, NegotiationCoachSellerAgent
from utils import create_logger, client_for_endpoint
from logging_framework.core import SimulationLogger
from engine.message_hook import get_hook_manager, HookedConversableAgent, FileStreamHook

logger = create_logger(__name__)

_utility_class_registry = {
    "UtilityAgent": UtilityAgent,
    "BuyerAgent":   BuyerAgent,
    "SellerAgent":  SellerAgent,
    "NegotiationCoachAgent": NegotiationCoachAgent,
    "NegotiationCoachBuyerAgent": NegotiationCoachBuyerAgent,
    "NegotiationCoachSellerAgent": NegotiationCoachSellerAgent,
}


class SimulationRun(BaseModel):
    run_id: str
    output_variables: Any  # Can be either Dict[str, Any] or List[Dict[str, Any]]
    system_prompts: Optional[Dict[str, str]] = None
    messages: Optional[List[Dict[str, str]]] = None


class SimulationEnvironment(BaseModel):
    runs: Optional[List[SimulationRun]] = []
    config: Optional[dict] = None


def validate_environment(env: dict) -> dict:
    try:
        SimulationEnvironment(**env)  # just validate
        return env
    except ValidationError as e:
        raise ValueError(f"Invalid environment format: {e}")


class SelectorGCSimulation:
    def __init__(
        self,
        config: dict,
        environment: dict,
        model: Optional[str] = None,
        log_dir: Optional[Path] = None,
        simulation_id: Optional[str] = None,
    ):
        model_name = model or config.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.config = config
        self.llm_config = LLMConfig(api_type="openai", model=model_name)
        self.min_messages = config.get("min_messages", 2)
        self.max_messages = config.get("max_messages", 25)
        self.run_id = str(uuid.uuid4())
        self.simulation_id = simulation_id or self.run_id  # Use provided simulation_id or generate one
        self.environment = validate_environment(environment)
        self.environment["config"] = self.config

        logger.info(f"Initializing SelectorGCSimulation {self.simulation_id} (run: {self.run_id}) with config: {self.config}")

        # set up SimulationLogger
        self.sim_logger = SimulationLogger(self.run_id, log_dir)
        simulation_info = {
            "simulation_id": self.run_id,
            "config": self.config,
            "model": model_name,
            "max_messages": self.max_messages,
            "min_messages": self.min_messages,
        }
        # persist simulation metadata
        with open(self.sim_logger.log_dir / "simulation_info.json", "w") as f:
            json.dump(simulation_info, f, indent=2)

        # registry for variable types
        type_mapping = {
            "String": "STRING",
            "Number": "NUMBER",
            "Boolean": "BOOLEAN",
            "Float": "FLOAT",
            "Date": "DATE",
            "List": "LIST",
            "Dict": "DICT",
        }

        # inject InformationReturnAgent if missing
        if not any(a["name"] == "InformationReturnAgent" for a in self.config["agents"]):
            config_dir = Path(__file__).resolve().parents[1] / "config"
            with open(config_dir / "InformationReturnAgent.json", "r", encoding="utf-8") as file:
                ira = json.load(file)
            ira["prompt"] = ira["prompt"].format(
                output_variables_str=(
                    "{\n"
                    + ",\n".join(
                        f'"{v["name"]}": "{type_mapping.get(v["type"], "UNKNOWN_TYPE")}"'
                        for v in self.config["output_variables"]
                    )
                    + "\n}"
                ),
                termination_condition=self.config.get("termination_condition", "TERMINATE"),
            )
            self.config["agents"].append(ira)

                                
        self.agents = []
        for agent_cfg in self.config["agents"]:
            cls_name = agent_cfg.get("utility_class", "UtilityAgent")
            AgentClass = _utility_class_registry[cls_name]
            ag = AgentClass(
                system_prompt=agent_cfg["prompt"],
                name=agent_cfg["name"],
                description=agent_cfg["description"],
                llm_config=self.llm_config,
                strategy=agent_cfg.get("strategy"),
                model=model or self.config.get("model"),
                optimization_prompt=config.get("optimization_prompt")
            )
            # Set simulation_id and run_id on the agent for optimization tracking
            ag.simulation_id = self.simulation_id
            ag.run_id = self.run_id
            
            # allow self-improvement from prior runs
            if self.environment.get("runs") and agent_cfg.get("self_improve", False):
                ag.learn_from_feedback(self.environment)
            self.agents.append(ag)

        # Use StreamingGroupChat for real-time message streaming
        self.group_chat = StreamingGroupChat(
            agents=self.agents,
            messages=[],
            simulation_id=self.simulation_id,
            run_id=self.run_id,
            max_round=self.max_messages,
            speaker_selection_method="auto",
        )
        self.manager = StreamingGroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
            is_termination_msg=lambda m: "TERMINATE" in m.get("content", "").upper(),
        )

        # wire up detailed per-agent logging
        self._setup_agent_logging()
        
        # Setup message hooks for real-time streaming
        self._setup_message_hooks()

    def _setup_agent_logging(self):
        """Attach a SimulationLogger per agent and record init utilities/actions."""
        for agent in self.agents:
            ag_logger = self.sim_logger.get_agent_logger(agent.name)
            agent._logger = ag_logger

            # log initial utility if supported
            if hasattr(agent, "compute_utility"):
                init_util = agent.compute_utility(self.environment.get("config", {}))
                ag_logger.log_utility(0, init_util, self.environment.get("config", {}))

            # log creation
            ag_logger.log_action(
                "initialization",
                f"Agent {agent.name} initialized with strategy: {getattr(agent, 'strategy', {})}"
            )
    
    def _setup_message_hooks(self):
        """Setup message hooks for real-time streaming."""
        # Check if streaming is enabled via environment or config
        enable_streaming = os.environ.get('ENABLE_REALTIME_STREAMING', 'true').lower() == 'true'
        
        if not enable_streaming:
            return
            
        # Add file stream hook
        hook_manager = get_hook_manager()
        stream_hook = FileStreamHook(self.sim_logger.log_dir)
        hook_manager.add_hook(stream_hook)
        
        # Wrap all agents with hooked versions
        for agent in self.agents:
            HookedConversableAgent(agent, self.simulation_id, self.run_id)
            
        # Also wrap the manager
        if hasattr(self, 'manager'):
            HookedConversableAgent(self.manager, self.simulation_id, self.run_id)

    def force_info_return(
        self,
        messages: dict,
        output_variables: list[str],
    ) -> dict:
        """Call the InfoReturn agent post-hoc to populate missing output variables."""
        client, model_name = client_for_endpoint()

        full_transcript = "\n".join(f"{m['agent']}: {m['message']}" for m in messages)
        
        prompt = textwrap.dedent(f"""
        You are an AI assistant tasked with analyzing a conversation between multiple LLM agents. 
        Your goal is to extract specific variables from the conversation and output them in JSON format when a specific termination condition is met.
        Monitor the conversation and track relevant details as messages are exchanged between the agents.
        Incase of output variables like string variables, comprehensively look at the conversation and output concise and objective information, i.e in case of a court case simulation demanding verdict as a str, output the verdict as the length of prison sentence etc, do not simply state that the verdict was reached

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
        if json_match is None:
            parsed_json = [None] * len(output_variables)
        else:
            parsed_json = json.loads(json_match.group(0))
        for variable in parsed_json:
            # Handle both None and "Unspecified" values
            value = parsed_json[variable]
            if value is None or value == "Unspecified":
                value = "Unspecified"
            output_variables.append({"name": variable, "value": value})
        return output_variables

    def _process_result(self, simulation_result):
        """Process chat history, extract variables, compute utilities, log everything."""
        # round counter
        current_round = 0

        if len(simulation_result.chat_history) < self.min_messages:
            self.sim_logger.logger.warning(
                f"Chat history too short: {len(simulation_result.chat_history)} < {self.min_messages}"
            )
            self.sim_logger.save_logs()
            return None

        # Get the hook manager for streaming
        hook_manager = get_hook_manager()

        # log each message and agent action
        messages = []
        for idx, msg in enumerate(simulation_result.chat_history):
            if idx > 0 and idx % len(self.agents) == 0:
                current_round += 1
                self.sim_logger.increment_round()

            agent_name = msg["name"]
            content = msg["content"]
            messages.append({"agent": agent_name, "message": content})

            # Stream the message in real-time
            hook_manager.on_message(agent_name, content, {
                "round": current_round,
                "index": idx,
                "simulation_id": self.simulation_id,
                "run_id": self.run_id
            })

            self.sim_logger.log_message(agent_name, content, {"round": current_round})
            for ag in self.agents:
                if ag.name == agent_name and hasattr(ag, "_logger"):
                    ag._logger.log_action("message", content, {"round": current_round})
                    break

        # extract output variables
        output_vars: List[Dict[str, Any]] = []
        last = messages[-1]
        if last["agent"] == "InformationReturnAgent":
            output_vars = self.parse_ira_message(last["message"], output_vars)
        else:
            self.sim_logger.logger.warning("Last message not from IRA, forcing info return")
            output_vars = self.force_info_return(
                messages=messages,
                output_variables=self.config["output_variables"]
            )

        # log metrics for each variable
        for var in output_vars:
            if var["value"] not in (None, "Unspecified"):
                self.sim_logger.metrics.record(var["name"], var["value"])

        # persist and return
        self.sim_logger.save_logs()
        strategies = {}

        for agent in self.agents:
            strat = self.environment.get(f"strategies_{agent.name.lower()}", [])
            if strat:
                strategies[agent.name.lower()] = strat
        if strategies:
            with open(self.sim_logger.log_dir / "strategies.json", "w") as f:
                json.dump(strategies, f, indent=2)
        return {
            "run_id": self.run_id,
            "system_prompts": {ag.name: ag.system_prompt for ag in self.agents},
            "messages": messages,
            "output_variables": output_vars,  # Return as list, not dict
        }
    def calculate_utility(self) -> None:
        for ag in self.agents:
            if len(self.environment.get("runs", [])) > 0:
                # we have >=1 runs to learn from
                self.environment = ag.compute_utility(self.environment)

    async def run(self):
        """Run the simulation, logging at start and completion."""
        self.sim_logger.logger.info(f"Starting simulation {self.run_id}")
        
        # Notify hooks of simulation start
        hook_manager = get_hook_manager()
        hook_manager.on_simulation_start(self.simulation_id, self.run_id)

        # pre-conversation log
        for ag in self.agents:
            if hasattr(ag, "_logger"):
                ag._logger.log_action(
                    "pre_conversation",
                    f"System prompt: {ag.system_prompt[:200]}..."
                )

        starter = ConversableAgent("starter", llm_config=self.llm_config, human_input_mode="NEVER")
        chat_result = await starter.a_initiate_chat(
            recipient=self.manager,
            message="Begin",
            max_turns=1,
            silent=True,
        )  # TODO: add private reflection
        # Use the group chat messages instead of the starter-manager conversation
        # Create a chat result-like object with the group chat messages
        class GroupChatResult:
            def __init__(self, messages):
                self.chat_history = messages
        
        group_chat_result = GroupChatResult(self.group_chat.messages)
        processed = self._process_result(group_chat_result)
        
        self.environment.get("runs", []).append(processed)
        self.calculate_utility()
        
        # Notify hooks of simulation end
        hook_manager.on_simulation_end(self.simulation_id, self.run_id, processed)

        # if processed:
        #     processed["private_reflections"] = [
        #         {"agent": h["agent"], "thought": h.get("thought", "")}
        #         for h in running_history
        #         if "thought" in h and h["agent"] != "InformationReturnAgent"
        #     ]
        return processed

