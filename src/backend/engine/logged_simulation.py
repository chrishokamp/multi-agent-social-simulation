"""
Enhanced simulation class with comprehensive logging.
"""
import re
import os
import json
import sys
from pathlib import Path

# Add logging_framework to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from autogen import ConversableAgent
from .simulation import SelectorGCSimulation
from logging_framework.core import SimulationLogger
from typing import Dict, Any, Optional


class LoggedSelectorGCSimulation(SelectorGCSimulation):
    """Extended simulation class with rich logging capabilities."""
    
    def __init__(self, config: dict, environment: dict, 
                 max_messages=25, min_messages=5, 
                 model: str | None = None,
                 log_dir: Optional[Path] = None):
        # Initialize parent class
        super().__init__(config, environment, max_messages, min_messages, model)
        
        # Initialize logger
        self.logger = SimulationLogger(self.run_id, log_dir)
        
        # Store simulation configuration
        simulation_info = {
            'simulation_id': self.run_id,
            'config': config,
            'model': model or config.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o"),
            'max_messages': max_messages,
            'min_messages': min_messages
        }
        
        with open(self.logger.log_dir / "simulation_info.json", 'w') as f:
            json.dump(simulation_info, f, indent=2)
        
        # Hook into agent initialization to add logging
        self._setup_agent_logging()
    
    def _setup_agent_logging(self):
        """Setup logging for each agent."""
        for agent in self.agents:
            # Create agent logger
            agent_logger = self.logger.get_agent_logger(agent.name)
            
            # Store reference to logger in agent
            agent._logger = agent_logger
            
            # Log initial utility if available
            if hasattr(agent, 'compute_utility'):
                initial_utility = agent.compute_utility(self.config.get('environment', {}))
                agent_logger.log_utility(0, initial_utility, self.config.get('environment', {}))
            
            # Log agent initialization
            agent_logger.log_action(
                'initialization',
                f"Agent {agent.name} initialized with strategy: {getattr(agent, 'strategy', {})}"
            )
    
    def _process_result(self, chat_result):
        """Process results with logging."""
        # Increment round for each message exchange
        current_round = 0
        
        if len(chat_result.chat_history) < self.min_messages:
            self.logger.logger.warning(f"Chat history too short: {len(chat_result.chat_history)} < {self.min_messages}")
            return None
        
        messages = []
        for idx, message in enumerate(chat_result.chat_history):
            # Determine round number (new round every time we cycle through agents)
            if idx > 0 and idx % len(self.agents) == 0:
                current_round += 1
                self.logger.increment_round()
            
            # Extract message info
            if isinstance(message, dict):
                agent_name = message.get("name", message.get("role"))
                content = message.get("content", "")
            else:
                agent_name = getattr(message, "source", getattr(message, "name", getattr(message, "role", "")))
                content = getattr(message, "content", "")
            
            messages.append({"agent": agent_name, "message": content})
            
            # Log message
            self.logger.log_message(agent_name, content, {'round': current_round})
            
            # Find corresponding agent and log action
            for agent in self.agents:
                if agent.name == agent_name:
                    if hasattr(agent, '_logger'):
                        agent._logger.log_action('message', content, {'round': current_round})
                    break
        
        # Process output variables
        output_variables = []
        information_return_agent_message = messages[-1]["message"]
        json_match = re.search(r'\{.*\}', information_return_agent_message, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
                for variable in parsed_json:
                    value = parsed_json[variable]
                    if value is None or value == "Unspecified":
                        value = "Unspecified"
                    output_variables.append({"name": variable, "value": value})
                
                # Log output variables as metrics
                for var in output_variables:
                    if var['value'] != "Unspecified":
                        self.logger.metrics.record(var['name'], var['value'])
                        
            except json.JSONDecodeError:
                self.logger.logger.error("Failed to parse output variables JSON")
                return None
        else:
            self.logger.logger.error("No JSON found in information return agent message")
            return None
        
        # Compute and log final utilities
        environment = {"outputs": {v["name"]: v["value"] for v in output_variables}}
        for agent in self.agents:
            if hasattr(agent, 'compute_utility'):
                final_utility = agent.compute_utility(environment)
                self.logger.log_utility_update(agent.name, final_utility, environment)
                
                if hasattr(agent, '_logger'):
                    agent._logger.log_action(
                        'final_evaluation',
                        f"Final utility: {final_utility:.4f}",
                        {'utility': final_utility, 'outputs': environment['outputs']}
                    )
        
        # Save all logs
        self.logger.save_logs()
        
        return {"messages": messages, "output_variables": output_variables}
    
    async def run(self):
        """Run simulation with logging."""
        self.logger.logger.info(f"Starting simulation {self.run_id}")
        
        # Log initial state
        for agent in self.agents:
            if hasattr(agent, '_logger'):
                agent._logger.log_action(
                    'pre_conversation',
                    f"System prompt: {agent.system_message[:200]}..."
                )
        
        # Run parent simulation
        starter = ConversableAgent(
            "starter", llm_config=self.llm_config, human_input_mode="NEVER"
        )
        chat_result = await starter.a_initiate_chat(
            recipient=self.manager,
            message="Begin",
            max_turns=1,
            silent=True,
        )
        
        result = self._process_result(chat_result)
        
        # Log completion
        self.logger.logger.info(f"Simulation {self.run_id} completed")
        
        return result