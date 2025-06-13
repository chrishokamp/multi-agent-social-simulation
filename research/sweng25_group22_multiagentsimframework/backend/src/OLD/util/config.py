import os
import json

class SimConfigLoader:
    def __init__(self, file_name):
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sim_configs")
        self.file_path = os.path.join(base_path, "sim_config.json")

    def _format_agent_data(self, agent):
        description = f"You are: {agent['description']}\n"
        parameters = f"parameters = {json.dumps(agent['parameters'], indent=2)}\n"
        free_prompt = f"\n{agent['free_prompt']}"
        return f"""{description}\n{parameters}\n{free_prompt}"""

    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        
        formatted_agents = [
            {
                "name": agent["name"],
                "description": agent["description"],
                "system_message": self._format_agent_data(agent)  # Now this can be found
            }
            for agent in config["agents"]
        ]

        return {
            "name": config["name"],
            "models": config["models"],
            "agents": formatted_agents,
            "termination_condition": config["termination_condition"],
            "output_variables": config["output_variables"]
        }
