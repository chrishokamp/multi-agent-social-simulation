import click
import json
from pathlib import Path
from pprint import pprint
from collections import defaultdict


@click.command()
@click.option('--results', 'result_dir', required=True, type=click.Path(exists=True), help='Path to config JSON file')
def main(result_dir):

    simulations = []
    for file_path in Path(result_dir).iterdir():
        assert file_path.suffix == '.json', f"Expected .json file, got {file_path.suffix}"
        with open(file_path, 'r') as f:
            result = json.load(f)
            simulations.append(result)

    agent_names = list(simulations[0]['runs'][-1]['output_variables']['utility'].keys())

    simulations.sort(key=lambda x: len(x['runs']), reverse=True)

    agent_to_utilities = defaultdict(list)
    for sim in simulations:
        for run in sim['runs']:
            # example: 'utility': {'Responder': 1.0, 'Walker': 0.0}}
            utility_scores = run['output_variables']['utility']
            for agent_name in agent_names:
                agent_to_utilities[agent_name].append(utility_scores.get(agent_name, 0.0))

    # Print the utilities for each agent
    for agent_name, utilities in agent_to_utilities.items():
        print(f"Agent: {agent_name}")
        print(f"Utilities: {utilities}")

    # plot results over runs (time) using matplotlib

    import matplotlib.pyplot as plt


if __name__ == "__main__":
  main()
