import asyncio
import click
import dotenv
import json
from engine.simulation import SelectorGCSimulation
from pprint import pprint

dotenv.load_dotenv()

@click.command()
@click.option('--config', 'config_path', required=True, type=click.Path(exists=True), help='Path to config JSON file')
@click.option('--max-messages', default=5, show_default=True, type=int, help='Maximum number of messages')
@click.option('--min-messages', default=1, show_default=True, type=int, help='Minimum number of messages')
def main(config_path, max_messages, min_messages):
  with open(config_path, 'r') as f:
    config = json.load(f)

  simulation = SelectorGCSimulation(
    config['config'],
    environment=config['initial_environment'],
    max_messages=max_messages,
    min_messages=min_messages
  )
  result = asyncio.run(simulation.run())
  pprint(result)


if __name__ == "__main__":
  main()
