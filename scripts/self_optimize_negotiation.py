import asyncio
import json
from pathlib import Path
from pprint import pprint

import click
import dotenv

import sys

# Allow running without installation by adjusting PYTHONPATH
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src" / "backend"))

from engine.simulation import SelectorGCSimulation

dotenv.load_dotenv()


async def run_once(config: dict, environment: dict, max_messages: int, min_messages: int):
    sim = SelectorGCSimulation(
        config,
        environment=environment,
        max_messages=max_messages,
        min_messages=min_messages,
    )
    result = await sim.run()
    return result, sim


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation configuration JSON",
)
@click.option("--max-messages", default=10, show_default=True, help="Maximum conversation length")
@click.option("--min-messages", default=1, show_default=True, help="Minimum messages for valid result")
def main(config_path: Path, max_messages: int, min_messages: int):
    """Run a self-optimising negotiation simulation."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = raw.get("config", raw)
    num_runs = raw.get("num_runs", 5)

    environment = {"runs": [], "outputs": {}}
    history = []

    for run_idx in range(1, num_runs + 1):
        print(f"=== Run {run_idx}/{num_runs} ===")
        result, sim = asyncio.run(run_once(config, environment, max_messages, min_messages))
        if not result:
            print("No result returned\n")
            continue

        pprint(result)

        outputs = {var["name"]: var["value"] for var in result["output_variables"]}
        history.append({"run_id": run_idx, "outputs": outputs})

        environment["runs"].append((run_idx, {"messages": result["messages"]}))
        environment["outputs"] = outputs

        # persist improved prompts for next run
        for agent in sim.agents:
            for cfg in config["agents"]:
                if cfg["name"] == agent.name:
                    cfg["prompt"] = getattr(agent, "system_prompt", cfg.get("prompt"))

    out_path = config_path.with_name("history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"History written to {out_path}")


if __name__ == "__main__":
    main()
