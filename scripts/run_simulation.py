import asyncio
import asyncio
import json
import logging
from pathlib import Path
from pprint import pprint
from datetime import datetime

import click
import dotenv

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src" / "backend"))

from engine.simulation import SelectorGCSimulation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


async def run_once(config: dict, environment: dict, max_messages: int, min_messages: int, model: str | None = None):
    sim = SelectorGCSimulation(
        config,
        environment=environment,
        max_messages=max_messages,
        min_messages=min_messages,
        model=model,
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
    """Run repeated simulations with self-improving agents."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = raw.get("config", raw)
    num_runs = raw.get("num_runs", 5)
    model = raw.get("model") or config.get("model")

    environment = {"runs": [], "outputs": {}}
    history = []

    for run_idx in range(1, num_runs + 1):
        print(f"=== Run {run_idx}/{num_runs} ===")
        result, sim = asyncio.run(run_once(config, environment, max_messages, min_messages, model))
        if result is None:
            print("No result returned\n")
            continue

        pprint(result)

        outputs = {var["name"]: var["value"] for var in result["output_variables"]}
        history.append({"run_id": run_idx, "outputs": outputs})

        environment["runs"].append((run_idx, {"messages": result["messages"]}))
        environment["outputs"] = outputs

        # agents learn from this run before next round
        for agent in sim.agents:
            utility = 0.0
            if hasattr(agent, "compute_utility"):
                utility = agent.compute_utility({"outputs": outputs})
            for cfg in config["agents"]:
                if cfg["name"] == agent.name and cfg.get("self_improve", False):
                    old_prompt = cfg.get("prompt", "")
                    agent.learn_from_feedback(utility, environment)
                    cfg["prompt"] = getattr(agent, "system_prompt", cfg.get("prompt"))
                    logger.info(
                        "Agent: %s -- optimising prompt for round %s, previous prompt: %s new prompt after optimisation: %s",
                        agent.name,
                        run_idx,
                        old_prompt,
                        cfg["prompt"],
                    )

    # Save individual run immediately
    run_output_path = config_path.with_name(f"run_{run_idx}_results.json")
    with open(run_output_path, "w", encoding="utf-8") as f:
        json.dump(history[-1], f, indent=2)

    # Save full history incrementally
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config_path.with_name(f"history_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


    # Persist updated prompts so future runs start with improved agents
    optimized_path = config_path.with_name(f"{config_path.stem}_optimized.json")
    with open(optimized_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "num_runs": num_runs, "model": model}, f, indent=2)
    print(f"Updated config written to {optimized_path}")


if __name__ == "__main__":
    main()
