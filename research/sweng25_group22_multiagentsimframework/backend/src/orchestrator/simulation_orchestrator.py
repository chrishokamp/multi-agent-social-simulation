import os
import sys
import time
import signal
import asyncio
import concurrent.futures
from pymongo import MongoClient

from db.simulation_queue import SimulationQueue
from db.simulation_results import SimulationResults
from db.simulation_catalog import SimulationCatalog
from engine.simulation import SelectorGCSimulation

from utils import create_logger
import json
logger = create_logger(__name__)

executor = None

def signal_handler(_sig, _frame):
    print("\nTermination signal received. Shutting down...")
    if executor:
        executor.shutdown(wait=False, cancel_futures=True)
    sys.exit(0)

def run_all_runs(simulation_id: str, simulation_config: dict, num_runs: int):
    """
        Run a simulation `num_runs` times synchronously, 
        passing the growing `env` into each new run so 
        self-improvment can happen
    """
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    results_store  = SimulationResults(mongo_client)
    catalog_store  = SimulationCatalog(mongo_client)

    env = {"runs": []}

    for i in range(num_runs):
        # each SelectorGCSimulation will call learn_from_feedback()
        # on the previous `env` and then replay with the updated prompt
        sim = SelectorGCSimulation(simulation_config, environment=env)
        simulation_result = asyncio.run(sim.run())

        if not simulation_result:
            print(f"Run {i} failed; retrying...")
            continue

        results_store.insert(simulation_id, simulation_result)
        catalog_store.update_progress(simulation_id)

        env["runs"].append({
            "run_id": sim.run_id,
            "messages": simulation_result["messages"],
            "outputs": {
                v["name"]: v["value"] 
                for v in simulation_result["output_variables"]
            }
        })
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sim_history_dir = "sim_history"
        os.mkdir(sim_history_dir, exist_ok=True)
        env_path = f"{sim_history_dir}/simulation_{simulation_id}_env_run_{i+1}_{timestamp}.json"
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2)
        logger.info(f"Saved environment to {env_path}")

    print(f"All {num_runs} runs of simulation {simulation_id} complete.")


def orchestrator():
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    queue = SimulationQueue(mongo_client)

    print("Listening for simulations…")
    while True:
        job = queue.retrieve_full_job()
        if job is None:
            time.sleep(5)
            logger.info("No jobs in queue, sleeping for 5 seconds...")
            continue

        sim_id, config, num_runs = job
        print(f"→ running simulation {sim_id} synchronously for {num_runs} runs…")
        run_all_runs(sim_id, config, num_runs)
