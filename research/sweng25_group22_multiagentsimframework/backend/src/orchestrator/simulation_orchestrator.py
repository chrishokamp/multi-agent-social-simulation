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

    print(f"All {num_runs} runs of simulation {simulation_id} complete.")


async def run_simulation(simulation_id, simulation_config):
    print(f"Starting run for simulation ID: {simulation_id}...")

    env = None
    while True:
        simulation = SelectorGCSimulation(simulation_config, environment=env)
        simulation_result = await simulation.run()
        if simulation_result:
            mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
            simulation_results = SimulationResults(mongo_client)
            simulation_catalog = SimulationCatalog(mongo_client)

            print(f"Successfull run for simulation ID {simulation_id}! Saving result...", end="")
            simulation_results.insert(simulation_id, simulation_result)
            simulation_catalog.update_progress(simulation_id)

            env["runs"] = env.get("runs", {})
            env["runs"].append(
                {
                    "run_id": simulation.run_id,
                    "messages": simulation_result["messages"],
                    "outputs": {
                        v["name"]: v["value"] for v in simulation_result["output_variables"]
                    },
                }
            )

            break
        else:
            print(f"Failed run for simulation ID {simulation_id}! Retrying...")

def start_simulation(simulation_id, simulation_config):
    asyncio.run(run_simulation(simulation_id, simulation_config))

def orchestrator():
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    queue = SimulationQueue(mongo_client)

    print("Listening for simulations…")
    while True:
        job = queue.retrieve_full_job()
        if job is None:
            time.sleep(5)
            continue

        sim_id, config, num_runs = job
        print(f"→ running simulation {sim_id} synchronously for {num_runs} runs…")
        run_all_runs(sim_id, config, num_runs)
