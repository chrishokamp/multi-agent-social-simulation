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
            print(" Done!")
            break
        else:
            print(f"Failed run for simulation ID {simulation_id}! Retrying...")

def start_simulation(simulation_id, simulation_config):
    asyncio.run(run_simulation(simulation_id, simulation_config))

def orchestrator(max_threads=4):
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    simulation_queue = SimulationQueue(mongo_client)

    print("Listening for simulations...")
    while True:
        job = simulation_queue.retrieve_full_job()
        if job is None:
            time.sleep(5)
            logger.info("No jobs in queue, sleeping for 5 seconds...")
            continue

        sim_id, config, num_runs = job
        print(f"→ running simulation {sim_id} synchronously for {num_runs} runs…")
        run_all_runs(sim_id, config, num_runs)


def run_all_runs(simulation_id: str, simulation_config: dict, num_runs: int, update_catalog=True):
    """
        Run a simulation `num_runs` times synchronously, 
        passing the growing `env` into each new run so 
        self-improvment can happen
    """
    mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
    results_store  = SimulationResults(mongo_client)
    catalog_store  = SimulationCatalog(mongo_client)
    
    # Set status to running
    if update_catalog:
        catalog_store.update_status(simulation_id, "running")

    env = {"runs": []}

    for i in range(num_runs):
        # each SelectorGCSimulation will call learn_from_feedback()
        # on the previous `env` and then replay with the updated prompt
        sim = SelectorGCSimulation(simulation_config, environment=env, simulation_id=simulation_id)
        simulation_result = asyncio.run(sim.run())

        if not simulation_result:
            print(f"Run {i} failed; retrying...")
            logger.error(f"Simulation {simulation_id} run {i+1} returned no result")
            continue

        logger.info(f"Simulation {simulation_id} run {i+1} completed with result type: {type(simulation_result)}")
        
        if update_catalog:
            try:
                print(f"[Orchestrator] About to call results_store.insert for {simulation_id}")
                insert_result = results_store.insert(simulation_id, simulation_result)
                print(f"[Orchestrator] insert_result = {insert_result}")
                if insert_result:
                    logger.info(f"Saved results for simulation {simulation_id} run {i+1}")
                else:
                    logger.error(f"Insert returned None for simulation {simulation_id} run {i+1}")
                catalog_store.update_progress(simulation_id)
                logger.info(f"Updated progress for simulation {simulation_id}")
            except Exception as e:
                logger.error(f"Failed to save results for simulation {simulation_id}: {e}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        sim_history_dir = f"simulations/{simulation_id}/history"
        os.makedirs(sim_history_dir, exist_ok=True)
        env_path = f"{sim_history_dir}/simulation_{simulation_id}_env_run_{i+1}_{timestamp}.json"
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2)
        logger.info(f"Saved environment to {env_path}")

    print(f"All {num_runs} runs of simulation {simulation_id} complete.")
