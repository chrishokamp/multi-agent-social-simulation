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

    global executor
    signal.signal(signal.SIGINT, signal_handler)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_threads)
    future_to_id = {}
    try:
        while True:
            next_simulation = simulation_queue.retrieve_next()
            
            if next_simulation:
                future = executor.submit(start_simulation, next_simulation[0], next_simulation[1])
                future_to_id[future] = next_simulation[0]
            
            done_futures = [f for f in future_to_id if f.done()]
            for f in done_futures:
                del future_to_id[f]
            
            time.sleep(5)
    except SystemExit:
        pass