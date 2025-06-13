import asyncio
from prefect import flow, task
from pymongo import MongoClient
from db.simulation_queue import SimulationQueue
from db.simulation_results import SimulationResults
from db.simulation_catalog import SimulationCatalog
import time

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "simulation_db"

# Initialize MongoDB Client
mongo_client = MongoClient(MONGO_URI)

# Initialize DB Components
simulation_queue = SimulationQueue(mongo_client)
simulation_results = SimulationResults(mongo_client)
simulation_catalog = SimulationCatalog(mongo_client)

@task
def insert_simulation_to_queue(config, num_runs):
    simulation_id = simulation_queue.insert(config, num_runs)
    return simulation_id

@task
def retrieve_simulation_from_queue():
    simulation = simulation_queue.retrieve_next()
    
    if not simulation:  # Check if no simulation is returned
        print("No simulation found in the queue.")
        return None, None  # or return empty dicts, depending on your flow's needs
    
    simulation_id, config = simulation
    print(f"Simulation retrieved: {simulation_id}")
    return simulation_id, config

@task
def execute_simulation(simulation_id, config):
    # Mock simulation logic: Here you would have your actual simulation code
    results = {
        "messages": [{"agent": "Agent1", "message": "Hello, world!"}],
        "output_variables": [{"name": "var1", "value": "123"}]
    }
    simulation_results.insert(simulation_id, results)
    return results

@task
def update_simulation_catalog(simulation_id):
    print(f"Looking for simulation_id: {simulation_id}")  # Debugging line
    query = {"simulation_id": simulation_id}
    result = simulation_catalog.catalog_collection.find_one(query)
    if result is None:
        print("No matching simulation found in catalog.")  # Debugging line
    else:
        print(f"Found simulation: {result}")  # Debugging line
    expected_runs = result["expected_runs"] if result else None
    return expected_runs

@flow
def simulation_flow(config, num_runs):
    # Insert a new simulation into the queue
    simulation_id = insert_simulation_to_queue(config, num_runs)

    # Retrieve the simulation from the queue and execute it
    simulation_id, config = retrieve_simulation_from_queue()

    if simulation_id is None or config is None:  # Check if no simulation was retrieved
        print("No simulation to run.")
        return {"status": "Queue is empty"}

    # Execute the simulation and store the results
    results = execute_simulation(simulation_id, config)

    # Update the simulation catalog progress
    progress = update_simulation_catalog(simulation_id)

    return {
        "simulation_id": simulation_id,
        "results": results,
        "progress": progress
    }

async def run_flow():
    # Sample simulation configuration
    sample_config = {
        "name": "Test Simulation",
        "agents": [{"name": "Agent1", "description": "First agent", "prompt": "Start"}],
        "termination_condition": "Condition Met",
        "output_variables": [{"name": "var1", "type": "Number"}]
    }
    num_runs = 5

    # Run the Prefect flow
    result = simulation_flow(config=sample_config, num_runs=num_runs)
    print(result)

if __name__ == "__main__":
    # Run the asynchronous flow
    asyncio.run(run_flow())

