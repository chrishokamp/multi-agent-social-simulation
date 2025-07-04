"""Utility functions for simulation results operations."""
import os
from pymongo import MongoClient
from db.simulation_results import SimulationResults


def get_simulation_results(simulation_id):
    """Get simulation results by ID."""
    try:
        # Get MongoDB connection
        conn_string = os.environ.get("DB_CONNECTION_STRING", "mongodb://localhost:27017")
        client = MongoClient(conn_string)
        
        # Create results DB instance
        results_db = SimulationResults(client)
        
        # Search for the simulation results
        results_collection = results_db.results_collection
        simulation = results_collection.find_one({"simulation_id": simulation_id})
        
        return simulation
    except Exception as e:
        print(f"Error retrieving simulation results: {e}")
        return None
    finally:
        if 'client' in locals():
            client.close()