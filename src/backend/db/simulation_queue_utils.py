"""Utility functions for simulation queue operations."""
import os
from pymongo import MongoClient
from db.simulation_queue import SimulationQueue


def get_simulation_from_queue(simulation_id):
    """Get a simulation from the queue by ID."""
    try:
        # Get MongoDB connection
        conn_string = os.environ.get("DB_CONNECTION_STRING", "mongodb://localhost:27017")
        client = MongoClient(conn_string)
        
        # Create queue DB instance
        queue_db = SimulationQueue(client)
        
        # Search for the simulation in the queue
        queue_collection = queue_db.queue_collection
        simulation = queue_collection.find_one({"simulation_id": simulation_id})
        
        return simulation
    except Exception as e:
        print(f"Error retrieving simulation from queue: {e}")
        return None
    finally:
        if 'client' in locals():
            client.close()