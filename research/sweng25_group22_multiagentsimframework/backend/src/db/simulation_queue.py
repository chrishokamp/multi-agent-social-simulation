import uuid
import time

from db.base import MongoBase

from utils import create_logger
logger = create_logger(__name__)

class SimulationQueue(MongoBase):
    def __init__(self, mongo_client):
        super().__init__(mongo_client)
        self.queue_collection = self.db["queue"]

    def insert(self, config, num_runs):
        simulation_id = str(uuid.uuid4())[:8]
        
        # First validate and process the config
        processed_id = self.insert_with_id(simulation_id, config, num_runs)
        
        if processed_id:
            print(f"Successfully inserted simulation with ID {simulation_id}")
            return processed_id
        else:
            print(f"Failed to insert simulation with ID {simulation_id}")
            return None
    
    def insert_with_id(self, simulation_id, config, num_runs):
        # validate simulation config
        if config and "name" in config and config["name"] and \
            "agents" in config and len(config["agents"]) >= 2 and \
            "termination_condition" in config and config["termination_condition"] and \
            "output_variables" in config and len(config["output_variables"]) >= 1:
            for agent in config["agents"]:
                if "name" in agent and agent["name"] and \
                    "description" in agent and agent["description"] and \
                    "prompt" in agent and agent["prompt"]:
                    # Replace spaces with underscores in agent names
                    agent["name"] = agent["name"].replace(" ", "_")
                    continue
                else:
                    return None

            for variable in config["output_variables"]:
                if "name" in variable and variable["name"] and \
                    "type" in variable:
                    continue
                else:
                    return None
        else:
            return None
        
        # validate number of runs
        if num_runs < 1:
            return None

        # insert into database
        self.queue_collection.insert_one({
            "simulation_id": simulation_id,
            "timestamp": int(time.time()),
            "remaining_runs": num_runs,
            "config": config
        })
        
        self.db["configs"].update_one(
            {"simulation_id": simulation_id},
            {"$set": {"config": config}},
            upsert=True,
        )
        
        print(f"Inserted simulation: {simulation_id}")
        
        return simulation_id
    
    def retrieve_next(self):
        # get oldest object
        print("Checking queue collection...")
        
        oldest_object = self.queue_collection.find_one(sort=[("timestamp", 1)])
        
        if not oldest_object:
            print("Queue is empty.")  
            return None  # return None or an empty dict if nothing is found

        remaining_runs = oldest_object["remaining_runs"] - 1
        query = {"simulation_id": oldest_object["simulation_id"]}
        self.queue_collection.update_one(query, {"$inc": {"remaining_runs": -1}})

        if remaining_runs <= 0:
            logger.info("Simulation %s has no remaining runs, deleting from queue.", oldest_object["simulation_id"])
            self.queue_collection.delete_one(query)

        simulation_id = oldest_object["simulation_id"]
        config = oldest_object["config"]

        print(f"Retrieved simulation: {simulation_id}, {config}")  
        return simulation_id, config
    
    def delete(self, simulation_id):
        result = self.queue_collection.delete_one({"simulation_id": simulation_id})
        return result.deleted_count > 0
