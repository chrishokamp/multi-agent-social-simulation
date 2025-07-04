from db.base import MongoBase
from utils import create_logger

logger = create_logger(__name__)

class SimulationResults(MongoBase):
    def __init__(self, mongo_client):
        super().__init__(mongo_client)
        self.results_collection = self.db["results"]

    def insert(self, simulation_id, results):
        # validate simulation results
        print(f"[SimulationResults.insert] Called for simulation {simulation_id}")
        logger.info(f"Attempting to insert results for simulation {simulation_id}")
        logger.info(f"Results type: {type(results)}")
        logger.info(f"Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
        
        if not results:
            logger.error(f"Results is falsy for simulation {simulation_id}")
            return None
            
        if "messages" not in results:
            logger.error(f"'messages' key not found in results for simulation {simulation_id}")
            return None
            
        if "output_variables" not in results:
            logger.error(f"'output_variables' key not found in results for simulation {simulation_id}")
            return None
            
        # Check format of output_variables
        output_vars = results["output_variables"]
        logger.info(f"output_variables type: {type(output_vars)}")
        logger.info(f"output_variables content: {output_vars}")
        
        # Handle both dict and list formats
        if isinstance(output_vars, dict):
            # Convert dict format to list format expected by database
            logger.info(f"Converting output_variables from dict to list format for simulation {simulation_id}")
            output_vars_list = [{"name": k, "value": v} for k, v in output_vars.items()]
            results = dict(results)  # Make a copy to avoid modifying the original
            results["output_variables"] = output_vars_list
            
        for i, message in enumerate(results["messages"]):
            if not ("agent" in message and message["agent"] and "message" in message):
                logger.error(f"Invalid message format at index {i} for simulation {simulation_id}: {message}")
                return None
                
        for i, variable in enumerate(results["output_variables"]):
            if not ("name" in variable and variable["name"] and "value" in variable):
                logger.error(f"Invalid output variable format at index {i} for simulation {simulation_id}: {variable}")
                return None
        
        # insert into database
        try:
            insert_result = self.results_collection.insert_one({
                "simulation_id": simulation_id,
                "messages": results["messages"],
                "output_variables": results["output_variables"]
            })
            logger.info(f"Successfully inserted results for simulation {simulation_id}, inserted_id: {insert_result.inserted_id}")
        except Exception as e:
            logger.error(f"Failed to insert results for simulation {simulation_id}: {e}")
            return None

        return simulation_id
    
    def retrieve(self, simulation_id):
        # retrieve all results of provided simulation
        query = {"simulation_id": simulation_id}
        logger.info(f"Retrieving results for simulation {simulation_id}")
        results = list(self.results_collection.find(query))
        logger.info(f"Found {len(results)} results for simulation {simulation_id}")
        return results
    
    def delete(self, simulation_id):
        result = self.results_collection.delete_one({"simulation_id": simulation_id})
        return result.deleted_count