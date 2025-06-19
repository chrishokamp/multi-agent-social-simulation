from db.base import MongoBase

class SimulationResults(MongoBase):
    def __init__(self, mongo_client):
        super().__init__(mongo_client)
        self.results_collection = self.db["results"]

    def insert(self, simulation_id, results):
        # validate simulation results
        if results and "messages" in results and "output_variables" in results:
            for message in results["messages"]:
                if "agent" in message and message["agent"] and "message" in message:
                    continue
                else:
                    return None
                
            for variable in results["output_variables"]:
                if "name" in variable and variable["name"] and "value" in variable:
                    continue
                else:
                    return None
        else:
            return None
        
        # insert into database
        self.results_collection.insert_one({
            "simulation_id": simulation_id,
            "messages": results["messages"],
            "output_variables": results["output_variables"]
        })

        return simulation_id
    
    def retrieve(self, simulation_id):
        # retrieve all results of provided simulation
        query = {"simulation_id": simulation_id}
        return list(self.results_collection.find(query))
    
    def delete(self, simulation_id):
        result = self.results_collection.delete_one({"simulation_id": simulation_id})
        return result.deleted_count