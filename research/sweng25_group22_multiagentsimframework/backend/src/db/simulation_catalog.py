from db.base import MongoBase
from db.simulation_results import SimulationResults

from utils import create_logger
logger = create_logger(__name__)

class SimulationCatalog(MongoBase):
    def __init__(self, mongo_client):
        super().__init__(mongo_client)
        self.catalog_collection = self.db["catalog"]
        self.simulation_results = SimulationResults(mongo_client)

    def insert(self, simulation_id, name, num_runs):
        if not simulation_id or not name or num_runs < 1:
            return None

        self.catalog_collection.insert_one({
            "simulation_id": simulation_id,
            "name": name,
            "expected_runs": num_runs,
            "progress_percentage": 0
        })

        return simulation_id
    
    def update_progress(self, simulation_id):
        query = {"simulation_id": simulation_id}

        expected_runs = self.catalog_collection.find_one(query)["expected_runs"]
        completed_runs = len(self.simulation_results.retrieve(simulation_id))
        progress_percentage = int((completed_runs / expected_runs) * 100)

        self.catalog_collection.update_one(query, {"$set": {"progress_percentage": progress_percentage}})

        return progress_percentage
    
    def get_all(self):
        try:
            data = self.catalog_collection.find()
        except Exception as e:
            logger.error("Error fetching data from catalog: %s", e)
            return []
        catalog = []

        for doc in data:
            catalog.append({
                "simulation_id": doc["simulation_id"],
                "name": doc["name"],
                "expected_runs": doc["expected_runs"],
                "progress_percentage": doc["progress_percentage"]
            })

        return catalog
    
    def exists(self, simulation_id):
        query = {"simulation_id": simulation_id}
        return self.catalog_collection.find_one(query) is not None
    
    def delete(self, simulation_id):
        result = self.catalog_collection.delete_one({"simulation_id": simulation_id})
        return result.deleted_count
