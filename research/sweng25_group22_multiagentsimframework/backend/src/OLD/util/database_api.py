from pymongo import MongoClient
import os

class DataBaseAPI:
    def __init__(self, conn_str):
        self.m_client = MongoClient(conn_str)
        self.m_database = self.m_client.sweng22_database
        self.m_coll_config = self.m_database.config
        self.m_coll_output = self.m_database.output

    def insert_config(self, json):
        if (self.m_coll_config.find_one({"id": json["id"]})):
            return
        
        result = self.m_coll_config.insert_one(json)
        print("Inserted document ID:", result.inserted_id)

    def query_config(self, id: str):
        return self.m_coll_config.find_one({"id": id})
    
    def insert_output(self, json):
        if (self.m_coll_output.find_one({"id": json["id"]})):
            return
        
        result = self.m_coll_output.insert_one(json)
        print("Inserted document ID:", result.inserted_id)

    def query_output(self, id: str):
        return self.m_coll_output.find_one({"id": id})
    
    def get_one_sim_id(self):
        doc = self.m_coll_config.find_one({}, {"id": 1})
        return doc["id"] if doc else None

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()
    db = DataBaseAPI(os.environ['DB_CONNECTION_STRING'])

    print("All sim IDs in config collection:")
    for config in db.m_coll_config.find():
        print(config["_id"])

for output in db.m_coll_output.find():
    print(output["_id"])
