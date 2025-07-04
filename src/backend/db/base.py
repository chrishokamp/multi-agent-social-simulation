class MongoBase:
    def __init__(self, mongo_client):
        self.mongo_client = mongo_client
        self.db = self.mongo_client["simulation_gym"]