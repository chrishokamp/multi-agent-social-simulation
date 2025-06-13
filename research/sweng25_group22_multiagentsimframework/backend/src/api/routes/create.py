import os
from pymongo import MongoClient
from flask import Blueprint, request, jsonify

from db.simulation_queue import SimulationQueue
from db.simulation_catalog import SimulationCatalog

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_queue = SimulationQueue(mongo_client)
simulation_catalog = SimulationCatalog(mongo_client)

create_bp = Blueprint("create", __name__)

@create_bp.route("/create", methods=["POST", "PUT"])
def create_simulation():
    request_json = request.get_json()

    if "num_runs" in request_json and "config" in request_json:
        simulation_id = simulation_queue.insert(request_json["config"], request_json["num_runs"])
        if simulation_id:
            if simulation_catalog.insert(simulation_id, request_json["config"]["name"], request_json["num_runs"]):
                return jsonify(
                    {"message": f"Successfully created simulation with ID: {simulation_id} and {request_json['num_runs']} runs."}
                ), 200

    return jsonify({"message": "Invalid request syntax."}), 400