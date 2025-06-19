import os
import json
from pathlib import Path
from pymongo import MongoClient
from flask import Blueprint, request, jsonify

from db.simulation_queue import SimulationQueue
from db.simulation_catalog import SimulationCatalog

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_queue = SimulationQueue(mongo_client)
simulation_catalog = SimulationCatalog(mongo_client)

create_bp = Blueprint("create", __name__)

from utils import create_logger
logger = create_logger(__name__)

@create_bp.route("/create", methods=["POST", "PUT"])
def create_simulation():
    request_json = request.get_json(force=True)

    # raw json config
    if "agents" in request_json:
        config = request_json

    # nested inside a {"config": ...} object
    elif "config" in request_json:
        config = request_json["config"]

    # load from file path
    elif "path" in request_json:
        path = Path(request_json["path"])
        if not path.exists():
            return jsonify({"error": f"File not found: {path}"}), 400
        with open(path, "r") as f:
            config = json.load(f)

    else:
        return jsonify({"error": "Invalid request. Provide either full config JSON, a 'config' field, or a 'path'."}), 400

    num_runs = request_json["num_runs"]
    logger.info(f"Received simulation config: {config}")
    request_json["config"] = config
    request_json["num_runs"] = num_runs

    if "num_runs" in request_json and "config" in request_json:
        simulation_id = simulation_queue.insert(request_json["config"], request_json["num_runs"])
        if simulation_id:
            if simulation_catalog.insert(simulation_id, request_json["config"]["name"], request_json["num_runs"]):
                return jsonify(
                    {"message": f"Successfully created simulation with ID: {simulation_id} and {request_json['num_runs']} runs."}
                ), 200

    return jsonify({"message": "Invalid request syntax."}), 400