import os
import re
import json
from pymongo import MongoClient
from flask import Blueprint, request, jsonify
from openai import AsyncOpenAI, AsyncAzureOpenAI
from utils import create_logger, client_for_endpoint
from flask import Blueprint, jsonify
from db.simulation_queue import SimulationQueue
from db.simulation_catalog import SimulationCatalog

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_queue = SimulationQueue(mongo_client)
simulation_catalog = SimulationCatalog(mongo_client)

logger = create_logger(__name__)

stop_bp = Blueprint("stop", __name__)

@stop_bp.route("/stop", methods=["POST"])
def stop_simulation():
    sim_id = request.get_json().get("id")
    if not sim_id:
        return jsonify({"error": "Missing simulation id"}), 400

    success = simulation_queue.delete(sim_id)
    if success:
        return jsonify({"message": f"Cancelled simulation {sim_id}"}), 200
    else:
        return jsonify({"message": f"No pending simulation found with id {sim_id}"}), 404
