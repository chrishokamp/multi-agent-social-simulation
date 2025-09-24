import os
import json
from pathlib import Path
from pymongo import MongoClient
from flask import Blueprint, request, jsonify

from db.simulation_queue import SimulationQueue
from db.simulation_catalog import SimulationCatalog
from config_utils import materialise_config, normalize_simulation_config

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_queue = SimulationQueue(mongo_client)
simulation_catalog = SimulationCatalog(mongo_client)

create_bp = Blueprint("create", __name__)

from utils import create_logger
logger = create_logger(__name__)

@create_bp.route("/create", methods=["POST", "PUT"])
def create_simulation():
    try:
        request_json = request.get_json(force=True)
        logger.info(f"Received simulation request: {request_json.keys()}")

        # Step 1: Normalize config structure
        try:
            config, variables, num_runs = normalize_simulation_config(request_json)
            logger.info(f"Normalized config: name={config.get('name', 'Unknown')}, has_variables={variables is not None}, num_runs={num_runs}")
        except ValueError as e:
            logger.error(f"Config normalization failed: {e}")
            return jsonify({"error": f"Invalid config format: {e}"}), 400

        # Step 2: Materialize variables if present
        if variables:
            logger.info("Processing dynamic variables in config")
            try:
                config = materialise_config({"config": config, "variables": variables})
                logger.info(f"Config after variable materialization: name={config.get('name', 'Unknown')}")
            except Exception as e:
                logger.error(f"Variable materialization failed: {e}")
                return jsonify({"error": f"Failed to process dynamic variables: {e}"}), 400

        # Step 3: Validate final config
        if not config.get("name"):
            return jsonify({"error": "Config must have a 'name' field"}), 400
        if not config.get("agents"):
            return jsonify({"error": "Config must have 'agents' field"}), 400

        # Step 4: Insert into queue and catalog
        logger.info(f"Inserting to queue: config={config['name']}, num_runs={num_runs}")
        simulation_id = simulation_queue.insert(config, num_runs)
        logger.info(f"Queue insert result: {simulation_id}")

        if simulation_id:
            logger.info(f"Inserting to catalog: id={simulation_id}, name={config['name']}, num_runs={num_runs}")
            catalog_result = simulation_catalog.insert(simulation_id, config["name"], num_runs)
            logger.info(f"Catalog insert result: {catalog_result}")

            if catalog_result:
                return jsonify({
                    "message": f"Successfully created simulation with ID: {simulation_id} and {num_runs} runs.",
                    "simulation_id": simulation_id,
                    "num_runs": num_runs
                }), 200
            else:
                logger.error(f"Catalog insertion failed for simulation {simulation_id}")
                return jsonify({"error": "Failed to insert simulation into catalog"}), 500
        else:
            logger.error("Queue insertion failed")
            return jsonify({"error": "Failed to insert simulation into queue"}), 500

    except Exception as e:
        logger.error(f"Unexpected error in create_simulation: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500