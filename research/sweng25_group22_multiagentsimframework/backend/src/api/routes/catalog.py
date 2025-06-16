import os
import json
from pymongo import MongoClient
from flask import Blueprint, jsonify

from db.simulation_catalog import SimulationCatalog

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_catalog = SimulationCatalog(mongo_client)

catalog_bp = Blueprint("catalog", __name__)

@catalog_bp.route("/catalog", methods=["GET"])
def create_simulation():
    return json.dumps(simulation_catalog.get_all(), indent=4)