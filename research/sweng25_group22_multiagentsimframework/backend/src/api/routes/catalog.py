import os
from pymongo import MongoClient
from flask import Blueprint, jsonify

from db.simulation_catalog import SimulationCatalog

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_catalog = SimulationCatalog(mongo_client)

catalog_bp = Blueprint("catalog", __name__)

@catalog_bp.route("/catalog", methods=["GET"])
def create_simulation():
    return jsonify(simulation_catalog.get_all())