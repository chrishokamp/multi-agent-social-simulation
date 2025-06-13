import os
from pymongo import MongoClient
from flask import Blueprint, request, jsonify

from db.simulation_results import SimulationResults
from db.simulation_catalog import SimulationCatalog

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_results = SimulationResults(mongo_client)
simulation_catalog = SimulationCatalog(mongo_client)

del_results_bp = Blueprint("del_results", __name__)

@del_results_bp.route("/del_results", methods=["POST"])
def del_results():
    request_json = request.get_json()

    sim_id = request_json["id"]
    if not sim_id:
        return jsonify({"message": "id not given"}), 400

    result = simulation_results.delete(sim_id)
    if (result > 0):
        return jsonify({"message":"Successful deletion of " + str(result) + " items"}), 200

    return jsonify({"message": "Failed deletion"}), 400