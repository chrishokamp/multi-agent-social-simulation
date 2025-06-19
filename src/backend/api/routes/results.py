import os
from pymongo import MongoClient
from flask import Blueprint, request, jsonify

from db.simulation_results import SimulationResults
from db.simulation_catalog import SimulationCatalog

mongo_client = MongoClient(os.environ["DB_CONNECTION_STRING"])
simulation_results = SimulationResults(mongo_client)
simulation_catalog = SimulationCatalog(mongo_client)

results_bp = Blueprint("results", __name__)

@results_bp.route("/results", methods=["GET"])
def get_results():
    simulation_id = request.args.get("id")
    index = request.args.get("i")
    show_messages = False if request.args.get("show_messages") == "no" else True

    results = simulation_results.retrieve(simulation_id)
    if len(results) == 0:
        if simulation_catalog.exists(simulation_id):
            return jsonify({
                "id": simulation_id,
                "num_runs": 0,
                "runs": []
            }), 200
        return jsonify([]), 404
    
    if index:
        index = int(index)
        if index >= 0 and index < len(results):
            response = {
                "num_messages": len(results[index]["messages"]),
                "messages": results[index]["messages"],
                "output_variables": results[index]["output_variables"]
            }

            if not show_messages:
                del response["messages"]
            
            return jsonify(response), 200
        else:
            return jsonify({}), 404

    response = {
        "id": simulation_id,
        "num_runs": len(results),
        "runs": [] 
    }

    for result in results:
        run_result = {
            "num_messages": len(result["messages"]),
            "messages": result["messages"],
            "output_variables": result["output_variables"]
        }

        if not show_messages:
            del run_result["messages"]

        response["runs"].append(run_result)

    return jsonify(response), 200