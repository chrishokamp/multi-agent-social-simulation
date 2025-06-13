import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from flask import Flask, request, jsonify

from dotenv import load_dotenv
load_dotenv()

from OLD.util import database_api

app = Flask(__name__)

db = database_api.DataBaseAPI(os.environ['DB_CONNECTION_STRING'])

@app.route('/report/config')
def report_config():
    sim_id = request.args.get('id')
    
    if not sim_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400
    
    config_data = db.query_config(sim_id)

    if (not config_data):
        return jsonify({"error": "Requested data not in database"}), 400

    if (config_data["_id"]):
        del config_data["_id"]

    return jsonify(config_data)

def remove_chat_log(data):
    if isinstance(data, dict):
        data = {k: remove_chat_log(v) for k, v in data.items() if k != "chat_log"}
    elif isinstance(data, list):
        data = [remove_chat_log(item) for item in data]
    return data
 
@app.route('/report/output')
def report_output():
    sim_id = request.args.get('id')
    i = request.args.get('i')
    log = request.args.get('log', 'yes')
    log_bool = log == "yes"
    
    if not sim_id:
        return jsonify({"error": "Missing 'id' parameter"}), 400

    output_data = db.query_output(sim_id)

    if (not output_data):
        return jsonify({"error": "Requested data not in database"}), 400
    
    if (output_data["_id"]):
        del output_data["_id"]

    if (not log_bool):
        output_data = remove_chat_log(output_data)

    if (i):
        return jsonify(output_data["runs"][int(i)])

    return jsonify(output_data)